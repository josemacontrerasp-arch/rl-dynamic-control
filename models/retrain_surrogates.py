#!/usr/bin/env python3
"""
Retrain GPR Surrogates on Wider Domain (Approach A)
====================================================
Addresses the GPR extrapolation problem: SAC operates in regions the
original 500-point LHS didn't cover (72-98% of steps flagged).

Strategy
--------
1. Load the trained SAC agent and run evaluation episodes to log the
   full state-action trajectory the agent actually visits.
2. Compute the bounding box of the agent's operating region vs the
   original LHS training domain.
3. Generate a new LHS sample biased toward the agent's operating region
   and combined with the original training data.
4. Use existing surrogates for pseudo-labelling (limitation: circular —
   the real fix would re-run Aspen for the new points).
5. Retrain GPR surrogates (Matern 5/2) on the combined dataset.
6. Save as versioned ``models/surrogates_v2/`` with full metrics.

Usage::

    python -m rl_dynamic_control.models.retrain_surrogates \\
        --episodes 50 --new-points 300 --seed 42

Run from the EURECHA root directory.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import ConvexHull
from scipy.stats.qmc import LatinHypercube
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Ensure EURECHA root is importable
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rl_dynamic_control.config import (
    DECISION_VAR_BOUNDS,
    PLANT_BOUNDS,
    RL_CFG,
    SIMPLIFIED_PROCESS_MODEL,
    T_REACTOR_NOMINAL,
    P_REACTOR_NOMINAL,
    MW_H2,
)
from rl_dynamic_control.environment.methanol_plant_env import MethanolPlantEnv
from rl_dynamic_control.models.surrogates import PlantSurrogates

LOGGER = logging.getLogger("retrain_surrogates")

# Directories
_RL_DIR = Path(__file__).resolve().parent.parent
_SAVED_MODELS = _RL_DIR / "saved_models"
_V2_DIR = _SAVED_MODELS / "surrogates_v2"


# ======================================================================
# Step 1: Trajectory logging
# ======================================================================

def collect_agent_trajectories(
    n_episodes: int = 50,
    seed: int = RL_CFG.seed,
    model_path: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Run the trained SAC agent and log every state-action pair.

    Returns
    -------
    dict with keys:
        load, T, P, h2_feed_kmol, h2_kg_hr, meoh_tph, elec_mw,
        price, reward, gpr_sigma
    Each value is a 1-D array of length (n_episodes × episode_length).
    """
    from stable_baselines3 import SAC

    model_path = model_path or str(_SAVED_MODELS / "sac_methanol.zip")
    LOGGER.info("Loading SAC model from %s", model_path)
    model = SAC.load(model_path)

    surrogates = PlantSurrogates(use_gpr=True)

    # Use default GB prices
    import pandas as pd
    data_dir = _RL_DIR / "data"
    real_path = data_dir / "electricity_prices_real.csv"
    synth_path = data_dir / "electricity_prices.csv"
    csv_path = real_path if real_path.exists() else synth_path
    prices = pd.read_csv(csv_path)["price_gbp_per_mwh"].values.astype(np.float32)

    # Collectors
    records: Dict[str, List[float]] = {
        "load": [], "T": [], "P": [],
        "h2_feed_kmol": [], "h2_kg_hr": [],
        "meoh_tph": [], "elec_mw": [],
        "price": [], "reward": [], "gpr_sigma": [],
    }

    for ep in range(n_episodes):
        env = MethanolPlantEnv(price_data=prices, surrogates=surrogates)
        obs, _ = env.reset(seed=seed + ep)

        for _ in range(RL_CFG.episode_length):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)

            state = info.get("state", {})
            load = state.get("load", 0.0)
            T = state.get("T", T_REACTOR_NOMINAL)
            P = state.get("P", P_REACTOR_NOMINAL)

            # Compute derived quantities
            h2_kg_hr = surrogates.predict_h2_production(load)
            h2_kmol_hr = h2_kg_hr / MW_H2

            records["load"].append(load)
            records["T"].append(T)
            records["P"].append(P)
            records["h2_feed_kmol"].append(h2_kmol_hr)
            records["h2_kg_hr"].append(h2_kg_hr)
            records["meoh_tph"].append(info.get("meoh_tph", 0.0))
            records["elec_mw"].append(info.get("elec_mw", 0.0))
            records["price"].append(info.get("price", 0.0))
            records["reward"].append(reward)

            # GPR predictive sigma at this point
            sigma = _get_gpr_sigma(surrogates, load, T, P)
            records["gpr_sigma"].append(sigma)

            if term or trunc:
                break

    return {k: np.array(v) for k, v in records.items()}


def _get_gpr_sigma(
    surrogates: PlantSurrogates,
    load: float,
    T: float,
    P: float,
) -> float:
    """Extract the GPR predictive std for the MeOH surrogate."""
    if not surrogates._gpr_loaded or surrogates._gp_meoh is None:
        return 0.0
    h2_kg_hr = surrogates.predict_h2_production(load)
    h2_kmol = np.clip(h2_kg_hr / MW_H2,
                      DECISION_VAR_BOUNDS[3, 0],
                      DECISION_VAR_BOUNDS[3, 1])
    X = np.array([[h2_kmol, T, P]])
    if surrogates._meoh_mu is not None:
        X = (X - surrogates._meoh_mu) / surrogates._meoh_sig
    try:
        _, sigma = surrogates._gp_meoh.predict(X, return_std=True)
        return float(sigma[0])
    except Exception:
        return 0.0


# ======================================================================
# Step 2-3: Domain gap analysis
# ======================================================================

def analyse_domain_gap(
    trajectories: Dict[str, np.ndarray],
) -> Dict[str, any]:
    """Compare agent trajectory bounding box vs original LHS bounds.

    Works in the 3-D surrogate input space: (h2_feed_kmol, T, P).

    Returns
    -------
    dict with keys: agent_bounds, lhs_bounds, gap_report (str),
                    agent_points (N×3 array)
    """
    h2 = trajectories["h2_feed_kmol"]
    T = trajectories["T"]
    P = trajectories["P"]

    agent_pts = np.column_stack([h2, T, P])

    # Original LHS bounds from DECISION_VAR_BOUNDS
    lhs_bounds = np.array([
        [DECISION_VAR_BOUNDS[3, 0], DECISION_VAR_BOUNDS[3, 1]],  # F_H2
        [DECISION_VAR_BOUNDS[0, 0], DECISION_VAR_BOUNDS[0, 1]],  # T
        [DECISION_VAR_BOUNDS[1, 0], DECISION_VAR_BOUNDS[1, 1]],  # P
    ])

    agent_bounds = np.array([
        [h2.min(), h2.max()],
        [T.min(), T.max()],
        [P.min(), P.max()],
    ])

    # Fraction of agent points outside original LHS bounding box
    outside = (
        (h2 < lhs_bounds[0, 0]) | (h2 > lhs_bounds[0, 1]) |
        (T < lhs_bounds[1, 0]) | (T > lhs_bounds[1, 1]) |
        (P < lhs_bounds[2, 0]) | (P > lhs_bounds[2, 1])
    )
    frac_outside = float(outside.mean())

    # Per-dimension analysis
    dim_names = ["H2_feed_kmol", "T_reactor_C", "P_reactor_bar"]
    lines = ["Domain Gap Analysis", "=" * 50]
    for i, name in enumerate(dim_names):
        lb, ub = lhs_bounds[i]
        ab_lo, ab_hi = agent_bounds[i]
        lines.append(
            f"  {name:20s}: LHS [{lb:.1f}, {ub:.1f}]  "
            f"Agent [{ab_lo:.1f}, {ab_hi:.1f}]  "
            f"gap_lo={lb - ab_lo:+.1f}  gap_hi={ab_hi - ub:+.1f}"
        )
    lines.append(f"  Points outside LHS bbox: {frac_outside:.1%}")
    report = "\n".join(lines)

    return {
        "agent_bounds": agent_bounds,
        "lhs_bounds": lhs_bounds,
        "gap_report": report,
        "agent_points": agent_pts,
        "frac_outside_bbox": frac_outside,
    }


# ======================================================================
# Step 4-5: Augmented LHS + pseudo-labelling + GPR retraining
# ======================================================================

def generate_original_training_data(
    n: int = 500,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Regenerate the original LHS training set (same as surrogate_optimization.py).

    Returns X (N×5), lcom, util, production arrays.
    """
    U = LatinHypercube(d=5, seed=seed).random(n)
    X = DECISION_VAR_BOUNDS[:, 0] + U * (DECISION_VAR_BOUNDS[:, 1] - DECISION_VAR_BOUNDS[:, 0])

    N = X.shape[0]
    lcom = np.empty(N)
    util = np.empty(N)
    prod = np.empty(N)
    for i in range(N):
        lcom[i], util[i], prod[i] = SIMPLIFIED_PROCESS_MODEL(X[i])

    return X, lcom, util, prod


def generate_augmented_lhs(
    trajectories: Dict[str, np.ndarray],
    gap_info: Dict,
    n_new: int = 300,
    seed: int = 123,
) -> np.ndarray:
    """Generate new LHS points biased toward the agent's operating region.

    The new points are generated in the 5-D decision variable space
    [T, P, f_purge, F_H2, P_col] with bounds extended to cover the
    agent's actual operating envelope.

    Parameters
    ----------
    trajectories : dict
        From ``collect_agent_trajectories()``.
    gap_info : dict
        From ``analyse_domain_gap()``.
    n_new : int
        Number of new points to generate.
    seed : int
        Random seed.

    Returns
    -------
    X_new : ndarray, shape (n_new, 5)
        New 5-D decision variable points.
    """
    agent_bounds = gap_info["agent_bounds"]  # (3, 2) for [H2, T, P]

    # Build extended bounds in 5-D space
    # Original: [T, P, f_purge, F_H2, P_col]
    ext_bounds = DECISION_VAR_BOUNDS.copy()

    # Extend T bounds (dim 0) to cover agent operating region
    agent_T_lo, agent_T_hi = agent_bounds[1]
    ext_bounds[0, 0] = min(ext_bounds[0, 0], agent_T_lo - 5.0)
    ext_bounds[0, 1] = max(ext_bounds[0, 1], agent_T_hi + 5.0)

    # Extend P bounds (dim 1)
    agent_P_lo, agent_P_hi = agent_bounds[2]
    ext_bounds[1, 0] = min(ext_bounds[1, 0], agent_P_lo - 2.0)
    ext_bounds[1, 1] = max(ext_bounds[1, 1], agent_P_hi + 2.0)

    # Extend F_H2 bounds (dim 3)
    agent_H2_lo, agent_H2_hi = agent_bounds[0]
    ext_bounds[3, 0] = min(ext_bounds[3, 0], agent_H2_lo - 50.0)
    ext_bounds[3, 1] = max(ext_bounds[3, 1], agent_H2_hi + 50.0)

    # Keep f_purge and P_col at original bounds (agent doesn't control them)
    # but ensure physical feasibility
    ext_bounds = np.clip(ext_bounds, 1e-6, None)

    LOGGER.info("Extended bounds:\n%s", ext_bounds)

    # Strategy: 70% of new points in the agent's operating region,
    # 30% in the extended envelope for smooth interpolation
    n_agent_region = int(0.7 * n_new)
    n_envelope = n_new - n_agent_region

    rng = np.random.default_rng(seed)

    # --- Agent-region points ---
    # Use the agent trajectory distribution to define a tighter box
    agent_T = trajectories["T"]
    agent_P = trajectories["P"]
    agent_H2 = trajectories["h2_feed_kmol"]

    # Sample T, P, H2 from the agent distribution (with slight jitter)
    agent_region_bounds = np.array([
        [np.percentile(agent_T, 2), np.percentile(agent_T, 98)],      # T
        [np.percentile(agent_P, 2), np.percentile(agent_P, 98)],      # P
        [DECISION_VAR_BOUNDS[2, 0], DECISION_VAR_BOUNDS[2, 1]],       # f_purge
        [np.percentile(agent_H2, 2), np.percentile(agent_H2, 98)],    # F_H2
        [DECISION_VAR_BOUNDS[4, 0], DECISION_VAR_BOUNDS[4, 1]],       # P_col
    ])

    U_agent = LatinHypercube(d=5, seed=seed).random(n_agent_region)
    X_agent = (agent_region_bounds[:, 0]
               + U_agent * (agent_region_bounds[:, 1] - agent_region_bounds[:, 0]))

    # --- Envelope points ---
    U_env = LatinHypercube(d=5, seed=seed + 1).random(n_envelope)
    X_env = ext_bounds[:, 0] + U_env * (ext_bounds[:, 1] - ext_bounds[:, 0])

    X_new = np.vstack([X_agent, X_env])
    LOGGER.info("Generated %d new points (%d agent-region + %d envelope)",
                n_new, n_agent_region, n_envelope)

    return X_new


def pseudo_label(
    X_new: np.ndarray,
    surrogates: PlantSurrogates,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate pseudo-labels for new points using EXISTING surrogates.

    .. warning::
        This is circular: we're using the old surrogates (which may be
        unreliable in the agent's operating region) to generate training
        data for the new surrogates.  The real fix would be running these
        points through Aspen Plus.  This approach smooths the surrogate
        surface in the agent's region but cannot correct systematic bias.

    For H2 production and energy consumption (1-D functions of load),
    we use the analytical fallback which is exact.

    For methanol output (the 3-D GPR), we use the simplified process
    model directly — this is the ground truth for pseudo-labels since
    the GPRs were trained on it.

    Parameters
    ----------
    X_new : ndarray, shape (N, 5)
        [T, P, f_purge, F_H2, P_col]

    Returns
    -------
    lcom, util, production : 1-D arrays of length N
    """
    N = X_new.shape[0]
    lcom = np.empty(N)
    util = np.empty(N)
    prod = np.empty(N)

    for i in range(N):
        lcom[i], util[i], prod[i] = SIMPLIFIED_PROCESS_MODEL(X_new[i])

    LOGGER.info(
        "Pseudo-labelled %d points via simplified_process_model "
        "(NOTE: this is the analytical model, not Aspen — see docstring)",
        N,
    )
    return lcom, util, prod


def retrain_gprs(
    X_combined: np.ndarray,
    lcom_combined: np.ndarray,
    util_combined: np.ndarray,
    n_restarts: int = 5,
    alpha: float = 1e-6,
) -> Tuple[GaussianProcessRegressor, GaussianProcessRegressor,
           np.ndarray, np.ndarray]:
    """Retrain GPR surrogates on the combined (original + augmented) dataset.

    Uses the same kernel structure as the originals:
    ConstantKernel × Matern(nu=2.5).

    Returns
    -------
    gp_lcom, gp_util, X_mu, X_sig
    """
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.ones(5), length_scale_bounds=(1e-2, 1e2), nu=2.5,
    )

    X_mu = X_combined.mean(axis=0)
    X_sig = X_combined.std(axis=0)
    X_sig[X_sig < 1e-10] = 1.0  # avoid division by zero
    X_n = (X_combined - X_mu) / X_sig

    LOGGER.info("Training GP-LCOM on %d points ...", len(X_combined))
    gp_lcom = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=n_restarts,
        alpha=alpha, normalize_y=True, random_state=0,
    )
    gp_lcom.fit(X_n, lcom_combined)

    LOGGER.info("Training GP-Util on %d points ...", len(X_combined))
    gp_util = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=n_restarts,
        alpha=alpha, normalize_y=True, random_state=0,
    )
    gp_util.fit(X_n, util_combined)

    return gp_lcom, gp_util, X_mu, X_sig


# ======================================================================
# Step 6: Metrics and saving
# ======================================================================

def evaluate_surrogates(
    gp_lcom: GaussianProcessRegressor,
    gp_util: GaussianProcessRegressor,
    X_mu: np.ndarray,
    X_sig: np.ndarray,
    X_test: np.ndarray,
    lcom_test: np.ndarray,
    util_test: np.ndarray,
    label: str = "test",
) -> Dict[str, float]:
    """Compute R² and RMSE on a test set."""
    X_n = (X_test - X_mu) / X_sig
    pred_lcom = gp_lcom.predict(X_n)
    pred_util = gp_util.predict(X_n)

    metrics = {
        f"{label}_lcom_r2": float(r2_score(lcom_test, pred_lcom)),
        f"{label}_lcom_rmse": float(np.sqrt(mean_squared_error(lcom_test, pred_lcom))),
        f"{label}_util_r2": float(r2_score(util_test, pred_util)),
        f"{label}_util_rmse": float(np.sqrt(mean_squared_error(util_test, pred_util))),
    }
    return metrics


def save_v2_surrogates(
    gp_lcom: GaussianProcessRegressor,
    gp_util: GaussianProcessRegressor,
    X_mu: np.ndarray,
    X_sig: np.ndarray,
    metrics: Dict[str, float],
    X_train: np.ndarray,
) -> Path:
    """Save retrained surrogates as v2 with metadata."""
    _V2_DIR.mkdir(parents=True, exist_ok=True)

    for name, model in [("gp_lcom_5d.pkl", gp_lcom), ("gp_util_5d.pkl", gp_util)]:
        data = {
            "model": model,
            "X_mu": X_mu,
            "X_sig": X_sig,
            "version": "v2",
            "n_train": len(X_train),
            "metrics": metrics,
        }
        path = _V2_DIR / name
        with open(path, "wb") as f:
            pickle.dump(data, f)
        LOGGER.info("Saved %s", path)

    # Also create a 3-D MeOH-output surrogate for the environment
    # The env uses (h2_feed_kmol, T, P) → meoh_tph
    # We train this on the production output from the process model
    _train_meoh_3d_surrogate(X_train, X_mu, X_sig)

    # Save metadata
    meta = {
        "version": "v2",
        "n_original": 500,
        "n_augmented": len(X_train) - 500,
        "n_total": len(X_train),
        "metrics": metrics,
        "bounds_original": DECISION_VAR_BOUNDS.tolist(),
        "X_mu": X_mu.tolist(),
        "X_sig": X_sig.tolist(),
        "note": (
            "Augmented training set includes pseudo-labels from the "
            "simplified process model (NOT Aspen). The agent-region "
            "points smooth the GPR surface but cannot correct systematic "
            "bias in the analytical model."
        ),
    }
    import json
    with open(_V2_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    LOGGER.info("v2 surrogates saved to %s", _V2_DIR)
    return _V2_DIR


def _train_meoh_3d_surrogate(
    X_5d: np.ndarray,
    X_mu_5d: np.ndarray,
    X_sig_5d: np.ndarray,
) -> None:
    """Train a 3-D GPR for MeOH output: (F_H2, T, P) → production (t/hr).

    This is what the environment actually queries via
    ``PlantSurrogates.predict_methanol_output()``.
    """
    # Extract the 3 relevant dims: T (col 0), P (col 1), F_H2 (col 3)
    X_3d = X_5d[:, [3, 0, 1]]  # → [F_H2, T, P]
    X_3d_mu = X_3d.mean(axis=0)
    X_3d_sig = X_3d.std(axis=0)
    X_3d_sig[X_3d_sig < 1e-10] = 1.0
    X_3d_n = (X_3d - X_3d_mu) / X_3d_sig

    # Labels: methanol production in t/hr
    N = X_5d.shape[0]
    meoh_tph = np.empty(N)
    for i in range(N):
        _, _, prod_tpy = SIMPLIFIED_PROCESS_MODEL(X_5d[i])
        meoh_tph[i] = prod_tpy / 8000.0

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.ones(3), length_scale_bounds=(1e-2, 1e2), nu=2.5,
    )
    gp_meoh = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=5,
        alpha=1e-6, normalize_y=True, random_state=0,
    )
    gp_meoh.fit(X_3d_n, meoh_tph)

    data = {
        "model": gp_meoh,
        "X_mu": X_3d_mu,
        "X_sig": X_3d_sig,
        "version": "v2",
        "n_train": N,
    }
    path = _V2_DIR / "gp_meoh_output.pkl"
    with open(path, "wb") as f:
        pickle.dump(data, f)
    LOGGER.info("Saved 3-D MeOH surrogate → %s (N=%d)", path, N)

    # Also retrain H2 production and energy consumption (1-D)
    _train_1d_surrogates(X_5d)


def _train_1d_surrogates(X_5d: np.ndarray) -> None:
    """Retrain the 1-D surrogates (H2 production, energy consumption).

    These are functions of electrolyser load only. Since we don't have
    load in X_5d, we generate a grid of load fractions and use the
    analytical model (which is exact for these).
    """
    from rl_dynamic_control.config import ELECTROLYSER_MW, ELEC_SPEC_POWER, W_TOTAL_COMP

    loads = np.linspace(0.0, 1.0, 200)

    # H2 production (kg/hr)
    full_h2_kg_hr = ELECTROLYSER_MW * 1e3 / ELEC_SPEC_POWER
    h2_prod = np.array([
        load * full_h2_kg_hr * (1.0 - 0.05 * (1.0 - load) ** 2)
        for load in loads
    ])

    # Energy consumption (MW)
    energy = np.array([
        load * ELECTROLYSER_MW + W_TOTAL_COMP * max(load, 0.1)
        for load in loads
    ])

    kernel_1d = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.ones(1), length_scale_bounds=(1e-2, 1e2), nu=2.5,
    )

    for name, y_data in [("gp_h2_production.pkl", h2_prod),
                          ("gp_energy_consumption.pkl", energy)]:
        gp = GaussianProcessRegressor(
            kernel=kernel_1d, n_restarts_optimizer=5,
            alpha=1e-8, normalize_y=True, random_state=0,
        )
        X = loads.reshape(-1, 1)
        gp.fit(X, y_data)
        data = {"model": gp, "version": "v2", "n_train": len(loads)}
        path = _V2_DIR / name
        with open(path, "wb") as f:
            pickle.dump(data, f)
        LOGGER.info("Saved 1-D surrogate → %s", path)


def print_comparison_table(
    metrics_v1: Dict[str, float],
    metrics_v2: Dict[str, float],
) -> None:
    """Print a formatted comparison of old vs new surrogate metrics."""
    print("\n" + "=" * 72)
    print("SURROGATE COMPARISON: v1 (original) vs v2 (augmented)")
    print("=" * 72)
    header = f"{'Metric':35s} {'v1':>12s} {'v2':>12s} {'Change':>12s}"
    print(header)
    print("-" * 72)

    for key in sorted(set(list(metrics_v1.keys()) + list(metrics_v2.keys()))):
        v1_val = metrics_v1.get(key, float("nan"))
        v2_val = metrics_v2.get(key, float("nan"))
        change = v2_val - v1_val
        sign = "+" if change > 0 else ""
        print(f"  {key:33s} {v1_val:12.4f} {v2_val:12.4f} {sign}{change:11.4f}")

    print("=" * 72)


# ======================================================================
# Main pipeline
# ======================================================================

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Retrain GPR surrogates on wider domain (Approach A)",
    )
    parser.add_argument("--episodes", type=int, default=50,
                        help="Number of evaluation episodes for trajectory logging")
    parser.add_argument("--new-points", type=int, default=300,
                        help="Number of new LHS points to generate")
    parser.add_argument("--seed", type=int, default=RL_CFG.seed)
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to trained SAC model (default: saved_models/sac_methanol.zip)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # --- Step 1: Collect trajectories ---
    print("\n[1/7] Collecting agent trajectories ...")
    traj = collect_agent_trajectories(
        n_episodes=args.episodes,
        seed=args.seed,
        model_path=args.model_path,
    )
    n_steps = len(traj["load"])
    print(f"      Logged {n_steps} state-action pairs "
          f"({args.episodes} episodes × {RL_CFG.episode_length} steps)")

    # --- Step 2: Domain gap analysis ---
    print("\n[2/7] Analysing domain gap ...")
    gap = analyse_domain_gap(traj)
    print(gap["gap_report"])

    # --- Step 3: Generate original training data ---
    print("\n[3/7] Regenerating original LHS training data (500 pts) ...")
    X_orig, lcom_orig, util_orig, prod_orig = generate_original_training_data()
    print(f"      LCOM range: [{lcom_orig.min():.0f}, {lcom_orig.max():.0f}]")

    # --- Step 4: Generate augmented LHS ---
    print(f"\n[4/7] Generating {args.new_points} new LHS points ...")
    X_new = generate_augmented_lhs(traj, gap, n_new=args.new_points, seed=args.seed + 100)

    # --- Step 5: Pseudo-label ---
    print("\n[5/7] Pseudo-labelling new points ...")
    lcom_new, util_new, prod_new = pseudo_label(X_new, PlantSurrogates(use_gpr=True))

    # Combine datasets
    X_combined = np.vstack([X_orig, X_new])
    lcom_combined = np.concatenate([lcom_orig, lcom_new])
    util_combined = np.concatenate([util_orig, util_new])

    print(f"      Combined dataset: {len(X_combined)} points "
          f"({len(X_orig)} original + {len(X_new)} new)")

    # --- Step 6: Retrain GPRs ---
    print("\n[6/7] Retraining GPR surrogates ...")
    gp_lcom_v2, gp_util_v2, X_mu_v2, X_sig_v2 = retrain_gprs(
        X_combined, lcom_combined, util_combined,
    )

    # --- Step 7: Evaluate and save ---
    print("\n[7/7] Evaluating and saving ...")

    # Generate a fresh test set
    X_test, lcom_test, util_test, _ = generate_original_training_data(n=100, seed=77)

    # v2 metrics on standard test set
    metrics_v2 = evaluate_surrogates(
        gp_lcom_v2, gp_util_v2, X_mu_v2, X_sig_v2,
        X_test, lcom_test, util_test, label="std_test",
    )

    # v2 metrics on agent-trajectory region
    # Create a test set from the agent's region
    n_agent_test = min(200, n_steps)
    idx = np.random.default_rng(77).choice(n_steps, size=n_agent_test, replace=False)
    X_agent_5d = np.column_stack([
        traj["T"][idx],
        traj["P"][idx],
        np.full(n_agent_test, 0.005),  # f_purge (base case)
        traj["h2_feed_kmol"][idx],
        np.full(n_agent_test, 1.5),    # P_col (base case)
    ])
    lcom_agent, util_agent, _ = pseudo_label(X_agent_5d, PlantSurrogates(use_gpr=True))
    metrics_v2_agent = evaluate_surrogates(
        gp_lcom_v2, gp_util_v2, X_mu_v2, X_sig_v2,
        X_agent_5d, lcom_agent, util_agent, label="agent_region",
    )
    metrics_v2.update(metrics_v2_agent)

    # Load v1 surrogates for comparison
    metrics_v1: Dict[str, float] = {}
    try:
        v1_lcom_path = _SAVED_MODELS / "gp_lcom_5d.pkl"
        v1_util_path = _SAVED_MODELS / "gp_util_5d.pkl"
        if v1_lcom_path.exists() and v1_util_path.exists():
            with open(v1_lcom_path, "rb") as f:
                v1_lcom_data = pickle.load(f)
            with open(v1_util_path, "rb") as f:
                v1_util_data = pickle.load(f)

            v1_mu = v1_lcom_data.get("X_mu", X_mu_v2)
            v1_sig = v1_lcom_data.get("X_sig", X_sig_v2)

            metrics_v1 = evaluate_surrogates(
                v1_lcom_data["model"], v1_util_data["model"],
                v1_mu, v1_sig,
                X_test, lcom_test, util_test, label="std_test",
            )
            metrics_v1_agent = evaluate_surrogates(
                v1_lcom_data["model"], v1_util_data["model"],
                v1_mu, v1_sig,
                X_agent_5d, lcom_agent, util_agent, label="agent_region",
            )
            metrics_v1.update(metrics_v1_agent)
    except Exception as e:
        LOGGER.warning("Could not evaluate v1 surrogates: %s", e)

    # Save
    save_v2_surrogates(gp_lcom_v2, gp_util_v2, X_mu_v2, X_sig_v2,
                       metrics_v2, X_combined)

    # Print comparison
    if metrics_v1:
        print_comparison_table(metrics_v1, metrics_v2)
    else:
        print("\nv2 Surrogate Metrics:")
        for k, v in sorted(metrics_v2.items()):
            print(f"  {k}: {v:.4f}")

    print(f"\nv2 surrogates saved to: {_V2_DIR}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
