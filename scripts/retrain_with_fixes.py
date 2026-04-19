#!/usr/bin/env python3
"""
Retrain SAC with Extrapolation Fixes
=====================================
Trains SAC under three configurations to isolate the effect of each fix:

    (a) v2 surrogates, NO variance penalty
        — tests whether wider-domain surrogates alone fix extrapolation
    (b) v1 surrogates, WITH variance penalty
        — tests whether the penalty alone steers the agent in-distribution
    (c) v2 surrogates, WITH variance penalty
        — belt-and-suspenders: both fixes combined

Also evaluates the ORIGINAL SAC agent on v2 surrogates to check whether
the old policy still looks good under the improved model.

Usage::

    python -m rl_dynamic_control.scripts.retrain_with_fixes \\
        --timesteps 500000 --eval-episodes 20

Run from the EURECHA root directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rl_dynamic_control.config import (
    RL_CFG,
    T_REACTOR_NOMINAL,
    P_REACTOR_NOMINAL,
)
from rl_dynamic_control.environment.methanol_plant_env import MethanolPlantEnv
from rl_dynamic_control.models.surrogate_manager import load_surrogates
from rl_dynamic_control.agents.sb3_agent import make_sb3_agent, train_sb3
from rl_dynamic_control.utils.variance_penalty import VariancePenaltyConfig

LOGGER = logging.getLogger("retrain_with_fixes")

_RL_DIR = Path(__file__).resolve().parent.parent
_SAVED = _RL_DIR / "saved_models"
_OUT_DIR = _RL_DIR / "outputs" / "extrapolation_fix"


# ======================================================================
# Helpers
# ======================================================================

def load_prices() -> np.ndarray:
    """Load GB electricity prices."""
    data_dir = _RL_DIR / "data"
    real_path = data_dir / "electricity_prices_real.csv"
    synth_path = data_dir / "electricity_prices.csv"
    csv_path = real_path if real_path.exists() else synth_path
    LOGGER.info("Loading prices from %s", csv_path.name)
    df = pd.read_csv(csv_path)
    return df["price_gbp_per_mwh"].values.astype(np.float32)


def make_env(
    prices: np.ndarray,
    surrogate_version: str = "v1",
    use_variance_penalty: bool = False,
    variance_penalty_config: Optional[VariancePenaltyConfig] = None,
) -> MethanolPlantEnv:
    """Create an environment with specified surrogate version and penalty."""
    surrogates = load_surrogates(version=surrogate_version)
    return MethanolPlantEnv(
        price_data=prices,
        surrogates=surrogates,
        episode_length=RL_CFG.episode_length,
        use_variance_penalty=use_variance_penalty,
        variance_penalty_config=variance_penalty_config,
    )


def evaluate_agent(
    model: Any,
    prices: np.ndarray,
    surrogate_version: str,
    use_variance_penalty: bool,
    var_config: Optional[VariancePenaltyConfig],
    n_episodes: int = 20,
    seed: int = RL_CFG.seed,
    is_rl: bool = True,
) -> Dict[str, Any]:
    """Run evaluation episodes and collect comprehensive statistics.

    Returns dict with: mean_reward, std_reward, mean_co2_util,
    mean_extrap_frac, per_step data (rewards, sigmas, penalties).
    """
    totals: List[float] = []
    co2s: List[float] = []
    extrps: List[float] = []
    all_step_data: List[Dict[str, float]] = []

    for ep in range(n_episodes):
        env = make_env(prices, surrogate_version, use_variance_penalty, var_config)
        obs, _ = env.reset(seed=seed + ep)
        ep_reward = 0.0

        for _ in range(RL_CFG.episode_length):
            if is_rl:
                action, _ = model.predict(obs, deterministic=True)
            else:
                # Baseline: constant full load
                action = np.array(
                    [1.0, T_REACTOR_NOMINAL, P_REACTOR_NOMINAL],
                    dtype=np.float32,
                )

            obs, r, term, trunc, info = env.step(action)
            ep_reward += r

            all_step_data.append({
                "episode": ep,
                "reward": r,
                "reward_before_penalty": info.get("reward_before_penalty", r),
                "gpr_sigma": info.get("gpr_sigma", 0.0),
                "variance_penalty": info.get("variance_penalty", 0.0),
                "load": info.get("state", {}).get("load", 0.0),
                "T": info.get("state", {}).get("T", 0.0),
                "P": info.get("state", {}).get("P", 0.0),
                "meoh_tph": info.get("meoh_tph", 0.0),
                "price": info.get("price", 0.0),
            })

            if term or trunc:
                break

        totals.append(ep_reward)
        co2s.append(env.mean_co2_utilisation())
        extrps.append(env.extrapolation_stats()["fraction"])

    return {
        "mean_reward": float(np.mean(totals)),
        "std_reward": float(np.std(totals)),
        "mean_co2_util": float(np.mean(co2s)),
        "mean_extrap_frac": float(np.mean(extrps)),
        "std_extrap_frac": float(np.std(extrps)),
        "episode_rewards": totals,
        "step_data": all_step_data,
    }


# ======================================================================
# Main pipeline
# ======================================================================

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=RL_CFG.total_timesteps,
                        help="Training timesteps per variant")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=RL_CFG.seed)
    parser.add_argument("--alpha", type=float, default=2.0,
                        help="Variance penalty alpha coefficient")
    parser.add_argument("--skip-training", action="store_true",
                        help="Only evaluate existing models (if saved)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_prices()

    var_config = VariancePenaltyConfig(
        alpha=args.alpha,
        sigma_threshold_mult=2.0,
        enabled=True,
    )

    # TensorBoard dir
    tb_dir = str(_RL_DIR / RL_CFG.tb_log_dir)

    results: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    # Variant (a): v2 surrogates, NO variance penalty
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Variant (a): v2 surrogates, NO variance penalty")
    print("=" * 60)

    save_path_a = _SAVED / "sac_extrap_fix_a"
    if not args.skip_training:
        env_a = make_env(prices, "v2", False, None)
        model_a = make_sb3_agent(env_a, algo="SAC", seed=args.seed,
                                 tb_log_dir=tb_dir)
        train_sb3(model_a, total_timesteps=args.timesteps,
                  save_path=save_path_a, tb_log_name="SAC_fix_a")
    else:
        from stable_baselines3 import SAC
        model_a = SAC.load(str(save_path_a))

    results["variant_a"] = evaluate_agent(
        model_a, prices, "v2", False, None,
        n_episodes=args.eval_episodes, seed=args.seed,
    )
    print(f"  Mean reward: {results['variant_a']['mean_reward']:.3f} "
          f"± {results['variant_a']['std_reward']:.3f}")
    print(f"  Extrap frac: {results['variant_a']['mean_extrap_frac']:.2%}")

    # ------------------------------------------------------------------
    # Variant (b): v1 surrogates, WITH variance penalty
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Variant (b): v1 surrogates, WITH variance penalty")
    print("=" * 60)

    save_path_b = _SAVED / "sac_extrap_fix_b"
    if not args.skip_training:
        env_b = make_env(prices, "v1", True, var_config)
        model_b = make_sb3_agent(env_b, algo="SAC", seed=args.seed,
                                 tb_log_dir=tb_dir)
        train_sb3(model_b, total_timesteps=args.timesteps,
                  save_path=save_path_b, tb_log_name="SAC_fix_b")
    else:
        from stable_baselines3 import SAC
        model_b = SAC.load(str(save_path_b))

    results["variant_b"] = evaluate_agent(
        model_b, prices, "v1", True, var_config,
        n_episodes=args.eval_episodes, seed=args.seed,
    )
    print(f"  Mean reward: {results['variant_b']['mean_reward']:.3f} "
          f"± {results['variant_b']['std_reward']:.3f}")
    print(f"  Extrap frac: {results['variant_b']['mean_extrap_frac']:.2%}")

    # ------------------------------------------------------------------
    # Variant (c): v2 surrogates, WITH variance penalty
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Variant (c): v2 surrogates, WITH variance penalty")
    print("=" * 60)

    save_path_c = _SAVED / "sac_extrap_fix_c"
    if not args.skip_training:
        env_c = make_env(prices, "v2", True, var_config)
        model_c = make_sb3_agent(env_c, algo="SAC", seed=args.seed,
                                 tb_log_dir=tb_dir)
        train_sb3(model_c, total_timesteps=args.timesteps,
                  save_path=save_path_c, tb_log_name="SAC_fix_c")
    else:
        from stable_baselines3 import SAC
        model_c = SAC.load(str(save_path_c))

    results["variant_c"] = evaluate_agent(
        model_c, prices, "v2", True, var_config,
        n_episodes=args.eval_episodes, seed=args.seed,
    )
    print(f"  Mean reward: {results['variant_c']['mean_reward']:.3f} "
          f"± {results['variant_c']['std_reward']:.3f}")
    print(f"  Extrap frac: {results['variant_c']['mean_extrap_frac']:.2%}")

    # ------------------------------------------------------------------
    # Original SAC on v1 (reference)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Original SAC on v1 surrogates (reference)")
    print("=" * 60)

    orig_path = _SAVED / "sac_methanol.zip"
    if orig_path.exists():
        from stable_baselines3 import SAC
        model_orig = SAC.load(str(orig_path))
        results["original_v1"] = evaluate_agent(
            model_orig, prices, "v1", False, None,
            n_episodes=args.eval_episodes, seed=args.seed,
        )
        print(f"  Mean reward: {results['original_v1']['mean_reward']:.3f} "
              f"± {results['original_v1']['std_reward']:.3f}")
        print(f"  Extrap frac: {results['original_v1']['mean_extrap_frac']:.2%}")

        # Also evaluate original SAC on v2 surrogates
        print("\n" + "=" * 60)
        print("Original SAC on v2 surrogates (zero-shot transfer)")
        print("=" * 60)
        results["original_on_v2"] = evaluate_agent(
            model_orig, prices, "v2", False, None,
            n_episodes=args.eval_episodes, seed=args.seed,
        )
        print(f"  Mean reward: {results['original_on_v2']['mean_reward']:.3f} "
              f"± {results['original_on_v2']['std_reward']:.3f}")
        print(f"  Extrap frac: {results['original_on_v2']['mean_extrap_frac']:.2%}")
    else:
        LOGGER.warning("Original SAC model not found at %s", orig_path)

    # ------------------------------------------------------------------
    # Baseline: constant full load
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Baseline: constant full load (v1 surrogates)")
    print("=" * 60)
    results["baseline"] = evaluate_agent(
        None, prices, "v1", False, None,
        n_episodes=args.eval_episodes, seed=args.seed, is_rl=False,
    )
    print(f"  Mean reward: {results['baseline']['mean_reward']:.3f}")
    print(f"  Extrap frac: {results['baseline']['mean_extrap_frac']:.2%}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    # Summary CSV
    rows = []
    for name, res in results.items():
        rows.append({
            "variant": name,
            "mean_reward": res["mean_reward"],
            "std_reward": res["std_reward"],
            "mean_co2_util": res["mean_co2_util"],
            "mean_extrap_frac": res["mean_extrap_frac"],
            "std_extrap_frac": res.get("std_extrap_frac", 0.0),
        })
    df = pd.DataFrame(rows)
    csv_path = _OUT_DIR / "extrapolation_fix_results.csv"
    df.to_csv(csv_path, index=False)

    # Step-level data for plotting
    for name, res in results.items():
        if res.get("step_data"):
            step_df = pd.DataFrame(res["step_data"])
            step_df.to_csv(_OUT_DIR / f"step_data_{name}.csv", index=False)

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("EXTRAPOLATION FIX — RESULTS SUMMARY")
    print("=" * 78)
    header = (f"{'Variant':25s} {'Mean Reward':>12s} {'Std':>8s} "
              f"{'CO2 Util':>10s} {'Extrap %':>10s}")
    print(header)
    print("-" * 78)
    for _, r in df.iterrows():
        print(f"  {r['variant']:23s} {r['mean_reward']:12.3f} {r['std_reward']:8.3f} "
              f"{r['mean_co2_util']:10.3f} {100*r['mean_extrap_frac']:9.2f}%")
    print("=" * 78)
    print(f"\nResults saved to: {_OUT_DIR}")
    print(f"  Summary CSV:   {csv_path}")
    print(f"  Step-level data: {_OUT_DIR}/step_data_*.csv")
    print(f"\nPlot with: python -m rl_dynamic_control.scripts.plot_extrapolation_fix")

    return 0


if __name__ == "__main__":
    sys.exit(main())
