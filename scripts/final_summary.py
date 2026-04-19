#!/usr/bin/env python3
"""
Final Summary — Definitive Results Table
=========================================
Runs the COMPLETE evaluation suite: all agents (original SAC, fix
variants a/b/c, Q-learning, PPO, baselines) on both GB and NL price
data.  Generates a single summary table with columns:

    agent, price_data, mean_reward, std, co2_util, gpr_extrap_rate, vs_baseline

This is the definitive results table for the EURECHA RL project.

Usage::

    python -m rl_dynamic_control.scripts.final_summary \\
        --episodes 20

Run from the EURECHA root directory.
"""

from __future__ import annotations

import argparse
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
from rl_dynamic_control.environment.methanol_plant_env import (
    MethanolPlantEnv,
    DiscretizedMethanolEnv,
)
from rl_dynamic_control.models.surrogate_manager import load_surrogates, v2_available
from rl_dynamic_control.agents.q_learning import QLearningAgent
from rl_dynamic_control.utils.variance_penalty import VariancePenaltyConfig

LOGGER = logging.getLogger("final_summary")

_RL_DIR = Path(__file__).resolve().parent.parent
_SAVED = _RL_DIR / "saved_models"
_OUT_DIR = _RL_DIR / "outputs"


# ======================================================================
# Evaluation helpers
# ======================================================================

def load_prices(dataset: str = "gb") -> np.ndarray:
    """Load price data by name."""
    data_dir = _RL_DIR / "data"
    if dataset in ("gb", "synthetic_gb"):
        real_path = data_dir / "electricity_prices_real.csv"
        synth_path = data_dir / "electricity_prices.csv"
        csv_path = real_path if real_path.exists() else synth_path
        df = pd.read_csv(csv_path)
        return df["price_gbp_per_mwh"].values.astype(np.float32)
    elif dataset in ("nl", "entso_nl"):
        nl_path = data_dir / "entso_nl_2023_2024.csv"
        if nl_path.exists():
            df = pd.read_csv(nl_path)
            col = [c for c in df.columns if "price" in c.lower()][0]
            return df[col].values.astype(np.float32)
    raise FileNotFoundError(f"Price dataset '{dataset}' not found in {data_dir}")


def evaluate_agent(
    model: Any,
    prices: np.ndarray,
    surr_version: str,
    n_episodes: int,
    seed: int,
    agent_type: str = "sb3",
    use_var_penalty: bool = False,
    var_config: Optional[VariancePenaltyConfig] = None,
) -> Dict[str, float]:
    """Evaluate an agent and return summary statistics."""
    surrogates = load_surrogates(version=surr_version)
    totals, co2s, extrps = [], [], []

    for ep in range(n_episodes):
        env = MethanolPlantEnv(
            price_data=prices,
            surrogates=surrogates,
            episode_length=RL_CFG.episode_length,
            use_variance_penalty=use_var_penalty,
            variance_penalty_config=var_config,
        )

        if agent_type == "q_learning":
            disc_env = DiscretizedMethanolEnv(env)
            obs, _ = disc_env.reset(seed=seed + ep)
            ep_reward = 0.0
            for _ in range(RL_CFG.episode_length):
                action = model.select_action(obs, greedy=True)
                obs, r, term, trunc, info = disc_env.step(action)
                ep_reward += r
                if term or trunc:
                    break
            # Access underlying env for stats
            totals.append(ep_reward)
            co2s.append(env.mean_co2_utilisation())
            extrps.append(env.extrapolation_stats()["fraction"])
        elif agent_type == "baseline":
            obs, _ = env.reset(seed=seed + ep)
            ep_reward = 0.0
            action = np.array([1.0, T_REACTOR_NOMINAL, P_REACTOR_NOMINAL],
                              dtype=np.float32)
            for _ in range(RL_CFG.episode_length):
                obs, r, term, trunc, info = env.step(action)
                ep_reward += r
                if term or trunc:
                    break
            totals.append(ep_reward)
            co2s.append(env.mean_co2_utilisation())
            extrps.append(env.extrapolation_stats()["fraction"])
        elif agent_type == "rule_based":
            obs, _ = env.reset(seed=seed + ep)
            ep_reward = 0.0
            for _ in range(RL_CFG.episode_length):
                price = obs[0]
                load = 0.3 if price > 65.0 else 1.0
                action = np.array([load, T_REACTOR_NOMINAL, P_REACTOR_NOMINAL],
                                  dtype=np.float32)
                obs, r, term, trunc, info = env.step(action)
                ep_reward += r
                if term or trunc:
                    break
            totals.append(ep_reward)
            co2s.append(env.mean_co2_utilisation())
            extrps.append(env.extrapolation_stats()["fraction"])
        else:  # sb3
            obs, _ = env.reset(seed=seed + ep)
            ep_reward = 0.0
            for _ in range(RL_CFG.episode_length):
                action, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(action)
                ep_reward += r
                if term or trunc:
                    break
            totals.append(ep_reward)
            co2s.append(env.mean_co2_utilisation())
            extrps.append(env.extrapolation_stats()["fraction"])

    return {
        "mean_reward": float(np.mean(totals)),
        "std_reward": float(np.std(totals)),
        "co2_util": float(np.mean(co2s)),
        "gpr_extrap_rate": float(np.mean(extrps)),
    }


# ======================================================================
# Agent registry
# ======================================================================

def build_agent_registry() -> List[Dict[str, Any]]:
    """Build list of agents to evaluate with their configurations."""
    agents: List[Dict[str, Any]] = []

    # --- Baselines ---
    agents.append({
        "name": "Full-load baseline",
        "model": None,
        "agent_type": "baseline",
        "surr_version": "v1",
        "use_var_penalty": False,
    })
    agents.append({
        "name": "Rule-based (threshold)",
        "model": None,
        "agent_type": "rule_based",
        "surr_version": "v1",
        "use_var_penalty": False,
    })

    # --- Original agents ---
    sb3_models = {
        "Original SAC": ("sac_methanol.zip", "SAC"),
        "Original PPO": ("ppo_methanol.zip", "PPO"),
    }
    for label, (fname, algo) in sb3_models.items():
        path = _SAVED / fname
        if path.exists():
            if algo == "SAC":
                from stable_baselines3 import SAC
                model = SAC.load(str(path))
            else:
                from stable_baselines3 import PPO
                model = PPO.load(str(path))
            agents.append({
                "name": label,
                "model": model,
                "agent_type": "sb3",
                "surr_version": "v1",
                "use_var_penalty": False,
            })

    # Q-learning
    q_path = _SAVED / "q_table.pkl"
    if q_path.exists():
        q_agent = QLearningAgent.load(q_path)
        agents.append({
            "name": "Q-learning",
            "model": q_agent,
            "agent_type": "q_learning",
            "surr_version": "v1",
            "use_var_penalty": False,
        })

    # --- Fix variants ---
    fix_models = {
        "Fix (a) v2 surr.": "sac_extrap_fix_a.zip",
        "Fix (b) v1+penalty": "sac_extrap_fix_b.zip",
        "Fix (c) v2+penalty": "sac_extrap_fix_c.zip",
    }
    var_config = VariancePenaltyConfig(alpha=2.0, sigma_threshold_mult=2.0, enabled=True)

    for label, fname in fix_models.items():
        path = _SAVED / fname
        if path.exists():
            from stable_baselines3 import SAC
            model = SAC.load(str(path))
            use_pen = "penalty" in label
            surr_v = "v2" if "v2" in label else "v1"
            agents.append({
                "name": label,
                "model": model,
                "agent_type": "sb3",
                "surr_version": surr_v,
                "use_var_penalty": use_pen,
                "var_config": var_config if use_pen else None,
            })

    # --- NL-trained agents ---
    nl_models = {
        "SAC (NL-trained)": "sac_methanol_nl.zip",
        "PPO (NL-trained)": "ppo_methanol_nl.zip",
    }
    for label, fname in nl_models.items():
        path = _SAVED / fname
        if path.exists():
            algo = "SAC" if "SAC" in label else "PPO"
            if algo == "SAC":
                from stable_baselines3 import SAC
                model = SAC.load(str(path))
            else:
                from stable_baselines3 import PPO
                model = PPO.load(str(path))
            agents.append({
                "name": label,
                "model": model,
                "agent_type": "sb3",
                "surr_version": "v1",
                "use_var_penalty": False,
            })

    q_nl_path = _SAVED / "q_table_nl.pkl"
    if q_nl_path.exists():
        agents.append({
            "name": "Q-learning (NL-trained)",
            "model": QLearningAgent.load(q_nl_path),
            "agent_type": "q_learning",
            "surr_version": "v1",
            "use_var_penalty": False,
        })

    return agents


# ======================================================================
# Main
# ======================================================================

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=RL_CFG.seed)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load price datasets
    price_datasets: Dict[str, np.ndarray] = {}
    for ds_name in ("gb", "nl"):
        try:
            price_datasets[ds_name] = load_prices(ds_name)
            LOGGER.info("Loaded %s prices: %d hours", ds_name, len(price_datasets[ds_name]))
        except FileNotFoundError:
            LOGGER.warning("Price dataset '%s' not available, skipping", ds_name)

    if not price_datasets:
        LOGGER.error("No price datasets found!")
        return 1

    agents = build_agent_registry()
    LOGGER.info("Evaluating %d agents × %d price datasets × %d episodes",
                len(agents), len(price_datasets), args.episodes)

    rows: List[Dict] = []

    for agent_info in agents:
        name = agent_info["name"]
        for ds_name, prices in price_datasets.items():
            print(f"  Evaluating {name:30s} on {ds_name} ... ", end="", flush=True)

            try:
                result = evaluate_agent(
                    model=agent_info["model"],
                    prices=prices,
                    surr_version=agent_info["surr_version"],
                    n_episodes=args.episodes,
                    seed=args.seed,
                    agent_type=agent_info["agent_type"],
                    use_var_penalty=agent_info.get("use_var_penalty", False),
                    var_config=agent_info.get("var_config"),
                )

                rows.append({
                    "agent": name,
                    "price_data": ds_name,
                    "mean_reward": result["mean_reward"],
                    "std_reward": result["std_reward"],
                    "co2_util": result["co2_util"],
                    "gpr_extrap_rate": result["gpr_extrap_rate"],
                })
                print(f"reward={result['mean_reward']:.3f} "
                      f"extrap={result['gpr_extrap_rate']:.1%}")
            except Exception as e:
                LOGGER.warning("Failed: %s", e)
                print(f"FAILED: {e}")

    df = pd.DataFrame(rows)

    # Compute vs_baseline column
    for ds_name in price_datasets:
        mask = (df["price_data"] == ds_name) & (df["agent"] == "Full-load baseline")
        if mask.any():
            baseline_reward = df.loc[mask, "mean_reward"].values[0]
            ds_mask = df["price_data"] == ds_name
            df.loc[ds_mask, "vs_baseline"] = (
                (df.loc[ds_mask, "mean_reward"] - baseline_reward)
                / abs(baseline_reward) * 100
            ).round(1)

    # Save CSV
    csv_path = _OUT_DIR / "final_summary.csv"
    df.to_csv(csv_path, index=False)

    # Console output
    print("\n" + "=" * 100)
    print("DEFINITIVE RESULTS TABLE — EURECHA RL Dynamic Control")
    print("=" * 100)
    header = (f"{'Agent':30s} {'Data':>6s} {'Mean Reward':>12s} {'Std':>8s} "
              f"{'CO2 Util':>9s} {'Extrap %':>9s} {'vs Base':>8s}")
    print(header)
    print("-" * 100)

    for _, r in df.iterrows():
        vs_base = r.get("vs_baseline", float("nan"))
        vs_str = f"{vs_base:+7.1f}%" if not np.isnan(vs_base) else "    N/A"
        print(
            f"  {r['agent']:28s} {r['price_data']:>6s} "
            f"{r['mean_reward']:12.3f} {r['std_reward']:8.3f} "
            f"{r['co2_util']:9.3f} {100*r['gpr_extrap_rate']:8.2f}% "
            f"{vs_str}"
        )

    print("=" * 100)
    print(f"\nSaved to: {csv_path}")
    print(f"Total evaluations: {len(df)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
