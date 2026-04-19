#!/usr/bin/env python3
"""
Evaluate Trained Agents & Baselines
====================================
Runs a trained RL agent (Q-learning or SB3) through the methanol-plant
environment and compares it against two baselines:

1. **Constant full-load** — electrolyser always at 100%, reactor at
   nominal T/P (the "do nothing" baseline).
2. **Price-threshold rule-based** — run full-load when price < threshold,
   reduce to 30% when price is high.

Generates plots:
    - actions vs electricity price over the evaluation week
    - cumulative profit comparison
    - action distribution histograms

Usage::

    python -m rl_dynamic_control.scripts.evaluate [--agent q_learning|ppo|sac]

Run from the EURECHA root directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rl_dynamic_control.config import RL_CFG, PLANT_BOUNDS, T_REACTOR_NOMINAL, P_REACTOR_NOMINAL
from rl_dynamic_control.environment.methanol_plant_env import (
    DiscretizedMethanolEnv,
    MethanolPlantEnv,
)
from rl_dynamic_control.agents.q_learning import QLearningAgent
from rl_dynamic_control.models.surrogates import PlantSurrogates


# ======================================================================
# Baselines
# ======================================================================

def run_constant_full_load(env: MethanolPlantEnv) -> Tuple[List[float], List[Dict]]:
    """Baseline 1: always run at 100% load, nominal T/P."""
    obs, info = env.reset(seed=0)
    rewards, infos = [], []
    action = np.array([1.0, T_REACTOR_NOMINAL, P_REACTOR_NOMINAL], dtype=np.float32)
    for _ in range(RL_CFG.episode_length):
        obs, r, term, trunc, info = env.step(action)
        rewards.append(r)
        infos.append(info)
        if term or trunc:
            break
    return rewards, infos


def run_price_threshold(
    env: MethanolPlantEnv,
    high_price: float = 65.0,
    low_load: float = 0.3,
) -> Tuple[List[float], List[Dict]]:
    """Baseline 2: reduce load when price exceeds threshold."""
    obs, info = env.reset(seed=0)
    rewards, infos = [], []
    for _ in range(RL_CFG.episode_length):
        price = obs[0]
        load = low_load if price > high_price else 1.0
        action = np.array([load, T_REACTOR_NOMINAL, P_REACTOR_NOMINAL], dtype=np.float32)
        obs, r, term, trunc, info = env.step(action)
        rewards.append(r)
        infos.append(info)
        if term or trunc:
            break
    return rewards, infos


def run_rl_agent(
    env: Any,
    agent: Any,
    agent_type: str = "q_learning",
) -> Tuple[List[float], List[Dict]]:
    """Run a trained RL agent through one episode."""
    obs, info = env.reset(seed=0)
    rewards, infos = [], []
    for _ in range(RL_CFG.episode_length):
        if agent_type == "q_learning":
            action = agent.select_action(obs, greedy=True)
        else:
            action, _ = agent.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        rewards.append(r)
        infos.append(info)
        if term or trunc:
            break
    return rewards, infos


# ======================================================================
# Plotting
# ======================================================================

def plot_evaluation(
    results: Dict[str, Tuple[List[float], List[Dict]]],
    save_dir: Path,
) -> None:
    """Generate comparison plots."""
    save_dir.mkdir(parents=True, exist_ok=True)
    hours = np.arange(RL_CFG.episode_length)

    # --- Cumulative profit ---
    fig, ax = plt.subplots(figsize=(12, 5))
    colours = {"Constant full-load": "#e63946", "Price-threshold": "#457b9d",
               "RL Agent": "#2a9d8f"}
    for name, (rews, _) in results.items():
        cum = np.cumsum(rews)
        ax.plot(hours[:len(cum)], cum, linewidth=2,
                label=f"{name} (total={cum[-1]:.2f})",
                color=colours.get(name, None))
    ax.set_xlabel("Hour")
    ax.set_ylabel("Cumulative reward (scaled £)")
    ax.set_title("Cumulative Profit — 1-Week Evaluation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "cumulative_profit.png", dpi=150)
    plt.close(fig)

    # --- Actions vs price (RL agent only, if available) ---
    if "RL Agent" in results:
        _, rl_infos = results["RL Agent"]
        prices = [info.get("price", 0) for info in rl_infos]
        loads = [info.get("state", {}).get("load", 0) for info in rl_infos]
        meoh = [info.get("meoh_tph", 0) for info in rl_infos]

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        axes[0].plot(hours[:len(prices)], prices, color="#e63946", linewidth=1)
        axes[0].set_ylabel("Elec. price (£/MWh)")
        axes[0].set_title("RL Agent Actions vs Electricity Price")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(hours[:len(loads)], loads, color="#457b9d", linewidth=1)
        axes[1].set_ylabel("Electrolyser load")
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(hours[:len(meoh)], meoh, color="#2a9d8f", linewidth=1)
        axes[2].set_ylabel("MeOH rate (t/hr)")
        axes[2].set_xlabel("Hour")
        axes[2].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(save_dir / "rl_actions_vs_price.png", dpi=150)
        plt.close(fig)

    print(f"Plots saved → {save_dir}/")


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RL agents")
    parser.add_argument("--agent", type=str, default="q_learning",
                        choices=["q_learning", "ppo", "sac"],
                        help="Which trained agent to evaluate")
    args = parser.parse_args()

    # Load prices — prefer real data if available
    data_dir = Path(__file__).resolve().parent.parent / "data"
    real_path = data_dir / "electricity_prices_real.csv"
    synth_path = data_dir / "electricity_prices.csv"
    csv_path = real_path if real_path.exists() else synth_path
    print(f"[prices] Using {csv_path.name}")
    df = pd.read_csv(csv_path)
    prices = df["price_gbp_per_mwh"].values.astype(np.float32)
    surrogates = PlantSurrogates(use_gpr=True)

    results: Dict[str, Tuple[List[float], List[Dict]]] = {}

    # --- Baselines ---
    print("Running Baseline 1: Constant full-load ...")
    env1 = MethanolPlantEnv(price_data=prices, surrogates=surrogates)
    results["Constant full-load"] = run_constant_full_load(env1)
    total1 = sum(results["Constant full-load"][0])
    print(f"  Total reward: {total1:.3f}")

    print("Running Baseline 2: Price-threshold rule ...")
    env2 = MethanolPlantEnv(price_data=prices, surrogates=surrogates)
    results["Price-threshold"] = run_price_threshold(env2)
    total2 = sum(results["Price-threshold"][0])
    print(f"  Total reward: {total2:.3f}")

    # --- RL Agent ---
    model_dir = Path(__file__).resolve().parent.parent / "saved_models"

    if args.agent == "q_learning":
        q_path = model_dir / "q_table.pkl"
        if q_path.exists():
            print(f"Loading Q-learning agent from {q_path} ...")
            agent = QLearningAgent.load(q_path)
            env_rl = MethanolPlantEnv(price_data=prices, surrogates=surrogates)
            disc_env = DiscretizedMethanolEnv(env_rl)
            results["RL Agent"] = run_rl_agent(disc_env, agent, "q_learning")
        else:
            print(f"⚠ No Q-table found at {q_path}. Run train_q_learning.py first.")
            print("  Evaluating baselines only.")

    else:  # SB3 (PPO or SAC)
        algo = args.agent.upper()
        model_path = model_dir / f"{args.agent}_methanol.zip"
        if model_path.exists():
            print(f"Loading {algo} model from {model_path} ...")
            if algo == "PPO":
                from stable_baselines3 import PPO
                model = PPO.load(str(model_path))
            else:
                from stable_baselines3 import SAC
                model = SAC.load(str(model_path))
            env_rl = MethanolPlantEnv(price_data=prices, surrogates=surrogates)
            results["RL Agent"] = run_rl_agent(env_rl, model, "sb3")
        else:
            print(f"⚠ No model found at {model_path}. Run train_sb3.py first.")
            print("  Evaluating baselines only.")

    if "RL Agent" in results:
        total_rl = sum(results["RL Agent"][0])
        print(f"  RL Agent total reward: {total_rl:.3f}")

    # --- Summary ---
    print(f"\n{'='*50}")
    print("EVALUATION SUMMARY")
    print(f"{'='*50}")
    for name, (rews, _) in results.items():
        total = sum(rews)
        print(f"  {name:25s}  total_reward = {total:+.3f}")
    print(f"{'='*50}")

    # --- Plots ---
    plot_dir = Path(__file__).resolve().parent.parent / "evaluation_plots"
    plot_evaluation(results, plot_dir)


if __name__ == "__main__":
    main()
