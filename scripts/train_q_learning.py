#!/usr/bin/env python3
"""
Train Tabular Q-Learning Agent
===============================
Trains an epsilon-greedy Q-learning agent on the discretised
methanol-plant environment, logs episode rewards, saves the Q-table,
and plots the training curve.

Usage::

    python -m rl_dynamic_control.scripts.train_q_learning [--episodes 2000]

Run from the EURECHA root directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure EURECHA root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rl_dynamic_control.config import RL_CFG, PLANT_BOUNDS
from rl_dynamic_control.environment.methanol_plant_env import (
    DiscretizedMethanolEnv,
    MethanolPlantEnv,
)
from rl_dynamic_control.agents.q_learning import QLearningAgent
from rl_dynamic_control.models.surrogates import PlantSurrogates

import pandas as pd


def load_prices() -> np.ndarray:
    """Load electricity price data — prefer real prices if available."""
    data_dir = Path(__file__).resolve().parent.parent / "data"
    real_path = data_dir / "electricity_prices_real.csv"
    synth_path = data_dir / "electricity_prices.csv"
    csv_path = real_path if real_path.exists() else synth_path
    print(f"[prices] Loading from {csv_path.name}")
    df = pd.read_csv(csv_path)
    return df["price_gbp_per_mwh"].values.astype(np.float32)


def make_env(prices: np.ndarray) -> DiscretizedMethanolEnv:
    """Create the discretised environment."""
    surrogates = PlantSurrogates(use_gpr=True)
    base_env = MethanolPlantEnv(
        price_data=prices,
        surrogates=surrogates,
        episode_length=RL_CFG.episode_length,
    )
    return DiscretizedMethanolEnv(
        base_env,
        n_load_bins=RL_CFG.n_load_bins,
        n_temp_bins=RL_CFG.n_temp_bins,
        n_pressure_bins=RL_CFG.n_pressure_bins,
    )


def train(n_episodes: int = 2000) -> QLearningAgent:
    """Run the Q-learning training loop."""
    prices = load_prices()
    env = make_env(prices)
    agent = QLearningAgent(n_actions=env.n_actions)

    print(f"Training Q-learning for {n_episodes} episodes "
          f"({RL_CFG.episode_length} steps each)")
    print(f"Action space: {env.n_actions} discrete actions "
          f"({RL_CFG.n_load_bins}×{RL_CFG.n_temp_bins}×{RL_CFG.n_pressure_bins})")

    for ep in range(n_episodes):
        state, _ = env.reset(seed=RL_CFG.seed + ep)
        total_reward = 0.0

        for step in range(RL_CFG.episode_length):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.update(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
            if done:
                break

        agent.log_episode_reward(total_reward)

        if (ep + 1) % 100 == 0:
            recent = agent.episode_rewards[-100:]
            mean_r = np.mean(recent)
            print(f"  Episode {ep+1:5d} | "
                  f"ε={agent.epsilon:.4f} | "
                  f"mean_reward(100)={mean_r:.3f} | "
                  f"Q-states={len(agent.q_table)}")

    return agent


def plot_training_curve(agent: QLearningAgent, save_path: Path) -> None:
    """Plot and save the training reward curve."""
    rewards = agent.episode_rewards
    window = 50
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rewards, alpha=0.3, color="steelblue", label="Episode reward")
    ax.plot(
        np.arange(window - 1, len(rewards)),
        smoothed,
        color="darkblue",
        linewidth=2,
        label=f"Rolling mean ({window} ep)",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_title("Q-Learning Training Curve — Methanol Plant Control")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Training curve saved → {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Q-learning agent")
    parser.add_argument("--episodes", type=int, default=2000)
    args = parser.parse_args()

    agent = train(n_episodes=args.episodes)

    # Save outputs
    out_dir = Path(__file__).resolve().parent.parent / "saved_models"
    out_dir.mkdir(exist_ok=True)

    agent.save(out_dir / "q_table.pkl")
    plot_training_curve(agent, out_dir / "q_learning_training_curve.png")

    print(f"\nFinal agent: {agent}")
    print(f"Best episode reward: {max(agent.episode_rewards):.3f}")


if __name__ == "__main__":
    main()
