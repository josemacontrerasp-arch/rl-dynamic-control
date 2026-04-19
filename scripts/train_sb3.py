#!/usr/bin/env python3
"""
Train Stable-Baselines3 Agent (PPO / SAC)
==========================================
Trains a deep-RL agent on the continuous methanol-plant environment
with TensorBoard logging and model checkpointing.

Usage::

    python -m rl_dynamic_control.scripts.train_sb3 [--algo PPO] [--timesteps 500000]

    # Monitor with TensorBoard:
    tensorboard --logdir rl_dynamic_control/tb_logs

Run from the EURECHA root directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure EURECHA root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rl_dynamic_control.config import RL_CFG
from rl_dynamic_control.environment.methanol_plant_env import MethanolPlantEnv
from rl_dynamic_control.models.surrogates import PlantSurrogates
from rl_dynamic_control.agents.sb3_agent import make_sb3_agent, train_sb3


def load_prices() -> np.ndarray:
    """Load electricity price data — prefer real prices if available."""
    data_dir = Path(__file__).resolve().parent.parent / "data"
    real_path = data_dir / "electricity_prices_real.csv"
    synth_path = data_dir / "electricity_prices.csv"
    csv_path = real_path if real_path.exists() else synth_path
    print(f"[prices] Loading from {csv_path.name}")
    df = pd.read_csv(csv_path)
    return df["price_gbp_per_mwh"].values.astype(np.float32)


def make_env(prices: np.ndarray) -> MethanolPlantEnv:
    """Create a continuous-action environment."""
    surrogates = PlantSurrogates(use_gpr=True)
    return MethanolPlantEnv(
        price_data=prices,
        surrogates=surrogates,
        episode_length=RL_CFG.episode_length,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SB3 agent")
    parser.add_argument("--algo", type=str, default=RL_CFG.sb3_algo,
                        choices=["PPO", "SAC"], help="Algorithm to use")
    parser.add_argument("--timesteps", type=int, default=RL_CFG.total_timesteps)
    parser.add_argument("--seed", type=int, default=RL_CFG.seed)
    args = parser.parse_args()

    prices = load_prices()
    env = make_env(prices)

    # TensorBoard log dir inside rl_dynamic_control/
    tb_dir = str(Path(__file__).resolve().parent.parent / RL_CFG.tb_log_dir)

    model = make_sb3_agent(
        env,
        algo=args.algo,
        learning_rate=RL_CFG.learning_rate,
        gamma=RL_CFG.gamma,
        seed=args.seed,
        tb_log_dir=tb_dir,
    )

    save_dir = Path(__file__).resolve().parent.parent / "saved_models"
    save_path = save_dir / f"{args.algo.lower()}_methanol"

    train_sb3(
        model,
        total_timesteps=args.timesteps,
        save_path=save_path,
        tb_log_name=f"{args.algo}_methanol",
    )

    # Quick evaluation
    from stable_baselines3.common.evaluation import evaluate_policy

    eval_env = make_env(prices)
    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=10,
                                     deterministic=True)
    print(f"\n{'='*50}")
    print(f"{args.algo} Evaluation (10 episodes): {mean_r:.2f} ± {std_r:.2f}")
    print(f"Model saved → {save_path}")
    print(f"TensorBoard → tensorboard --logdir {tb_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
