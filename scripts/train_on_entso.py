#!/usr/bin/env python3
"""
Retrain SAC / PPO / Q-learning on ENTSO-E NL prices
====================================================
Loads the Dutch day-ahead price series via ``PriceLoader`` and retrains
each agent on it.  Saves weights under
``saved_models/<algo>_methanol_nl.zip`` (or ``.pkl`` for Q-learning)
so the originals remain available for transfer comparisons.

Usage::

    python -m rl_dynamic_control.scripts.train_on_entso --algos SAC PPO Q \\
        --sac-steps 200000 --ppo-steps 200000 --q-episodes 2000 \\
        --price-dataset entso_nl_all

Run from the EURECHA root directory.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rl_dynamic_control.config import RL_CFG
from rl_dynamic_control.data.price_loader import get_loader
from rl_dynamic_control.environment.methanol_plant_env import (
    DiscretizedMethanolEnv, MethanolPlantEnv,
)
from rl_dynamic_control.models.surrogates import PlantSurrogates
from rl_dynamic_control.agents.sb3_agent import make_sb3_agent, train_sb3

LOGGER = logging.getLogger("train_on_entso")


def train_sb3_on(prices: np.ndarray, algo: str, steps: int, seed: int, save_dir: Path):
    surrogates = PlantSurrogates(use_gpr=True)
    env = MethanolPlantEnv(price_data=prices, surrogates=surrogates)
    tb_dir = str(save_dir.parent / RL_CFG.tb_log_dir)
    model = make_sb3_agent(env, algo=algo, seed=seed, tb_log_dir=tb_dir)
    save_path = save_dir / f"{algo.lower()}_methanol_nl"
    train_sb3(model, total_timesteps=steps, save_path=save_path,
              tb_log_name=f"{algo}_methanol_nl")
    return model, env


def train_q_on(prices: np.ndarray, episodes: int, seed: int, save_dir: Path):
    from rl_dynamic_control.agents.q_learning import QLearningAgent

    surrogates = PlantSurrogates(use_gpr=True)
    base_env = MethanolPlantEnv(price_data=prices, surrogates=surrogates, seed=seed)
    env = DiscretizedMethanolEnv(base_env)
    agent = QLearningAgent(
        n_actions=env.n_actions,
        alpha=RL_CFG.q_learning_rate,
        gamma=RL_CFG.gamma,
        epsilon_start=RL_CFG.epsilon_start,
        epsilon_end=RL_CFG.epsilon_end,
        epsilon_decay=RL_CFG.epsilon_decay,
        seed=seed,
    )
    LOGGER.info(f"Training Q-learning for {episodes} episodes "
                f"({RL_CFG.episode_length} steps each)")
    for ep in range(episodes):
        state, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        for _ in range(RL_CFG.episode_length):
            action = agent.select_action(state)
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            agent.update(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
            if done:
                break
        agent.log_episode_reward(total_reward)
        if (ep + 1) % max(1, episodes // 10) == 0:
            recent = np.mean(agent.episode_rewards[-min(100, ep + 1):])
            LOGGER.info(f"  ep {ep+1:5d}  ε={agent.epsilon:.3f}  "
                        f"mean100={recent:.3f}  Q-states={len(agent.q_table)}")

    save_path = save_dir / "q_table_nl.pkl"
    agent.save(save_path)
    LOGGER.info(f"Q-learning saved → {save_path}")
    return agent


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algos", nargs="+", default=["SAC", "PPO", "Q"],
                        choices=["SAC", "PPO", "Q"])
    parser.add_argument("--price-dataset", default="entso_nl_all")
    parser.add_argument("--sac-steps", type=int, default=200_000)
    parser.add_argument("--ppo-steps", type=int, default=200_000)
    parser.add_argument("--q-episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=RL_CFG.seed)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    loader = get_loader()
    if args.price_dataset not in loader.available():
        LOGGER.error("Dataset %s not found. Available: %s",
                     args.price_dataset, loader.available())
        LOGGER.error("Run `python -m rl_dynamic_control.data.fetch_entso_prices` first.")
        return 2

    prices = loader.load(args.price_dataset)
    LOGGER.info(f"Training on {args.price_dataset}: "
                f"{len(prices)} hours, mean={prices.mean():.2f}, std={prices.std():.2f}")

    save_dir = Path(__file__).resolve().parent.parent / "saved_models"
    save_dir.mkdir(exist_ok=True)

    for algo in args.algos:
        LOGGER.info(f"--- Training {algo} ---")
        if algo == "Q":
            train_q_on(prices, args.q_episodes, args.seed, save_dir)
        elif algo == "SAC":
            train_sb3_on(prices, "SAC", args.sac_steps, args.seed, save_dir)
        elif algo == "PPO":
            train_sb3_on(prices, "PPO", args.ppo_steps, args.seed, save_dir)

    print("\n" + "=" * 60)
    print("Retraining summary")
    print("=" * 60)
    print(f"Dataset : {args.price_dataset}  (n={len(prices)})")
    print(f"Output  : {save_dir}")
    print("Models  : " + ", ".join(args.algos))
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
