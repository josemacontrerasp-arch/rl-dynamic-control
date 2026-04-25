#!/usr/bin/env python3
"""
Pareto Sweep v2: profit vs electricity consumption
==================================================

Train a fresh SAC agent for each electricity-penalty weight:

    lambda_elec in [0, 1]
    lambda_profit = 1 - lambda_elec

The scalarised training reward is:

    R = lambda_profit * profit - lambda_elec * electricity_cost_per_step

Each trained agent is evaluated on profit-only reward over 20 episodes while
recording:
    - mean episode profit
    - mean episode electricity consumption (MWh)
    - mean reward
    - action variance

Original PPO and SAC models are evaluated as overlay baselines, together with
the existing handcrafted baselines.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Callable, Dict, List

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rl_dynamic_control.agents.sb3_agent import make_sb3_agent, train_sb3
from rl_dynamic_control.config import P_REACTOR_NOMINAL, RL_CFG, T_REACTOR_NOMINAL
from rl_dynamic_control.data.price_loader import load_prices
from rl_dynamic_control.environment.methanol_plant_env import MethanolPlantEnv
from rl_dynamic_control.models.surrogates import PlantSurrogates

LOGGER = logging.getLogger("pareto_sweep_v2")


def evaluate_episode(
    env: MethanolPlantEnv,
    act_fn: Callable[[np.ndarray], np.ndarray],
    seed: int,
) -> Dict[str, float]:
    obs, _ = env.reset(seed=seed)
    reward_sum = 0.0
    profit_sum = 0.0
    elec_sum = 0.0
    actions: List[np.ndarray] = []

    for _ in range(RL_CFG.episode_length):
        action = np.asarray(act_fn(obs), dtype=np.float32).reshape(-1)
        actions.append(action.copy())
        obs, reward, term, trunc, info = env.step(action)
        reward_sum += float(reward)
        profit_sum += float(info["profit_reward"])
        elec_sum += float(info.get("elec_consumption_mwh", 0.0))
        if term or trunc:
            break

    if actions:
        action_var = np.var(np.vstack(actions), axis=0)
    else:
        action_var = np.zeros(3, dtype=np.float32)

    return {
        "mean_reward": reward_sum,
        "profit": profit_sum,
        "elec_mwh": elec_sum,
        "action_var_load": float(action_var[0]),
        "action_var_T": float(action_var[1]),
        "action_var_P": float(action_var[2]),
        "action_var_mean": float(np.mean(action_var)),
    }


def evaluate_policy(
    act_fn: Callable[[np.ndarray], np.ndarray],
    prices: np.ndarray,
    surrogates: PlantSurrogates,
    n_episodes: int,
    seed: int,
) -> Dict[str, float]:
    rows: List[Dict[str, float]] = []
    for ep in range(n_episodes):
        env = MethanolPlantEnv(
            price_data=prices,
            surrogates=surrogates,
            lambda_profit=1.0,
            lambda_elec=0.0,
        )
        rows.append(evaluate_episode(env, act_fn, seed + ep))

    df = pd.DataFrame(rows)
    return {
        "mean_reward": float(df["mean_reward"].mean()),
        "std_reward": float(df["mean_reward"].std(ddof=0)),
        "profit_mean": float(df["profit"].mean()),
        "profit_std": float(df["profit"].std(ddof=0)),
        "elec_mean_mwh": float(df["elec_mwh"].mean()),
        "elec_std_mwh": float(df["elec_mwh"].std(ddof=0)),
        "action_var_load": float(df["action_var_load"].mean()),
        "action_var_T": float(df["action_var_T"].mean()),
        "action_var_P": float(df["action_var_P"].mean()),
        "action_var_mean": float(df["action_var_mean"].mean()),
    }


def evaluate_model(
    model,
    prices: np.ndarray,
    surrogates: PlantSurrogates,
    n_episodes: int,
    seed: int,
) -> Dict[str, float]:
    def act(obs: np.ndarray, _model=model) -> np.ndarray:
        action, _ = _model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32).reshape(-1)

    return evaluate_policy(act, prices, surrogates, n_episodes, seed)


def evaluate_handcrafted_baselines(
    prices: np.ndarray,
    surrogates: PlantSurrogates,
    n_episodes: int,
    seed: int,
) -> List[Dict[str, float]]:
    act_full = np.array([1.0, T_REACTOR_NOMINAL, P_REACTOR_NOMINAL], dtype=np.float32)
    price_threshold = float(np.median(prices))

    def full_load(_obs: np.ndarray) -> np.ndarray:
        return act_full

    def rule(obs: np.ndarray, _thr: float = price_threshold) -> np.ndarray:
        load = 0.3 if float(obs[0]) > _thr else 1.0
        return np.array([load, T_REACTOR_NOMINAL, P_REACTOR_NOMINAL], dtype=np.float32)

    rows = []
    for kind, act_fn in (("full_load", full_load), ("rule", rule)):
        stats = evaluate_policy(act_fn, prices, surrogates, n_episodes, seed)
        stats["kind"] = kind
        rows.append(stats)
    return rows


def evaluate_saved_model_baselines(
    prices: np.ndarray,
    surrogates: PlantSurrogates,
    n_episodes: int,
    seed: int,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    models_dir = Path(__file__).resolve().parent.parent / "saved_models"
    model_specs = [
        ("original_ppo", "PPO", models_dir / "ppo_methanol.zip"),
        ("original_sac", "SAC", models_dir / "sac_methanol.zip"),
    ]

    for kind, algo, path in model_specs:
        if not path.exists():
            LOGGER.warning("Skipping missing baseline model: %s", path)
            continue

        if algo == "PPO":
            from stable_baselines3 import PPO

            model = PPO.load(str(path))
        else:
            from stable_baselines3 import SAC

            model = SAC.load(str(path))

        stats = evaluate_model(model, prices, surrogates, n_episodes, seed)
        stats["kind"] = kind
        rows.append(stats)

    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-weights", type=int, default=9)
    parser.add_argument("--steps", type=int, default=60_000)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--price-dataset", default="synthetic_gb")
    parser.add_argument("--seed", type=int, default=RL_CFG.seed)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--keep-models", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(__file__).resolve().parent.parent / "outputs" / "pareto_v2"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "models"
    models_dir.mkdir(exist_ok=True)

    prices = load_prices(args.price_dataset)
    surrogates = PlantSurrogates(use_gpr=True)
    lambda_elec_values = np.linspace(0.0, 1.0, args.n_weights)

    rows: List[Dict[str, float]] = []
    for idx, lambda_elec in enumerate(lambda_elec_values):
        lambda_profit = 1.0 - float(lambda_elec)
        LOGGER.info(
            "[%d/%d] Training SAC with lambda_profit=%.3f lambda_elec=%.3f",
            idx + 1,
            args.n_weights,
            lambda_profit,
            float(lambda_elec),
        )
        env = MethanolPlantEnv(
            price_data=prices,
            surrogates=surrogates,
            lambda_profit=lambda_profit,
            lambda_elec=float(lambda_elec),
        )
        model = make_sb3_agent(
            env,
            algo="SAC",
            seed=args.seed + idx,
            tb_log_dir=str(out_dir / "tb"),
        )
        save_path = models_dir / f"sac_lambda_elec_{lambda_elec:.3f}"
        train_sb3(
            model,
            total_timesteps=args.steps,
            save_path=save_path if args.keep_models else None,
            tb_log_name=f"sac_lambda_elec_{lambda_elec:.2f}",
        )

        stats = evaluate_model(
            model=model,
            prices=prices,
            surrogates=surrogates,
            n_episodes=args.episodes,
            seed=args.seed + 1_000 * idx,
        )
        stats.update(
            lambda_profit=lambda_profit,
            lambda_elec=float(lambda_elec),
            train_steps=int(args.steps),
            eval_episodes=int(args.episodes),
            algorithm="SAC",
        )
        rows.append(stats)
        LOGGER.info(
            "lambda_elec=%.2f -> reward=%.3f, profit=%.3f, elec=%.3f MWh, action_var=%.4f",
            float(lambda_elec),
            stats["mean_reward"],
            stats["profit_mean"],
            stats["elec_mean_mwh"],
            stats["action_var_mean"],
        )

    results_df = pd.DataFrame(rows).sort_values("lambda_elec").reset_index(drop=True)
    results_df.to_csv(out_dir / "pareto_results.csv", index=False)

    baseline_rows = []
    baseline_rows.extend(
        evaluate_handcrafted_baselines(
            prices=prices,
            surrogates=surrogates,
            n_episodes=args.episodes,
            seed=args.seed,
        )
    )
    baseline_rows.extend(
        evaluate_saved_model_baselines(
            prices=prices,
            surrogates=surrogates,
            n_episodes=args.episodes,
            seed=args.seed + 50_000,
        )
    )
    baselines_df = pd.DataFrame(baseline_rows)
    baselines_df.to_csv(out_dir / "baselines.csv", index=False)

    print("\n" + "=" * 104)
    print(
        f"{'lambda_elec':>12s} {'lambda_profit':>14s} {'mean_reward':>13s} "
        f"{'profit_mean':>13s} {'elec_mean_mwh':>15s} {'action_var_mean':>17s}"
    )
    print("-" * 104)
    for _, row in results_df.iterrows():
        print(
            f"{row['lambda_elec']:12.3f} {row['lambda_profit']:14.3f} "
            f"{row['mean_reward']:13.3f} {row['profit_mean']:13.3f} "
            f"{row['elec_mean_mwh']:15.3f} {row['action_var_mean']:17.5f}"
        )
    print("-" * 104)
    for _, row in baselines_df.iterrows():
        print(
            f"[{row['kind']:>12s}] reward={row['mean_reward']:.3f} "
            f"profit={row['profit_mean']:.3f} elec={row['elec_mean_mwh']:.3f} "
            f"action_var={row['action_var_mean']:.5f}"
        )
    print("=" * 104)
    print(f"Results -> {out_dir / 'pareto_results.csv'}")
    print(f"Baselines -> {out_dir / 'baselines.csv'}")

    if not args.skip_plots:
        from rl_dynamic_control.scripts.plot_pareto import main as plot_main

        plot_main(["--in-dir", str(out_dir)])

    return 0


if __name__ == "__main__":
    sys.exit(main())
