#!/usr/bin/env python3
"""
Train SAC with the edge-state GNN uncertainty penalty.

This is the focused runner for the GNN-penalty variant.  It trains only
``variant_gnn`` on v2 surrogates, evaluates it, writes step-level data, and
updates ``outputs/extrapolation_fix/extrapolation_fix_results.csv`` so it can
be plotted alongside Fix (c).

Run from the EURECHA root directory:

    python -m rl_dynamic_control.scripts.train_gnn_penalty_sac \
        --timesteps 500000 --eval-episodes 20
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from rl_dynamic_control.agents.sb3_agent import (
    make_sb3_agent,
    maybe_make_vec_env,
    train_sb3,
)
from rl_dynamic_control.config import RL_CFG
from rl_dynamic_control.scripts.retrain_with_fixes import (
    _OUT_DIR,
    _SAVED,
    load_prices,
    make_env,
)
from rl_dynamic_control.utils.gnn_confidence import GNNPenaltyConfig

LOGGER = logging.getLogger("train_gnn_penalty_sac")


def _make_gnn_env(prices, gnn_config: GNNPenaltyConfig):
    return make_env(
        prices,
        surrogate_version="v2",
        use_variance_penalty=False,
        variance_penalty_config=None,
        use_gnn_penalty=True,
        gnn_penalty_config=gnn_config,
    )


def _make_train_env(prices, gnn_config: GNNPenaltyConfig):
    env = maybe_make_vec_env(
        env_factory=lambda: _make_gnn_env(prices, gnn_config),
        normalize=True,
        training=True,
    )
    if hasattr(env, "norm_reward"):
        env.norm_reward = False
    return env


def _make_eval_env(prices, gnn_config: GNNPenaltyConfig, train_env):
    env = maybe_make_vec_env(
        env_factory=lambda: _make_gnn_env(prices, gnn_config),
        normalize=True,
        training=False,
    )
    env.obs_rms = copy.deepcopy(train_env.obs_rms)
    env.ret_rms = copy.deepcopy(train_env.ret_rms)
    env.training = False
    env.norm_reward = False
    return env


def evaluate_vec_model(model, prices, gnn_config: GNNPenaltyConfig, n_episodes: int, seed: int):
    """Evaluate a VecNormalize-trained SAC model and keep step-level logs."""

    eval_env = _make_eval_env(prices, gnn_config, model.env)
    totals = []
    co2s = []
    extrps = []
    all_step_data = []

    for ep in range(n_episodes):
        eval_env.seed(seed + ep)
        obs = eval_env.reset()
        done = np.array([False])
        ep_reward = 0.0
        ep_co2 = []
        last_extrap = 0.0

        while not bool(done[0]):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = eval_env.step(action)
            r = float(np.asarray(reward).reshape(-1)[0])
            info = infos[0]
            ep_reward += r
            ep_co2.append(float(info.get("co2_utilisation", 0.0)))
            last_extrap = float(info.get("gpr_extrapolation_frac", last_extrap))

            state = info.get("state", {})
            all_step_data.append({
                "episode": ep,
                "reward": r,
                "reward_before_penalty": info.get("reward_before_penalty", r),
                "gpr_sigma": info.get("gpr_sigma", 0.0),
                "variance_penalty": info.get("variance_penalty", 0.0),
                "gnn_uncertainty": info.get("gnn_uncertainty", 0.0),
                "gnn_confidence": info.get("gnn_confidence", 1.0),
                "gnn_penalty": info.get("gnn_penalty", 0.0),
                "load": state.get("load", 0.0),
                "T": state.get("T", 0.0),
                "P": state.get("P", 0.0),
                "meoh_tph": info.get("meoh_tph", 0.0),
                "price": info.get("price", 0.0),
            })

        totals.append(ep_reward)
        co2s.append(float(np.mean(ep_co2)) if ep_co2 else 0.0)
        extrps.append(last_extrap)
        print(
            f"[eval] episode={ep + 1:02d}/{n_episodes} "
            f"reward={ep_reward:.3f} extrap={last_extrap:.2%}"
        )

    eval_env.close()
    return {
        "mean_reward": float(np.mean(totals)),
        "std_reward": float(np.std(totals)),
        "mean_co2_util": float(np.mean(co2s)),
        "mean_extrap_frac": float(np.mean(extrps)),
        "std_extrap_frac": float(np.std(extrps)),
        "step_data": all_step_data,
        "mean_gnn_uncertainty": float(np.mean([
            row["gnn_uncertainty"] for row in all_step_data
        ])) if all_step_data else 0.0,
        "mean_gnn_penalty": float(np.mean([
            row["gnn_penalty"] for row in all_step_data
        ])) if all_step_data else 0.0,
    }


def _upsert_summary(summary_path: Path, row: Dict[str, Any]) -> pd.DataFrame:
    """Insert/update the variant row in the shared comparison CSV."""

    if summary_path.exists():
        df = pd.read_csv(summary_path)
        df = df[df["variant"] != row["variant"]]
    else:
        df = pd.DataFrame()

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(summary_path, index=False)
    return df


def _print_fix_c_comparison(df: pd.DataFrame, row: Dict[str, Any]) -> None:
    if "variant" not in df.columns or "variant_c" not in set(df["variant"]):
        print("\nFix (c) row not found in the summary CSV yet.")
        print("Run retrain_with_fixes.py first if you want the automatic delta.")
        return

    c = df[df["variant"] == "variant_c"].iloc[0]
    delta_mean = row["mean_reward"] - float(c["mean_reward"])
    delta_std = row["std_reward"] - float(c["std_reward"])
    delta_extrap_pp = 100.0 * (
        row["mean_extrap_frac"] - float(c["mean_extrap_frac"])
    )

    print("\nGNN penalty vs Fix (c):")
    print(f"  Delta mean reward:        {delta_mean:+.3f}")
    print(f"  Delta reward std:         {delta_std:+.3f}")
    print(f"  Delta extrapolation rate: {delta_extrap_pp:+.2f} percentage points")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=RL_CFG.total_timesteps)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=RL_CFG.seed)
    parser.add_argument("--gnn-beta", type=float, default=2.0)
    parser.add_argument("--gnn-threshold", type=float, default=1e-3)
    parser.add_argument("--gnn-mc-samples", type=int, default=4)
    parser.add_argument("--gnn-dropout-p", type=float, default=0.05)
    parser.add_argument("--gnn-max-penalty", type=float, default=10.0)
    parser.add_argument("--gnn-device", type=str, default="auto")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--ent-coef", type=str, default="auto_0.1")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    _SAVED.mkdir(parents=True, exist_ok=True)

    prices = load_prices()
    gnn_config = GNNPenaltyConfig(
        beta=args.gnn_beta,
        threshold=args.gnn_threshold,
        enabled=True,
        mc_samples=args.gnn_mc_samples,
        dropout_p=args.gnn_dropout_p,
        max_penalty=args.gnn_max_penalty,
        device=args.gnn_device,
    )

    print("\n" + "=" * 72)
    print("SAC variant: v2 surrogates + edge-state GNN uncertainty penalty")
    print("=" * 72)
    print(f"timesteps={args.timesteps:,}  eval_episodes={args.eval_episodes}")
    print(f"beta={gnn_config.beta}  threshold={gnn_config.threshold}  "
          f"mc_samples={gnn_config.mc_samples}  dropout_p={gnn_config.dropout_p}  "
          f"max_penalty={gnn_config.max_penalty}")
    print(f"SAC learning_rate={args.learning_rate}  ent_coef={args.ent_coef}  "
          f"device={args.device}  gnn_device={args.gnn_device}")

    save_path = _SAVED / "sac_gnn_penalty"
    vec_path = _SAVED / "sac_gnn_penalty_vecnormalize.pkl"
    checkpoint_dir = _SAVED / "sac_gnn_penalty_checkpoints"
    tb_dir = str(Path(__file__).resolve().parent.parent / RL_CFG.tb_log_dir)

    if not args.skip_training:
        env = _make_train_env(prices, gnn_config)
        model = make_sb3_agent(
            env,
            algo="SAC",
            seed=args.seed,
            tb_log_dir=tb_dir,
            learning_rate=args.learning_rate,
            ent_coef=args.ent_coef,
            device=args.device,
        )
        train_sb3(
            model,
            total_timesteps=args.timesteps,
            save_path=save_path,
            vecnormalize_path=vec_path,
            checkpoint_dir=checkpoint_dir,
            checkpoint_freq=max(10_000, args.timesteps // 10),
            tb_log_name="SAC_gnn_penalty",
        )
    else:
        from stable_baselines3 import SAC
        model = SAC.load(str(save_path))
        if vec_path.exists():
            from stable_baselines3.common.vec_env import VecNormalize
            train_env = _make_train_env(prices, gnn_config)
            model.set_env(VecNormalize.load(str(vec_path), train_env.venv))
            model.env.training = False
            model.env.norm_reward = False
        else:
            raise FileNotFoundError(f"VecNormalize stats not found: {vec_path}")

    result = evaluate_vec_model(
        model,
        prices,
        gnn_config,
        args.eval_episodes,
        args.seed,
    )

    row = {
        "variant": "variant_gnn",
        "mean_reward": result["mean_reward"],
        "std_reward": result["std_reward"],
        "mean_co2_util": result["mean_co2_util"],
        "mean_extrap_frac": result["mean_extrap_frac"],
        "std_extrap_frac": result.get("std_extrap_frac", 0.0),
        "mean_gnn_uncertainty": result.get("mean_gnn_uncertainty", 0.0),
        "mean_gnn_penalty": result.get("mean_gnn_penalty", 0.0),
    }

    summary_path = _OUT_DIR / "extrapolation_fix_results.csv"
    df = _upsert_summary(summary_path, row)

    step_df = pd.DataFrame(result["step_data"])
    step_path = _OUT_DIR / "step_data_variant_gnn.csv"
    step_df.to_csv(step_path, index=False)

    print("\nResults:")
    print(f"  Mean reward:          {row['mean_reward']:.3f}")
    print(f"  Reward std:           {row['std_reward']:.3f}")
    print(f"  Extrapolation rate:   {100 * row['mean_extrap_frac']:.2f}%")
    print(f"  Mean GNN uncertainty: {row['mean_gnn_uncertainty']:.6f}")
    print(f"  Mean GNN penalty:     {row['mean_gnn_penalty']:.6f}")
    _print_fix_c_comparison(df, row)

    print(f"\nSaved summary: {summary_path}")
    print(f"Saved step data: {step_path}")
    print("Plot with: python -m rl_dynamic_control.scripts.plot_extrapolation_fix")
    return 0


if __name__ == "__main__":
    sys.exit(main())
