#!/usr/bin/env python3
"""
Train and compare Stable-Baselines3 agents on the methanol plant.
"""

from __future__ import annotations

import argparse
import copy
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from rl_dynamic_control.agents.sb3_agent import make_sb3_agent, maybe_make_vec_env, train_sb3
from rl_dynamic_control.config import RL_CFG
from rl_dynamic_control.data.price_loader import load_prices
from rl_dynamic_control.environment.methanol_plant_env import MethanolPlantEnv
from rl_dynamic_control.models.surrogates import PlantSurrogates


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    algo: str
    learning_rate: float
    n_steps: Optional[int] = None
    n_epochs: Optional[int] = None
    normalize_obs: bool = False
    normalize_reward: bool = False
    notes: str = ""


def build_experiments() -> List[ExperimentSpec]:
    return [
        ExperimentSpec(name="ppo_original", algo="PPO", learning_rate=RL_CFG.learning_rate),
        ExperimentSpec(
            name="ppo_nsteps_4096",
            algo="PPO",
            learning_rate=RL_CFG.learning_rate,
            n_steps=4096,
            notes="Increase rollout horizon.",
        ),
        ExperimentSpec(
            name="ppo_vecnormalize",
            algo="PPO",
            learning_rate=RL_CFG.learning_rate,
            normalize_obs=True,
            normalize_reward=True,
            notes="Normalize observations and rewards.",
        ),
        ExperimentSpec(
            name="ppo_lr_1e4",
            algo="PPO",
            learning_rate=1e-4,
            notes="Lower optimizer step size.",
        ),
        ExperimentSpec(
            name="ppo_epochs_20",
            algo="PPO",
            learning_rate=RL_CFG.learning_rate,
            n_epochs=20,
            notes="More epochs per rollout batch.",
        ),
        ExperimentSpec(
            name="ppo_fixed_combo",
            algo="PPO",
            learning_rate=1e-4,
            n_steps=4096,
            n_epochs=20,
            normalize_obs=True,
            normalize_reward=True,
            notes="Combined PPO fix.",
        ),
        ExperimentSpec(
            name="sac_original",
            algo="SAC",
            learning_rate=RL_CFG.learning_rate,
            normalize_obs=True,
            normalize_reward=False,
            notes="SAC with observation normalization for numerical stability.",
        ),
    ]


def filter_experiments(all_specs: Iterable[ExperimentSpec], requested: Optional[str]) -> List[ExperimentSpec]:
    specs = list(all_specs)
    if not requested:
        return specs
    wanted = [item.strip() for item in requested.split(",") if item.strip()]
    wanted_set = set(wanted)
    missing = [name for name in wanted if name not in {spec.name for spec in specs}]
    if missing:
        raise ValueError(f"Unknown experiments: {missing}")
    return [spec for spec in specs if spec.name in wanted_set]


def make_env(prices: np.ndarray, seed: int) -> MethanolPlantEnv:
    return MethanolPlantEnv(
        price_data=prices,
        surrogates=PlantSurrogates(use_gpr=True),
        episode_length=RL_CFG.episode_length,
        seed=seed,
    )


def make_train_env(prices: np.ndarray, seed: int, spec: ExperimentSpec) -> Any:
    env = maybe_make_vec_env(
        env_factory=lambda: make_env(prices, seed),
        normalize=spec.normalize_obs or spec.normalize_reward,
        training=True,
    )
    if hasattr(env, "norm_reward"):
        env.norm_reward = spec.normalize_reward
    return env


def make_eval_env(prices: np.ndarray, seed: int, train_env: Any | None) -> Any:
    normalize = train_env is not None and hasattr(train_env, "obs_rms")
    eval_env = maybe_make_vec_env(
        env_factory=lambda: make_env(prices, seed),
        normalize=normalize,
        training=False,
    )
    if normalize:
        eval_env.obs_rms = copy.deepcopy(train_env.obs_rms)
        eval_env.ret_rms = copy.deepcopy(train_env.ret_rms)
        eval_env.training = False
        eval_env.norm_reward = False
    return eval_env


def evaluate_model(model: Any, prices: np.ndarray, seed: int, n_episodes: int) -> Dict[str, float]:
    train_env = model.env if hasattr(model, "env") else None
    eval_env = make_eval_env(prices, seed, train_env)
    episode_rewards: List[float] = []
    action_variances: List[np.ndarray] = []
    mean_co2_utils: List[float] = []

    for episode_idx in range(n_episodes):
        obs = eval_env.reset()
        done = np.array([False])
        ep_reward = 0.0
        ep_actions: List[np.ndarray] = []
        ep_co2_utils: List[float] = []

        while not bool(done[0]):
            action, _ = model.predict(obs, deterministic=True)
            flat_action = np.asarray(action, dtype=np.float32).reshape(-1)
            ep_actions.append(flat_action.copy())
            obs, reward, done, infos = eval_env.step(action)
            ep_reward += float(np.asarray(reward).reshape(-1)[0])
            info = infos[0]
            if "co2_utilisation" in info:
                ep_co2_utils.append(float(info["co2_utilisation"]))

        episode_rewards.append(ep_reward)
        mean_co2_utils.append(float(np.mean(ep_co2_utils)) if ep_co2_utils else 0.0)
        if ep_actions:
            action_variances.append(np.var(np.vstack(ep_actions), axis=0))
        else:
            action_variances.append(np.zeros(model.action_space.shape[0], dtype=np.float32))

        print(
            f"[eval] episode={episode_idx + 1:02d}/{n_episodes} "
            f"reward={ep_reward:.3f} mean_co2_util={mean_co2_utils[-1]:.4f}"
        )

    eval_env.close()
    action_var = np.mean(np.vstack(action_variances), axis=0)
    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_co2_utilisation": float(np.mean(mean_co2_utils)),
        "std_co2_utilisation": float(np.std(mean_co2_utils)),
        "action_var_load": float(action_var[0]),
        "action_var_T": float(action_var[1]),
        "action_var_P": float(action_var[2]),
        "action_var_mean": float(np.mean(action_var)),
    }


def load_existing_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "experiment" not in df.columns:
        return pd.DataFrame()
    if "status" not in df.columns:
        df["status"] = np.where(df.get("mean_reward").notna(), "completed", "unknown")
    return df


def persist_results(rows: List[Dict[str, Any]], output_path: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if not df.empty and "mean_reward" in df.columns:
        df = df.sort_values(by=["status", "mean_reward"], ascending=[True, False], na_position="last")
    df.to_csv(output_path, index=False)
    return df


def experiment_artifact_paths(models_dir: Path, spec: ExperimentSpec) -> Dict[str, Path]:
    model_path = models_dir / f"{spec.name}.zip"
    vec_path = models_dir / f"{spec.name}_vecnormalize.pkl"
    checkpoint_dir = models_dir / f"{spec.name}_checkpoints"
    return {"model": model_path, "vecnormalize": vec_path, "checkpoint_dir": checkpoint_dir}


def train_experiment(
    spec: ExperimentSpec,
    prices: np.ndarray,
    timesteps: int,
    eval_episodes: int,
    seed: int,
    tb_dir: Path,
    models_dir: Path,
) -> Dict[str, Any]:
    print(f"\n{'=' * 72}")
    print(f"Training {spec.name} ({spec.algo}, {timesteps:,} steps)")
    print(f"{'=' * 72}")

    artifacts = experiment_artifact_paths(models_dir, spec)
    train_env = make_train_env(prices, seed, spec)
    model_kwargs: Dict[str, Any] = {}
    if spec.algo.upper() == "PPO":
        if spec.n_steps is not None:
            model_kwargs["n_steps"] = spec.n_steps
        if spec.n_epochs is not None:
            model_kwargs["n_epochs"] = spec.n_epochs

    model = make_sb3_agent(
        train_env,
        algo=spec.algo,
        learning_rate=spec.learning_rate,
        gamma=RL_CFG.gamma,
        seed=seed,
        tb_log_dir=str(tb_dir),
        **model_kwargs,
    )
    checkpoint_freq = max(10_000, timesteps // 10)
    train_sb3(
        model,
        total_timesteps=timesteps,
        save_path=artifacts["model"],
        tb_log_name=spec.name,
        vecnormalize_path=artifacts["vecnormalize"] if (spec.normalize_obs or spec.normalize_reward) else None,
        checkpoint_dir=artifacts["checkpoint_dir"],
        checkpoint_freq=checkpoint_freq,
    )
    metrics = evaluate_model(model=model, prices=prices, seed=seed + 10_000, n_episodes=eval_episodes)
    train_env.close()

    return {
        "experiment": spec.name,
        "algo": spec.algo,
        "status": "completed",
        "notes": spec.notes,
        "learning_rate": spec.learning_rate,
        "n_steps": spec.n_steps or np.nan,
        "n_epochs": spec.n_epochs or np.nan,
        "normalize_obs": spec.normalize_obs,
        "normalize_reward": spec.normalize_reward,
        "model_path": str(artifacts["model"]),
        "vecnormalize_path": (
            str(artifacts["vecnormalize"])
            if (spec.normalize_obs or spec.normalize_reward)
            else ""
        ),
        **metrics,
    }


def save_best_ppo_model(
    results_df: pd.DataFrame,
    experiments: List[ExperimentSpec],
    prices: np.ndarray,
    timesteps: int,
    seed: int,
    tb_dir: Path,
    models_dir: Path,
    save_path: Path,
) -> pd.Series:
    completed = results_df[results_df["status"] == "completed"]
    ppo_results = completed[completed["algo"] == "PPO"].sort_values(
        by=["mean_reward", "action_var_mean"],
        ascending=[False, False],
    )
    if ppo_results.empty:
        raise RuntimeError("No completed PPO results available for final export.")

    best_row = ppo_results.iloc[0]
    best_spec = next(spec for spec in experiments if spec.name == best_row["experiment"])
    artifacts = experiment_artifact_paths(models_dir, best_spec)

    print(f"\n[best] Re-training {best_spec.name} for final export -> {save_path}")
    train_env = make_train_env(prices, seed, best_spec)
    model_kwargs: Dict[str, Any] = {}
    if best_spec.n_steps is not None:
        model_kwargs["n_steps"] = best_spec.n_steps
    if best_spec.n_epochs is not None:
        model_kwargs["n_epochs"] = best_spec.n_epochs

    model = make_sb3_agent(
        train_env,
        algo=best_spec.algo,
        learning_rate=best_spec.learning_rate,
        gamma=RL_CFG.gamma,
        seed=seed,
        tb_log_dir=str(tb_dir),
        **model_kwargs,
    )
    train_sb3(
        model,
        total_timesteps=timesteps,
        save_path=save_path,
        tb_log_name=f"{best_spec.name}_final",
        vecnormalize_path=save_path.with_name(f"{save_path.stem}_vecnormalize.pkl")
        if (best_spec.normalize_obs or best_spec.normalize_reward)
        else None,
        checkpoint_dir=artifacts["checkpoint_dir"],
        checkpoint_freq=max(10_000, timesteps // 10),
    )
    train_env.close()
    return best_row


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare SB3 agents")
    parser.add_argument("--timesteps", type=int, default=RL_CFG.total_timesteps)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=RL_CFG.seed)
    parser.add_argument("--dataset", type=str, default="synthetic_gb")
    parser.add_argument("--experiments", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-final-export", action="store_true")
    args = parser.parse_args()

    prices = load_prices(args.dataset)
    tb_dir = Path(__file__).resolve().parent.parent / RL_CFG.tb_log_dir
    outputs_dir = Path(__file__).resolve().parent.parent / "outputs"
    models_dir = Path(__file__).resolve().parent.parent / "saved_models"
    output_path = outputs_dir / "ppo_fix_results.csv"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    experiments = filter_experiments(build_experiments(), args.experiments)
    existing_df = load_existing_results(output_path)
    rows: List[Dict[str, Any]] = existing_df.to_dict("records") if not existing_df.empty else []
    completed_names = {
        row["experiment"]
        for row in rows
        if row.get("status") == "completed"
    }

    for spec in experiments:
        if args.resume and spec.name in completed_names:
            print(f"[resume] Skipping {spec.name}; completed row already exists in {output_path.name}")
            continue

        rows = [row for row in rows if row.get("experiment") != spec.name]
        try:
            row = train_experiment(
                spec=spec,
                prices=prices,
                timesteps=args.timesteps,
                eval_episodes=args.eval_episodes,
                seed=args.seed,
                tb_dir=tb_dir,
                models_dir=models_dir,
            )
        except Exception as exc:
            row = {
                "experiment": spec.name,
                "algo": spec.algo,
                "status": "failed",
                "notes": spec.notes,
                "learning_rate": spec.learning_rate,
                "n_steps": spec.n_steps or np.nan,
                "n_epochs": spec.n_epochs or np.nan,
                "normalize_obs": spec.normalize_obs,
                "normalize_reward": spec.normalize_reward,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
            print(f"[error] {spec.name} failed: {exc}")

        rows.append(row)
        results_df = persist_results(rows, output_path)
        print(f"[progress] Wrote intermediate results -> {output_path}")

    results_df = persist_results(rows, output_path)
    if not args.skip_final_export:
        try:
            best_row = save_best_ppo_model(
                results_df=results_df,
                experiments=experiments,
                prices=prices,
                timesteps=args.timesteps,
                seed=args.seed,
                tb_dir=tb_dir,
                models_dir=models_dir,
                save_path=models_dir / "ppo_methanol_fixed.zip",
            )
            results_df.loc[:, "best_ppo_exported"] = (
                results_df["experiment"] == best_row["experiment"]
            )
            results_df.to_csv(output_path, index=False)
        except Exception as exc:
            print(f"[warning] Final PPO export skipped: {exc}")

    print(f"\nResults saved -> {output_path}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
