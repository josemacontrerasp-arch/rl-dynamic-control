"""
Stable-Baselines3 Agent Wrapper
================================
Convenience functions for training PPO and SAC on the methanol-plant
environment, with TensorBoard logging and model comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Type

import numpy as np

from ..config import RL_CFG


def make_sb3_agent(
    env: Any,
    algo: str = RL_CFG.sb3_algo,
    learning_rate: float = RL_CFG.learning_rate,
    gamma: float = RL_CFG.gamma,
    seed: int = RL_CFG.seed,
    tb_log_dir: str = RL_CFG.tb_log_dir,
    **kwargs: Any,
) -> Any:
    """Create an SB3 agent (PPO or SAC) for the given environment.

    Parameters
    ----------
    env : gymnasium.Env
        Must have a continuous (Box) action space.
    algo : str
        ``"PPO"`` or ``"SAC"``.
    learning_rate : float
    gamma : float
    seed : int
    tb_log_dir : str
        TensorBoard log directory.
    **kwargs
        Extra arguments forwarded to the SB3 algorithm constructor.

    Returns
    -------
    stable_baselines3.common.base_class.BaseAlgorithm
    """
    from stable_baselines3 import PPO, SAC

    algo_map: Dict[str, Type] = {"PPO": PPO, "SAC": SAC}
    algo_upper = algo.upper()
    if algo_upper not in algo_map:
        raise ValueError(f"Unsupported algorithm: {algo!r}. Choose 'PPO' or 'SAC'.")

    AlgoCls = algo_map[algo_upper]

    common_kwargs: Dict[str, Any] = dict(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        gamma=gamma,
        seed=seed,
        verbose=1,
        tensorboard_log=tb_log_dir,
    )

    # Algorithm-specific defaults
    if algo_upper == "PPO":
        common_kwargs.setdefault("n_steps", 2048)
        common_kwargs.setdefault("batch_size", 64)
        common_kwargs.setdefault("n_epochs", 10)
        common_kwargs.setdefault("clip_range", 0.2)
    elif algo_upper == "SAC":
        common_kwargs.setdefault("batch_size", 256)
        common_kwargs.setdefault("buffer_size", 100_000)
        common_kwargs.setdefault("learning_starts", 1000)
        common_kwargs.setdefault("tau", 0.005)

    common_kwargs.update(kwargs)
    model = AlgoCls(**common_kwargs)
    print(f"[SB3] Created {algo_upper} agent  (lr={learning_rate}, γ={gamma})")
    return model


def train_sb3(
    model: Any,
    total_timesteps: int = RL_CFG.total_timesteps,
    save_path: Optional[str | Path] = None,
    tb_log_name: Optional[str] = None,
) -> Any:
    """Train an SB3 model and optionally save it.

    Parameters
    ----------
    model : BaseAlgorithm
    total_timesteps : int
    save_path : str or Path, optional
        If provided, save the trained model here.
    tb_log_name : str, optional
        Run name for TensorBoard.

    Returns
    -------
    model : BaseAlgorithm
        The trained model.
    """
    log_name = tb_log_name or model.__class__.__name__
    model.learn(total_timesteps=total_timesteps, tb_log_name=log_name)
    print(f"[SB3] Training complete — {total_timesteps:,} timesteps.")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(save_path))
        print(f"[SB3] Model saved → {save_path}")

    return model


def compare_algorithms(
    env_factory: Any,
    algos: tuple[str, ...] = ("PPO", "SAC"),
    total_timesteps: int = RL_CFG.total_timesteps,
    n_eval_episodes: int = RL_CFG.n_eval_episodes,
    seed: int = RL_CFG.seed,
) -> Dict[str, Dict[str, float]]:
    """Train and evaluate multiple SB3 algorithms, returning summary stats.

    Parameters
    ----------
    env_factory : callable
        ``env_factory()`` should return a fresh ``MethanolPlantEnv``.
    algos : tuple of str
        Algorithm names to compare.
    total_timesteps : int
    n_eval_episodes : int
    seed : int

    Returns
    -------
    dict
        ``{algo_name: {"mean_reward": ..., "std_reward": ...}}``
    """
    from stable_baselines3.common.evaluation import evaluate_policy

    results: Dict[str, Dict[str, float]] = {}
    for algo_name in algos:
        print(f"\n{'='*60}")
        print(f"Training {algo_name}  ({total_timesteps:,} steps)")
        print(f"{'='*60}")

        train_env = env_factory()
        model = make_sb3_agent(train_env, algo=algo_name, seed=seed)
        train_sb3(model, total_timesteps=total_timesteps,
                  save_path=f"saved_models/{algo_name.lower()}_methanol")

        eval_env = env_factory()
        mean_r, std_r = evaluate_policy(
            model, eval_env, n_eval_episodes=n_eval_episodes, deterministic=True
        )
        results[algo_name] = {"mean_reward": float(mean_r), "std_reward": float(std_r)}
        print(f"[{algo_name}] Eval: mean={mean_r:.2f} ± {std_r:.2f}")

    return results
