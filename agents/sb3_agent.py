"""
Stable-Baselines3 Agent Wrapper
===============================
Convenience helpers for training PPO and SAC on the methanol plant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type

from ..config import RL_CFG


def maybe_make_vec_env(
    env: Any | None = None,
    env_factory: Optional[Callable[[], Any]] = None,
    *,
    normalize: bool = False,
    training: bool = True,
) -> Any:
    """Wrap an environment in SB3 vectorized helpers."""
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    if env is None and env_factory is None:
        raise ValueError("Provide either env or env_factory.")

    if env_factory is None:
        env_factory = lambda: env

    vec_env = DummyVecEnv([lambda: Monitor(env_factory())])
    if not normalize:
        return vec_env

    return VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=training,
        clip_obs=RL_CFG.vecnorm_clip_obs,
        gamma=RL_CFG.gamma,
        training=training,
    )


def make_sb3_agent(
    env: Any,
    algo: str = RL_CFG.sb3_algo,
    learning_rate: float = RL_CFG.learning_rate,
    gamma: float = RL_CFG.gamma,
    seed: int = RL_CFG.seed,
    tb_log_dir: str = RL_CFG.tb_log_dir,
    **kwargs: Any,
) -> Any:
    """Create an SB3 agent (PPO or SAC) for the given environment."""
    from stable_baselines3 import PPO, SAC

    algo_map: Dict[str, Type] = {"PPO": PPO, "SAC": SAC}
    algo_upper = algo.upper()
    if algo_upper not in algo_map:
        raise ValueError(f"Unsupported algorithm: {algo!r}. Choose 'PPO' or 'SAC'.")

    algo_cls = algo_map[algo_upper]
    common_kwargs: Dict[str, Any] = dict(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        gamma=gamma,
        seed=seed,
        verbose=1,
        tensorboard_log=tb_log_dir,
    )

    if algo_upper == "PPO":
        common_kwargs.setdefault("n_steps", RL_CFG.ppo_n_steps)
        common_kwargs.setdefault("batch_size", RL_CFG.ppo_batch_size)
        common_kwargs.setdefault("n_epochs", RL_CFG.ppo_n_epochs)
        common_kwargs.setdefault("clip_range", RL_CFG.ppo_clip_range)
    elif algo_upper == "SAC":
        common_kwargs.setdefault("batch_size", RL_CFG.sac_batch_size)
        common_kwargs.setdefault("buffer_size", RL_CFG.sac_buffer_size)
        common_kwargs.setdefault("learning_starts", RL_CFG.sac_learning_starts)
        common_kwargs.setdefault("tau", RL_CFG.sac_tau)

    common_kwargs.update(kwargs)
    model = algo_cls(**common_kwargs)
    print(f"[SB3] Created {algo_upper} agent (lr={learning_rate}, gamma={gamma})")
    return model


def train_sb3(
    model: Any,
    total_timesteps: int = RL_CFG.total_timesteps,
    save_path: Optional[str | Path] = None,
    tb_log_name: Optional[str] = None,
    vecnormalize_path: Optional[str | Path] = None,
    checkpoint_dir: Optional[str | Path] = None,
    checkpoint_freq: Optional[int] = None,
) -> Any:
    """Train an SB3 model and optionally save checkpoints and final state."""
    from stable_baselines3.common.callbacks import CheckpointCallback

    log_name = tb_log_name or model.__class__.__name__
    callback = None
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        callback = CheckpointCallback(
            save_freq=max(1, int(checkpoint_freq or 50_000)),
            save_path=str(checkpoint_dir),
            name_prefix=log_name,
            save_vecnormalize=vecnormalize_path is not None,
        )

    model.learn(total_timesteps=total_timesteps, tb_log_name=log_name, callback=callback)
    print(f"[SB3] Training complete - {total_timesteps:,} timesteps.")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(save_path))
        if vecnormalize_path is not None and hasattr(model.env, "save"):
            vecnormalize_path = Path(vecnormalize_path)
            vecnormalize_path.parent.mkdir(parents=True, exist_ok=True)
            model.env.save(str(vecnormalize_path))
            print(f"[SB3] VecNormalize stats saved -> {vecnormalize_path}")
        print(f"[SB3] Model saved -> {save_path}")

    return model
