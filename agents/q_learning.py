"""
Tabular Q-Learning Agent
=========================
Epsilon-greedy Q-learning with save/load for the discretised
methanol-plant environment.
"""

from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..config import RL_CFG


class QLearningAgent:
    """Tabular Q-learning with epsilon-greedy exploration.

    Parameters
    ----------
    n_actions : int
        Size of the discrete action space.
    alpha : float
        Learning rate.
    gamma : float
        Discount factor.
    epsilon_start, epsilon_end, epsilon_decay : float
        Epsilon-greedy schedule (multiplicative decay per step).
    seed : int
        Random seed.
    """

    def __init__(
        self,
        n_actions: int,
        alpha: float = RL_CFG.q_learning_rate,
        gamma: float = RL_CFG.gamma,
        epsilon_start: float = RL_CFG.epsilon_start,
        epsilon_end: float = RL_CFG.epsilon_end,
        epsilon_decay: float = RL_CFG.epsilon_decay,
        seed: int = RL_CFG.seed,
    ) -> None:
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Q-table: state (tuple) → array of Q-values per action
        self.q_table: Dict[Tuple[int, ...], np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions)
        )

        self._rng = np.random.default_rng(seed)
        self._step_count: int = 0
        self._episode_rewards: List[float] = []

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def select_action(self, state: Tuple[int, ...], greedy: bool = False) -> int:
        """Epsilon-greedy action selection.

        Parameters
        ----------
        state : tuple of ints
            Discretised observation.
        greedy : bool
            If True, always pick argmax (no exploration).
        """
        if not greedy and self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.n_actions))
        return int(np.argmax(self.q_table[state]))

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    def update(
        self,
        state: Tuple[int, ...],
        action: int,
        reward: float,
        next_state: Tuple[int, ...],
        done: bool,
    ) -> float:
        """One-step Q-learning update.

        Returns
        -------
        td_error : float
            The temporal-difference error (for diagnostics).
        """
        q_current = self.q_table[state][action]
        q_next = 0.0 if done else float(np.max(self.q_table[next_state]))
        td_target = reward + self.gamma * q_next
        td_error = td_target - q_current
        self.q_table[state][action] += self.alpha * td_error

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self._step_count += 1

        return float(td_error)

    # ------------------------------------------------------------------
    # Episode tracking
    # ------------------------------------------------------------------
    def log_episode_reward(self, total_reward: float) -> None:
        """Append episode reward for training-curve logging."""
        self._episode_rewards.append(total_reward)

    @property
    def episode_rewards(self) -> List[float]:
        return list(self._episode_rewards)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Save Q-table and metadata to a pickle file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "q_table": dict(self.q_table),
            "epsilon": self.epsilon,
            "step_count": self._step_count,
            "episode_rewards": self._episode_rewards,
            "config": {
                "n_actions": self.n_actions,
                "alpha": self.alpha,
                "gamma": self.gamma,
                "epsilon_end": self.epsilon_end,
                "epsilon_decay": self.epsilon_decay,
            },
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"[QLearning] Saved Q-table ({len(self.q_table)} states) → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "QLearningAgent":
        """Load a saved Q-table."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        cfg = data["config"]
        agent = cls(
            n_actions=cfg["n_actions"],
            alpha=cfg["alpha"],
            gamma=cfg["gamma"],
            epsilon_start=data["epsilon"],
            epsilon_end=cfg["epsilon_end"],
            epsilon_decay=cfg["epsilon_decay"],
        )
        for k, v in data["q_table"].items():
            agent.q_table[k] = v
        agent._step_count = data["step_count"]
        agent._episode_rewards = data.get("episode_rewards", [])
        print(f"[QLearning] Loaded Q-table ({len(agent.q_table)} states) ← {path}")
        return agent

    def __repr__(self) -> str:
        return (
            f"QLearningAgent(n_actions={self.n_actions}, "
            f"states={len(self.q_table)}, ε={self.epsilon:.4f}, "
            f"steps={self._step_count})"
        )
