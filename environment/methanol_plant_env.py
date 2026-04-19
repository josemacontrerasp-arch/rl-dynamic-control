"""
Gymnasium Environment — CO₂-to-Methanol Plant
==============================================
Simulates hourly operation of the methanol plant under fluctuating
electricity prices.  The RL agent chooses electrolyser load, reactor
temperature, and reactor pressure each hour.

Two flavours:
    ``MethanolPlantEnv``         — continuous Box action space (for SB3)
    ``DiscretizedMethanolEnv``   — Discrete action space (for tabular Q)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..config import (
    RL_CFG,
    PLANT_BOUNDS,
    ELECTROLYSER_MW,
    H2_FEED_KMOL,
    MW_H2,
    MEOH_PROD_TPH,
    CO2_FEED_TPH,
)
from ..models.surrogates import PlantSurrogates
from ..utils.reward import compute_reward, RewardConfig
from ..utils.variance_penalty import (
    VariancePenaltyConfig,
    compute_variance_penalty,
)


class MethanolPlantEnv(gym.Env):
    """Hourly-step methanol-plant control environment.

    Observation (Box, 8-dim)
    ------------------------
    0. electricity_price       £/MWh
    1. electrolyser_load       fraction [0, 1]
    2. h2_buffer_level         kg  [0, buffer_max]
    3. reactor_T               °C  [T_min, T_max]
    4. reactor_P               bar [P_min, P_max]
    5. methanol_rate           t/hr
    6. hour_of_day             [0, 23]
    7. day_of_week             [0, 6]

    Action (Box, 3-dim)
    --------------------
    0. electrolyser_load_setpoint   [0, 1]
    1. reactor_T_setpoint           [T_min, T_max]  (°C)
    2. reactor_P_setpoint           [P_min, P_max]  (bar)

    Parameters
    ----------
    price_data : np.ndarray
        1-D array of hourly electricity prices (£/MWh), length ≥ episode_length.
    surrogates : PlantSurrogates, optional
        If None a fresh instance is created.
    reward_config : RewardConfig, optional
        Override default reward parameters.
    episode_length : int
        Number of steps per episode (default 168 = 1 week).
    seed : int
        Random seed.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        price_data: Optional[np.ndarray] = None,
        surrogates: Optional[PlantSurrogates] = None,
        reward_config: Optional[RewardConfig] = None,
        episode_length: int = RL_CFG.episode_length,
        seed: int = RL_CFG.seed,
        price_dataset: Optional[str] = None,
        lambda_profit: float = 1.0,
        lambda_co2: float = 0.0,
        gpr_warn_sigma_mult: float = 2.0,
        use_variance_penalty: bool = False,
        variance_penalty_config: Optional[VariancePenaltyConfig] = None,
    ) -> None:
        """
        Parameters
        ----------
        price_data : np.ndarray, optional
            Hourly electricity prices (£ or €/MWh).  Mutually exclusive with
            ``price_dataset``.
        price_dataset : str, optional
            Name of a dataset to pull from ``rl_dynamic_control.data.PriceLoader``
            (e.g. ``"entso_nl_2023"``, ``"synthetic_gb"``).  Allows switching
            price series without any other code changes.
        lambda_profit, lambda_co2 : float
            Multi-objective scalarisation weights.  The reward becomes
            ``λ_profit * profit + λ_co2 * co2_utilisation_component``.
            Default (1,0) reproduces the original profit-only reward.
        gpr_warn_sigma_mult : float
            Multiplier on the training-set std used to flag GPR
            extrapolation during evaluation.  See ``extrapolation_stats``.
        use_variance_penalty : bool
            If True, apply the GPR-variance penalty to the reward
            (Approach B for extrapolation fix).  Off by default to
            preserve backward compatibility.
        variance_penalty_config : VariancePenaltyConfig, optional
            Configuration for the variance penalty.  If None and
            ``use_variance_penalty`` is True, uses default config.
        """
        super().__init__()
        # ---- Price data source ------------------------------------------
        if price_data is None:
            if price_dataset is None:
                from ..data.price_loader import load_prices
                price_data = load_prices("auto")
            else:
                from ..data.price_loader import get_loader
                price_data = get_loader().load(price_dataset)
        self.price_data = np.asarray(price_data, dtype=np.float32)
        assert len(self.price_data) >= episode_length, (
            f"Price data ({len(self.price_data)} hrs) shorter than "
            f"episode ({episode_length} hrs)"
        )

        self.surrogates = surrogates or PlantSurrogates(use_gpr=True)
        self.reward_cfg = reward_config or RewardConfig()
        self.episode_length = episode_length

        self._np_random: Optional[np.random.Generator] = None
        self._seed = seed

        # Bounds from existing project
        pb = PLANT_BOUNDS

        # --- Observation space ---
        self.observation_space = spaces.Box(
            low=np.array([
                0.0,              # price (£/MWh) — can be negative in practice
                pb.load_min,      # load fraction
                pb.h2_buffer_min, # H₂ buffer (kg)
                pb.T_min,         # reactor T (°C)
                pb.P_min,         # reactor P (bar)
                pb.meoh_rate_min, # MeOH rate (t/hr)
                0.0,              # hour of day
                0.0,              # day of week
            ], dtype=np.float32),
            high=np.array([
                500.0,            # price ceiling (generous)
                pb.load_max,
                pb.h2_buffer_max,
                pb.T_max,
                pb.P_max,
                pb.meoh_rate_max,
                23.0,
                6.0,
            ], dtype=np.float32),
        )

        # --- Action space (continuous) ---
        self.action_space = spaces.Box(
            low=np.array([pb.load_min, pb.T_min, pb.P_min], dtype=np.float32),
            high=np.array([pb.load_max, pb.T_max, pb.P_max], dtype=np.float32),
        )

        # Internal state
        self._step_idx: int = 0
        self._episode_start: int = 0
        self._state: Dict[str, float] = {}
        self._prev_load: float = 0.5
        self._cumulative_profit: float = 0.0

        # Multi-objective weights
        self.lambda_profit = float(lambda_profit)
        self.lambda_co2 = float(lambda_co2)

        # GPR extrapolation monitoring
        self.gpr_warn_sigma_mult = float(gpr_warn_sigma_mult)
        self._extrap_count = 0
        self._extrap_total_steps = 0
        # CO₂ utilisation accumulator for episode
        self._co2_utilisation_sum = 0.0
        self._co2_utilisation_n = 0

        # Variance penalty (Approach B)
        self.use_variance_penalty = use_variance_penalty
        if use_variance_penalty:
            self._var_penalty_cfg = variance_penalty_config or VariancePenaltyConfig()
        else:
            self._var_penalty_cfg = VariancePenaltyConfig(enabled=False)
        self._cumulative_var_penalty = 0.0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment: sample a random week from price data."""
        super().reset(seed=seed)
        if self._np_random is None:
            self._np_random = np.random.default_rng(seed or self._seed)

        max_start = len(self.price_data) - self.episode_length
        self._episode_start = int(self._np_random.integers(0, max(max_start, 1)))
        self._step_idx = 0
        self._cumulative_profit = 0.0
        self._extrap_count = 0
        self._extrap_total_steps = 0
        self._co2_utilisation_sum = 0.0
        self._co2_utilisation_n = 0
        self._cumulative_var_penalty = 0.0

        pb = PLANT_BOUNDS

        # Randomise initial state within safe operating bounds
        self._state = {
            "load": float(self._np_random.uniform(0.3, 0.8)),
            "h2_buffer": float(self._np_random.uniform(
                0.2 * pb.h2_buffer_max, 0.6 * pb.h2_buffer_max
            )),
            "T": float(self._np_random.uniform(
                pb.T_min + 10, pb.T_max - 10
            )),
            "P": float(self._np_random.uniform(
                pb.P_min + 5, pb.P_max - 5
            )),
            "meoh_rate": float(MEOH_PROD_TPH * 0.5),
        }
        self._prev_load = self._state["load"]

        return self._get_obs(), self._get_info()

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one hourly time step.

        action : ndarray, shape (3,)
            [load_setpoint, T_setpoint, P_setpoint]
        """
        load_sp = float(np.clip(action[0], PLANT_BOUNDS.load_min, PLANT_BOUNDS.load_max))
        T_sp = float(np.clip(action[1], PLANT_BOUNDS.T_min, PLANT_BOUNDS.T_max))
        P_sp = float(np.clip(action[2], PLANT_BOUNDS.P_min, PLANT_BOUNDS.P_max))

        # --- Dynamics: first-order lag toward setpoint (1-hr step) ---
        tau_load = 0.3   # load responds fast
        tau_T = 0.5      # temperature: ~2-hr time constant
        tau_P = 0.7      # pressure: faster than T

        self._state["load"] += tau_load * (load_sp - self._state["load"])
        self._state["T"] += tau_T * (T_sp - self._state["T"])
        self._state["P"] += tau_P * (P_sp - self._state["P"])

        load = float(np.clip(self._state["load"], 0.0, 1.0))
        T = float(np.clip(self._state["T"], PLANT_BOUNDS.T_min, PLANT_BOUNDS.T_max))
        P = float(np.clip(self._state["P"], PLANT_BOUNDS.P_min, PLANT_BOUNDS.P_max))

        # --- Surrogate predictions ---
        h2_kg_hr, meoh_tph, elec_mw = self.surrogates.step(load, T, P)

        # --- H₂ buffer dynamics ---
        h2_consumed_kg = (meoh_tph / (MW_H2 * 1e-3)) * MW_H2  # rough
        # More precisely: stoich 3 mol H₂ per mol MeOH
        meoh_kmol_hr = meoh_tph * 1000.0 / 32.04
        h2_consumed_kg_hr = meoh_kmol_hr * 3.0 * MW_H2
        h2_net = h2_kg_hr - h2_consumed_kg_hr  # kg/hr net into buffer
        self._state["h2_buffer"] = float(np.clip(
            self._state["h2_buffer"] + h2_net,
            PLANT_BOUNDS.h2_buffer_min,
            PLANT_BOUNDS.h2_buffer_max,
        ))

        self._state["meoh_rate"] = meoh_tph

        # --- Electricity price ---
        price_idx = self._episode_start + self._step_idx
        price = float(self.price_data[price_idx % len(self.price_data)])

        # --- Reward ---
        ramp = abs(load - self._prev_load)
        constraint_violations = self._check_constraints()
        profit_reward = compute_reward(
            meoh_tph=meoh_tph,
            elec_mw=elec_mw,
            elec_price=price,
            ramp=ramp,
            constraint_violations=constraint_violations,
            config=self.reward_cfg,
        )

        # CO₂ utilisation component: fraction of available CO₂ (fresh feed,
        # CO2_FEED_TPH t/hr) actually converted to methanol in this step.
        co2_consumed_tph = meoh_tph * self.reward_cfg.co2_per_t_meoh
        co2_utilisation = float(np.clip(co2_consumed_tph / max(CO2_FEED_TPH, 1e-6),
                                        0.0, 1.5))
        self._co2_utilisation_sum += co2_utilisation
        self._co2_utilisation_n += 1

        reward_before_penalty = (self.lambda_profit * profit_reward
                                + self.lambda_co2 * co2_utilisation)

        # --- Variance penalty (Approach B) --------------------------------
        var_penalty = 0.0
        gpr_sigma_val = 0.0
        if self.use_variance_penalty and self._var_penalty_cfg.enabled:
            var_penalty, gpr_sigma_val = compute_variance_penalty(
                self.surrogates, load, T, P, self._var_penalty_cfg,
            )
            # Scale penalty by reward_scale for consistency
            var_penalty *= self.reward_cfg.reward_scale
        self._cumulative_var_penalty += var_penalty

        reward = reward_before_penalty - var_penalty

        self._cumulative_profit += profit_reward
        self._prev_load = load

        # --- GPR extrapolation monitoring --------------------------------
        self._extrap_total_steps += 1
        try:
            flagged = self.surrogates.flag_extrapolation(
                load, T, P, sigma_mult=self.gpr_warn_sigma_mult,
            )
        except Exception:
            flagged = False
        if flagged:
            self._extrap_count += 1

        # --- Step bookkeeping ---
        self._step_idx += 1
        terminated = False
        truncated = self._step_idx >= self.episode_length

        obs = self._get_obs()
        info = self._get_info()
        info.update({
            "h2_produced_kg": h2_kg_hr,
            "meoh_tph": meoh_tph,
            "elec_mw": elec_mw,
            "price": price,
            "reward": reward,
            "reward_before_penalty": reward_before_penalty,
            "profit_reward": profit_reward,
            "co2_utilisation": co2_utilisation,
            "cumulative_profit": self._cumulative_profit,
            "gpr_extrapolation_frac": (
                self._extrap_count / max(self._extrap_total_steps, 1)
            ),
            "gpr_sigma": gpr_sigma_val,
            "variance_penalty": var_penalty,
            "cumulative_var_penalty": self._cumulative_var_penalty,
        })

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        """Build the 8-dim observation vector."""
        price_idx = self._episode_start + self._step_idx
        price = float(self.price_data[price_idx % len(self.price_data)])
        hour_of_day = (price_idx % 24)
        day_of_week = (price_idx // 24) % 7

        return np.array([
            price,
            self._state["load"],
            self._state["h2_buffer"],
            self._state["T"],
            self._state["P"],
            self._state["meoh_rate"],
            hour_of_day,
            day_of_week,
        ], dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        return {
            "step": self._step_idx,
            "episode_start_hour": self._episode_start,
            "state": dict(self._state),
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def extrapolation_stats(self) -> Dict[str, float]:
        """Return GPR-extrapolation stats for the current episode."""
        n = max(self._extrap_total_steps, 1)
        return {
            "extrapolation_steps": int(self._extrap_count),
            "total_steps": int(self._extrap_total_steps),
            "fraction": self._extrap_count / n,
        }

    def mean_co2_utilisation(self) -> float:
        """Mean per-step CO₂ utilisation over the episode (0..1+)."""
        if self._co2_utilisation_n == 0:
            return 0.0
        return self._co2_utilisation_sum / self._co2_utilisation_n

    def _check_constraints(self) -> int:
        """Count constraint violations for the current state."""
        violations = 0
        pb = PLANT_BOUNDS

        if self._state["h2_buffer"] <= pb.h2_buffer_min + 1.0:
            violations += 1  # buffer empty — reactor must shut down
        if self._state["h2_buffer"] >= pb.h2_buffer_max - 1.0:
            violations += 1  # buffer full — venting H₂
        if self._state["T"] < pb.T_min + 1 or self._state["T"] > pb.T_max - 1:
            violations += 1  # near temperature limits
        if self._state["P"] < pb.P_min + 1 or self._state["P"] > pb.P_max - 1:
            violations += 1  # near pressure limits

        return violations


# ======================================================================
# Discretised wrapper for tabular Q-learning
# ======================================================================

class DiscretizedMethanolEnv(gym.Wrapper):
    """Wraps ``MethanolPlantEnv`` with a Discrete action space.

    Actions are a flat index into the Cartesian product of:
        load_bins × T_bins × P_bins

    Observations are also discretised into bin indices for
    Q-table lookup (returned as a tuple for dict-based Q-table).

    Parameters
    ----------
    env : MethanolPlantEnv
        The continuous environment to wrap.
    n_load_bins, n_temp_bins, n_pressure_bins : int
        Number of bins per action dimension.
    n_obs_bins : int
        Number of bins per observation dimension for state discretisation.
    """

    def __init__(
        self,
        env: MethanolPlantEnv,
        n_load_bins: int = RL_CFG.n_load_bins,
        n_temp_bins: int = RL_CFG.n_temp_bins,
        n_pressure_bins: int = RL_CFG.n_pressure_bins,
        n_obs_bins: int = 8,
    ) -> None:
        super().__init__(env)
        self.n_load = n_load_bins
        self.n_temp = n_temp_bins
        self.n_pres = n_pressure_bins
        self.n_obs_bins = n_obs_bins

        self.n_actions = n_load_bins * n_temp_bins * n_pressure_bins
        self.action_space = spaces.Discrete(self.n_actions)

        pb = PLANT_BOUNDS
        # Build the discrete action grid
        self._load_grid = np.linspace(0.1, pb.load_max, n_load_bins)
        self._temp_grid = np.linspace(pb.T_min, pb.T_max, n_temp_bins)
        self._pres_grid = np.linspace(pb.P_min, pb.P_max, n_pressure_bins)

        # Obs bin edges for each dimension
        obs_low = env.observation_space.low
        obs_high = env.observation_space.high
        self._obs_edges = [
            np.linspace(obs_low[i], obs_high[i], n_obs_bins + 1)
            for i in range(len(obs_low))
        ]

    def action_to_continuous(self, action_idx: int) -> np.ndarray:
        """Convert flat Discrete index → continuous 3-D action."""
        p_idx = action_idx % self.n_pres
        remainder = action_idx // self.n_pres
        t_idx = remainder % self.n_temp
        l_idx = remainder // self.n_temp
        return np.array([
            self._load_grid[min(l_idx, self.n_load - 1)],
            self._temp_grid[min(t_idx, self.n_temp - 1)],
            self._pres_grid[min(p_idx, self.n_pres - 1)],
        ], dtype=np.float32)

    def discretize_obs(self, obs: np.ndarray) -> Tuple[int, ...]:
        """Digitise continuous obs into a tuple of bin indices."""
        bins = []
        for i, val in enumerate(obs):
            b = int(np.digitize(val, self._obs_edges[i])) - 1
            b = max(0, min(b, self.n_obs_bins - 1))
            bins.append(b)
        return tuple(bins)

    def reset(self, **kwargs) -> Tuple[Tuple[int, ...], Dict[str, Any]]:  # type: ignore[override]
        obs, info = self.env.reset(**kwargs)
        return self.discretize_obs(obs), info

    def step(self, action: int) -> Tuple[Tuple[int, ...], float, bool, bool, Dict[str, Any]]:  # type: ignore[override]
        cont_action = self.action_to_continuous(action)
        obs, reward, terminated, truncated, info = self.env.step(cont_action)
        return self.discretize_obs(obs), reward, terminated, truncated, info
