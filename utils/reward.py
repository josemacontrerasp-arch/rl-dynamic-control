"""
Reward Function for RL Dynamic Control
======================================
Computes the economic reward for operating the methanol plant.

    reward = methanol_revenue + co2_credits - electricity_cost
             - water_cost - ramp_penalty - constraint_penalty

All monetary values are converted to a per-step quantity, then scaled by
``reward_scale`` to keep returns ~O(1) for stable RL training.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import (
    CO2_FEED_TPH,
    E_MEOH_LOW,
    MEOH_PROD_TPH,
    RL_CFG,
    UTILITY_PER_T_MEOH,
)


@dataclass
class RewardConfig:
    """Tunable reward parameters derived from project economics."""

    meoh_price: float = E_MEOH_LOW
    co2_credit_price: float = 80.0
    co2_per_t_meoh: float = CO2_FEED_TPH / MEOH_PROD_TPH
    water_and_utility_per_t: float = UTILITY_PER_T_MEOH
    constraint_penalty: float = RL_CFG.constraint_penalty
    ramp_penalty_coeff: float = RL_CFG.ramp_penalty_coeff
    reward_scale: float = RL_CFG.reward_scale
    use_reward_shaping: bool = RL_CFG.use_reward_shaping


def compute_electricity_cost(
    elec_mw: float,
    elec_price: float,
    config: RewardConfig | None = None,
) -> float:
    """Return the scaled electricity cost for one control step."""
    cfg = config or RewardConfig()
    electricity_cost = elec_mw * elec_price * RL_CFG.dt_hours
    return float(electricity_cost * cfg.reward_scale)


def compute_electricity_consumption_mwh(elec_mw: float) -> float:
    """Return electricity consumed during one control step in MWh."""
    return float(elec_mw * RL_CFG.dt_hours)


def compute_reward(
    meoh_tph: float,
    elec_mw: float,
    elec_price: float,
    ramp: float = 0.0,
    constraint_violations: int = 0,
    config: RewardConfig | None = None,
) -> float:
    """Compute the scaled scalar reward for one control step."""
    cfg = config or RewardConfig()

    methanol_revenue = meoh_tph * cfg.meoh_price * RL_CFG.dt_hours
    co2_utilised_tph = meoh_tph * cfg.co2_per_t_meoh
    co2_credits = co2_utilised_tph * cfg.co2_credit_price * RL_CFG.dt_hours
    electricity_cost = elec_mw * elec_price * RL_CFG.dt_hours
    water_cost = meoh_tph * cfg.water_and_utility_per_t * RL_CFG.dt_hours
    ramp_penalty = cfg.ramp_penalty_coeff * ramp
    violation_penalty = cfg.constraint_penalty * constraint_violations

    profit = (
        methanol_revenue
        + co2_credits
        - electricity_cost
        - water_cost
        - ramp_penalty
        + violation_penalty
    )

    return float(profit * cfg.reward_scale)
