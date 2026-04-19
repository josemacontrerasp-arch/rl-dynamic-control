"""
Reward Function for RL Dynamic Control
=======================================
Computes the hourly economic reward for operating the methanol plant.

    reward = methanol_revenue + co2_credits − electricity_cost
             − water_cost − ramp_penalty − constraint_penalty

All monetary values are in **£/hr** then scaled by ``reward_scale``
to keep returns ~O(1) for stable RL training.

Economic parameters are imported from the master
``EURECHA/figures/scripts/constants.py`` via ``config.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import (
    E_MEOH_LOW,
    COMMODITY_MEOH,
    CO2_CAPTURE_COST,
    CO2_FEED_TPH,
    MEOH_PROD_TPH,
    CW_PRICE_M3,
    UTILITY_PER_T_MEOH,
    RL_CFG,
)


@dataclass
class RewardConfig:
    """Tunable reward parameters.

    Defaults are derived from real economics in the EURECHA project.
    """

    # Revenue per tonne MeOH (£/t).  Use e-methanol low estimate as
    # default; commodity price available as alternative baseline.
    meoh_price: float = E_MEOH_LOW                     # £800/t

    # CO₂ credit price (£/t CO₂ utilised).
    # UK ETS ~£50–100/t;  use a mid estimate.
    co2_credit_price: float = 80.0                     # £/t CO₂

    # CO₂ consumed per tonne MeOH (from constants.py)
    co2_per_t_meoh: float = CO2_FEED_TPH / MEOH_PROD_TPH  # ~1.49 t CO₂/t MeOH

    # Water cost (£/t MeOH) — from constants UTILITY_PER_T_MEOH
    # The water component is small relative to electricity; lump it in.
    water_and_utility_per_t: float = UTILITY_PER_T_MEOH   # £/t MeOH

    # Constraint violation penalty per event per step
    constraint_penalty: float = RL_CFG.constraint_penalty  # −500

    # Ramp penalty coefficient (£ per unit load change per step)
    ramp_penalty_coeff: float = RL_CFG.ramp_penalty_coeff  # 5.0

    # Overall scale factor to bring reward to ~O(1)
    reward_scale: float = RL_CFG.reward_scale              # 1e-3

    # Reward shaping: bonus for keeping H₂ buffer at 40-60% capacity
    use_reward_shaping: bool = RL_CFG.use_reward_shaping


def compute_reward(
    meoh_tph: float,
    elec_mw: float,
    elec_price: float,
    ramp: float = 0.0,
    constraint_violations: int = 0,
    config: RewardConfig | None = None,
) -> float:
    """Compute the scalar reward for one hourly time step.

    Parameters
    ----------
    meoh_tph : float
        Methanol production rate (t/hr).
    elec_mw : float
        Total electricity consumption (MW).
    elec_price : float
        Spot electricity price this hour (£/MWh).
    ramp : float
        Absolute change in electrolyser load fraction since last step.
    constraint_violations : int
        Number of constraint violations this step.
    config : RewardConfig, optional

    Returns
    -------
    float
        Scaled hourly reward (£ × reward_scale).
    """
    cfg = config or RewardConfig()

    # ── Revenue (£/hr) ─────────────────────────────────────────────────
    methanol_revenue = meoh_tph * cfg.meoh_price

    # ── CO₂ credits (£/hr) ────────────────────────────────────────────
    co2_utilised_tph = meoh_tph * cfg.co2_per_t_meoh
    co2_credits = co2_utilised_tph * cfg.co2_credit_price

    # ── Electricity cost (£/hr) ───────────────────────────────────────
    electricity_cost = elec_mw * elec_price   # MW × £/MWh = £/hr

    # ── Water & utility cost (£/hr) ───────────────────────────────────
    water_cost = meoh_tph * cfg.water_and_utility_per_t

    # ── Penalties ──────────────────────────────────────────────────────
    ramp_penalty = cfg.ramp_penalty_coeff * ramp
    violation_penalty = cfg.constraint_penalty * constraint_violations

    # ── Total ──────────────────────────────────────────────────────────
    profit = (
        methanol_revenue
        + co2_credits
        - electricity_cost
        - water_cost
        - ramp_penalty
        + violation_penalty      # negative when violations > 0
    )

    return float(profit * cfg.reward_scale)
