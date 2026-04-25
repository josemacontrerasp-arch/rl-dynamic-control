"""
RL Dynamic Control — Configuration
===================================
Imports plant parameters from the existing EURECHA constants module
and adds RL-specific hyperparameters on top.

All physical / economic values are single-sourced from:
    ``EURECHA/figures/scripts/constants.py``
    ``EURECHA/scripts/surrogate_optimization.py``
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# Make the EURECHA package importable so we can pull from existing modules.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent          # …/EURECHA
_FIGURES_SCRIPTS = _PROJECT_ROOT / "figures" / "scripts"
_SCRIPTS = _PROJECT_ROOT / "scripts"

for _p in (_PROJECT_ROOT, _FIGURES_SCRIPTS, _SCRIPTS):
    _p_str = str(_p)
    if _p_str not in sys.path:
        sys.path.insert(0, _p_str)

# ---------------------------------------------------------------------------
# Import ALL plant parameters from the master constants module.
# ---------------------------------------------------------------------------
import constants as _c                       # figures/scripts/constants.py
import surrogate_optimization as _so         # scripts/surrogate_optimization.py

# ── Feed & product (from constants.py) ────────────────────────────────────
CO2_FEED_KMOL: float       = _c.CO2_FEED_KMOL        # 899.0 kmol/hr
H2_FEED_KMOL: float        = _c.H2_FEED_KMOL         # 2704.0 kmol/hr
MEOH_PROD_KMOL: float      = _c.MEOH_PROD_KMOL       # 830.0 kmol/hr
H2_CO2_RATIO: float        = _c.H2_CO2_RATIO         # 3.00
CO2_FEED_TPH: float        = _c.CO2_FEED_TPH         # 39.6 t/hr
H2_FEED_TPH: float         = _c.H2_FEED_TPH          # 5.45 t/hr
MEOH_PROD_TPH: float       = _c.MEOH_PROD_TPH        # 26.6 t/hr
MEOH_ANNUAL_T: float       = _c.MEOH_ANNUAL_T        # 212 800 t/yr
OPERATING_HOURS: float     = _c.OPERATING_HOURS       # 8000 hr/yr

# ── Reactor (from constants.py) ───────────────────────────────────────────
T_REACTOR_NOMINAL: float   = _c.T_REACTOR             # 230 °C
P_REACTOR_NOMINAL: float   = _c.P_REACTOR             # 70 bar
PERPASS_CONV: float        = _c.PERPASS_CONV           # 0.629
OVERALL_CONV: float        = _c.OVERALL_CONV           # 0.997

# ── Electrolyser (from constants.py) ──────────────────────────────────────
ELECTROLYSER_MW: float     = _c.ELECTROLYSER_MW       # 270 MW
ELEC_SPEC_POWER: float     = _c.ELEC_SPEC_POWER       # 50 kWh/kg H2
ELEC_CAPEX_PER_KW: float   = _c.ELEC_CAPEX_PER_KW    # £1500/kW

# ── Economics (from constants.py) ─────────────────────────────────────────
TOTAL_CAPEX_M: float       = _c.TOTAL_CAPEX_M         # £503 M
ANNUALISED_CAPEX: float    = _c.ANNUALISED_CAPEX      # £/t MeOH
CO2_CAPTURE_COST: float    = _c.CO2_CAPTURE_COST      # £60/t CO₂
OTHER_OPEX: float          = _c.OTHER_OPEX            # £50/t MeOH
COMMODITY_MEOH: float      = _c.COMMODITY_MEOH        # £575/t
E_MEOH_LOW: float          = _c.E_MEOH_LOW            # £800/t
E_MEOH_HIGH: float         = _c.E_MEOH_HIGH           # £1200/t
ELEC_PRICE_SCENARIOS: Dict[str, float] = _c.ELEC_PRICE_SCENARIOS

# ── Utility costs (from constants.py) ─────────────────────────────────────
CW_PRICE_M3: float         = _c.CW_PRICE_M3           # £0.03/m³
NG_PRICE_MWH: float        = _c.NG_PRICE_MWH          # £30/MWh
STEAM_COST_PER_MWH: float  = _c.STEAM_COST_PER_MWH    # £/MWh(steam)
UTILITY_PER_T_MEOH: float  = _c.UTILITY_PER_T_MEOH    # £/t MeOH

# ── Duties (from constants.py) ────────────────────────────────────────────
W_TOTAL_COMP: float        = _c.W_TOTAL_COMP          # 6.14 MW
Q_STRIPPER_REBOILER: float = _c.Q_STRIPPER_REBOILER   # 50.0 MW

# ── Existing LCOM function ────────────────────────────────────────────────
lcom = _c.lcom  # lcom(elec_price) → £/t MeOH

# ── Surrogate optimisation bounds (from surrogate_optimization.py) ────────
import numpy as np

DECISION_VAR_BOUNDS: np.ndarray = _so.BOUNDS.copy()
# [T_rxn(°C), P_rxn(bar), f_purge, F_H2(kmol/h), P_col(bar)]
#  [210, 280], [50, 100], [0.003, 0.10], [2515, 3147], [1.0, 3.0]

SIMPLIFIED_PROCESS_MODEL = _so.simplified_process_model
# simplified_process_model(x) → (lcom, util, production)

# Molecular weights (from surrogate_optimization.py)
MW_H2: float   = _so.MW_H2     # 2.016 kg/kmol
MW_MEOH: float = _so.MW_MeOH   # 32.04 kg/kmol

# ===================================================================
# RL-SPECIFIC CONFIGURATION
# ===================================================================


@dataclass
class RLConfig:
    """Hyperparameters for RL training."""

    # ── Episode structure ──────────────────────────────────────────────
    episode_length: int = 168          # steps (= 1 week at hourly resolution)
    dt_hours: float = 1.0              # time step in hours

    # ── Discount & learning ────────────────────────────────────────────
    gamma: float = 0.99                # discount factor
    learning_rate: float = 3e-4        # for SB3 (Adam)
    q_learning_rate: float = 0.1       # for tabular Q-learning (α)

    # ── Epsilon-greedy (tabular Q-learning) ────────────────────────────
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.9995      # multiplicative per step

    # ── SB3 algorithm choice ───────────────────────────────────────────
    sb3_algo: str = "PPO"              # "PPO" or "SAC"
    total_timesteps: int = 500_000
    n_eval_episodes: int = 10
    ppo_n_steps: int = 2048
    ppo_batch_size: int = 64
    ppo_n_epochs: int = 10
    ppo_clip_range: float = 0.2
    sac_batch_size: int = 256
    sac_buffer_size: int = 100_000
    sac_learning_starts: int = 1000
    sac_tau: float = 0.005
    vecnorm_clip_obs: float = 10.0

    # ── Q-learning discretisation ──────────────────────────────────────
    n_load_bins: int = 10              # electrolyser load [0.1 … 1.0]
    n_temp_bins: int = 5               # reactor T bins
    n_pressure_bins: int = 5           # reactor P bins

    # ── Reward shaping ─────────────────────────────────────────────────
    constraint_penalty: float = -500.0          # per-step penalty for violations
    ramp_penalty_coeff: float = 5.0             # £ per 10% load change
    reward_scale: float = 1e-3                  # scale reward to ~O(1)
    use_reward_shaping: bool = True

    # ── Variance penalty (Approach B: extrapolation fix) ──────────────
    use_variance_penalty: bool = False           # off by default for backward compat
    variance_penalty_alpha: float = 2.0          # penalty coefficient
    variance_penalty_sigma_mult: float = 2.0     # threshold = mult × mean_train_sigma

    # ── Surrogate versioning (Approach A: extrapolation fix) ──────────
    surrogate_version: str = "auto"              # "v1", "v2", or "auto"

    # ── TensorBoard ────────────────────────────────────────────────────
    tb_log_dir: str = "tb_logs"

    # ── Random seed ────────────────────────────────────────────────────
    seed: int = 42


@dataclass
class PlantBounds:
    """Operating bounds for the methanol plant (from existing project).

    All values sourced from EURECHA/scripts/surrogate_optimization.py BOUNDS
    and EURECHA/figures/scripts/constants.py.
    """

    # Electrolyser load fraction [0, 1] where 1 = full 270 MW
    load_min: float = 0.0
    load_max: float = 1.0

    # Reactor temperature (°C) — from DECISION_VAR_BOUNDS row 0
    T_min: float = float(DECISION_VAR_BOUNDS[0, 0])   # 210
    T_max: float = float(DECISION_VAR_BOUNDS[0, 1])   # 280

    # Reactor pressure (bar) — from DECISION_VAR_BOUNDS row 1
    P_min: float = float(DECISION_VAR_BOUNDS[1, 0])   # 50
    P_max: float = float(DECISION_VAR_BOUNDS[1, 1])   # 100

    # H₂ buffer tank level (kg) — sized for ~2 h of full production
    h2_buffer_min: float = 0.0
    h2_buffer_max: float = H2_FEED_TPH * 1000 * 2.0   # ~10 900 kg

    # Methanol production rate (t/hr) — 0 to ~30 t/hr
    meoh_rate_min: float = 0.0
    meoh_rate_max: float = MEOH_PROD_TPH * 1.15        # 15% headroom

    # H₂ feed range (kmol/hr) — from DECISION_VAR_BOUNDS row 3
    F_H2_min: float = float(DECISION_VAR_BOUNDS[3, 0])  # 2515
    F_H2_max: float = float(DECISION_VAR_BOUNDS[3, 1])  # 3147

    # Methanol purity specification (mass fraction)
    meoh_purity_spec: float = 0.995


@dataclass
class ElectricityPriceConfig:
    """Configuration for electricity price data."""

    # Path to price CSV (relative to rl_dynamic_control/)
    csv_path: str = "data/electricity_prices.csv"

    # Price statistics (European day-ahead, £/MWh)
    mean_price: float = 50.0           # ~midpoint of scenarios
    scenarios: Dict[str, float] = field(
        default_factory=lambda: dict(ELEC_PRICE_SCENARIOS)
    )


# ── Convenience singletons ────────────────────────────────────────────────
RL_CFG = RLConfig()
PLANT_BOUNDS = PlantBounds()
PRICE_CFG = ElectricityPriceConfig()
