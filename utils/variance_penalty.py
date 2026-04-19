"""
Variance-Penalised Reward (Approach B)
======================================
Adds an exploration-safety penalty to the base reward that discourages
the RL agent from operating in regions where the GPR surrogates have
high predictive uncertainty.

    R_safe = R_base - alpha * max(0, sigma_gpr - sigma_threshold)

where:
    sigma_gpr       = GPR predictive std at the current state
    sigma_threshold  = 2 * mean_training_sigma (the extrapolation
                       warning threshold already used by
                       ``PlantSurrogates.flag_extrapolation()``)
    alpha            = penalty coefficient (configurable)

The penalty magnitude is logged separately so it can be analysed
independently of the base reward.

Usage::

    from rl_dynamic_control.utils.variance_penalty import (
        VariancePenaltyConfig,
        compute_variance_penalty,
    )

    cfg = VariancePenaltyConfig(alpha=0.5)
    penalty, sigma = compute_variance_penalty(surrogates, load, T, P, cfg)
    reward_safe = reward_base - penalty
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..config import DECISION_VAR_BOUNDS, MW_H2

LOGGER = logging.getLogger("variance_penalty")


@dataclass
class VariancePenaltyConfig:
    """Configuration for the GPR variance penalty.

    Parameters
    ----------
    alpha : float
        Penalty coefficient.  The penalty is
        ``alpha * max(0, sigma - sigma_threshold)``.
        Good starting value: ``0.5 * mean_reward / mean_excess_sigma``
        estimated from evaluation logs.  Default 2.0 is a reasonable
        starting point for the methanol plant (scaled reward ~O(1)).
    sigma_threshold_mult : float
        Multiplier on the mean training-set sigma.  Points with
        sigma > mult * mean_train_sigma are penalised.  Default 2.0
        matches the existing extrapolation warning threshold.
    enabled : bool
        If False, no penalty is applied (backward compatible).
    """

    alpha: float = 2.0
    sigma_threshold_mult: float = 2.0
    enabled: bool = True


def compute_variance_penalty(
    surrogates: "PlantSurrogates",
    load_fraction: float,
    T_C: float,
    P_bar: float,
    config: Optional[VariancePenaltyConfig] = None,
) -> Tuple[float, float]:
    """Compute the GPR-variance penalty for a single state.

    Parameters
    ----------
    surrogates : PlantSurrogates
        Must have GPR models loaded.
    load_fraction : float
        Electrolyser load [0, 1].
    T_C : float
        Reactor temperature (deg C).
    P_bar : float
        Reactor pressure (bar).
    config : VariancePenaltyConfig, optional

    Returns
    -------
    penalty : float
        Non-negative penalty to subtract from the base reward.
        Zero if the config is disabled or surrogates have no GPR.
    sigma_gpr : float
        Raw GPR predictive std at this point (for logging).
    """
    cfg = config or VariancePenaltyConfig()

    if not cfg.enabled:
        return 0.0, 0.0

    # Get the GPR predictive sigma
    sigma_gpr = _get_gpr_sigma(surrogates, load_fraction, T_C, P_bar)

    if sigma_gpr <= 0.0:
        return 0.0, 0.0

    # Get the training-set reference sigma
    sigma_ref = _get_train_sigma(surrogates)
    if sigma_ref <= 0.0:
        return 0.0, sigma_gpr

    sigma_threshold = cfg.sigma_threshold_mult * sigma_ref
    excess = max(0.0, sigma_gpr - sigma_threshold)
    penalty = cfg.alpha * excess

    return float(penalty), float(sigma_gpr)


def estimate_alpha(
    mean_reward: float,
    mean_excess_sigma: float,
    scale: float = 0.5,
) -> float:
    """Estimate a good alpha from evaluation statistics.

    Parameters
    ----------
    mean_reward : float
        Mean per-step reward from evaluation logs.
    mean_excess_sigma : float
        Mean of ``max(0, sigma - threshold)`` from evaluation logs.
    scale : float
        Fraction of reward that the penalty should represent at the
        mean excess sigma.  0.5 means the penalty equals half the
        reward when sigma is at the mean excess level.

    Returns
    -------
    float
        Suggested alpha value.
    """
    if mean_excess_sigma <= 0:
        LOGGER.warning("No excess sigma detected — alpha estimation not possible")
        return 2.0  # safe default

    alpha = scale * abs(mean_reward) / mean_excess_sigma
    LOGGER.info(
        "Estimated alpha = %.3f (mean_reward=%.4f, mean_excess_sigma=%.4f, scale=%.2f)",
        alpha, mean_reward, mean_excess_sigma, scale,
    )
    return float(alpha)


def _get_gpr_sigma(
    surrogates: "PlantSurrogates",
    load: float,
    T: float,
    P: float,
) -> float:
    """Extract GPR predictive std for the MeOH surrogate."""
    if not surrogates._gpr_loaded or surrogates._gp_meoh is None:
        return 0.0

    h2_kg_hr = surrogates.predict_h2_production(load)
    h2_kmol = np.clip(
        h2_kg_hr / MW_H2,
        DECISION_VAR_BOUNDS[3, 0],
        DECISION_VAR_BOUNDS[3, 1],
    )
    X = np.array([[h2_kmol, T, P]])
    if surrogates._meoh_mu is not None:
        X = (X - surrogates._meoh_mu) / surrogates._meoh_sig
    try:
        _, sigma = surrogates._gp_meoh.predict(X, return_std=True)
        return float(sigma[0])
    except Exception:
        return 0.0


def _get_train_sigma(surrogates: "PlantSurrogates") -> float:
    """Get or compute the training-set reference sigma."""
    if surrogates._train_sigma_meoh is None:
        surrogates._estimate_train_sigma()
    return surrogates._train_sigma_meoh or 0.0
