"""
Surrogate Version Manager
=========================
Version-aware loader for GPR surrogates.  Supports ``v1`` (original
500-point LHS) and ``v2`` (augmented with agent-trajectory data).

Usage::

    from rl_dynamic_control.models.surrogate_manager import load_surrogates

    surrogates = load_surrogates(version="v2")   # prefer v2
    surrogates = load_surrogates(version="v1")   # force original
    surrogates = load_surrogates(version="auto")  # v2 if available, else v1
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

from .surrogates import PlantSurrogates

LOGGER = logging.getLogger("surrogate_manager")

_RL_DIR = Path(__file__).resolve().parent.parent
_SAVED_MODELS = _RL_DIR / "saved_models"
_V2_DIR = _SAVED_MODELS / "surrogates_v2"

# Minimum files required for a valid v2 installation
_V2_REQUIRED = ["gp_meoh_output.pkl", "gp_lcom_5d.pkl", "gp_util_5d.pkl"]


def v2_available() -> bool:
    """Check whether v2 surrogates exist on disk."""
    return all((_V2_DIR / f).exists() for f in _V2_REQUIRED)


def load_surrogates(
    version: str = "auto",
    use_gpr: bool = True,
) -> PlantSurrogates:
    """Load surrogates with version control.

    Parameters
    ----------
    version : str
        ``"v1"`` — original surrogates from ``saved_models/``.
        ``"v2"`` — augmented surrogates from ``saved_models/surrogates_v2/``.
        ``"auto"`` — v2 if available, otherwise v1.
    use_gpr : bool
        If False, use analytical fallback regardless of version.

    Returns
    -------
    PlantSurrogates
        Configured surrogate instance.
    """
    if version == "auto":
        version = "v2" if v2_available() else "v1"
        LOGGER.info("Auto-selected surrogate version: %s", version)

    if version == "v2":
        if not v2_available():
            LOGGER.warning(
                "v2 surrogates not found at %s — falling back to v1. "
                "Run `python -m rl_dynamic_control.models.retrain_surrogates` first.",
                _V2_DIR,
            )
            return _load_v1(use_gpr)
        return _load_v2(use_gpr)

    if version == "v1":
        return _load_v1(use_gpr)

    raise ValueError(f"Unknown surrogate version: {version!r}. Use 'v1', 'v2', or 'auto'.")


def _load_v1(use_gpr: bool) -> PlantSurrogates:
    """Load original v1 surrogates (standard PlantSurrogates init)."""
    LOGGER.info("Loading v1 surrogates from %s", _SAVED_MODELS)
    surr = PlantSurrogates(use_gpr=use_gpr)
    surr._surrogate_version = "v1"
    return surr


def _load_v2(use_gpr: bool) -> PlantSurrogates:
    """Load v2 surrogates from the versioned directory."""
    LOGGER.info("Loading v2 surrogates from %s", _V2_DIR)

    surr = PlantSurrogates(use_gpr=False)  # skip default loading
    surr._surrogate_version = "v2"

    if not use_gpr:
        LOGGER.info("GPR disabled — using analytical fallback")
        return surr

    # Load v2 models manually
    loaded = 0
    expected = {
        "gp_h2_production.pkl": "_gp_h2",
        "gp_meoh_output.pkl": "_gp_meoh",
        "gp_energy_consumption.pkl": "_gp_energy",
        "gp_lcom_5d.pkl": "_gp_lcom",
        "gp_util_5d.pkl": "_gp_util",
    }

    for fname, attr in expected.items():
        path = _V2_DIR / fname
        if path.exists():
            with open(path, "rb") as f:
                data = pickle.load(f)
            setattr(surr, attr, data["model"])

            # Normalisation stats
            if "X_mu" in data and fname == "gp_meoh_output.pkl":
                surr._meoh_mu = data["X_mu"]
                surr._meoh_sig = data["X_sig"]
            elif "X_mu" in data and "5d" in fname:
                surr._X_mu = data["X_mu"]
                surr._X_sig = data["X_sig"]

            loaded += 1

    if loaded >= 3:
        surr._gpr_loaded = True
        LOGGER.info("Loaded %d v2 GPR models", loaded)
    else:
        LOGGER.warning("Only %d v2 models found — falling back to analytical", loaded)
        surr._gpr_loaded = False

    # Re-estimate training sigma for the new model
    surr._train_sigma_meoh = None  # force re-estimation on first call
    surr._estimate_train_sigma()

    return surr


def get_surrogate_version(surrogates: PlantSurrogates) -> str:
    """Return the version string of a loaded surrogate instance."""
    return getattr(surrogates, "_surrogate_version", "v1")
