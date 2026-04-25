"""
Surrogate Version Manager
=========================
Version-aware loader for GPR surrogates.  Supports ``v1`` (original
500-point LHS), ``v2`` (augmented with agent-trajectory data), and
``v3`` (LCOM/util retrained with extended F_H2 coverage).

Usage::

    from rl_dynamic_control.models.surrogate_manager import load_surrogates

    surrogates = load_surrogates(version="v2")   # prefer v2
    surrogates = load_surrogates(version="v3")   # force v3
    surrogates = load_surrogates(version="v1")   # force original
    surrogates = load_surrogates(version="auto")  # newest available, else v1
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
_V3_DIR = _SAVED_MODELS / "surrogates_v3"

# Minimum files required for a valid versioned installation
_VERSION_REQUIRED = ["gp_meoh_output.pkl", "gp_lcom_5d.pkl", "gp_util_5d.pkl"]


def _version_dir(version: str) -> Path:
    if version == "v2":
        return _V2_DIR
    if version == "v3":
        return _V3_DIR
    raise ValueError(f"Unsupported versioned surrogate directory: {version!r}")


def _version_available(version: str) -> bool:
    model_dir = _version_dir(version)
    return all((model_dir / f).exists() for f in _VERSION_REQUIRED)


def v2_available() -> bool:
    """Check whether v2 surrogates exist on disk."""
    return _version_available("v2")


def v3_available() -> bool:
    """Check whether v3 surrogates exist on disk."""
    return _version_available("v3")


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
        ``"v3"`` — retrained surrogates from ``saved_models/surrogates_v3/``.
        ``"auto"`` — newest available version, otherwise v1.
    use_gpr : bool
        If False, use analytical fallback regardless of version.

    Returns
    -------
    PlantSurrogates
        Configured surrogate instance.
    """
    if version == "auto":
        if v3_available():
            version = "v3"
        elif v2_available():
            version = "v2"
        else:
            version = "v1"
        LOGGER.info("Auto-selected surrogate version: %s", version)

    if version == "v3":
        if not v3_available():
            LOGGER.warning(
                "v3 surrogates not found at %s — falling back to v2/v1.",
                _V3_DIR,
            )
            if v2_available():
                return _load_versioned("v2", use_gpr)
            return _load_v1(use_gpr)
        return _load_versioned("v3", use_gpr)

    if version == "v2":
        if not v2_available():
            LOGGER.warning(
                "v2 surrogates not found at %s — falling back to v1. "
                "Run `python -m rl_dynamic_control.models.retrain_surrogates` first.",
                _V2_DIR,
            )
            return _load_v1(use_gpr)
        return _load_versioned("v2", use_gpr)

    if version == "v1":
        return _load_v1(use_gpr)

    raise ValueError(
        f"Unknown surrogate version: {version!r}. Use 'v1', 'v2', 'v3', or 'auto'."
    )


def _load_v1(use_gpr: bool) -> PlantSurrogates:
    """Load original v1 surrogates (standard PlantSurrogates init)."""
    LOGGER.info("Loading v1 surrogates from %s", _SAVED_MODELS)
    surr = PlantSurrogates(use_gpr=use_gpr)
    surr._surrogate_version = "v1"
    return surr


def _load_versioned(version: str, use_gpr: bool) -> PlantSurrogates:
    """Load versioned surrogates from their dedicated directory."""
    model_dir = _version_dir(version)
    LOGGER.info("Loading %s surrogates from %s", version, model_dir)

    surr = PlantSurrogates(use_gpr=False)  # skip default loading
    surr._surrogate_version = version

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
        path = model_dir / fname
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
        LOGGER.info("Loaded %d %s GPR models", loaded, version)
    else:
        LOGGER.warning(
            "Only %d %s models found — falling back to analytical", loaded, version
        )
        surr._gpr_loaded = False

    # Re-estimate training sigma for the new model
    surr._train_sigma_meoh = None  # force re-estimation on first call
    surr._estimate_train_sigma()

    return surr


def get_surrogate_version(surrogates: PlantSurrogates) -> str:
    """Return the version string of a loaded surrogate instance."""
    return getattr(surrogates, "_surrogate_version", "v1")
