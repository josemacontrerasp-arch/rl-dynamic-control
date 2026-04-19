"""
Plant Surrogate Models
======================
Wraps the existing simplified process model from
``EURECHA/scripts/surrogate_optimization.py`` and provides
RL-friendly prediction methods.

If pre-trained GPR .pkl/.joblib files are found in the project tree they
are loaded automatically.  Otherwise the existing analytical simplified
model is used directly (fast enough for RL), with placeholder hooks for
future GPR drop-in.

Methods required by the Gymnasium environment:
    predict_h2_production(load_fraction) → H₂ produced (kg/hr)
    predict_methanol_output(h2_feed_kmol, T_C, P_bar) → MeOH (t/hr)
    predict_energy_consumption(load_fraction) → electricity consumed (MW)
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Import from existing project via config (which sets up sys.path)
from ..config import (
    SIMPLIFIED_PROCESS_MODEL,
    ELECTROLYSER_MW,
    ELEC_SPEC_POWER,
    H2_FEED_KMOL,
    H2_FEED_TPH,
    MEOH_PROD_TPH,
    CO2_FEED_KMOL,
    MW_H2,
    MW_MEOH,
    T_REACTOR_NOMINAL,
    P_REACTOR_NOMINAL,
    W_TOTAL_COMP,
    DECISION_VAR_BOUNDS,
    PLANT_BOUNDS,
)

# Paths where trained GP models might live
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_MODEL_SEARCH_PATTERNS = [
    str(_PROJECT_ROOT / "**" / "*.pkl"),
    str(_PROJECT_ROOT / "**" / "*.joblib"),
]


def _find_saved_models() -> list[str]:
    """Search the EURECHA tree for serialised surrogate models."""
    found: list[str] = []
    for pattern in _MODEL_SEARCH_PATTERNS:
        found.extend(glob.glob(pattern, recursive=True))
    return found


class PlantSurrogates:
    """Unified surrogate interface for the CO₂-to-methanol plant.

    Parameters
    ----------
    use_gpr : bool
        If True and .pkl/.joblib files exist, load and use GP regressors.
        Otherwise fall back to the analytical simplified process model from
        ``surrogate_optimization.py``.
    """

    def __init__(self, use_gpr: bool = True) -> None:
        self._gpr_loaded = False
        self._gp_h2 = None
        self._gp_meoh = None
        self._gp_energy = None
        self._gp_lcom = None
        self._gp_util = None
        self._meoh_mu: Optional[np.ndarray] = None
        self._meoh_sig: Optional[np.ndarray] = None
        self._X_mu: Optional[np.ndarray] = None
        self._X_sig: Optional[np.ndarray] = None

        # Training-set predictive-std reference for extrapolation warnings.
        # Populated lazily the first time flag_extrapolation() is called.
        self._train_sigma_meoh: Optional[float] = None

        if use_gpr:
            self._try_load_gpr()

    # ------------------------------------------------------------------
    # GPR loading
    # ------------------------------------------------------------------
    def _try_load_gpr(self) -> None:
        """Load serialised GP models from saved_models/ directory."""
        import pickle

        model_dir = Path(__file__).resolve().parent.parent / "saved_models"
        expected = {
            "gp_h2_production.pkl": "_gp_h2",
            "gp_meoh_output.pkl": "_gp_meoh",
            "gp_energy_consumption.pkl": "_gp_energy",
            "gp_lcom_5d.pkl": "_gp_lcom",
            "gp_util_5d.pkl": "_gp_util",
        }

        loaded_count = 0
        for fname, attr in expected.items():
            path = model_dir / fname
            if path.exists():
                with open(path, "rb") as f:
                    data = pickle.load(f)
                setattr(self, attr, data["model"])

                # Store normalisation stats where available
                if "X_mu" in data and fname == "gp_meoh_output.pkl":
                    self._meoh_mu = data["X_mu"]
                    self._meoh_sig = data["X_sig"]
                elif "X_mu" in data and "5d" in fname:
                    self._X_mu = data["X_mu"]
                    self._X_sig = data["X_sig"]

                loaded_count += 1

        if loaded_count >= 3:  # need at least H2, MeOH, energy
            self._gpr_loaded = True
            print(f"[surrogates] Loaded {loaded_count} GPR models from {model_dir}")
        elif loaded_count > 0:
            print(f"[surrogates] Only {loaded_count}/5 models found — "
                  "falling back to analytical model.")
            self._gpr_loaded = False
        else:
            print("[surrogates] No saved .pkl models found — "
                  "using analytical simplified model.")

    # ------------------------------------------------------------------
    # Core prediction methods
    # ------------------------------------------------------------------
    def predict_h2_production(self, load_fraction: float) -> float:
        """Predict H₂ production rate from electrolyser load.

        Parameters
        ----------
        load_fraction : float
            Electrolyser operating point in [0, 1].

        Returns
        -------
        float
            H₂ produced in **kg/hr**.
        """
        load = np.clip(load_fraction, 0.0, 1.0)

        if self._gpr_loaded and self._gp_h2 is not None:
            pred = self._gp_h2.predict(np.array([[load]]))[0]
            return float(max(pred, 0.0))

        # Analytical fallback
        full_load_h2_kg_hr = ELECTROLYSER_MW * 1e3 / ELEC_SPEC_POWER  # 5400
        efficiency = 1.0 - 0.05 * (1.0 - load) ** 2
        return float(load * full_load_h2_kg_hr * efficiency)

    def predict_methanol_output(
        self,
        h2_feed_kmol: float,
        T_C: float,
        P_bar: float,
    ) -> float:
        """Predict methanol production rate.

        Uses the calibrated simplified process model from
        ``surrogate_optimization.py`` (which matches the Aspen base case
        LCOM = 938 £/t at T=230°C, P=70 bar, F_H2=2704 kmol/h).

        Parameters
        ----------
        h2_feed_kmol : float
            H₂ feed to reactor (kmol/hr).
        T_C : float
            Reactor temperature (°C).
        P_bar : float
            Reactor pressure (bar).

        Returns
        -------
        float
            Methanol production in **t/hr**.
        """
        # Clamp to optimisation bounds
        T_C = np.clip(T_C, DECISION_VAR_BOUNDS[0, 0], DECISION_VAR_BOUNDS[0, 1])
        P_bar = np.clip(P_bar, DECISION_VAR_BOUNDS[1, 0], DECISION_VAR_BOUNDS[1, 1])
        h2_feed_kmol = np.clip(
            h2_feed_kmol,
            DECISION_VAR_BOUNDS[3, 0],
            DECISION_VAR_BOUNDS[3, 1],
        )

        if self._gpr_loaded and self._gp_meoh is not None:
            X = np.array([[h2_feed_kmol, T_C, P_bar]])
            X_n = (X - self._meoh_mu) / self._meoh_sig
            pred = self._gp_meoh.predict(X_n)[0]
            return float(max(pred, 0.0))

        # Analytical fallback
        f_purge = 0.005   # base-case value
        P_col = 1.5       # base-case value
        x = np.array([T_C, P_bar, f_purge, h2_feed_kmol, P_col])
        _lcom, util, production_tpy = SIMPLIFIED_PROCESS_MODEL(x)
        meoh_tph = production_tpy / 8000.0  # t/yr → t/hr

        return float(meoh_tph)

    def predict_energy_consumption(self, load_fraction: float) -> float:
        """Total plant electricity consumption at given electrolyser load.

        Parameters
        ----------
        load_fraction : float
            Electrolyser load in [0, 1].

        Returns
        -------
        float
            Electricity consumed in **MW**.

        Notes
        -----
        Electrolyser dominates: ``load × 270 MW``.
        Compressors (H₂ + CO₂ + pump = 6.14 MW from constants.py)
        scale roughly with throughput.
        """
        load = np.clip(load_fraction, 0.0, 1.0)

        if self._gpr_loaded and self._gp_energy is not None:
            pred = self._gp_energy.predict(np.array([[load]]))[0]
            return float(max(pred, 0.0))

        # Analytical fallback
        elec_mw = load * ELECTROLYSER_MW                      # 0–270 MW
        comp_mw = W_TOTAL_COMP * max(load, 0.1)               # ~0.6–6.14 MW
        return float(elec_mw + comp_mw)

    # ------------------------------------------------------------------
    # Convenience: full step prediction for the environment
    # ------------------------------------------------------------------
    def step(
        self,
        load_fraction: float,
        T_C: float,
        P_bar: float,
    ) -> Tuple[float, float, float]:
        """Run all surrogates for one environment step.

        Returns
        -------
        h2_kg_hr : float
        meoh_tph : float
        elec_mw  : float
        """
        h2_kg_hr = self.predict_h2_production(load_fraction)
        h2_kmol_hr = h2_kg_hr / MW_H2
        meoh_tph = self.predict_methanol_output(h2_kmol_hr, T_C, P_bar)
        elec_mw = self.predict_energy_consumption(load_fraction)
        return h2_kg_hr, meoh_tph, elec_mw

    # ------------------------------------------------------------------
    # Extrapolation monitoring
    # ------------------------------------------------------------------
    def _estimate_train_sigma(self) -> float:
        """Estimate the baseline predictive-std of the MeOH GP at
        a grid of in-bounds points.  Returns the *mean* of these sigmas
        which we treat as the expected "in-distribution" uncertainty.
        """
        if not self._gpr_loaded or self._gp_meoh is None:
            self._train_sigma_meoh = 0.0
            return 0.0

        # Grid across the 3-D operating envelope
        T_lo, T_hi = DECISION_VAR_BOUNDS[0]
        P_lo, P_hi = DECISION_VAR_BOUNDS[1]
        H_lo, H_hi = DECISION_VAR_BOUNDS[3]
        T = np.linspace(T_lo, T_hi, 6)
        P = np.linspace(P_lo, P_hi, 5)
        H = np.linspace(H_lo, H_hi, 5)
        pts = np.array([[h, t, p] for h in H for t in T for p in P])
        try:
            if self._meoh_mu is not None:
                X_n = (pts - self._meoh_mu) / self._meoh_sig
            else:
                X_n = pts
            _, sigmas = self._gp_meoh.predict(X_n, return_std=True)
            self._train_sigma_meoh = float(np.mean(sigmas))
        except Exception:
            self._train_sigma_meoh = 0.0
        return self._train_sigma_meoh

    def flag_extrapolation(
        self,
        load_fraction: float,
        T_C: float,
        P_bar: float,
        sigma_mult: float = 2.0,
    ) -> bool:
        """Return True if the MeOH GP predictive std at this operating
        point exceeds ``sigma_mult * mean_training_sigma``.  Analytical
        fallback never flags (no uncertainty available)."""
        if not self._gpr_loaded or self._gp_meoh is None:
            return False
        if self._train_sigma_meoh is None:
            self._estimate_train_sigma()
        ref = self._train_sigma_meoh or 0.0
        if ref <= 0.0:
            return False

        h2_kg_hr = self.predict_h2_production(load_fraction)
        h2_kmol_hr = h2_kg_hr / MW_H2
        h2_kmol_hr = np.clip(h2_kmol_hr,
                             DECISION_VAR_BOUNDS[3, 0],
                             DECISION_VAR_BOUNDS[3, 1])
        X = np.array([[h2_kmol_hr, T_C, P_bar]])
        if self._meoh_mu is not None:
            X = (X - self._meoh_mu) / self._meoh_sig
        try:
            _, sigma = self._gp_meoh.predict(X, return_std=True)
            return bool(sigma[0] > sigma_mult * ref)
        except Exception:
            return False

    def __repr__(self) -> str:
        mode = "GPR" if self._gpr_loaded else "Analytical (simplified_process_model)"
        return f"PlantSurrogates(mode={mode!r})"
