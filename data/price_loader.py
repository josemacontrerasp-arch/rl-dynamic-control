"""
Unified Electricity Price Loader
=================================
Auto-detects all price CSVs in ``rl_dynamic_control/data/`` and exposes
them as named datasets with train/test splits by date range or season.

Registered datasets (discovered dynamically):
    * ``synthetic_gb``   — data/electricity_prices.csv
    * ``real_gb``        — data/electricity_prices_real.csv
    * ``entso_nl_all``   — data/entso_nl_2023_2024.csv   (full 2 years)
    * ``entso_nl_2023``  — slice of entso_nl_all
    * ``entso_nl_2024``  — slice of entso_nl_all

API
---
    loader = PriceLoader()
    loader.available()                    -> list[str]
    loader.load(name)                     -> np.ndarray  (£/MWh float32)
    loader.load_df(name)                  -> pd.DataFrame
    loader.split(name, by="season", ...)  -> {"Q1": ndarray, ...}
    loader.split(name, by="date", train_end="2024-06-30")
    loader.currency_units(name)           -> "EUR" | "GBP"

The returned ``np.ndarray`` uses the same magnitude as the legacy CSVs:
the GBP per-MWh column when present, otherwise EUR per-MWh.  Environment
and reward code continue to treat this as a price-per-MWh regardless of
the currency label, so relative comparisons (and all training dynamics)
are unaffected.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("price_loader")

_DATA_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
def _pick_price_column(df: pd.DataFrame) -> str:
    """Return the preferred numerical price column name."""
    for col in ("price_gbp_per_mwh", "price_eur_mwh", "price"):
        if col in df.columns:
            return col
    raise KeyError(f"No price column found; columns={list(df.columns)}")


def _currency_of(df: pd.DataFrame) -> str:
    col = _pick_price_column(df)
    if "eur" in col:
        return "EUR"
    return "GBP"


class PriceLoader:
    """Unified loader that auto-discovers price CSVs."""

    def __init__(self, data_dir: Path | str = _DATA_DIR) -> None:
        self.data_dir = Path(data_dir)
        self._registry: Dict[str, Path] = {}
        self._cache: Dict[str, pd.DataFrame] = {}
        self._discover()

    # -- discovery ----------------------------------------------------------
    def _discover(self) -> None:
        mapping = {
            "synthetic_gb":    "electricity_prices.csv",
            "real_gb":         "electricity_prices_real.csv",
            "entso_nl_all":    "entso_nl_2023_2024.csv",
        }
        for name, fname in mapping.items():
            p = self.data_dir / fname
            if p.exists():
                self._registry[name] = p

        # Derived yearly slices
        if "entso_nl_all" in self._registry:
            self._registry["entso_nl_2023"] = self._registry["entso_nl_all"]
            self._registry["entso_nl_2024"] = self._registry["entso_nl_all"]

        LOGGER.info(f"Discovered price datasets: {sorted(self._registry)}")

    # -- public API ---------------------------------------------------------
    def available(self) -> List[str]:
        return sorted(self._registry)

    def load_df(self, name: str) -> pd.DataFrame:
        if name not in self._registry:
            raise KeyError(f"Unknown dataset {name!r}. Available: {self.available()}")
        if name in self._cache:
            df = self._cache[name]
        else:
            df = pd.read_csv(self._registry[name])
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            self._cache[name] = df

        # Apply year filter for derived slices
        if name == "entso_nl_2023":
            return df[df["timestamp"].dt.year == 2023].reset_index(drop=True)
        if name == "entso_nl_2024":
            return df[df["timestamp"].dt.year == 2024].reset_index(drop=True)
        return df

    def load(self, name: str) -> np.ndarray:
        """Return a 1-D float32 price array (magnitude compatible with env)."""
        df = self.load_df(name)
        col = _pick_price_column(df)
        return df[col].to_numpy(dtype=np.float32)

    def currency_units(self, name: str) -> str:
        return _currency_of(self.load_df(name))

    # -- splits -------------------------------------------------------------
    def split(
        self,
        name: str,
        by: str = "season",
        train_end: Optional[str] = None,
        train_months: Optional[List[int]] = None,
    ) -> Dict[str, np.ndarray]:
        """Return a dict of named splits suitable for robustness testing.

        ``by="season"``    -> {"Q1": ..., "Q2": ..., "Q3": ..., "Q4": ...}
        ``by="halves"``    -> {"H1": Q1+Q2, "H2": Q3+Q4}
        ``by="date"``      -> {"train": ..., "test": ...} using ``train_end``
        ``by="months"``    -> {"train": months ∈ train_months, "test": rest}
        """
        df = self.load_df(name)
        col = _pick_price_column(df)
        if "timestamp" not in df.columns:
            # Synthetic CSVs without timestamps get a synthetic hourly index
            df = df.copy()
            df["timestamp"] = pd.date_range(
                "2024-01-01", periods=len(df), freq="1h",
            )

        t = df["timestamp"]
        p = df[col].to_numpy(dtype=np.float32)

        if by == "season":
            out = {}
            for q, months in {"Q1": [1, 2, 3], "Q2": [4, 5, 6],
                              "Q3": [7, 8, 9], "Q4": [10, 11, 12]}.items():
                mask = t.dt.month.isin(months).to_numpy()
                out[q] = p[mask]
            return out

        if by == "halves":
            h1 = t.dt.month.isin([1, 2, 3, 4, 5, 6]).to_numpy()
            return {"H1": p[h1], "H2": p[~h1]}

        if by == "date":
            if train_end is None:
                raise ValueError("by='date' requires train_end")
            mask = (t <= pd.Timestamp(train_end)).to_numpy()
            return {"train": p[mask], "test": p[~mask]}

        if by == "months":
            if not train_months:
                raise ValueError("by='months' requires train_months list")
            mask = t.dt.month.isin(train_months).to_numpy()
            return {"train": p[mask], "test": p[~mask]}

        raise ValueError(f"Unknown split mode {by!r}")


# Convenience singleton ------------------------------------------------------
_default_loader: Optional[PriceLoader] = None


def get_loader() -> PriceLoader:
    global _default_loader
    if _default_loader is None:
        _default_loader = PriceLoader()
    return _default_loader


def load_prices(name: str = "auto") -> np.ndarray:
    """Shortcut used throughout the package.

    ``name="auto"`` picks in order: entso_nl_all, real_gb, synthetic_gb.
    """
    loader = get_loader()
    if name == "auto":
        for candidate in ("entso_nl_all", "real_gb", "synthetic_gb"):
            if candidate in loader.available():
                LOGGER.info(f"[price_loader] auto-selected {candidate}")
                return loader.load(candidate)
        raise FileNotFoundError("No price CSVs found in data/")
    return loader.load(name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    L = PriceLoader()
    print("Available datasets:", L.available())
    for n in L.available():
        a = L.load(n)
        print(f"  {n:18s}  n={len(a):6d}  mean={a.mean():7.2f}  "
              f"std={a.std():6.2f}  min={a.min():7.2f}  max={a.max():7.2f}  "
              f"({L.currency_units(n)})")
