#!/usr/bin/env python3
"""
Generate Aspen Plus validation points from the Fix (c) agent trajectory.

Reads the per-step data of the Fix (c) SAC policy
(`outputs/extrapolation_fix/step_data_variant_c.csv`), clusters the
agent's visited operating points into `n_centroids` k-means centroids on
`[load, T, P]`, expands each centroid to the Aspen input tuple
`[electrolyser_load, reactor_T_C, reactor_P_bar, F_H2_kmolhr]`, and
writes the results to `outputs/aspen_validation/validation_points.csv`
together with a companion `aspen_variables.txt` file that lists the
Aspen Plus variables the user should set manually (or script via COM).

This is the minimal set of Aspen Plus runs required to spot-check the
v2/v3 surrogate accuracy in the region where the SAC policy actually
operates.

Usage
-----

    python -m rl_dynamic_control.scripts.generate_aspen_validation_points \
        --n-centroids 20 --seed 42

Run from the EURECHA root directory.

Dependencies: numpy, pandas, scikit-learn.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

LOGGER = logging.getLogger("generate_aspen_validation_points")

# Mapping from the graph-builder: F_H2 (kmol/hr) = capacity_factor * H2_FEED_KMOL.
# H2_FEED_KMOL is defined in figures/scripts/constants.py (2704 kmol/hr at nominal load).
# We pull it from rl_dynamic_control.config.H2_FEED_KMOL when available and
# fall back to the nominal value so this script runs even if the heavy
# process-model imports in config.py fail (e.g. missing scipy).
_H2_FEED_KMOL_DEFAULT = 2704.0


def _h2_feed_kmol() -> float:
    """Resolve the nominal PEM H2 flow in kmol/hr."""

    try:  # best-effort import; config.py pulls scipy indirectly
        from rl_dynamic_control import config as _cfg  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        return _H2_FEED_KMOL_DEFAULT
    for attr in ("H2_FEED_KMOL", "h2_feed_kmol", "H2_KMOL_HR"):
        value = getattr(_cfg, attr, None)
        if value is not None:
            return float(value)
    return _H2_FEED_KMOL_DEFAULT


def _load_step_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Fix (c) step data at {path}. "
            "Run `scripts/retrain_with_fixes.py` first."
        )
    df = pd.read_csv(path)
    required = {"load", "T", "P"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing)}"
        )
    return df[["load", "T", "P"]].copy()


def _kmeans(points: np.ndarray, n_centroids: int, seed: int) -> np.ndarray:
    """Cluster `points` into `n_centroids` centroids.

    Uses scikit-learn if available, otherwise falls back to a numpy-only
    implementation of Lloyd's algorithm so the script runs in minimal
    environments.
    """

    try:
        from sklearn.cluster import KMeans  # type: ignore

        km = KMeans(n_clusters=n_centroids, random_state=seed, n_init=10)
        km.fit(points)
        return km.cluster_centers_
    except Exception:  # pragma: no cover - fallback path
        LOGGER.warning(
            "scikit-learn not available; falling back to numpy k-means."
        )
        rng = np.random.default_rng(seed)
        # k-means++ init
        idx0 = int(rng.integers(0, points.shape[0]))
        centres = [points[idx0]]
        for _ in range(1, n_centroids):
            d = np.min(
                np.stack(
                    [np.linalg.norm(points - c, axis=1) ** 2 for c in centres],
                    axis=0,
                ),
                axis=0,
            )
            probs = d / d.sum()
            idx = int(rng.choice(points.shape[0], p=probs))
            centres.append(points[idx])
        centres_arr = np.stack(centres, axis=0)
        for _ in range(50):
            d_all = np.linalg.norm(
                points[:, None, :] - centres_arr[None, :, :], axis=2
            )
            labels = np.argmin(d_all, axis=1)
            new_centres = np.stack(
                [
                    points[labels == k].mean(axis=0)
                    if np.any(labels == k)
                    else centres_arr[k]
                    for k in range(n_centroids)
                ],
                axis=0,
            )
            if np.allclose(new_centres, centres_arr, atol=1e-6):
                centres_arr = new_centres
                break
            centres_arr = new_centres
        return centres_arr


def _build_output(centroids: np.ndarray, h2_feed: float) -> pd.DataFrame:
    load = np.clip(centroids[:, 0], 0.0, 1.0)
    T_C = centroids[:, 1]
    P_bar = centroids[:, 2]
    f_h2_kmolhr = load * h2_feed
    return pd.DataFrame(
        {
            "centroid_id": np.arange(1, len(load) + 1),
            "electrolyser_load": load,
            "reactor_T_C": T_C,
            "reactor_P_bar": P_bar,
            "F_H2_kmolhr": f_h2_kmolhr,
        }
    )


def _write_aspen_variables(path: Path) -> None:
    text = """
# Aspen Plus validation variable list
# -----------------------------------
# Set each of the following Aspen Plus manipulated variables per
# centroid in validation_points.csv, then record the resulting KPIs
# for comparison against the v2/v3 GPR surrogates.
#
# Aspen Plus block names follow the base-case flowsheet in
# aspenfiles/; they correspond to the nodes in
# rl_dynamic_control.flowsheet_graph.FlowsheetGraph.
#
# Manipulated variables
#   PEM.FEED-RATE           = F_H2_kmolhr        (kmol/hr)
#   PEM.CAPACITY-FACTOR     = electrolyser_load   (--)
#   REACT.TEMP              = reactor_T_C         (deg C)
#   REACT.PRES              = reactor_P_bar       (bar)
#
# Quantities to record (per centroid run)
#   Levelised cost of methanol    LCOM           (GBP/t)
#   Total annualised cost          TAC            (GBP/yr)
#   Specific utility consumption   util_per_tMeOH (kWh/t)
#   Specific methanol yield        meoh_tph       (t/hr)
#   Carbon efficiency              carbon_eff     (mol MeOH / mol CO2)
#   Recycle-loop CO2 mole fraction x_CO2_recycle  (--)
#
# Suggested Aspen export: one .bkp per centroid named
#   aspen_validation_points_<centroid_id>.bkp
# or a sensitivity study driven by validation_points.csv inside a
# single .bkp. Record results in a companion .xlsx so surrogate
# parity plots can be generated automatically.
"""
    path.write_text(text.lstrip(), encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step-data",
        type=Path,
        default=_ROOT
        / "rl_dynamic_control"
        / "outputs"
        / "extrapolation_fix"
        / "step_data_variant_c.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_ROOT
        / "rl_dynamic_control"
        / "outputs"
        / "aspen_validation",
    )
    parser.add_argument("--n-centroids", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    df = _load_step_data(args.step_data)
    LOGGER.info("Loaded %d Fix (c) step rows from %s", len(df), args.step_data)

    centroids = _kmeans(df.to_numpy(), n_centroids=args.n_centroids, seed=args.seed)
    h2_feed = _h2_feed_kmol()
    out_df = _build_output(centroids, h2_feed=h2_feed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "validation_points.csv"
    out_df.to_csv(csv_path, index=False)
    LOGGER.info("Wrote %d centroids to %s", len(out_df), csv_path)

    vars_path = args.out_dir / "aspen_variables.txt"
    _write_aspen_variables(vars_path)
    LOGGER.info("Wrote Aspen variable cheat-sheet to %s", vars_path)

    print(f"\n{out_df.to_string(index=False)}")
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {vars_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
