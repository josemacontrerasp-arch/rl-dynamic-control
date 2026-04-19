#!/usr/bin/env python3
"""
Plot the Pareto front produced by ``pareto_sweep.py``
======================================================
Reads ``outputs/pareto/pareto_results.csv`` and
``outputs/pareto/baselines.csv`` and produces:

* ``pareto_front.png``   — mean weekly profit (€) vs mean CO₂ utilisation (%),
  each point labelled by its λ_profit; baselines overlaid.
* ``tradeoff_curve.png`` — two-axis sweep of profit and CO₂ utilisation
  as functions of λ_profit.

Colour scheme matches the existing evaluate.py plots.

Usage::

    python -m rl_dynamic_control.scripts.plot_pareto [--in-dir outputs/pareto]

Run from the EURECHA root.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOGGER = logging.getLogger("plot_pareto")

COLORS = {
    "pareto":    "#2a9d8f",
    "full_load": "#e63946",
    "rule":      "#457b9d",
    "profit":    "#1d3557",
    "co2":       "#e9c46a",
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir",
                        default=str(Path(__file__).resolve().parent.parent
                                    / "outputs" / "pareto"))
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    in_dir = Path(args.in_dir)
    res_path = in_dir / "pareto_results.csv"
    base_path = in_dir / "baselines.csv"
    if not res_path.exists():
        LOGGER.error(f"Missing {res_path}. Run pareto_sweep.py first.")
        return 2

    df = pd.read_csv(res_path).sort_values("lambda_profit").reset_index(drop=True)
    baselines = pd.read_csv(base_path) if base_path.exists() else pd.DataFrame()

    # ---- 1. Pareto front plot ----
    fig, ax = plt.subplots(figsize=(8, 6))
    co2_pct = df["co2_mean"].values * 100
    ax.errorbar(
        co2_pct, df["profit_mean"],
        xerr=df["co2_std"] * 100, yerr=df["profit_std"],
        fmt="o-", color=COLORS["pareto"], linewidth=2, markersize=8,
        capsize=3, label="SAC Pareto (λ sweep)",
    )
    for _, row in df.iterrows():
        ax.annotate(f"{row['lambda_profit']:.2f}",
                    (row["co2_mean"] * 100, row["profit_mean"]),
                    fontsize=8, xytext=(5, 5), textcoords="offset points")

    for _, b in baselines.iterrows():
        c = COLORS.get(b["kind"], "grey")
        ax.scatter(b["co2_mean"] * 100, b["profit_mean"],
                   marker="*", s=220, color=c, edgecolors="black",
                   zorder=5, label=f"Baseline: {b['kind']}")

    ax.set_xlabel("Mean CO₂ utilisation (%)")
    ax.set_ylabel("Mean weekly profit (scaled £)")
    ax.set_title("Pareto Front — Profit vs CO₂ Utilisation (SAC)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(in_dir / "pareto_front.png", dpi=150)
    plt.close(fig)

    # ---- 2. Trade-off curve ----
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    lam = df["lambda_profit"].values
    ax1.plot(lam, df["profit_mean"], "o-", color=COLORS["profit"],
             linewidth=2, label="Profit")
    ax1.fill_between(lam, df["profit_mean"] - df["profit_std"],
                     df["profit_mean"] + df["profit_std"],
                     color=COLORS["profit"], alpha=0.15)
    ax2.plot(lam, df["co2_mean"] * 100, "s--", color=COLORS["co2"],
             linewidth=2, label="CO₂ utilisation (%)")
    ax2.fill_between(lam, (df["co2_mean"] - df["co2_std"]) * 100,
                     (df["co2_mean"] + df["co2_std"]) * 100,
                     color=COLORS["co2"], alpha=0.15)
    ax1.set_xlabel("λ_profit  (λ_co2 = 1 − λ_profit)")
    ax1.set_ylabel("Mean weekly profit (scaled £)", color=COLORS["profit"])
    ax2.set_ylabel("Mean CO₂ utilisation (%)", color=COLORS["co2"])
    ax1.set_title("Trade-off curve: profit ↕ CO₂ utilisation")
    ax1.grid(True, alpha=0.3)
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc="center right")
    fig.tight_layout()
    fig.savefig(in_dir / "tradeoff_curve.png", dpi=150)
    plt.close(fig)

    print(f"Plots saved → {in_dir}/pareto_front.png, tradeoff_curve.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
