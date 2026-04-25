#!/usr/bin/env python3
"""
Plot Pareto sweep results.

This script supports:
    - v2 profit vs electricity consumption outputs
    - legacy profit vs CO2 utilisation outputs
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

LOGGER = logging.getLogger("plot_pareto")

COLORS = {
    "pareto": "#2a9d8f",
    "profit": "#1d3557",
    "elec": "#e76f51",
    "co2": "#e9c46a",
    "full_load": "#e63946",
    "rule": "#457b9d",
    "original_ppo": "#e9c46a",
    "original_sac": "#264653",
}


def plot_v2(in_dir: Path, df: pd.DataFrame, baselines: pd.DataFrame, report_figure: Path) -> None:
    df = df.sort_values("lambda_elec").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    ax.errorbar(
        df["elec_mean_mwh"],
        df["profit_mean"],
        xerr=df["elec_std_mwh"],
        yerr=df["profit_std"],
        fmt="o-",
        color=COLORS["pareto"],
        linewidth=2,
        markersize=7,
        capsize=3,
        label="SAC lambda sweep",
    )
    for _, row in df.iterrows():
        ax.annotate(
            f"{row['lambda_elec']:.2f}",
            (row["elec_mean_mwh"], row["profit_mean"]),
            fontsize=8,
            xytext=(5, 5),
            textcoords="offset points",
        )

    for _, row in baselines.iterrows():
        color = COLORS.get(row["kind"], "gray")
        ax.scatter(
            row["elec_mean_mwh"],
            row["profit_mean"],
            s=180,
            marker="*",
            color=color,
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
            label=row["kind"].replace("_", " "),
        )

    ax.set_xlabel("Mean electricity consumption per episode (MWh)")
    ax.set_ylabel("Mean episode profit (scaled reward units)")
    ax.set_title("Pareto Front v2: Profit vs Electricity Consumption")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    pareto_path = in_dir / "pareto_front_elec.png"
    fig.savefig(pareto_path, dpi=170)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(9.0, 5.0))
    ax2 = ax1.twinx()
    lam = df["lambda_elec"]
    ax1.plot(lam, df["profit_mean"], "o-", color=COLORS["profit"], linewidth=2, label="Profit")
    ax1.fill_between(
        lam,
        df["profit_mean"] - df["profit_std"],
        df["profit_mean"] + df["profit_std"],
        color=COLORS["profit"],
        alpha=0.15,
    )
    ax2.plot(
        lam,
        df["elec_mean_mwh"],
        "s--",
        color=COLORS["elec"],
        linewidth=2,
        label="Electricity consumption",
    )
    ax2.fill_between(
        lam,
        df["elec_mean_mwh"] - df["elec_std_mwh"],
        df["elec_mean_mwh"] + df["elec_std_mwh"],
        color=COLORS["elec"],
        alpha=0.15,
    )
    ax1.set_xlabel("lambda_elec (lambda_profit = 1 - lambda_elec)")
    ax1.set_ylabel("Mean episode profit", color=COLORS["profit"])
    ax2.set_ylabel("Mean electricity consumption (MWh)", color=COLORS["elec"])
    ax1.set_title("Trade-off curve: profit vs electricity consumption")
    ax1.grid(True, alpha=0.3)
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc="best")
    fig.tight_layout()

    tradeoff_path = in_dir / "tradeoff_curve_elec.png"
    fig.savefig(tradeoff_path, dpi=170)
    plt.close(fig)

    report_figure.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pareto_path, report_figure)
    print(f"Plots saved -> {pareto_path}, {tradeoff_path}")
    print(f"Report figure -> {report_figure}")


def plot_legacy(in_dir: Path, df: pd.DataFrame, baselines: pd.DataFrame) -> None:
    df = df.sort_values("lambda_profit").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(
        df["co2_mean"] * 100.0,
        df["profit_mean"],
        xerr=df["co2_std"] * 100.0,
        yerr=df["profit_std"],
        fmt="o-",
        color=COLORS["pareto"],
        linewidth=2,
        markersize=8,
        capsize=3,
        label="SAC Pareto (legacy)",
    )
    for _, row in df.iterrows():
        ax.annotate(
            f"{row['lambda_profit']:.2f}",
            (row["co2_mean"] * 100.0, row["profit_mean"]),
            fontsize=8,
            xytext=(5, 5),
            textcoords="offset points",
        )
    for _, row in baselines.iterrows():
        color = COLORS.get(row["kind"], "gray")
        ax.scatter(
            row["co2_mean"] * 100.0,
            row["profit_mean"],
            marker="*",
            s=220,
            color=color,
            edgecolors="black",
            zorder=5,
            label=row["kind"],
        )
    ax.set_xlabel("Mean CO2 utilisation (%)")
    ax.set_ylabel("Mean episode profit")
    ax.set_title("Pareto Front: Profit vs CO2 Utilisation")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(in_dir / "pareto_front.png", dpi=150)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    lam = df["lambda_profit"]
    ax1.plot(lam, df["profit_mean"], "o-", color=COLORS["profit"], linewidth=2, label="Profit")
    ax2.plot(
        lam,
        df["co2_mean"] * 100.0,
        "s--",
        color=COLORS["co2"],
        linewidth=2,
        label="CO2 utilisation",
    )
    ax1.set_xlabel("lambda_profit")
    ax1.set_ylabel("Mean episode profit", color=COLORS["profit"])
    ax2.set_ylabel("Mean CO2 utilisation (%)", color=COLORS["co2"])
    ax1.set_title("Trade-off curve: profit vs CO2 utilisation")
    ax1.grid(True, alpha=0.3)
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc="best")
    fig.tight_layout()
    fig.savefig(in_dir / "tradeoff_curve.png", dpi=150)
    plt.close(fig)

    print(f"Legacy plots saved -> {in_dir}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in-dir",
        default=str(Path(__file__).resolve().parent.parent / "outputs" / "pareto_v2"),
    )
    parser.add_argument(
        "--report-figure",
        default=str(
            Path(__file__).resolve().parent.parent / "report" / "figures" / "fig_pareto_elec.png"
        ),
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    in_dir = Path(args.in_dir)
    report_figure = Path(args.report_figure)

    res_path = in_dir / "pareto_results.csv"
    base_path = in_dir / "baselines.csv"
    if not res_path.exists():
        LOGGER.error("Missing %s. Run pareto_sweep.py first.", res_path)
        return 2

    df = pd.read_csv(res_path)
    baselines = pd.read_csv(base_path) if base_path.exists() else pd.DataFrame()

    if {"lambda_elec", "elec_mean_mwh", "elec_std_mwh"}.issubset(df.columns):
        plot_v2(in_dir, df, baselines, report_figure)
    elif {"lambda_profit", "co2_mean", "co2_std"}.issubset(df.columns):
        plot_legacy(in_dir, df, baselines)
    else:
        LOGGER.error("Unrecognised pareto_results.csv schema in %s", res_path)
        return 3

    return 0


if __name__ == "__main__":
    sys.exit(main())
