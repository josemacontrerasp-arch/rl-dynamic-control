#!/usr/bin/env python3
"""
Plot Extrapolation Fix Results
==============================
Generates four publication-quality figures from the retraining experiment:

1. **Bar chart — Extrapolation warning rate** (% of steps) for original
   SAC vs variants (a), (b), (c).
2. **Bar chart — Mean reward** for original vs variants vs baselines.
3. **Scatter plot — GPR σ vs reward** per timestep, coloured by variant.
   Shows whether the penalty steers the agent away from high-σ regions.
4. **State-space coverage** — 2D projections (load vs T, load vs P, T vs P)
   showing original LHS points, v2 LHS points, and agent trajectories.

Reads data from ``outputs/extrapolation_fix/``.

Usage::

    python -m rl_dynamic_control.scripts.plot_extrapolation_fix

Run from the EURECHA root directory.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rl_dynamic_control.config import DECISION_VAR_BOUNDS, RL_CFG

LOGGER = logging.getLogger("plot_extrapolation_fix")

_RL_DIR = Path(__file__).resolve().parent.parent
_OUT_DIR = _RL_DIR / "outputs" / "extrapolation_fix"

# Publication-quality plot style
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# Consistent colour palette
COLOURS = {
    "original_v1": "#e63946",
    "original_on_v2": "#f4a261",
    "variant_a": "#2a9d8f",
    "variant_b": "#457b9d",
    "variant_c": "#264653",
    "variant_gnn": "#7b2cbf",
    "baseline": "#adb5bd",
}

LABELS = {
    "original_v1": "Original SAC (v1)",
    "original_on_v2": "Original SAC (on v2)",
    "variant_a": "(a) v2 surrogates",
    "variant_b": "(b) v1 + var. penalty",
    "variant_c": "(c) v2 + var. penalty",
    "variant_gnn": "v2 + edge-state GNN penalty",
    "baseline": "Full-load baseline",
}


# ======================================================================
# Plot 1: Extrapolation warning rate
# ======================================================================

def plot_extrap_rate(df: pd.DataFrame, save_dir: Path) -> None:
    """Bar chart of extrapolation warning rate per variant."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    names = df["variant"].values
    rates = df["mean_extrap_frac"].values * 100
    colours = [COLOURS.get(n, "#999999") for n in names]
    labels = [LABELS.get(n, n) for n in names]

    bars = ax.bar(range(len(names)), rates, color=colours, edgecolor="k",
                  linewidth=0.5, width=0.65)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("GPR Extrapolation Rate (%)")
    ax.set_title("GPR Extrapolation Warning Rate by Variant")
    ax.set_ylim(0, 105)

    # Value labels on bars
    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    path = save_dir / "extrap_rate_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    LOGGER.info("Saved %s", path)


# ======================================================================
# Plot 2: Mean reward comparison
# ======================================================================

def plot_reward_comparison(df: pd.DataFrame, save_dir: Path) -> None:
    """Bar chart of mean reward per variant with error bars."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    names = df["variant"].values
    rewards = df["mean_reward"].values
    stds = df["std_reward"].values
    colours = [COLOURS.get(n, "#999999") for n in names]
    labels = [LABELS.get(n, n) for n in names]

    bars = ax.bar(range(len(names)), rewards, yerr=stds,
                  color=colours, edgecolor="k", linewidth=0.5,
                  width=0.65, capsize=4, error_kw={"lw": 1})

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Mean Episode Reward (scaled £)")
    ax.set_title("Mean Reward by Variant")

    # Value labels
    for bar, val, s in zip(bars, rewards, stds):
        y = bar.get_height() + s + 0.05 * abs(max(rewards))
        ax.text(bar.get_x() + bar.get_width() / 2, y,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    path = save_dir / "reward_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    LOGGER.info("Saved %s", path)


# ======================================================================
# Plot 3: GPR σ vs reward scatter
# ======================================================================

def plot_sigma_vs_reward(save_dir: Path) -> None:
    """Scatter plot of GPR σ vs reward per timestep, coloured by variant."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    plot_order = ["original_v1", "variant_a", "variant_b", "variant_c", "variant_gnn"]
    max_points = 2000  # subsample for readability

    for name in plot_order:
        step_path = save_dir / f"step_data_{name}.csv"
        if not step_path.exists():
            continue

        step_df = pd.read_csv(step_path)
        if len(step_df) > max_points:
            step_df = step_df.sample(n=max_points, random_state=42)

        ax.scatter(
            step_df["gpr_sigma"],
            step_df["reward"],
            c=COLOURS.get(name, "#999999"),
            label=LABELS.get(name, name),
            alpha=0.35,
            s=12,
            edgecolors="none",
        )

    ax.set_xlabel(r"GPR Predictive $\sigma$")
    ax.set_ylabel("Per-step Reward (scaled £)")
    ax.set_title(r"GPR Uncertainty ($\sigma$) vs Reward by Variant")
    ax.legend(fontsize=9, markerscale=3, framealpha=0.9)

    fig.tight_layout()
    path = save_dir / "sigma_vs_reward.png"
    fig.savefig(path)
    plt.close(fig)
    LOGGER.info("Saved %s", path)


# ======================================================================
# Plot 4: State-space coverage
# ======================================================================

def plot_state_space_coverage(save_dir: Path) -> None:
    """2D projections showing LHS points, v2 points, and agent trajectories."""
    from scipy.stats.qmc import LatinHypercube

    # Regenerate original LHS (same seed as surrogate_optimization.py)
    U_orig = LatinHypercube(d=5, seed=42).random(500)
    X_orig = DECISION_VAR_BOUNDS[:, 0] + U_orig * (DECISION_VAR_BOUNDS[:, 1] - DECISION_VAR_BOUNDS[:, 0])
    T_orig = X_orig[:, 0]
    P_orig = X_orig[:, 1]
    H2_orig = X_orig[:, 3]

    # Load agent trajectory data
    projections = [
        ("T", "P", "T_reactor_C", "P_reactor_bar", T_orig, P_orig),
        ("load", "T", "Load_fraction", "T_reactor_C", None, T_orig),
        ("load", "P", "Load_fraction", "P_reactor_bar", None, P_orig),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    variant_order = ["original_v1", "variant_c", "variant_gnn"]

    for ax_idx, (x_col, y_col, x_label, y_label, x_lhs, y_lhs) in enumerate(projections):
        ax = axes[ax_idx]

        # Plot original LHS points (if applicable in this projection)
        if x_lhs is not None and y_lhs is not None:
            ax.scatter(x_lhs, y_lhs, c="#adb5bd", s=10, alpha=0.4,
                       label="Original LHS (500 pts)", zorder=1, edgecolors="none")

        # Plot agent trajectories
        for name in variant_order:
            step_path = save_dir / f"step_data_{name}.csv"
            if not step_path.exists():
                continue
            step_df = pd.read_csv(step_path)
            if len(step_df) > 1500:
                step_df = step_df.sample(n=1500, random_state=42)

            ax.scatter(
                step_df[x_col], step_df[y_col],
                c=COLOURS.get(name, "#999999"),
                label=LABELS.get(name, name),
                s=8, alpha=0.4, edgecolors="none", zorder=2,
            )

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        # Draw original LHS bounds as dashed rectangle (for T, P projections)
        if x_col == "T" and y_col == "P":
            from matplotlib.patches import Rectangle
            rect = Rectangle(
                (DECISION_VAR_BOUNDS[0, 0], DECISION_VAR_BOUNDS[1, 0]),
                DECISION_VAR_BOUNDS[0, 1] - DECISION_VAR_BOUNDS[0, 0],
                DECISION_VAR_BOUNDS[1, 1] - DECISION_VAR_BOUNDS[1, 0],
                linewidth=1.5, edgecolor="black", facecolor="none",
                linestyle="--", label="LHS bounds", zorder=3,
            )
            ax.add_patch(rect)

        if ax_idx == 0:
            ax.legend(fontsize=8, markerscale=3, framealpha=0.9, loc="upper left")

    fig.suptitle("State-Space Coverage: Original LHS vs Agent Trajectories",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    path = save_dir / "state_space_coverage.png"
    fig.savefig(path)
    plt.close(fig)
    LOGGER.info("Saved %s", path)


# ======================================================================
# Main
# ======================================================================

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory with extrapolation fix results")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    data_dir = Path(args.data_dir) if args.data_dir else _OUT_DIR

    # Load summary CSV
    csv_path = data_dir / "extrapolation_fix_results.csv"
    if not csv_path.exists():
        LOGGER.error("Results CSV not found at %s. Run retrain_with_fixes.py first.", csv_path)
        return 1

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} variants from {csv_path}")

    # Generate all plots
    print("\nGenerating plots ...")
    plot_extrap_rate(df, data_dir)
    plot_reward_comparison(df, data_dir)
    plot_sigma_vs_reward(data_dir)
    plot_state_space_coverage(data_dir)

    print(f"\nAll plots saved to: {data_dir}/")
    print("  - extrap_rate_comparison.png")
    print("  - reward_comparison.png")
    print("  - sigma_vs_reward.png")
    print("  - state_space_coverage.png")

    return 0


if __name__ == "__main__":
    sys.exit(main())
