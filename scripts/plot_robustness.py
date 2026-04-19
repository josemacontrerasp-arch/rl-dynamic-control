#!/usr/bin/env python3
"""
Plot robustness results
========================
Reads ``outputs/robustness/robustness_results.csv`` and writes:

* ``robustness_bar.png``         — mean reward per fold (retrained vs
                                     zero-shot vs full-load baseline)
* ``robustness_box.png``         — box plot of rewards across folds per method
* ``robustness_heatmap.png``     — train-period × eval-period heatmap
                                     (LOSO season matrix)
* ``robustness_seasonal.png``    — seasonal performance breakdown

Colour scheme matches the rest of the project.

Usage::

    python -m rl_dynamic_control.scripts.plot_robustness [--in-dir outputs/robustness]
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

LOGGER = logging.getLogger("plot_robustness")

METHOD_COLORS = {
    "retrained":           "#2a9d8f",
    "zero_shot":           "#f4a261",
    "baseline_full_load":  "#e63946",
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir",
                        default=str(Path(__file__).resolve().parent.parent
                                    / "outputs" / "robustness"))
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    in_dir = Path(args.in_dir)
    csv_path = in_dir / "robustness_results.csv"
    if not csv_path.exists():
        LOGGER.error(f"Missing {csv_path}. Run robustness_test.py first.")
        return 2
    df = pd.read_csv(csv_path)

    # ---- 1. Bar chart per fold ----
    fig, ax = plt.subplots(figsize=(12, 5))
    folds = df["fold"].unique()
    methods = [m for m in METHOD_COLORS if m in df["method"].unique()]
    x = np.arange(len(folds))
    width = 0.8 / max(len(methods), 1)
    for i, m in enumerate(methods):
        sub = df[df["method"] == m].set_index("fold").reindex(folds)
        ax.bar(x + i * width, sub["mean_reward"], width,
               yerr=sub["std_reward"],
               label=m.replace("_", " "),
               color=METHOD_COLORS[m], alpha=0.9, capsize=3)
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(folds, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Mean episode reward (scaled £)")
    ax.set_title("Robustness across Temporal Folds")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(in_dir / "robustness_bar.png", dpi=150)
    plt.close(fig)

    # ---- 2. Box plot across folds per method ----
    fig, ax = plt.subplots(figsize=(7, 5))
    data = [df[df["method"] == m]["mean_reward"].values for m in methods]
    bp = ax.boxplot(data, labels=[m.replace("_", " ") for m in methods],
                    patch_artist=True)
    for patch, m in zip(bp["boxes"], methods):
        patch.set_facecolor(METHOD_COLORS[m])
        patch.set_alpha(0.85)
    ax.set_ylabel("Mean episode reward (scaled £)")
    ax.set_title("Reward distribution across all folds")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(in_dir / "robustness_box.png", dpi=150)
    plt.close(fig)

    # ---- 3. LOSO seasonal heatmap (train-period × eval-period) ----
    # Extract LOSO rows — train=three other seasons, eval=held-out season.
    loso_df = df[df["fold"].str.startswith("loso_leave_")]
    if not loso_df.empty:
        seasons = ["Q1", "Q2", "Q3", "Q4"]
        # rows = "trained on everything except X"  →  held-out season
        # We tabulate the retrained rows' mean_reward as the diagonal-off.
        mat = np.full((len(seasons), len(seasons)), np.nan)
        method_for_heatmap = ("retrained" if (loso_df["method"] == "retrained").any()
                              else "zero_shot" if (loso_df["method"] == "zero_shot").any()
                              else None)
        heatmap_rows = (loso_df[loso_df["method"] == method_for_heatmap]
                        if method_for_heatmap else loso_df.iloc[0:0])
        for _, row in heatmap_rows.iterrows():
            q = row["fold"].split("_")[-1]
            j = seasons.index(q)
            # "train=3 others, eval=held-out" → mark col=held season
            for i in range(len(seasons)):
                if seasons[i] != q:
                    mat[i, j] = row["mean_reward"]  # any train row has same value
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(mat, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(seasons)))
        ax.set_xticklabels([f"Eval: {s}" for s in seasons])
        ax.set_yticks(range(len(seasons)))
        ax.set_yticklabels([f"Train∋{s}" for s in seasons])
        for i in range(len(seasons)):
            for j in range(len(seasons)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                            color="black", fontsize=8)
        plt.colorbar(im, ax=ax, label="Mean reward")
        title_suffix = f" ({method_for_heatmap})" if method_for_heatmap else ""
        ax.set_title(f"Leave-one-season-out performance matrix{title_suffix}")
        fig.tight_layout()
        fig.savefig(in_dir / "robustness_heatmap.png", dpi=150)
        plt.close(fig)

    # ---- 4. Seasonal breakdown ----
    season_rows = df[df["fold"].str.startswith("loso_leave_")].copy()
    # str.extract returns a DataFrame; squeeze to Series and drop NaN rows
    extracted = season_rows["fold"].str.extract(r"leave_(Q\d)")[0]
    season_rows = season_rows.loc[extracted.notna()].copy()
    season_rows["season"] = extracted.dropna().values
    if not season_rows.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        pivoted = season_rows.pivot_table(index="season", columns="method",
                                          values="mean_reward", aggfunc="mean")
        pivoted = pivoted.reindex(["Q1", "Q2", "Q3", "Q4"])
        pivoted.plot(kind="bar", ax=ax,
                     color=[METHOD_COLORS.get(c, "#888") for c in pivoted.columns])
        ax.set_ylabel("Mean episode reward (scaled £)")
        ax.set_xlabel("Held-out season")
        ax.set_title("Seasonal performance breakdown (LOSO)")
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(title="Method")
        plt.xticks(rotation=0)
        fig.tight_layout()
        fig.savefig(in_dir / "robustness_seasonal.png", dpi=150)
        plt.close(fig)

    print(f"Plots saved → {in_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
