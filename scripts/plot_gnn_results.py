#!/usr/bin/env python3
"""
Plot GNN results for the technical report.

Generates three publication-quality figures:
1. Grouped bar chart: GNN vs MLP-A vs MLP-B on KPI R^2.
2. Heatmap: per-edge/per-feature R^2 for the edge-state GNN.
3. Parity panels: predicted vs actual for the three best and three worst
   edge-feature combinations in the edge-state task.

Outputs are written to ``rl_dynamic_control/report/figures/`` by default.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


_HERE = Path(__file__).resolve().parent
_RL_DIR = _HERE.parent
_ROOT = _RL_DIR.parent

if str(_RL_DIR) not in sys.path:
    sys.path.insert(0, str(_RL_DIR))

import flowsheet_graph as fg  # noqa: E402
from flowsheet_graph import STREAMS  # noqa: E402
from train_gnn_edge_state import (  # noqa: E402
    EdgeStatePredictor,
    PHYS_FEATURE_NAMES,
    transform_for_edge_prediction,
)
from train_gnn_sweep import build_dataset_with_logging  # noqa: E402


ACCENTBLUE = "#264653"
ACCENTTEAL = "#2A9D8F"
ACCENTRED = "#E63946"
ACCENTORANGE = "#F4A261"
LIGHTGREY = "#F5F5F5"

MODEL_COLORS = {
    "GNN": ACCENTBLUE,
    "MLP-A": ACCENTTEAL,
    "MLP-B": ACCENTORANGE,
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": ":",
    "axes.facecolor": "white",
})


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scale = max(float(np.mean(np.abs(y_true))), 1.0)
    if float(np.std(y_true)) < 1e-6 * scale:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def load_model_comparison(out_dir: Path) -> pd.DataFrame:
    sweep = pd.read_csv(out_dir / "sweep_results.csv")
    mlp = pd.read_csv(out_dir / "mlp_baseline_results.csv")

    best_gnn = sweep.sort_values("best_val_loss", ascending=True).iloc[0]
    mlp_a = mlp.loc[mlp["baseline"] == "A"].iloc[0]
    mlp_b = mlp.loc[mlp["baseline"] == "B"].iloc[0]

    rows = [
        {
            "model": "GNN",
            "R2_TAC": float(best_gnn["test_r2_TAC"]),
            "R2_carbon": float(best_gnn["test_r2_carbon_eff"]),
            "R2_LCOM": float(best_gnn["test_r2_LCOM"]),
        },
        {
            "model": "MLP-A",
            "R2_TAC": float(mlp_a["test_r2_TAC"]),
            "R2_carbon": float(mlp_a["test_r2_carbon_eff"]),
            "R2_LCOM": float(mlp_a["test_r2_LCOM"]),
        },
        {
            "model": "MLP-B",
            "R2_TAC": float(mlp_b["test_r2_TAC"]),
            "R2_carbon": float(mlp_b["test_r2_carbon_eff"]),
            "R2_LCOM": float(mlp_b["test_r2_LCOM"]),
        },
    ]
    return pd.DataFrame(rows)


def rebuild_edge_state_predictions(out_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ckpt = torch.load(out_dir / "edge_state_model.pt", map_location="cpu")
    seed = 42
    n_samples = 2000

    orig_ranges = list(fg.NODE_FEATURE_SPEC[3]["ranges"])
    try:
        if ckpt.get("fix_capfactor"):
            fg.NODE_FEATURE_SPEC[3]["ranges"] = orig_ranges[:3] + [(0.93, 1.0)]

        dataset = build_dataset_with_logging(n_samples=n_samples, seed=seed)
        transform_for_edge_prediction(dataset)

        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_samples)
        n_train = int(round(n_samples * 0.70))
        n_val = int(round(n_samples * 0.15))
        idx_test = perm[n_train + n_val:]
        test_ds = [dataset[i] for i in idx_test]

        mu_y = ckpt["mu_y"].clone().detach().cpu()
        sig_y = ckpt["sig_y"].clone().detach().cpu()
        for d in test_ds:
            d.edge_y = (d.edge_y - mu_y) / sig_y

        model = EdgeStatePredictor(
            node_dim=int(ckpt["node_dim"]),
            hidden_dim=int(ckpt["hidden_dim"]),
            n_layers=int(ckpt["n_layers"]),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        from torch_geometric.loader import DataLoader

        loader = DataLoader(test_ds, batch_size=64)
        pred_list = []
        true_list = []
        with torch.no_grad():
            for batch in loader:
                pred_list.append(model(batch).cpu())
                true_list.append(batch.edge_y.cpu())

        y_pred_norm = torch.cat(pred_list, dim=0).numpy()
        y_true_norm = torch.cat(true_list, dim=0).numpy()

        mu = mu_y.numpy()
        sig = sig_y.numpy()
        y_pred = y_pred_norm * sig + mu
        y_true = y_true_norm * sig + mu

        n_edges = len(STREAMS)
        n_graphs = y_true.shape[0] // n_edges
        y_true_g = y_true.reshape(n_graphs, n_edges, len(PHYS_FEATURE_NAMES))
        y_pred_g = y_pred.reshape(n_graphs, n_edges, len(PHYS_FEATURE_NAMES))

        r2_grid = np.zeros((n_edges, len(PHYS_FEATURE_NAMES)), dtype=float)
        for e in range(n_edges):
            for f in range(len(PHYS_FEATURE_NAMES)):
                r2_grid[e, f] = _r2(y_true_g[:, e, f], y_pred_g[:, e, f])

        return y_true_g, y_pred_g, r2_grid
    finally:
        fg.NODE_FEATURE_SPEC[3]["ranges"] = orig_ranges


def plot_comparison(df: pd.DataFrame, save_path: Path) -> None:
    targets = [
        ("R2_TAC", "TAC"),
        ("R2_carbon", "Carbon efficiency"),
        ("R2_LCOM", "LCOM"),
    ]
    x = np.arange(len(targets))
    width = 0.24

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for i, model in enumerate(df["model"]):
        vals = [df.loc[df["model"] == model, key].iloc[0] for key, _ in targets]
        ax.bar(
            x + (i - 1) * width,
            vals,
            width=width,
            color=MODEL_COLORS[model],
            edgecolor="black",
            linewidth=0.6,
            label=model,
            zorder=3,
        )
        for j, val in enumerate(vals):
            ax.text(
                x[j] + (i - 1) * width,
                val + 0.015,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in targets])
    ax.set_ylabel(r"Test-set $R^2$")
    ax.set_ylim(0, 1.08)
    ax.set_title("Graph-level KPI prediction: GNN vs flat-MLP baselines")
    ax.legend(ncol=3, frameon=False, loc="upper center")
    ax.set_axisbelow(True)

    fig.savefig(save_path)
    plt.close(fig)


def plot_heatmap(r2_grid: np.ndarray, save_path: Path) -> None:
    edge_labels = [f"{i}: {name}" for i, (_, _, name) in enumerate(STREAMS)]
    feature_labels = [
        r"$\dot{m}$",
        r"$T$",
        r"$P$",
        r"$x_{\mathrm{CO_2}}$",
        r"$x_{\mathrm{H_2}}$",
        r"$x_{\mathrm{MeOH}}$",
        r"$x_{\mathrm{H_2O}}$",
        r"$x_{\mathrm{inert}}$",
    ]

    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad(color="#d9d9d9")

    fig, ax = plt.subplots(figsize=(9.6, 6.2))
    masked = np.ma.masked_invalid(r2_grid)
    im = ax.imshow(masked, cmap=cmap, vmin=0.97, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(feature_labels)))
    ax.set_xticklabels(feature_labels)
    ax.set_yticks(np.arange(len(edge_labels)))
    ax.set_yticklabels(edge_labels)
    ax.set_title(r"Edge-state GNN: per-edge / per-feature $R^2$")
    ax.set_xlabel("Predicted physical feature")
    ax.set_ylabel("Flowsheet edge")

    for i in range(r2_grid.shape[0]):
        for j in range(r2_grid.shape[1]):
            if np.isfinite(r2_grid[i, j]):
                color = "white" if r2_grid[i, j] < 0.988 else "black"
                label = f"{r2_grid[i, j]:.3f}"
            else:
                color = "black"
                label = "n/a"
            ax.text(j, i, label, ha="center", va="center", fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(r"$R^2$")
    fig.savefig(save_path)
    plt.close(fig)


def _combo_rankings(y_true_g: np.ndarray, y_pred_g: np.ndarray) -> list[dict]:
    combos = []
    for e, (_, _, edge_name) in enumerate(STREAMS):
        for f, feature_name in enumerate(PHYS_FEATURE_NAMES):
            r2 = _r2(y_true_g[:, e, f], y_pred_g[:, e, f])
            if not np.isfinite(r2):
                continue
            combos.append({
                "edge_idx": e,
                "edge_name": edge_name,
                "feature_idx": f,
                "feature_name": feature_name,
                "r2": r2,
                "y_true": y_true_g[:, e, f],
                "y_pred": y_pred_g[:, e, f],
            })
    combos.sort(key=lambda item: item["r2"])
    worst = combos[:3]
    best = combos[-3:][::-1]
    return best + worst


def plot_parity(y_true_g: np.ndarray, y_pred_g: np.ndarray, save_path: Path) -> None:
    selected = _combo_rankings(y_true_g, y_pred_g)
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.2))
    axes = axes.ravel()

    for idx, (ax, combo) in enumerate(zip(axes, selected)):
        is_best = idx < 3
        color = ACCENTTEAL if is_best else ACCENTRED
        y_true = combo["y_true"]
        y_pred = combo["y_pred"]
        ax.scatter(
            y_true,
            y_pred,
            s=18,
            alpha=0.65,
            color=color,
            edgecolors="none",
        )
        lo = float(min(np.min(y_true), np.min(y_pred)))
        hi = float(max(np.max(y_true), np.max(y_pred)))
        pad = 0.03 * (hi - lo if hi > lo else 1.0)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=ACCENTBLUE, linewidth=1.2)
        title_group = "Best" if is_best else "Worst"
        ax.set_title(
            f"{title_group}: {combo['edge_name']} / {combo['feature_name']}\n"
            f"$R^2={combo['r2']:.3f}$",
            fontsize=10,
        )
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_facecolor(LIGHTGREY)

    fig.suptitle("Edge-state parity: three strongest and three weakest output channels", y=1.01)
    fig.savefig(save_path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(_RL_DIR / "outputs" / "gnn_sweep"),
        help="Directory containing GNN/MLP result artefacts.",
    )
    parser.add_argument(
        "--fig-dir",
        type=str,
        default=str(_RL_DIR / "report" / "figures"),
        help="Directory where report figures are written.",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    comparison = load_model_comparison(out_dir)
    plot_comparison(comparison, fig_dir / "fig_gnn_comparison.png")

    y_true_g, y_pred_g, r2_grid = rebuild_edge_state_predictions(out_dir)
    plot_heatmap(r2_grid, fig_dir / "fig_edge_heatmap.png")
    plot_parity(y_true_g, y_pred_g, fig_dir / "fig_edge_parity.png")

    print("Saved:")
    print(fig_dir / "fig_gnn_comparison.png")
    print(fig_dir / "fig_edge_heatmap.png")
    print(fig_dir / "fig_edge_parity.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
