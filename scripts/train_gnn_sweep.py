"""
Level-4 GNN — longer training + hyperparameter sweep
====================================================

Standalone driver that:
    1. Generates a large LHS dataset via the (now properly LHS)
       ``flowsheet_graph.generate_lhs_samples``.
    2. Splits 70/15/15 train / val / test with a fixed seed (test set
       is held out for the entire sweep — used only for final metrics
       so HP selection doesn't leak into it).
    3. Sweeps a 12-config grid over ``hidden_dim × n_layers × dropout``.
    4. Early stops per config on val loss (patience 40, max 500 epochs).
    5. Reports MAE / R² per target in *physical* units (TAC in £/yr,
       carbon_eff as a fraction, LCOM in £/t) so the numbers are
       interpretable.
    6. Saves per-config checkpoints, a CSV of sweep results, and a
       summary plot.  Copies the overall best model to
       ``best_model.pt``.

Run:
    cd EURECHA/rl_dynamic_control/
    python scripts/train_gnn_sweep.py                    # full sweep
    python scripts/train_gnn_sweep.py --smoke            # tiny smoke run
    python scripts/train_gnn_sweep.py --n_samples 5000   # scale up

Author: Pepe (Jose Maria Contreras Prada)
Project: EURECHA 2026 Process Design Contest
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────
# Make the parent package importable when run as a script
# ─────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_PKG  = _HERE.parent                               # .../rl_dynamic_control
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from flowsheet_graph import (                                               # noqa: E402
    FlowsheetGNN,
    build_base_graph,
    evaluate_with_surrogates,
    generate_lhs_samples,
)


# ─────────────────────────────────────────────────────────────────────
# Default sweep grid — 12 configs.  Override via --grid json.
# ─────────────────────────────────────────────────────────────────────

DEFAULT_GRID = {
    "hidden_dim": [64, 128],
    "n_layers":   [2, 3, 4],
    "dropout":    [0.0, 0.1],
}

DEFAULT_SMOKE_GRID = {
    "hidden_dim": [32],
    "n_layers":   [2, 3],
    "dropout":    [0.0],
}


# ─────────────────────────────────────────────────────────────────────
# Dataset generation
# ─────────────────────────────────────────────────────────────────────

def build_dataset_with_logging(n_samples: int, seed: int) -> list:
    """Generate ``n_samples`` graphs with real surrogate targets and
    per-sample edge features (via ``compute_edge_features``).  Prints
    progress every 500 samples."""
    t0 = time.perf_counter()
    samples = generate_lhs_samples(n_samples, seed=seed)
    dataset = []
    real_count = 0
    for i, params in enumerate(samples):
        res = evaluate_with_surrogates(params)
        if res.get("_source") == "real":
            real_count += 1
        graph = build_base_graph(
            node_params=params,
            edge_features=res["stream_features"],
            targets={"TAC": res["TAC"], "carbon_eff": res["carbon_eff"], "LCOM": res["LCOM"]},
        )
        dataset.append(graph)
        if (i + 1) % 500 == 0:
            print(f"  [{i+1:>5}/{n_samples}]  real-surrogate ratio so far: {real_count/(i+1):.1%}")
    dt = time.perf_counter() - t0
    print(f"  Dataset generated: {len(dataset)} graphs in {dt:.1f} s "
          f"({real_count}/{n_samples} real-surrogate, rest synthetic)")
    return dataset


def split_dataset(dataset: list, seed: int,
                  train_frac: float = 0.70, val_frac: float = 0.15):
    """Deterministic 70/15/15 split by a seeded permutation."""
    n = len(dataset)
    n_train = int(round(n * train_frac))
    n_val   = int(round(n * val_frac))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    idx_train = perm[:n_train]
    idx_val   = perm[n_train:n_train + n_val]
    idx_test  = perm[n_train + n_val:]
    train = [dataset[i] for i in idx_train]
    val   = [dataset[i] for i in idx_val]
    test  = [dataset[i] for i in idx_test]
    return train, val, test


def normalise_targets(dataset: list):
    """Fit z-score normalisation on the union of targets, apply it, and
    return (μ, σ) so predictions can be inverted later."""
    all_y = torch.stack([d.y.squeeze(0) for d in dataset])
    mu = all_y.mean(dim=0)
    sig = all_y.std(dim=0) + 1e-8
    for d in dataset:
        d.y = (d.y - mu) / sig
    return mu, sig


# ─────────────────────────────────────────────────────────────────────
# Training loop — one config
# ─────────────────────────────────────────────────────────────────────

def train_one_config(
    train_data, val_data,
    hidden_dim: int, n_layers: int, dropout: float,
    max_epochs: int = 500, patience: int = 40,
    lr: float = 1e-3, batch_size: int = 32, seed: int = 42,
    loss_weights=None,
):
    """Train a single HP config and return (model, history, best_val).

    Parameters
    ----------
    loss_weights : tensor-like of shape (n_targets,), optional
        Per-target weights applied to the MSE.  Defaults to uniform
        ``[1, 1, …, 1]`` — i.e. identical to plain ``F.mse_loss``.
        Normalised internally so the *average* weight is 1, which
        keeps the optimiser's effective learning rate unchanged.
        Weights apply identically to train and val losses so val_loss
        remains a fair early-stopping signal.
    """
    from torch_geometric.loader import DataLoader

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=batch_size)

    model = FlowsheetGNN(
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
        pool="mean+add",
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=15, min_lr=1e-5
    )

    # ---- Prepare per-target loss weights --------------------------
    if loss_weights is None:
        w_tensor = None  # use plain F.mse_loss (fastest)
    else:
        w = torch.as_tensor(loss_weights, dtype=torch.float32)
        # Normalise so the mean weight is 1 — keeps loss scale & LR sane.
        w = w / w.mean()
        w_tensor = w.view(1, -1)  # broadcastable to (batch, n_targets)

    def _weighted_mse(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if w_tensor is None:
            return F.mse_loss(pred, y)
        return ((pred - y) ** 2 * w_tensor).mean()

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None
    bad = 0
    last_epoch = 0
    for epoch in range(max_epochs):
        last_epoch = epoch
        # Train
        model.train()
        tl = 0.0
        for batch in train_loader:
            optimiser.zero_grad()
            pred = model(batch)
            loss = _weighted_mse(pred, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            tl += loss.item() * batch.num_graphs
        tl /= len(train_data)

        # Validate — use the SAME weighting so val_loss is monotone
        # with train_loss and early stopping is well-defined.
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for batch in val_loader:
                vl += _weighted_mse(model(batch), batch.y).item() * batch.num_graphs
        vl /= len(val_data)

        scheduler.step(vl)
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)

        if vl < best_val:
            best_val = vl
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_val, last_epoch


# ─────────────────────────────────────────────────────────────────────
# Denormalised metrics: MAE + R² per target in physical units
# ─────────────────────────────────────────────────────────────────────

_TARGET_NAMES = ("TAC", "carbon_eff", "LCOM")


def evaluate_on_test(model, test_data, mu, sig) -> dict:
    """Compute denormalised MAE and R² per target plus overall MSE
    (in normalised space, so it's comparable across configs)."""
    from torch_geometric.loader import DataLoader

    model.eval()
    loader = DataLoader(test_data, batch_size=64)
    preds, truths = [], []
    mse_norm = 0.0
    with torch.no_grad():
        for batch in loader:
            p = model(batch)
            mse_norm += F.mse_loss(p, batch.y, reduction="sum").item()
            preds.append(p)
            truths.append(batch.y)
    mse_norm /= (len(test_data) * 3)     # per-target mean

    y_pred_norm = torch.cat(preds, dim=0)
    y_true_norm = torch.cat(truths, dim=0)

    # Invert the z-score to physical units
    y_pred = y_pred_norm * sig + mu
    y_true = y_true_norm * sig + mu

    metrics = {"test_mse_norm": float(mse_norm)}
    for i, name in enumerate(_TARGET_NAMES):
        err = (y_pred[:, i] - y_true[:, i])
        mae = err.abs().mean().item()
        ss_res = (err ** 2).sum().item()
        ss_tot = ((y_true[:, i] - y_true[:, i].mean()) ** 2).sum().item()
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        metrics[f"test_mae_{name}"] = float(mae)
        metrics[f"test_r2_{name}"]  = float(r2)
    return metrics


# ─────────────────────────────────────────────────────────────────────
# Sweep summary plot
# ─────────────────────────────────────────────────────────────────────

def plot_sweep(results: list[dict], out_path: Path) -> None:
    """Bar chart of test R² per config, one subplot per target."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        warnings.warn(f"matplotlib not available ({e}); skipping plot.")
        return

    labels = [
        f"h{r['hidden_dim']}-l{r['n_layers']}-d{r['dropout']:.1f}"
        for r in results
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=False)
    for i, name in enumerate(_TARGET_NAMES):
        vals = [r[f"test_r2_{name}"] for r in results]
        best_idx = int(np.argmax(vals))
        colors = ["#2b7a78" if j != best_idx else "#d62728" for j in range(len(vals))]
        axes[i].bar(labels, vals, color=colors)
        axes[i].set_title(f"R² — {name}")
        axes[i].axhline(0.0, color="k", lw=0.5)
        axes[i].set_ylim(min(min(vals) - 0.05, -0.1), max(1.0, max(vals) + 0.02))
        axes[i].tick_params(axis="x", rotation=60, labelsize=8)
        axes[i].grid(axis="y", ls=":", alpha=0.4)
    fig.suptitle("GNN HP sweep — test-set R² per target (red = best config per target)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n_samples", type=int, default=2000,
                        help="Number of LHS samples (default: 2000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out_dir", type=str, default=str(_PKG / "outputs" / "gnn_sweep"))
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny run (n_samples=100, 2-config grid) for plumbing validation.")
    args = parser.parse_args()

    if args.smoke:
        args.n_samples = 100
        args.max_epochs = 30
        args.patience = 10
        grid = DEFAULT_SMOKE_GRID
    else:
        grid = DEFAULT_GRID

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ─── Dataset ─────────────────────────────────────────────────────
    print("=" * 70)
    print(f"GNN HP sweep  |  n_samples={args.n_samples}  seed={args.seed}")
    print(f"Output dir:    {out_dir}")
    print("=" * 70)
    print("\n[1/4] Building dataset...")
    dataset = build_dataset_with_logging(args.n_samples, args.seed)

    print("\n[2/4] Splitting 70/15/15 (fixed seed)...")
    train, val, test = split_dataset(dataset, seed=args.seed)
    print(f"      train={len(train)}  val={len(val)}  test={len(test)}")

    # Normalise on TRAIN+VAL combined — test set stays in physical units
    # until we invert predictions for reporting.
    mu, sig = normalise_targets(train + val)
    # Apply same μ, σ to test labels for loss computation, then invert
    # for reporting metrics.
    for d in test:
        d.y = (d.y - mu) / sig

    # ─── Sweep ───────────────────────────────────────────────────────
    configs = [
        {"hidden_dim": h, "n_layers": l, "dropout": dp}
        for h in grid["hidden_dim"]
        for l in grid["n_layers"]
        for dp in grid["dropout"]
    ]
    print(f"\n[3/4] Sweeping {len(configs)} configs...")
    print(f"      grid: {grid}")

    results = []
    for cfg_idx, cfg in enumerate(configs):
        t0 = time.perf_counter()
        model, history, best_val, last_epoch = train_one_config(
            train, val,
            hidden_dim=cfg["hidden_dim"],
            n_layers=cfg["n_layers"],
            dropout=cfg["dropout"],
            max_epochs=args.max_epochs,
            patience=args.patience,
            lr=args.lr,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        dt = time.perf_counter() - t0
        test_metrics = evaluate_on_test(model, test, mu, sig)

        row = dict(cfg, best_val_loss=best_val, final_epoch=last_epoch,
                   wall_time_s=dt, **test_metrics)
        # Store μ/σ alongside the model so inference is reproducible.
        ckpt_name = f"ckpt_h{cfg['hidden_dim']}_l{cfg['n_layers']}_d{cfg['dropout']:.1f}.pt"
        ckpt_path = out_dir / ckpt_name
        torch.save({
            "state_dict": model.state_dict(),
            "config": cfg,
            "y_mean": mu,
            "y_std":  sig,
        }, ckpt_path)
        row["ckpt"] = ckpt_name
        results.append(row)

        r2_line = " ".join(f"R²_{n}={row[f'test_r2_{n}']:+.3f}" for n in _TARGET_NAMES)
        print(f"  [{cfg_idx+1:>2}/{len(configs)}] h{cfg['hidden_dim']:>3} l{cfg['n_layers']} "
              f"d{cfg['dropout']:.1f}  val={best_val:.4f}  {r2_line}  ({dt:.0f}s)")

    # ─── Save results + best model + plot ───────────────────────────
    print("\n[4/4] Saving results...")
    results.sort(key=lambda r: r["best_val_loss"])
    best = results[0]
    shutil.copy(out_dir / best["ckpt"], out_dir / "best_model.pt")

    fieldnames = list(results[0].keys())
    with open(out_dir / "sweep_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in results:
            w.writerow(row)
    with open(out_dir / "sweep_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    plot_sweep(results, out_dir / "sweep_summary.png")

    print("\n" + "=" * 70)
    print(f"Best config:  hidden={best['hidden_dim']}  layers={best['n_layers']}  dropout={best['dropout']}")
    print(f"  best_val_loss:   {best['best_val_loss']:.4f}  (normalised MSE)")
    for name in _TARGET_NAMES:
        print(f"  test R² ({name:<10}): {best[f'test_r2_{name}']:+.4f}")
        print(f"  test MAE({name:<10}): {best[f'test_mae_{name}']:.4g}")
    print(f"\nBest model copied to: {out_dir / 'best_model.pt'}")
    print(f"Results CSV:          {out_dir / 'sweep_results.csv'}")
    print(f"Summary plot:         {out_dir / 'sweep_summary.png'}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
