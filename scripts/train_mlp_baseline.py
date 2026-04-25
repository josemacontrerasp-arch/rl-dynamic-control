"""
MLP baselines — representation vs. signal diagnostic
=====================================================

Background
----------
2026-04-20 sweep found best GNN R²_LCOM = 0.249 (h=64, l=2, d=0.0).
2026-04-21 loss-weighting failed ([1,1,3] → 0.213, [1,1,10] → 0.246).
2026-04-21 target-swap failed (derived R²_LCOM = 0.210; root cause
R²_production = 0.52).

Before adding more architectural complexity (edge-level readouts,
dedicated LCOM heads, wider LHS), we need to answer a more basic
question: **is the problem representation or signal?**

This script runs two strictly-controlled MLP baselines on the SAME
2000-graph dataset, SAME 70/15/15 split, SAME seed, SAME target
normalisation, SAME optimiser recipe, and reports R²/MAE in physical
units for TAC, carbon_eff, and LCOM.

    A — Design-only MLP
        Input: flat 36-dim continuous design vector (concatenated
        node params, stripped of zero-padding). Asks: is the DESIGN
        VECTOR ALONE sufficient to predict LCOM?

    B — Design + flattened edge-features MLP
        Input: 36 design dims + 13×10 edge features = 166 dims.
        Asks: does graph structure add anything over plain feature
        concatenation?

Interpretation matrix
---------------------
    R²_LCOM(A)   R²_LCOM(B)   GNN baseline   → Diagnosis
    -----------  -----------  -------------  -----------------------
      ≤ 0.30       ≤ 0.30       0.249         Signal-limited. LCOM is
                                              not well-determined by
                                              the inputs we have. Move
                                              to widening LHS or adding
                                              derived features.
      ≤ 0.30       ≥ 0.50       0.249         Edge features are the
                                              critical signal. GNN's
                                              node-pooling readout is
                                              washing them out.
                                              Edge-level readout fix.
      ≥ 0.50        —           0.249         Design vector alone is
                                              enough. The GNN's
                                              relational inductive
                                              bias is actively hurting.
                                              Replace with flat MLP
                                              surrogate, or fundamentally
                                              rethink the pooling.

Usage
-----
    cd EURECHA/rl_dynamic_control/
    python scripts/train_mlp_baseline.py                # full (2000)
    python scripts/train_mlp_baseline.py --smoke        # n=100, fast
    python scripts/train_mlp_baseline.py --which A      # only design-only

Outputs
-------
    outputs/gnn_sweep/mlp_baseline_results.csv
    outputs/gnn_sweep/mlp_baseline_results.json
    outputs/gnn_sweep/mlp_baseline_A.pt
    outputs/gnn_sweep/mlp_baseline_B.pt

Author: Pepe (Jose Maria Contreras Prada)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Sequence

# Force UTF-8 stdio so Unicode doesn't crash under Windows cp1252.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────
# Parent package importability when run as script
# ─────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_PKG = _HERE.parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from flowsheet_graph import (                                                # noqa: E402
    NODE_FEATURE_SPEC,
    MAX_CONT_FEATURES,
    EDGE_FEATURE_DIM,
)
from train_gnn_sweep import build_dataset_with_logging                       # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# Defaults — mirror the GNN sweep so comparisons are apples-to-apples
# ─────────────────────────────────────────────────────────────────────
N_SAMPLES   = 2000
SEED        = 42
TRAIN_FRAC  = 0.70
VAL_FRAC    = 0.15
BATCH_SIZE  = 32
LR          = 1e-3
MAX_EPOCHS  = 500
PATIENCE    = 40
WEIGHT_DECAY = 1e-5
HIDDEN_DIM   = 128  # ~same param count order as the h=64 GNN
N_HIDDEN_LAYERS = 3  # slightly deeper than the GNN head to compensate
                    # for the absence of message passing

TARGET_NAMES = ("TAC", "carbon_eff", "LCOM")


# ─────────────────────────────────────────────────────────────────────
# Feature extractors
# ─────────────────────────────────────────────────────────────────────

def _design_only_dim() -> int:
    """Total number of actual (non-padded) continuous design params."""
    return sum(len(spec["names"]) for spec in NODE_FEATURE_SPEC.values())


def extract_design_vector(graph) -> np.ndarray:
    """Pull just the real continuous design params from graph.x,
    stripping the zero-padding + one-hot-type block.

    graph.x has shape (11, MAX_CONT_FEATURES + N_TYPES). The first
    MAX_CONT_FEATURES columns are continuous params, right-padded
    with zeros per-node. We grab only the ``len(spec['names'])``
    real entries for each node and concatenate node-wise.
    """
    x = graph.x.numpy()                        # (11, MAX+N_TYPES)
    pieces = []
    for node_id, spec in NODE_FEATURE_SPEC.items():
        n = len(spec["names"])
        pieces.append(x[node_id, :n])
    return np.concatenate(pieces, axis=0)       # (_design_only_dim(),)


def extract_design_plus_edges(graph) -> np.ndarray:
    """Design vector + flattened edge-feature matrix.

    Shape: _design_only_dim() + (13 × EDGE_FEATURE_DIM).
    """
    d = extract_design_vector(graph)
    e = graph.edge_attr.numpy().reshape(-1)     # (130,)
    return np.concatenate([d, e], axis=0)


# ─────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────

class BaselineMLP(nn.Module):
    """Plain feed-forward regressor. BatchNorm between Linear+ReLU to
    match the regularisation profile of the GNN (which uses BN after
    every message-passing step). Deliberately NO dropout — the GNN
    lost 13/12 d=0.1 configs to the dropout-BN variance-shift bug, so
    mirroring the fix here keeps the baselines honest.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = HIDDEN_DIM,
        n_hidden_layers: int = N_HIDDEN_LAYERS,
        n_targets: int = 3,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(n_hidden_layers):
            layers.extend([
                nn.Linear(prev, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            ])
            prev = hidden_dim
        layers.append(nn.Linear(prev, n_targets))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────
# Train / eval
# ─────────────────────────────────────────────────────────────────────

def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.abs(y_true - y_pred).mean())


def train_mlp(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    in_dim: int,
    hidden_dim: int = HIDDEN_DIM,
    n_hidden_layers: int = N_HIDDEN_LAYERS,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
    y_val_t   = torch.tensor(y_val,   dtype=torch.float32)

    n_train = X_train_t.shape[0]
    model = BaselineMLP(in_dim, hidden_dim, n_hidden_layers, n_targets=y_train.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=15, min_lr=1e-5
    )

    best_val = float("inf")
    best_state = None
    bad = 0
    last_epoch = 0
    for epoch in range(max_epochs):
        last_epoch = epoch
        model.train()
        # Shuffle each epoch
        perm = torch.randperm(n_train)
        tl = 0.0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_train_t[idx], y_train_t[idx]
            if xb.shape[0] < 2:  # BN needs ≥ 2 in the batch
                continue
            opt.zero_grad()
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item() * xb.shape[0]
        tl /= n_train

        model.eval()
        with torch.no_grad():
            vl = F.mse_loss(model(X_val_t), y_val_t).item()
        sched.step(vl)

        if vl < best_val - 1e-6:
            best_val = vl
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val, last_epoch


def eval_mlp(model: nn.Module, X: np.ndarray, y_true_phys: np.ndarray,
             mu: torch.Tensor, sig: torch.Tensor) -> dict:
    model.eval()
    with torch.no_grad():
        y_pred_norm = model(torch.tensor(X, dtype=torch.float32))
    y_pred_phys = (y_pred_norm * sig + mu).numpy()
    out: dict = {}
    for i, name in enumerate(TARGET_NAMES):
        out[f"test_r2_{name}"]  = _r2 (y_true_phys[:, i], y_pred_phys[:, i])
        out[f"test_mae_{name}"] = _mae(y_true_phys[:, i], y_pred_phys[:, i])
    return out


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n_samples", type=int, default=N_SAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--hidden_dim", type=int, default=HIDDEN_DIM)
    parser.add_argument("--n_hidden_layers", type=int, default=N_HIDDEN_LAYERS)
    parser.add_argument("--max_epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--which", choices=["A", "B", "both"], default="both",
                        help="Which baseline(s) to run.")
    parser.add_argument("--smoke", action="store_true",
                        help="n=100, shorter training — plumbing check.")
    parser.add_argument("--out_dir", type=str,
                        default=str(_PKG / "outputs" / "gnn_sweep"))
    args = parser.parse_args()

    if args.smoke:
        args.n_samples = 100
        args.max_epochs = 60
        args.patience = 15

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"MLP baselines (diagnostic)  |  n_samples={args.n_samples}  seed={args.seed}")
    print(f"Hidden: {args.hidden_dim} × {args.n_hidden_layers}  "
          f"lr={args.lr}  batch={args.batch_size}  patience={args.patience}")
    print(f"Running baseline(s): {args.which}")
    print("=" * 72)

    # ─── Dataset (IDENTICAL to the GNN sweep) ───────────────────────
    print("\n[1/4] Building dataset...")
    dataset = build_dataset_with_logging(args.n_samples, args.seed)

    # ─── Split (IDENTICAL permutation to split_dataset in the sweep) ─
    print("\n[2/4] Splitting 70/15/15 (same seed as the sweep)...")
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(args.n_samples)
    n_train = int(round(args.n_samples * TRAIN_FRAC))
    n_val = int(round(args.n_samples * VAL_FRAC))
    idx_train = perm[:n_train]
    idx_val   = perm[n_train:n_train + n_val]
    idx_test  = perm[n_train + n_val:]
    print(f"      train={len(idx_train)}  val={len(idx_val)}  test={len(idx_test)}")

    # Targets in physical units (before normalisation) — (N, 3)
    y_all_phys = np.stack([d.y.squeeze(0).numpy() for d in dataset]).astype(np.float64)

    # Normalise on train+val combined, in float32 tensors for the model.
    y_trainval = y_all_phys[np.concatenate([idx_train, idx_val])]
    mu = torch.tensor(y_trainval.mean(axis=0), dtype=torch.float32)
    sig = torch.tensor(y_trainval.std(axis=0) + 1e-8, dtype=torch.float32)

    def to_norm(y_phys: np.ndarray) -> np.ndarray:
        return ((torch.tensor(y_phys, dtype=torch.float32) - mu) / sig).numpy()

    y_train_norm = to_norm(y_all_phys[idx_train])
    y_val_norm   = to_norm(y_all_phys[idx_val])
    y_test_phys  = y_all_phys[idx_test]

    # ─── Define the two feature extractors and run whichever was asked ───
    extractors = {
        "A": ("design-only",           extract_design_vector),
        "B": ("design + edge-flat",    extract_design_plus_edges),
    }
    if args.which == "both":
        plan: Sequence[str] = ("A", "B")
    else:
        plan = (args.which,)

    results: list[dict] = []

    print("\n[3/4] Training requested baseline(s)...")
    for tag in plan:
        name, fn = extractors[tag]
        print(f"\n  → Baseline {tag} ({name})")

        X_all = np.stack([fn(d) for d in dataset]).astype(np.float32)
        in_dim = X_all.shape[1]
        # Z-score inputs using train+val stats to avoid leakage.
        X_tv = X_all[np.concatenate([idx_train, idx_val])]
        x_mu  = X_tv.mean(axis=0)
        x_std = X_tv.std(axis=0) + 1e-8
        X_all_n = (X_all - x_mu) / x_std

        X_train = X_all_n[idx_train]
        X_val   = X_all_n[idx_val]
        X_test  = X_all_n[idx_test]

        print(f"      in_dim={in_dim}  "
              f"train={X_train.shape[0]}  val={X_val.shape[0]}  test={X_test.shape[0]}")

        t0 = time.perf_counter()
        model, best_val, last_epoch = train_mlp(
            X_train, y_train_norm, X_val, y_val_norm,
            in_dim=in_dim,
            hidden_dim=args.hidden_dim,
            n_hidden_layers=args.n_hidden_layers,
            lr=args.lr,
            max_epochs=args.max_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        dt = time.perf_counter() - t0

        metrics = eval_mlp(model, X_test, y_test_phys, mu, sig)
        print(f"      trained in {dt/60:.2f} min  |  best_val={best_val:.4f}  "
              f"epochs={last_epoch + 1}")
        for n in TARGET_NAMES:
            print(f"        R²_{n:<11}  {metrics[f'test_r2_{n}']:+.4f}    "
                  f"MAE_{n:<11}  {metrics[f'test_mae_{n}']:.4g}")

        row = {
            "baseline":       tag,
            "name":           name,
            "in_dim":         in_dim,
            "hidden_dim":     args.hidden_dim,
            "n_hidden_layers": args.n_hidden_layers,
            "best_val_loss":  best_val,
            "final_epoch":    last_epoch,
            "wall_time_s":    dt,
            **metrics,
        }
        results.append(row)

        # Save checkpoint for reproducibility / later inference.
        ckpt_path = out_dir / f"mlp_baseline_{tag}.pt"
        torch.save({
            "state_dict":     model.state_dict(),
            "in_dim":         in_dim,
            "hidden_dim":     args.hidden_dim,
            "n_hidden_layers": args.n_hidden_layers,
            "y_mean":         mu,
            "y_std":          sig,
            "x_mean":         torch.tensor(x_mu, dtype=torch.float32),
            "x_std":          torch.tensor(x_std, dtype=torch.float32),
            "baseline_tag":   tag,
            "baseline_name":  name,
        }, ckpt_path)

    # ─── Persist results ────────────────────────────────────────────
    print("\n[4/4] Saving results...")
    out_csv  = out_dir / "mlp_baseline_results.csv"
    out_json = out_dir / "mlp_baseline_results.json"
    fieldnames = list(results[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in results:
            w.writerow(row)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=float)

    # ─── Summary table vs. GNN baseline ─────────────────────────────
    # GNN baseline numbers come from the 2026-04-20 unweighted sweep,
    # h=64 l=2 d=0.0 config (the same one target-swap used).
    gnn_baseline = {
        "test_r2_TAC":        0.9915,
        "test_r2_carbon_eff": 0.8990,
        "test_r2_LCOM":       0.2487,
    }

    print("\n" + "=" * 72)
    print("COMPARISON: R² per target (test set)")
    print("=" * 72)
    print(f"{'strategy':<28} {'R²_TAC':>9} {'R²_carb':>9} {'R²_LCOM':>9}")
    print(f"{'GNN baseline (unweighted)':<28} "
          f"{gnn_baseline['test_r2_TAC']:>+9.4f} "
          f"{gnn_baseline['test_r2_carbon_eff']:>+9.4f} "
          f"{gnn_baseline['test_r2_LCOM']:>+9.4f}")
    for row in results:
        tag = row["baseline"]; nm = row["name"]
        print(f"{'MLP ' + tag + ' — ' + nm:<28} "
              f"{row['test_r2_TAC']:>+9.4f} "
              f"{row['test_r2_carbon_eff']:>+9.4f} "
              f"{row['test_r2_LCOM']:>+9.4f}")

    # ─── Diagnostic verdict ─────────────────────────────────────────
    print("\n" + "=" * 72)
    print("DIAGNOSIS")
    print("=" * 72)
    r2s = {r["baseline"]: r["test_r2_LCOM"] for r in results}
    a = r2s.get("A")
    b = r2s.get("B")
    gnn = gnn_baseline["test_r2_LCOM"]

    if a is not None and a >= 0.50:
        print(f"  ✓ Design vector alone reaches R²_LCOM={a:+.3f}.")
        print("    → The GNN's relational inductive bias is HURTING. ")
        print("      Consider: flat-MLP surrogate as the production model,")
        print("      or rethink graph pooling to not wash out design signal.")
    elif b is not None and b >= 0.50 and (a is None or a < 0.30):
        print(f"  ~ Flat MLP with edge features reaches R²_LCOM={b:+.3f} "
              f"while design-only stays at R²_LCOM={a if a is not None else float('nan'):+.3f}.")
        print("    → The GNN's node-pooling readout is washing out edge signal.")
        print("      Next move: edge-level readout (gather DIST product edge into head).")
    elif max(v for v in (a, b) if v is not None) < gnn + 0.05:
        print(f"  ✗ Both flat baselines match or undershoot the GNN (R²_LCOM ≤ {gnn:+.3f}).")
        print("    → Signal-limited, NOT representation-limited.")
        print("      The inputs genuinely don't determine LCOM tightly.")
        print("      Next moves: widen LHS ranges, audit physics of the LCOM formula,")
        print("      or check if production is being computed correctly downstream.")
    else:
        # Partial — MLP moves the needle but doesn't blow past 0.5
        winner_tag = "A" if (a is not None and (b is None or a >= b)) else "B"
        winner_val = r2s[winner_tag]
        print(f"  ~ Best MLP baseline ({winner_tag}) at R²_LCOM={winner_val:+.3f} vs "
              f"GNN {gnn:+.3f}.")
        print("    → Intermediate: flat features help, but nothing gets to 0.5.")
        print("      Likely a mix of narrow-range signal AND readout-shape issues.")
        print("      Recommended order: edge-level readout first, then LHS widening.")

    print("\nResults saved:")
    print(f"  {out_csv}")
    print(f"  {out_json}")
    for tag in plan:
        print(f"  {out_dir / f'mlp_baseline_{tag}.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
