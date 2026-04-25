"""
Step 2 — Edge-state prediction reframe
======================================

What changes from the original task
-----------------------------------
Instead of predicting a graph-level 3-vector [TAC, carbon_eff, LCOM],
the GNN now predicts the full per-edge physical state:

    for each stream i ∈ {0,...,12}:
        y[i] = (m_dot, T, P, x_CO2, x_H2, x_MeOH, x_H2O, x_inert)

That is, the model is asked to reconstruct the 13 × 8 operating-point
matrix from only the design vector and the flowsheet topology.  The 2
structural flags (is_recycle, is_energy) remain as edge INPUTS so
message passing knows which edge is which.

Why this is the right shape for a GNN
-------------------------------------
- The answer is per-edge: every edge gets its own 8-dim prediction, so
  the model can't "wash out" edge-specific signal the way mean+add
  pooling did to the graph-level readout.
- Recycle loops couple upstream and downstream — a quantity on edge i
  depends on conditions at neighboring edges via shared nodes. This is
  exactly what message passing was invented for.
- The target is equivariant to graph automorphisms: if you permuted the
  node ordering, the predicted edge states would permute correspondingly,
  and the loss would be unchanged. Permutation-invariant readout hurt
  the LCOM task; permutation-equivariant readout helps this one.

Data leakage guardrail
----------------------
``edge_attr`` used as INPUT is reduced to the 2 structural-flag columns
only (is_recycle, is_energy). The 8 physical-state columns that the
GNN is asked to predict are removed from the input and only retained
as the target (``edge_y``). Without this split, ``edge_attr`` would
literally contain the answer.

Run:
    cd EURECHA/rl_dynamic_control/
    python scripts/train_gnn_edge_state.py                 # default
    python scripts/train_gnn_edge_state.py --fix-capfactor # with step-1 fix
    python scripts/train_gnn_edge_state.py --smoke         # quick check

Output:
    outputs/gnn_sweep/edge_state_results.csv
    outputs/gnn_sweep/edge_state_results.json
    outputs/gnn_sweep/edge_state_model.pt

Author: Pepe (Jose Maria Contreras Prada)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_PKG = _HERE.parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

import flowsheet_graph as fg                                                 # noqa: E402
from flowsheet_graph import (                                                # noqa: E402
    EDGE_FEATURE_DIM,
    EDGE_FEATURE_NAMES,
    MAX_CONT_FEATURES,
)
from train_gnn_sweep import build_dataset_with_logging                       # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────
N_SAMPLES    = 2000
SEED         = 42
MAX_EPOCHS   = 500
PATIENCE     = 40
BATCH_SIZE   = 32
LR           = 1e-3
WEIGHT_DECAY = 1e-5
HIDDEN_DIM   = 64
N_LAYERS     = 3     # one more than the graph-level sweep winner — edge
                     # prediction needs slightly deeper propagation so that
                     # downstream edges see the upstream design.

# Physical state columns — everything except the two structural flags.
PHYSICAL_IDX   = list(range(8))          # 0..7 of the 10-dim edge feature
STRUCTURAL_IDX = [8, 9]                  # is_recycle, is_energy
PHYS_FEATURE_NAMES = [EDGE_FEATURE_NAMES[i] for i in PHYSICAL_IDX]
STRUCT_FEATURE_NAMES = [EDGE_FEATURE_NAMES[i] for i in STRUCTURAL_IDX]
PHYS_DIM   = len(PHYSICAL_IDX)           # 8
STRUCT_DIM = len(STRUCTURAL_IDX)         # 2

# Narrowed cap_factor range (from step 1 findings).
FIXED_CAP_RANGE = (0.93, 1.0)

# Index of the DIST product edge (annual methanol stream) — hard-coded in
# the pipeline; see user's original analysis.
DIST_PRODUCT_EDGE = 12


# ─────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────

class EdgeStatePredictor(nn.Module):
    """GINE trunk → per-edge head.

    Trunk mirrors FlowsheetGNN: shared node_embed, shared edge_embed (but
    over the 2-dim structural flag vector, not the 10-dim state vector),
    GINE + BatchNorm + residual per layer.

    Head: for each edge (src → dst), we concatenate the final node
    embeddings at src and dst with the raw structural flags for that
    edge, and run the concatenation through an MLP that outputs the
    8-dim physical state.
    """

    def __init__(
        self,
        node_dim: int,
        struct_dim: int = STRUCT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        n_layers: int = N_LAYERS,
        out_dim: int = PHYS_DIM,
    ):
        super().__init__()
        from torch_geometric.nn import GINEConv

        self.node_embed = nn.Linear(node_dim, hidden_dim)
        self.edge_embed = nn.Linear(struct_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(n_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, edge_dim=hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Per-edge head: [src_embed | dst_embed | struct_flags] → 8
        head_in = 2 * hidden_dim + struct_dim
        self.edge_head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_dim),
        )
        self.n_layers = n_layers

    def forward(self, data) -> torch.Tensor:
        x = self.node_embed(data.x)
        e = self.edge_embed(data.edge_attr)

        for i in range(self.n_layers):
            x_res = x
            x = self.convs[i](x, data.edge_index, e)
            x = self.bns[i](x)
            x = F.relu(x + x_res)

        src, dst = data.edge_index
        edge_repr = torch.cat([x[src], x[dst], data.edge_attr], dim=-1)
        return self.edge_head(edge_repr)                      # (num_edges_total, 8)


# ─────────────────────────────────────────────────────────────────────
# Dataset transform: split edge_attr into (input structural, target phys)
# ─────────────────────────────────────────────────────────────────────

def transform_for_edge_prediction(dataset: list) -> None:
    """In-place: move physical edge features into ``edge_y`` (the target)
    and keep only the 2 structural flags on ``edge_attr`` (the input).
    Drops the old graph-level ``y`` since we're not using it.
    """
    for d in dataset:
        full = d.edge_attr                                    # (13, 10)
        d.edge_y    = full[:, PHYSICAL_IDX].clone()            # (13, 8) target
        d.edge_attr = full[:, STRUCTURAL_IDX].clone()          # (13, 2) input
        # PyG auto-handles per-edge attribute batching when the first dim
        # equals num_edges; edge_y has 13 rows per graph, matching
        # edge_index.shape[1] = 13, so no __cat_dim__ override needed.
        d.y = None


# ─────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────

def train(
    train_data, val_data,
    node_dim: int,
    hidden_dim: int = HIDDEN_DIM,
    n_layers: int = N_LAYERS,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
):
    from torch_geometric.loader import DataLoader

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=batch_size)

    model = EdgeStatePredictor(
        node_dim=node_dim,
        struct_dim=STRUCT_DIM,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        out_dim=PHYS_DIM,
    )
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
        tl = 0.0
        n_seen = 0
        for batch in train_loader:
            opt.zero_grad()
            pred = model(batch)                               # (batch_edges, 8)
            loss = F.mse_loss(pred, batch.edge_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item() * batch.edge_y.shape[0]
            n_seen += batch.edge_y.shape[0]
        tl /= max(n_seen, 1)

        model.eval()
        vl = 0.0
        n_seen_v = 0
        with torch.no_grad():
            for batch in val_loader:
                vl += F.mse_loss(model(batch), batch.edge_y).item() * batch.edge_y.shape[0]
                n_seen_v += batch.edge_y.shape[0]
        vl /= max(n_seen_v, 1)
        sched.step(vl)

        if vl < best_val - 1e-7:
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


# ─────────────────────────────────────────────────────────────────────
# Eval
# ─────────────────────────────────────────────────────────────────────

def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.abs(y_true - y_pred).mean())


def evaluate(model, test_data, mu_y: torch.Tensor, sig_y: torch.Tensor) -> dict:
    """Return per-feature R²/MAE in PHYSICAL units, plus per-edge mean R²
    averaged over all 13 edges.
    """
    from torch_geometric.loader import DataLoader

    model.eval()
    loader = DataLoader(test_data, batch_size=64)

    pred_list, true_list = [], []
    with torch.no_grad():
        for batch in loader:
            pred_norm = model(batch)                                  # (E_batch, 8)
            pred_list.append(pred_norm)
            true_list.append(batch.edge_y)
    y_pred_norm = torch.cat(pred_list, dim=0).cpu().numpy()
    y_true_norm = torch.cat(true_list, dim=0).cpu().numpy()

    # Invert z-score: these are ALREADY normalised values. Restore physical
    # units for interpretable metrics.
    mu = mu_y.cpu().numpy()
    sig = sig_y.cpu().numpy()
    y_pred = y_pred_norm * sig + mu
    y_true = y_true_norm * sig + mu

    out: dict = {}
    for j, fname in enumerate(PHYS_FEATURE_NAMES):
        out[f"r2_{fname}"]  = _r2 (y_true[:, j], y_pred[:, j])
        out[f"mae_{fname}"] = _mae(y_true[:, j], y_pred[:, j])

    # Per-edge R² (averaged over features): reshape (total_edges, 8) →
    # (n_test, 13, 8) using the fact that every graph has 13 edges.
    n_edges_per_graph = 13
    n_graphs = y_true.shape[0] // n_edges_per_graph
    y_true_g = y_true.reshape(n_graphs, n_edges_per_graph, PHYS_DIM)
    y_pred_g = y_pred.reshape(n_graphs, n_edges_per_graph, PHYS_DIM)
    per_edge_r2 = []
    for e in range(n_edges_per_graph):
        # R² across the n_graphs samples × 8 features for edge e
        per_edge_r2.append(_r2(y_true_g[:, e, :].ravel(), y_pred_g[:, e, :].ravel()))
    out["per_edge_r2"] = per_edge_r2

    # Derived KPIs using the predicted DIST product edge
    out["_y_true"] = y_true_g
    out["_y_pred"] = y_pred_g
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
    parser.add_argument("--n_layers", type=int, default=N_LAYERS)
    parser.add_argument("--max_epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--fix-capfactor", action="store_true",
                        help="Apply step-1 cap_factor fix (narrow to 0.93–1.0).")
    parser.add_argument("--smoke", action="store_true",
                        help="n=100 + short training — plumbing check.")
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
    print(f"STEP 2 — edge-state GNN  |  n_samples={args.n_samples}  seed={args.seed}")
    print(f"Config: h={args.hidden_dim}  l={args.n_layers}  lr={args.lr}  "
          f"batch={args.batch_size}")
    print(f"Target: per-edge 8-dim physical state {PHYS_FEATURE_NAMES}")
    print(f"Input:  node design vector + 2-dim structural edge flags")
    print(f"fix-capfactor: {args.fix_capfactor}")
    print("=" * 72)

    # Optionally apply the step-1 fix.
    orig_ranges = list(fg.NODE_FEATURE_SPEC[3]["ranges"])
    patched = False
    if args.fix_capfactor:
        fg.NODE_FEATURE_SPEC[3]["ranges"] = orig_ranges[:3] + [FIXED_CAP_RANGE]
        patched = True
        print(f"[patched] NODE_FEATURE_SPEC[3]['ranges'][3] = {FIXED_CAP_RANGE}")

    try:
        # ─── Dataset ────────────────────────────────────────────────
        print("\n[1/5] Building dataset...")
        dataset = build_dataset_with_logging(args.n_samples, args.seed)

        # ─── Reshape for edge prediction (strip leakage) ────────────
        print("\n[2/5] Transforming: move physical edge features to target,")
        print("       keep only structural flags as input (leakage guardrail)...")
        transform_for_edge_prediction(dataset)

        # ─── Split (same perm as the sweep) ─────────────────────────
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(args.n_samples)
        n_train = int(round(args.n_samples * 0.70))
        n_val   = int(round(args.n_samples * 0.15))
        idx_train = perm[:n_train]
        idx_val   = perm[n_train:n_train + n_val]
        idx_test  = perm[n_train + n_val:]
        train_ds = [dataset[i] for i in idx_train]
        val_ds   = [dataset[i] for i in idx_val]
        test_ds  = [dataset[i] for i in idx_test]
        print(f"      train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

        # Per-feature normalisation on train+val's physical edge features.
        all_edge_y = torch.cat([d.edge_y for d in (train_ds + val_ds)], dim=0)  # (n*13, 8)
        mu_y  = all_edge_y.mean(dim=0)
        sig_y = all_edge_y.std(dim=0) + 1e-8
        print(f"      physical-target per-feature mu: {mu_y.tolist()}")
        print(f"      physical-target per-feature sig: {sig_y.tolist()}")
        for d in train_ds + val_ds + test_ds:
            d.edge_y = (d.edge_y - mu_y) / sig_y

        # Derive node_dim from the first graph.
        node_dim = train_ds[0].x.shape[1]

        # ─── Train ──────────────────────────────────────────────────
        print(f"\n[3/5] Training edge-state GNN (node_dim={node_dim})...")
        t0 = time.perf_counter()
        model, best_val, last_epoch = train(
            train_ds, val_ds,
            node_dim=node_dim,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            max_epochs=args.max_epochs,
            patience=args.patience,
            lr=args.lr,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        dt = time.perf_counter() - t0
        print(f"      trained in {dt/60:.1f} min  |  best_val={best_val:.4f}  "
              f"epochs={last_epoch+1}")

        # ─── Evaluate ───────────────────────────────────────────────
        print("\n[4/5] Evaluating on test set (per-feature + per-edge R²)...")
        metrics = evaluate(model, test_ds, mu_y, sig_y)

        print("\n  Per-FEATURE R² (across all test edges):")
        for fname in PHYS_FEATURE_NAMES:
            r = metrics[f"r2_{fname}"]
            m = metrics[f"mae_{fname}"]
            print(f"    {fname:<12}  R²={r:>+7.4f}   MAE={m:.4g}")

        print("\n  Per-EDGE R² (across samples × 8 features per edge):")
        # Edge names from STREAMS in flowsheet_graph — fetch lazily if
        # available, else just index.
        try:
            from flowsheet_graph import STREAMS
            edge_labels = [s[0] if isinstance(s, tuple) else str(s) for s in STREAMS]
        except Exception:
            edge_labels = [f"edge_{i}" for i in range(13)]
        per_edge = metrics["per_edge_r2"]
        for i, (lbl, r) in enumerate(zip(edge_labels, per_edge)):
            marker = " ← DIST product" if i == DIST_PRODUCT_EDGE else ""
            print(f"    edge {i:>2} {str(lbl)[:20]:<20}  R²={r:>+7.4f}{marker}")

        # ─── Derived KPIs from predicted edge state ─────────────────
        # meoh_tph proxy = m_dot × x_MeOH on edge 12 (DIST product).
        # Indices of m_dot (0) and x_MeOH (5) inside PHYS_FEATURE_NAMES.
        M_DOT_IDX = PHYS_FEATURE_NAMES.index("m_dot_total")
        X_MEOH_IDX = PHYS_FEATURE_NAMES.index("x_MeOH")
        y_true_g = metrics["_y_true"]     # (n_graphs, 13, 8) physical
        y_pred_g = metrics["_y_pred"]
        meoh_rate_true = y_true_g[:, DIST_PRODUCT_EDGE, M_DOT_IDX] * \
                        y_true_g[:, DIST_PRODUCT_EDGE, X_MEOH_IDX]
        meoh_rate_pred = y_pred_g[:, DIST_PRODUCT_EDGE, M_DOT_IDX] * \
                        y_pred_g[:, DIST_PRODUCT_EDGE, X_MEOH_IDX]
        derived_r2   = _r2 (meoh_rate_true, meoh_rate_pred)
        derived_mae  = _mae(meoh_rate_true, meoh_rate_pred)

        print("\n  Derived KPI: MeOH production rate proxy (m_dot × x_MeOH @ DIST product)")
        print(f"    R²  = {derived_r2:+.4f}")
        print(f"    MAE = {derived_mae:.4g}  (in whatever units m_dot uses)")
        print(f"    NB: TAC in your current pipeline is `const + α × meoh_tph`, so this")
        print(f"        R² is approximately the ceiling for derived TAC as well.")

        # ─── Persist ────────────────────────────────────────────────
        row = {
            "experiment":    "edge_state_prediction",
            "n_samples":     args.n_samples,
            "hidden_dim":    args.hidden_dim,
            "n_layers":      args.n_layers,
            "fix_capfactor": args.fix_capfactor,
            "best_val_loss": best_val,
            "final_epoch":   last_epoch,
            "wall_time_s":   dt,
            "derived_r2_meoh_rate_proxy":  derived_r2,
            "derived_mae_meoh_rate_proxy": derived_mae,
        }
        for fname in PHYS_FEATURE_NAMES:
            row[f"r2_{fname}"]  = metrics[f"r2_{fname}"]
            row[f"mae_{fname}"] = metrics[f"mae_{fname}"]
        for i, r in enumerate(metrics["per_edge_r2"]):
            row[f"per_edge_r2_{i}"] = r

        print("\n[5/5] Saving results...")
        out_csv  = out_dir / "edge_state_results.csv"
        out_json = out_dir / "edge_state_results.json"
        out_ckpt = out_dir / "edge_state_model.pt"

        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerow(row)
        with open(out_json, "w") as f:
            json.dump(row, f, indent=2, default=float)
        torch.save({
            "state_dict":  model.state_dict(),
            "hidden_dim":  args.hidden_dim,
            "n_layers":    args.n_layers,
            "node_dim":    node_dim,
            "mu_y":        mu_y,
            "sig_y":       sig_y,
            "phys_features": PHYS_FEATURE_NAMES,
            "struct_features": STRUCT_FEATURE_NAMES,
            "fix_capfactor": args.fix_capfactor,
        }, out_ckpt)

        # ─── Verdict ────────────────────────────────────────────────
        mean_edge_r2 = float(np.mean(metrics["per_edge_r2"]))
        print("\n" + "=" * 72)
        print(f"SUMMARY:")
        print(f"  Mean R² across 13 edges:     {mean_edge_r2:+.4f}")
        print(f"  R² on DIST product edge:     {metrics['per_edge_r2'][DIST_PRODUCT_EDGE]:+.4f}")
        print(f"  Derived MeOH-rate proxy R²:  {derived_r2:+.4f}")

        if mean_edge_r2 >= 0.8:
            print("\n  ✓ GNN fits the operating state well end-to-end.")
            print("    Paper narrative: the GNN predicts plant state (edges);")
            print("    graph-level KPIs are computed downstream by existing formulas.")
            print("    The flat-MLP LCOM ablation is an *ablation*, not a counterexample:")
            print("    LCOM is a 5-D GPR distillation and doesn't need message passing.")
        elif mean_edge_r2 >= 0.5:
            print("\n  ~ GNN works for most edges but some are weak. Check the per-edge")
            print("    table — low-R² edges are typically recycle loops or the DIST")
            print("    product edge (scale-sensitive). Consider 2 more GINE layers or")
            print("    an attention-weighted pool.")
        else:
            print("\n  ✗ Edge-state prediction isn't landing either. At this point")
            print("    the signal may not be there — consider widening LHS ranges or")
            print("    adding process-simulator-grade training data.")
        print("=" * 72)
        print(f"\nResults saved:")
        print(f"  {out_csv}")
        print(f"  {out_json}")
        print(f"  {out_ckpt}")
        return 0

    finally:
        if patched:
            fg.NODE_FEATURE_SPEC[3]["ranges"] = orig_ranges


if __name__ == "__main__":
    sys.exit(main())
