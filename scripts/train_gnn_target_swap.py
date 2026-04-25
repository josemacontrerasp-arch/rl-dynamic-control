"""
Target-swap: predict meoh_production, derive LCOM at inference.
================================================================

Background
----------
The 2026-04-20 sweep found LCOM R² = 0.249.  The 2026-04-21 loss-
weighting experiment (``scripts/train_gnn_lcom_fix.py``) failed to
move it — w=[1,1,3] even regressed R²_LCOM to 0.213 while dragging
TAC from 0.992 → 0.978.  Diagnosis: the shared trunk lacks capacity
to represent LCOM directly, so reweighting the loss just redistributes
the same impoverished embedding.

Strategy
--------
Instead of predicting LCOM = TAC / annual_production directly, we:

    1. Swap LCOM out of the target vector and put annual methanol
       production in its place.  Production is a physical quantity
       — tonnes/yr of MeOH out of the column — that scales with
       clean, monotone levers (reactor conversion, feed flows).
       Hypothesis: the trunk can learn this far more easily than
       a ratio of two partially-correlated outputs.

    2. Train the winning config (h=64, l=2, d=0.0) with uniform
       loss weights on the new target vector [TAC, carbon_eff,
       production].

    3. At inference: predict the three primary quantities, then
       derive LCOM_pred = TAC_pred / production_pred.  Report
       metrics against the ORIGINAL LCOM truth values (which we
       stash before the swap) so the numbers are directly
       comparable to the baseline and weighted experiments.

Note on computing production: we don't touch ``flowsheet_graph.py``.
By definition LCOM = TAC / annual_production, so we recover production
from the existing targets as ``production = TAC / LCOM`` in physical
units.  No unit conversion, no approximation — it's exact.

Success criterion
-----------------
Derived LCOM R² ≥ 0.80 on the test set, with TAC R² ≥ 0.90 and
carbon_eff R² ≥ 0.80 preserved.  If derived LCOM R² is still stuck
below 0.5, the issue isn't representation-capacity at all and we'll
need a dedicated LCOM head branching off the trunk.

Run:
    cd EURECHA/rl_dynamic_control/
    python scripts/train_gnn_target_swap.py

Output:
    outputs/gnn_sweep/target_swap_results.csv
    outputs/gnn_sweep/target_swap_results.json
    outputs/gnn_sweep/target_swap_model.pt

Author: Pepe (Jose Maria Contreras Prada)
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

# Force UTF-8 stdio so → ² ≥ etc. don't crash under Windows' cp1252
# console, which bit us in the 2026-04-21 run_all.ps1 run.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

import numpy as np
import torch
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────
# Parent package importability
# ─────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_PKG  = _HERE.parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from train_gnn_sweep import (                                               # noqa: E402
    build_dataset_with_logging,
    train_one_config,
)


# Same 5 numbers as the sweep so the dataset & split are identical.
N_SAMPLES   = 2000
SEED        = 42
MAX_EPOCHS  = 500
PATIENCE    = 40
BATCH_SIZE  = 32
LR          = 1e-3

# Winning config from the 2026-04-20 sweep.
BEST_CFG = {"hidden_dim": 64, "n_layers": 2, "dropout": 0.0}

# New target vector ordering — LCOM replaced with production.
_SWAP_TARGET_NAMES = ("TAC", "carbon_eff", "production")


def main() -> int:
    out_dir = _PKG / "outputs" / "gnn_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv  = out_dir / "target_swap_results.csv"
    out_json = out_dir / "target_swap_results.json"
    out_ckpt = out_dir / "target_swap_model.pt"

    print("=" * 72)
    print(f"LCOM target-swap experiment  |  n_samples={N_SAMPLES}  seed={SEED}")
    print(f"Config: h={BEST_CFG['hidden_dim']}  l={BEST_CFG['n_layers']}  "
          f"d={BEST_CFG['dropout']}")
    print(f"Targets: {_SWAP_TARGET_NAMES}  (LCOM derived at inference)")
    print("=" * 72)

    # ─── Dataset (identical to the sweep) ───────────────────────────
    print("\n[1/5] Building dataset...")
    dataset = build_dataset_with_logging(N_SAMPLES, SEED)

    # ─── Swap LCOM -> production ────────────────────────────────────
    # Physical units going in:
    #   y[:, 0] = TAC        (£/yr)
    #   y[:, 1] = carbon_eff (-)
    #   y[:, 2] = LCOM       (£/t)
    # By definition LCOM = TAC / annual_production, so
    #   production (t/yr) = TAC / LCOM.
    # We keep the original LCOM vector aside to evaluate derived LCOM
    # against the ground truth after inference.
    print("\n[2/5] Swapping LCOM → production in the target vector...")
    orig_tac  = np.array([float(d.y[0, 0]) for d in dataset], dtype=np.float64)
    orig_carb = np.array([float(d.y[0, 1]) for d in dataset], dtype=np.float64)
    orig_lcom = np.array([float(d.y[0, 2]) for d in dataset], dtype=np.float64)

    # Guard against degenerate rows where LCOM ≤ 0 (shouldn't happen in
    # a valid flowsheet, but be defensive — fall back to large-but-finite
    # number rather than propagating NaN/Inf into training).
    safe_lcom = np.where(orig_lcom > 1e-9, orig_lcom, 1e-9)
    orig_prod = orig_tac / safe_lcom

    # Sanity: production should be strictly positive and of reasonable
    # magnitude (kt/yr scale for this plant).
    prod_min, prod_med, prod_max = orig_prod.min(), np.median(orig_prod), orig_prod.max()
    print(f"      production  (t/yr)  min={prod_min:.3g}  med={prod_med:.3g}  max={prod_max:.3g}")
    print(f"      LCOM orig   (£/t)   min={orig_lcom.min():.3g}  med={np.median(orig_lcom):.3g}  max={orig_lcom.max():.3g}")
    print(f"      TAC         (£/yr)  min={orig_tac.min():.3g}  med={np.median(orig_tac):.3g}  max={orig_tac.max():.3g}")

    # Overwrite slot 2 with production; y shape stays (1, 3).
    for i, d in enumerate(dataset):
        d.y = torch.tensor(
            [[orig_tac[i], orig_carb[i], orig_prod[i]]],
            dtype=torch.float32,
        )

    # ─── Split (replicate split_dataset's permutation so we can index
    # orig_lcom[idx_test] directly) ─────────────────────────────────
    print("\n[3/5] Splitting 70/15/15 (same seed as the sweep)...")
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(N_SAMPLES)
    n_train = int(round(N_SAMPLES * 0.70))
    n_val   = int(round(N_SAMPLES * 0.15))
    idx_train = perm[:n_train]
    idx_val   = perm[n_train:n_train + n_val]
    idx_test  = perm[n_train + n_val:]

    train = [dataset[i] for i in idx_train]
    val   = [dataset[i] for i in idx_val]
    test  = [dataset[i] for i in idx_test]
    print(f"      train={len(train)}  val={len(val)}  test={len(test)}")

    # Ground-truth physical values on the test set (for final metrics).
    test_tac_truth  = orig_tac [idx_test]
    test_carb_truth = orig_carb[idx_test]
    test_prod_truth = orig_prod[idx_test]
    test_lcom_truth = orig_lcom[idx_test]

    # Normalise on train+val — fit μ, σ from the new [TAC, carb, prod]
    # vector so the model learns a well-conditioned regression.
    all_y = torch.stack([d.y.squeeze(0) for d in (train + val)])
    mu  = all_y.mean(dim=0)
    sig = all_y.std(dim=0) + 1e-8
    for d in train + val + test:
        d.y = (d.y - mu) / sig

    # ─── Train ──────────────────────────────────────────────────────
    print(f"\n[4/5] Training best config with target-swap...")
    t0 = time.perf_counter()
    model, _, best_val, last_epoch = train_one_config(
        train, val,
        hidden_dim=BEST_CFG["hidden_dim"],
        n_layers=BEST_CFG["n_layers"],
        dropout=BEST_CFG["dropout"],
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        lr=LR,
        batch_size=BATCH_SIZE,
        seed=SEED,
        loss_weights=None,  # uniform — point of the swap is that uniform works
    )
    dt = time.perf_counter() - t0
    print(f"      trained in {dt/60:.1f} min  |  best_val={best_val:.4f}  "
          f"epochs_until_stop={last_epoch+1}")

    # ─── Inference + derive LCOM ────────────────────────────────────
    print("\n[5/5] Evaluating on test set + deriving LCOM...")
    from torch_geometric.loader import DataLoader
    model.eval()
    loader = DataLoader(test, batch_size=64)
    preds_norm = []
    with torch.no_grad():
        for batch in loader:
            preds_norm.append(model(batch))
    y_pred_norm = torch.cat(preds_norm, dim=0)                # (N_test, 3)
    # Invert z-score → physical units
    y_pred = (y_pred_norm * sig + mu).cpu().numpy()

    pred_tac  = y_pred[:, 0]
    pred_carb = y_pred[:, 1]
    pred_prod = y_pred[:, 2]

    # Derive LCOM at inference.  Guard against near-zero/negative
    # predicted production, which would make the ratio blow up or
    # flip sign.  If the trunk is doing its job this branch is empty.
    safe_pred_prod = np.where(pred_prod > 1e-9, pred_prod, 1e-9)
    pred_lcom = pred_tac / safe_pred_prod
    n_prod_neg = int((pred_prod <= 1e-9).sum())
    if n_prod_neg:
        print(f"      WARNING: {n_prod_neg}/{len(pred_prod)} predicted productions "
              f"≤ 0 — clamped before ratio")

    # ─── Metrics for all four quantities ────────────────────────────
    def _r2(y_true, y_pred) -> float:
        ss_res = float(((y_true - y_pred) ** 2).sum())
        ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    def _mae(y_true, y_pred) -> float:
        return float(np.abs(y_true - y_pred).mean())

    metrics = {
        "test_r2_TAC":         _r2 (test_tac_truth,  pred_tac),
        "test_mae_TAC":        _mae(test_tac_truth,  pred_tac),
        "test_r2_carbon_eff":  _r2 (test_carb_truth, pred_carb),
        "test_mae_carbon_eff": _mae(test_carb_truth, pred_carb),
        "test_r2_production":  _r2 (test_prod_truth, pred_prod),
        "test_mae_production": _mae(test_prod_truth, pred_prod),
        "test_r2_LCOM_derived":  _r2 (test_lcom_truth, pred_lcom),
        "test_mae_LCOM_derived": _mae(test_lcom_truth, pred_lcom),
    }

    # ─── Persist ────────────────────────────────────────────────────
    row = {
        "experiment":     "target_swap",
        "config":         BEST_CFG,
        "best_val_loss":  best_val,
        "final_epoch":    last_epoch,
        "wall_time_s":    dt,
        "n_pred_prod_nonpositive": n_prod_neg,
        **metrics,
    }
    # CSV with flat keys only — unfold `config` for grep-ability.
    flat_row = {
        "experiment":     "target_swap",
        "hidden_dim":     BEST_CFG["hidden_dim"],
        "n_layers":       BEST_CFG["n_layers"],
        "dropout":        BEST_CFG["dropout"],
        "best_val_loss":  best_val,
        "final_epoch":    last_epoch,
        "wall_time_s":    dt,
        "n_pred_prod_nonpositive": n_prod_neg,
        **metrics,
    }
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat_row.keys()))
        w.writeheader()
        w.writerow(flat_row)
    with open(out_json, "w") as f:
        json.dump(row, f, indent=2, default=float)

    # Checkpoint — includes orig_prod stats so inference code can sanity-
    # check future production predictions against the training regime.
    torch.save({
        "state_dict":      model.state_dict(),
        "config":          BEST_CFG,
        "target_names":    _SWAP_TARGET_NAMES,
        "y_mean":          mu,
        "y_std":           sig,
        "production_stats": {"min": float(prod_min), "median": float(prod_med), "max": float(prod_max)},
    }, out_ckpt)

    # ─── Before-vs-after summary ────────────────────────────────────
    # Baseline (2026-04-20 unweighted sweep) + 2026-04-21 weighting runs.
    baseline = {
        "test_r2_TAC":        0.9915,
        "test_r2_carbon_eff": 0.8990,
        "test_r2_LCOM":       0.2487,
        "test_mae_LCOM":      33.21,
    }
    w_1_1_3 = {
        "test_r2_TAC":        0.9780,
        "test_r2_carbon_eff": 0.8720,
        "test_r2_LCOM":       0.2130,
    }
    w_1_1_10 = {
        "test_r2_TAC":        0.9670,
        "test_r2_carbon_eff": 0.8590,
        "test_r2_LCOM":       0.2460,
    }

    print("\n" + "=" * 72)
    print("COMPARISON: LCOM R² across strategies")
    print("=" * 72)
    print(f"{'strategy':<22} {'R²_TAC':>9} {'R²_carb':>9} {'R²_LCOM':>9}")
    print(f"{'baseline (unweighted)':<22} "
          f"{baseline['test_r2_TAC']:>+9.4f} "
          f"{baseline['test_r2_carbon_eff']:>+9.4f} "
          f"{baseline['test_r2_LCOM']:>+9.4f}")
    print(f"{'loss w=[1,1,3]':<22} "
          f"{w_1_1_3['test_r2_TAC']:>+9.4f} "
          f"{w_1_1_3['test_r2_carbon_eff']:>+9.4f} "
          f"{w_1_1_3['test_r2_LCOM']:>+9.4f}")
    print(f"{'loss w=[1,1,10]':<22} "
          f"{w_1_1_10['test_r2_TAC']:>+9.4f} "
          f"{w_1_1_10['test_r2_carbon_eff']:>+9.4f} "
          f"{w_1_1_10['test_r2_LCOM']:>+9.4f}")
    print(f"{'target-swap (new)':<22} "
          f"{metrics['test_r2_TAC']:>+9.4f} "
          f"{metrics['test_r2_carbon_eff']:>+9.4f} "
          f"{metrics['test_r2_LCOM_derived']:>+9.4f}")
    print()
    print(f"  Aux: test R²_production = {metrics['test_r2_production']:+.4f} "
          f"(MAE = {metrics['test_mae_production']:.3g} t/yr)")
    print(f"  Aux: test MAE_LCOM_derived = {metrics['test_mae_LCOM_derived']:.3g} £/t  "
          f"(baseline MAE_LCOM = {baseline['test_mae_LCOM']:.3g} £/t)")

    # ─── Verdict ────────────────────────────────────────────────────
    r2_lcom_new = metrics["test_r2_LCOM_derived"]
    r2_tac_new  = metrics["test_r2_TAC"]
    r2_carb_new = metrics["test_r2_carbon_eff"]
    print("\n" + "=" * 72)
    if r2_lcom_new >= 0.80 and r2_tac_new >= 0.90 and r2_carb_new >= 0.80:
        print(f"VERDICT: ✓ SUCCESS — derived LCOM R² = {r2_lcom_new:+.3f}  "
              f"(TAC {r2_tac_new:+.3f} ≥ 0.90, carb {r2_carb_new:+.3f} ≥ 0.80)")
    elif r2_lcom_new >= 0.50:
        print(f"VERDICT: ~ PARTIAL — derived LCOM R² = {r2_lcom_new:+.3f}.  "
              f"Next step: dedicated LCOM head or larger trunk.")
    else:
        print(f"VERDICT: ✗ TARGET-SWAP INSUFFICIENT — R²_LCOM = {r2_lcom_new:+.3f}.  "
              f"Likely a capacity/coupling problem, not a target-shape one.")
    print("=" * 72)
    print(f"\nResults saved:\n  {out_csv}\n  {out_json}\n  {out_ckpt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
