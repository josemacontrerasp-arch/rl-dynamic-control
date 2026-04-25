"""
Lift LCOM R² via weighted MSE on the best config.
==================================================

The 2026-04-20 sweep found LCOM R² = 0.25 — the model collapses LCOM
predictions toward the mean because TAC (R² 0.99) and carbon_eff
(R² 0.90) dominate the shared trunk's representation. The LCOM
diagnostic (``scripts/diagnose_lcom.py``) rules out narrow variance
and outliers as causes.

This script retrains the winning config (h=64, l=2, d=0.0) with
two weighting schemes and compares to the unweighted baseline:

    * exp_1 : weights [1, 1, 3]   — mild boost for LCOM
    * exp_2 : weights [1, 1, 10]  — strong boost for LCOM

Weights are normalised internally so the *average* is 1 (keeps the
optimiser's effective learning rate unchanged). Same dataset, same
split, same seed as the sweep — the only difference is the loss.

Run:
    cd EURECHA/rl_dynamic_control/
    python scripts/train_gnn_lcom_fix.py

Output:
    outputs/gnn_sweep/lcom_fix_results.csv
    outputs/gnn_sweep/lcom_fix_results.json
    outputs/gnn_sweep/lcom_fix_w1_1_3.pt
    outputs/gnn_sweep/lcom_fix_w1_1_10.pt

Interpretation guide
--------------------
We call the fix a success if test R² on LCOM climbs from ~0.25 to
≥ 0.80 **without** dragging TAC or carbon_eff below 0.90. If the
best we can get is ~0.5, the next step is target-swap: replace LCOM
in the loss with ``meoh_production`` and derive LCOM at inference.

Author: Pepe (Jose Maria Contreras Prada)
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

# Force UTF-8 stdio — Windows' default cp1252 can't encode Unicode
# characters like → ² ≥ which appear in our verdict strings.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

import torch

# ─────────────────────────────────────────────────────────────────────
# Parent package importability
# ─────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_PKG  = _HERE.parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from train_gnn_sweep import (                                               # noqa: E402
    build_dataset_with_logging,
    split_dataset,
    normalise_targets,
    train_one_config,
    evaluate_on_test,
    _TARGET_NAMES,
)


# Same 5 numbers as the sweep so the dataset & split are identical.
N_SAMPLES   = 2000
SEED        = 42
MAX_EPOCHS  = 500
PATIENCE    = 40
BATCH_SIZE  = 32
LR          = 1e-3

# The winning config from the 2026-04-20 sweep.
BEST_CFG = {"hidden_dim": 64, "n_layers": 2, "dropout": 0.0}

# Weighting experiments — (name, weights tensor). Order: [TAC, carbon_eff, LCOM].
EXPERIMENTS = [
    ("w1_1_3",  [1.0, 1.0, 3.0]),
    ("w1_1_10", [1.0, 1.0, 10.0]),
]


def main() -> int:
    out_dir = _PKG / "outputs" / "gnn_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv  = out_dir / "lcom_fix_results.csv"
    out_json = out_dir / "lcom_fix_results.json"

    print("=" * 72)
    print(f"LCOM loss-weighting experiment  |  n_samples={N_SAMPLES}  seed={SEED}")
    print(f"Config: h={BEST_CFG['hidden_dim']}  l={BEST_CFG['n_layers']}  "
          f"d={BEST_CFG['dropout']}")
    print(f"Running {len(EXPERIMENTS)} weighting variants")
    print("=" * 72)

    # ─── Dataset (identical to the sweep) ───────────────────────────
    print("\n[1/3] Building dataset...")
    dataset = build_dataset_with_logging(N_SAMPLES, SEED)

    print("\n[2/3] Splitting 70/15/15...")
    train, val, test = split_dataset(dataset, seed=SEED)
    print(f"      train={len(train)}  val={len(val)}  test={len(test)}")
    mu, sig = normalise_targets(train + val)
    for d in test:
        d.y = (d.y - mu) / sig

    # ─── Run each weighting experiment ──────────────────────────────
    print(f"\n[3/3] Training {len(EXPERIMENTS)} variants...\n")
    results = []
    for idx, (name, weights) in enumerate(EXPERIMENTS):
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
            loss_weights=weights,
        )
        dt = time.perf_counter() - t0
        metrics = evaluate_on_test(model, test, mu, sig)

        row = {
            "experiment":     name,
            "weights_TAC":    weights[0],
            "weights_carbon": weights[1],
            "weights_LCOM":   weights[2],
            "best_val_loss":  best_val,
            "final_epoch":    last_epoch,
            "wall_time_s":    dt,
            **metrics,
        }
        results.append(row)

        # Save checkpoint with μ/σ so inference is reproducible.
        ckpt_path = out_dir / f"lcom_fix_{name}.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "config":     BEST_CFG,
            "loss_weights": weights,
            "y_mean":     mu,
            "y_std":      sig,
        }, ckpt_path)

        r2_line = " ".join(f"R²_{n}={row[f'test_r2_{n}']:+.3f}" for n in _TARGET_NAMES)
        print(f"  [{idx+1}/{len(EXPERIMENTS)}] {name:<8} weights={weights}  "
              f"val={best_val:.4f}  {r2_line}  ({dt:.0f}s)")

    # ─── CSV + JSON ─────────────────────────────────────────────────
    fieldnames = list(results[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(r)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=float)

    # ─── Before-vs-after summary ───────────────────────────────────
    # Baseline numbers come from the 2026-04-20 unweighted sweep
    # (row 0 of outputs/gnn_sweep/sweep_results.csv).
    baseline = {
        "test_r2_TAC":         0.9915,
        "test_r2_carbon_eff":  0.8990,
        "test_r2_LCOM":        0.2487,
        "test_mae_TAC":        1.494e5,
        "test_mae_carbon_eff": 0.01505,
        "test_mae_LCOM":       33.21,
    }
    print("\n" + "=" * 72)
    print("COMPARISON: unweighted baseline (2026-04-20) vs new weighted runs")
    print("=" * 72)
    print(f"{'run':<12} {'R²_TAC':>9} {'R²_carb':>9} {'R²_LCOM':>9} "
          f"{'MAE_LCOM':>10}")
    print(f"{'baseline':<12} "
          f"{baseline['test_r2_TAC']:>+9.4f} "
          f"{baseline['test_r2_carbon_eff']:>+9.4f} "
          f"{baseline['test_r2_LCOM']:>+9.4f} "
          f"{baseline['test_mae_LCOM']:>10.2f}")
    for r in results:
        print(f"{r['experiment']:<12} "
              f"{r['test_r2_TAC']:>+9.4f} "
              f"{r['test_r2_carbon_eff']:>+9.4f} "
              f"{r['test_r2_LCOM']:>+9.4f} "
              f"{r['test_mae_LCOM']:>10.2f}")

    # ─── Verdict ────────────────────────────────────────────────────
    best_r2_lcom = max(r["test_r2_LCOM"] for r in results)
    print("\n" + "=" * 72)
    if best_r2_lcom >= 0.80:
        # and TAC/carb haven't crashed
        worst_tac  = min(r["test_r2_TAC"]         for r in results)
        worst_carb = min(r["test_r2_carbon_eff"]  for r in results)
        if worst_tac >= 0.90 and worst_carb >= 0.80:
            print(f"VERDICT: ✓ SUCCESS — best R²_LCOM = {best_r2_lcom:+.3f}  "
                  f"(TAC stayed ≥0.90, carb stayed ≥0.80)")
        else:
            print(f"VERDICT: ~ PARTIAL — R²_LCOM reached {best_r2_lcom:+.3f}, "
                  f"but TAC fell to {worst_tac:+.3f} / carb to {worst_carb:+.3f}")
    elif best_r2_lcom >= 0.50:
        print(f"VERDICT: ~ MARGINAL — best R²_LCOM = {best_r2_lcom:+.3f}. "
              f"Next try target-swap (predict meoh_production, derive LCOM).")
    else:
        print(f"VERDICT: ✗ WEIGHTING ALONE INSUFFICIENT "
              f"(best R²_LCOM = {best_r2_lcom:+.3f}).  "
              f"Move to target-swap.")
    print("=" * 72)
    print(f"\nResults saved:\n  {out_csv}\n  {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
