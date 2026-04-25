"""
Verify the dropout=0.1 collapse is fixed.
=========================================

History
-------
Attempt #1 (2026-04-20): removed ``F.dropout`` from the GINE
message-passing loop. Did not fix the collapse — the 2026-04-21
run showed every d=0.1 config still hit val_loss ≈ 0.927 (predict-
the-mean), proving the BatchNorm variance-shift hypothesis was
wrong, or at least incomplete.

Attempt #2 (2026-04-21): replaced the head ``nn.Dropout(dropout)``
with ``nn.Identity()`` regardless of the dropout argument. Empirical
evidence: head dropout alone is sufficient to drive the model into
a "predict the mean" local minimum it cannot escape. The dropout
argument is retained on the public API for backwards compatibility
but is now a no-op; regularisation is via weight_decay=1e-5 in the
optimiser.

Expected outcome
----------------
With dropout now a no-op, d=0.0 and d=0.1 configs should produce
essentially identical val_loss and test R² (within floating-point /
batch-order noise). Specifically, d=0.1 should now reach val_loss
~0.30 and R²_TAC ~0.99 — matching what the 2026-04-20 d=0.0 configs
achieved.

Run:
    cd EURECHA/rl_dynamic_control/
    python scripts/verify_dropout_fix.py

Output:
    outputs/gnn_sweep/verify_dropout_fix.csv
    outputs/gnn_sweep/verify_dropout_fix.json

Author: Pepe (Jose Maria Contreras Prada)
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

# Force UTF-8 stdio so print statements containing Unicode characters
# (→, ², ≤, etc.) don't crash under Windows' default cp1252 console,
# which bit us in the 2026-04-21 run_all.ps1 execution.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass  # non-reconfigurable stream (notebooks, test harness, etc.)

import numpy as np
import torch

# ─────────────────────────────────────────────────────────────────────
# Make the parent package importable
# ─────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_PKG  = _HERE.parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

# Import the exact helpers the sweep uses, so the dataset and the
# split are byte-identical to the 2026-04-20 sweep.
from train_gnn_sweep import (                                                # noqa: E402
    build_dataset_with_logging,
    split_dataset,
    normalise_targets,
    train_one_config,
    evaluate_on_test,
    _TARGET_NAMES,
)


# Same six configs that collapsed in the previous sweep.
CONFIGS = [
    {"hidden_dim":  64, "n_layers": 2, "dropout": 0.1},
    {"hidden_dim":  64, "n_layers": 3, "dropout": 0.1},
    {"hidden_dim":  64, "n_layers": 4, "dropout": 0.1},
    {"hidden_dim": 128, "n_layers": 2, "dropout": 0.1},
    {"hidden_dim": 128, "n_layers": 3, "dropout": 0.1},
    {"hidden_dim": 128, "n_layers": 4, "dropout": 0.1},
]

# Matches the sweep's defaults — do NOT change these or the comparison
# becomes apples-to-oranges.
N_SAMPLES   = 2000
SEED        = 42
MAX_EPOCHS  = 500
PATIENCE    = 40
BATCH_SIZE  = 32
LR          = 1e-3


def main() -> int:
    out_dir = _PKG / "outputs" / "gnn_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv  = out_dir / "verify_dropout_fix.csv"
    out_json = out_dir / "verify_dropout_fix.json"

    print("=" * 70)
    print(f"Dropout-fix verification  |  n_samples={N_SAMPLES}  seed={SEED}")
    print(f"Rerunning {len(CONFIGS)} d=0.1 configs on patched FlowsheetGNN")
    print("=" * 70)

    print("\n[1/3] Building dataset (same seed as the sweep)...")
    dataset = build_dataset_with_logging(N_SAMPLES, SEED)

    print("\n[2/3] Splitting 70/15/15 (same seed)...")
    train, val, test = split_dataset(dataset, seed=SEED)
    print(f"      train={len(train)}  val={len(val)}  test={len(test)}")

    mu, sig = normalise_targets(train + val)
    for d in test:
        d.y = (d.y - mu) / sig

    print(f"\n[3/3] Retraining {len(CONFIGS)} configs...\n")
    results = []
    for idx, cfg in enumerate(CONFIGS):
        t0 = time.perf_counter()
        model, _, best_val, last_epoch = train_one_config(
            train, val,
            hidden_dim=cfg["hidden_dim"],
            n_layers=cfg["n_layers"],
            dropout=cfg["dropout"],
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
            lr=LR,
            batch_size=BATCH_SIZE,
            seed=SEED,
        )
        dt = time.perf_counter() - t0
        test_metrics = evaluate_on_test(model, test, mu, sig)
        row = dict(cfg,
                   best_val_loss=best_val,
                   final_epoch=last_epoch,
                   wall_time_s=dt,
                   **test_metrics)
        results.append(row)

        r2_line = " ".join(f"R²_{n}={row[f'test_r2_{n}']:+.3f}" for n in _TARGET_NAMES)
        print(f"  [{idx+1:>1}/{len(CONFIGS)}] h{cfg['hidden_dim']:>3} l{cfg['n_layers']} "
              f"d{cfg['dropout']:.1f}  val={best_val:.4f}  {r2_line}  ({dt:.0f}s)")

    # ─── Save CSV + JSON ────────────────────────────────────────────
    fieldnames = list(results[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in results:
            w.writerow(row)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=float)

    # ─── Summary comparison vs the pre-fix numbers from 2026-04-20 ─
    # (hard-coded so the summary works without parsing the old CSV)
    old_snapshot = {
        (64,  2, 0.1): {"val": 0.9267, "r2_TAC": -0.002, "r2_carb": -0.000, "r2_LCOM": -0.000},
        (64,  3, 0.1): {"val": 0.9274, "r2_TAC": -0.003, "r2_carb": -0.002, "r2_LCOM": -0.001},
        (64,  4, 0.1): {"val": 0.9278, "r2_TAC": -0.000, "r2_carb": -0.001, "r2_LCOM": -0.000},
        (128, 2, 0.1): {"val": 0.9271, "r2_TAC": -0.000, "r2_carb": -0.001, "r2_LCOM": -0.000},
        (128, 3, 0.1): {"val": 0.9271, "r2_TAC": -0.002, "r2_carb": -0.002, "r2_LCOM": -0.004},
        (128, 4, 0.1): {"val": 0.9276, "r2_TAC": -0.000, "r2_carb": -0.001, "r2_LCOM": -0.000},
    }
    print("\n" + "=" * 70)
    print("BEFORE-vs-AFTER COMPARISON (pre-fix 2026-04-20 → post-fix now)")
    print("=" * 70)
    print(f"{'config':<14} {'val(before→after)':<22} {'R²_TAC':<22}")
    for row in results:
        key = (row["hidden_dim"], row["n_layers"], row["dropout"])
        old = old_snapshot.get(key, {})
        val_str  = f"{old.get('val', float('nan')):.4f} → {row['best_val_loss']:.4f}"
        r2T_str  = f"{old.get('r2_TAC', float('nan')):+.3f} → {row['test_r2_TAC']:+.3f}"
        cfg_str  = f"h{row['hidden_dim']} l{row['n_layers']} d{row['dropout']:.1f}"
        print(f"{cfg_str:<14} {val_str:<22} {r2T_str:<22}")

    print(f"\nResults saved:\n  {out_csv}\n  {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
