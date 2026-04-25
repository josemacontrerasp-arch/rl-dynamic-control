"""
Step 1 — cap_factor LHS fix + re-sweep on winning config
=========================================================

Background
----------
In ``flowsheet_graph._node_params_to_surrogate_input`` (line 527), the
5-D LCOM surrogate's F_H2 input is derived from node 3's capacity_factor:

    F_H2 = capacity_factor * 2704   (kmol/hr)

NODE_FEATURE_SPEC[3]['ranges'][3] currently samples capacity_factor
from (0.3, 1.0), which produces F_H2 ∈ (811, 2704). But the 5-D GPR's
F_H2 training bounds are [2515, 3147] (``_SURR_BOUNDS_5D[3]``), and
line 535 silently ``np.clip``s out-of-range values. Result: every
sample with capacity_factor < 0.93 has F_H2 pinned to 2515 before the
GPR sees it — ~93% of the LHS dataset has one of the 5 LCOM-determining
dimensions collapsed to a constant.

This script runs the SAME winning config (h=64, l=2, d=0.0) on a
dataset where capacity_factor is LHS-sampled from (0.93, 1.0) instead,
so F_H2 stays inside the GPR's training domain for every sample.

What this doesn't touch
-----------------------
The underlying ``flowsheet_graph.py`` file is unmodified — the
NODE_FEATURE_SPEC narrowing is done via a monkey-patch at runtime, so
the RL pipeline (which legitimately uses the full (0.3, 1.0) capacity
range) remains unaffected.

Decision gate
-------------
- If R²_LCOM ≥ 0.70 on the fixed dataset, the clipping bug was the
  whole story — GNN architecture was fine all along, just evaluated on
  a dataset with a collapsed dimension. Ship the GNN as-is on the new
  data.
- If R²_LCOM stays near 0.25, the clipping wasn't the bottleneck — it's
  the GNN's architectural inability to extract LCOM's 5 specific input
  scalars via mean+add pooling, and you'd move on to step 2 (edge-state
  prediction).

Run:
    cd EURECHA/rl_dynamic_control/
    python scripts/train_gnn_capfactor_fix.py

Output:
    outputs/gnn_sweep/capfactor_fix_results.csv
    outputs/gnn_sweep/capfactor_fix_results.json
    outputs/gnn_sweep/capfactor_fix_model.pt

Author: Pepe (Jose Maria Contreras Prada)
"""

from __future__ import annotations

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
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_PKG = _HERE.parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

import flowsheet_graph as fg                                                 # noqa: E402
from train_gnn_sweep import (                                                # noqa: E402
    build_dataset_with_logging,
    split_dataset,
    normalise_targets,
    train_one_config,
    evaluate_on_test,
)


# Same settings as the sweep + target-swap runs so comparisons are clean.
N_SAMPLES   = 2000
SEED        = 42
MAX_EPOCHS  = 500
PATIENCE    = 40
BATCH_SIZE  = 32
LR          = 1e-3
BEST_CFG    = {"hidden_dim": 64, "n_layers": 2, "dropout": 0.0}

# The fix: narrowed capacity_factor range. 0.93 ≈ 2515 / 2704, the ratio
# at which F_H2 first reaches the surrogate's lower training bound.
ORIG_CAP_RANGE = (0.3, 1.0)
FIXED_CAP_RANGE = (0.93, 1.0)


def main() -> int:
    out_dir = _PKG / "outputs" / "gnn_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv  = out_dir / "capfactor_fix_results.csv"
    out_json = out_dir / "capfactor_fix_results.json"
    out_ckpt = out_dir / "capfactor_fix_model.pt"

    print("=" * 72)
    print(f"STEP 1 — cap_factor LHS fix  |  n_samples={N_SAMPLES}  seed={SEED}")
    print(f"Config: h={BEST_CFG['hidden_dim']}  l={BEST_CFG['n_layers']}  "
          f"d={BEST_CFG['dropout']}")
    print(f"capacity_factor range: {ORIG_CAP_RANGE} → {FIXED_CAP_RANGE}")
    print("=" * 72)

    # ─── Monkey-patch NODE_FEATURE_SPEC[3] ──────────────────────────
    # Node 3 is the PEM electrolyser. Its 4 continuous params are
    # [P_elec_MW, eta_elec, T_stack, capacity_factor]; we narrow only
    # the last one.
    orig_ranges = list(fg.NODE_FEATURE_SPEC[3]["ranges"])
    before = orig_ranges[3]
    if before != ORIG_CAP_RANGE:
        print(f"WARNING: expected capacity_factor range {ORIG_CAP_RANGE}, "
              f"found {before}. Proceeding with the fix anyway.")
    fixed_ranges = orig_ranges[:3] + [FIXED_CAP_RANGE]
    fg.NODE_FEATURE_SPEC[3]["ranges"] = fixed_ranges
    print(f"[patched] NODE_FEATURE_SPEC[3]['ranges'][3] = {FIXED_CAP_RANGE}")

    try:
        # ─── Dataset + split ────────────────────────────────────────
        print("\n[1/4] Building dataset with fixed cap_factor range...")
        dataset = build_dataset_with_logging(N_SAMPLES, SEED)

        # Diagnostics: confirm F_H2 actually varies now. We reconstruct
        # the surrogate input for each sample and check the 4th dim.
        print("\n[diagnostic] Inspecting F_H2 distribution post-fix:")
        f_h2s = []
        for d in dataset[:1000]:
            # Reconstruct node_params dict from graph.x
            node_params = {}
            for nid, spec in fg.NODE_FEATURE_SPEC.items():
                n = len(spec["names"])
                node_params[nid] = d.x[nid, :n].tolist()
            try:
                x5 = fg._node_params_to_surrogate_input(node_params)
                f_h2s.append(float(x5[3]))
            except Exception:
                pass
        if f_h2s:
            arr = np.asarray(f_h2s)
            print(f"  F_H2 range over first 1000 samples: "
                  f"min={arr.min():.1f}  med={np.median(arr):.1f}  max={arr.max():.1f}")
            print(f"  Fraction clipped to 2515.0: {(arr <= 2515.0 + 1e-3).mean():.1%}")
            print(f"  (should be near 0%; if still high the fix didn't take hold)")

        print("\n[2/4] Splitting 70/15/15 (same seed as the sweep)...")
        train, val, test = split_dataset(dataset, seed=SEED)
        print(f"      train={len(train)}  val={len(val)}  test={len(test)}")

        mu, sig = normalise_targets(train + val)
        for d in test:
            d.y = (d.y - mu) / sig

        # ─── Train ──────────────────────────────────────────────────
        print(f"\n[3/4] Training winning config on fixed dataset...")
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
        )
        dt = time.perf_counter() - t0
        print(f"      trained in {dt/60:.1f} min  |  best_val={best_val:.4f}  "
              f"epochs={last_epoch+1}")

        # ─── Evaluate ───────────────────────────────────────────────
        print("\n[4/4] Evaluating on test set...")
        metrics = evaluate_on_test(model, test, mu, sig)

        row = {
            "experiment":    "capfactor_fix",
            "hidden_dim":    BEST_CFG["hidden_dim"],
            "n_layers":      BEST_CFG["n_layers"],
            "dropout":       BEST_CFG["dropout"],
            "cap_range":     list(FIXED_CAP_RANGE),
            "best_val_loss": best_val,
            "final_epoch":   last_epoch,
            "wall_time_s":   dt,
            **metrics,
        }

        # ─── Persist ────────────────────────────────────────────────
        fieldnames = list(row.keys())
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            # CSV-friendlify list fields
            flat = {k: (json.dumps(v) if isinstance(v, list) else v) for k, v in row.items()}
            w.writerow(flat)
        with open(out_json, "w") as f:
            json.dump(row, f, indent=2, default=float)
        torch.save({
            "state_dict":   model.state_dict(),
            "config":       BEST_CFG,
            "cap_range":    FIXED_CAP_RANGE,
            "y_mean":       mu,
            "y_std":        sig,
        }, out_ckpt)

        # ─── Before / after comparison ──────────────────────────────
        baseline = {
            "test_r2_TAC":        0.9915,
            "test_r2_carbon_eff": 0.8990,
            "test_r2_LCOM":       0.2487,
        }
        print("\n" + "=" * 72)
        print("COMPARISON vs. 2026-04-20 baseline (same config, unfixed dataset)")
        print("=" * 72)
        print(f"{'metric':<24} {'baseline':>10} {'fixed':>10} {'Δ':>10}")
        for name in ("TAC", "carbon_eff", "LCOM"):
            b = baseline[f"test_r2_{name}"]
            n = metrics[f"test_r2_{name}"]
            print(f"{'test_r2_' + name:<24} {b:>+10.4f} {n:>+10.4f} {n - b:>+10.4f}")

        # ─── Verdict ────────────────────────────────────────────────
        r2_lcom = metrics["test_r2_LCOM"]
        print("\n" + "=" * 72)
        if r2_lcom >= 0.70:
            print(f"VERDICT: ✓ CLIPPING WAS THE STORY — R²_LCOM = {r2_lcom:+.3f}")
            print("          GNN architecture is fine on a non-collapsed dataset.")
            print("          Recommend: ship GNN as-is, retrain on the full pipeline")
            print("          with the narrowed cap_factor range (or retrain the 5-D")
            print("          LCOM GPR with extended F_H2 domain to support cap_factor")
            print("          down to 0.3 for the RL environment).")
        elif r2_lcom >= 0.50:
            print(f"VERDICT: ~ PARTIAL — R²_LCOM = {r2_lcom:+.3f}")
            print("          The clipping WAS costing you signal, but it wasn't the")
            print("          whole story. Architecture matters too. Consider running")
            print("          step 2 (edge-state reframe) to see if a different target")
            print("          shape gives the GNN something it can do well.")
        else:
            print(f"VERDICT: ✗ CLIPPING WASN'T THE ISSUE — R²_LCOM = {r2_lcom:+.3f}")
            print("          The GNN's mean+add pool over shared-weight node embeddings")
            print("          genuinely can't extract LCOM's 5 specific design scalars.")
            print("          Move on to step 2 (edge-state reframe).")
        print("=" * 72)

        print(f"\nResults saved:\n  {out_csv}\n  {out_json}\n  {out_ckpt}")
        return 0

    finally:
        # ALWAYS restore NODE_FEATURE_SPEC so downstream imports in the
        # same Python process (notebooks, interactive sessions) aren't
        # silently affected by this script.
        fg.NODE_FEATURE_SPEC[3]["ranges"] = orig_ranges


if __name__ == "__main__":
    sys.exit(main())
