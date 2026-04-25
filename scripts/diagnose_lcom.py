"""
Diagnose the low LCOM R² from the GNN sweep.
=============================================

Hypothesis: LCOM has low variance across LHS samples (and/or outliers),
so even a model with small MAE looks bad on R².

What this script does
---------------------
1. Loads ``best_model.pt`` (the winning config from the sweep).
2. Rebuilds the same 2000-graph LHS dataset (seed=42) and re-splits
   it 70/15/15 — this is DETERMINISTIC so we get the exact test set
   that was used during the sweep.
3. Runs inference and denormalises predictions.
4. Generates a 4-panel diagnostic plot:
      ┌─────────────────────────┬─────────────────────────┐
      │ (a) LCOM truth histogram│ (b) Parity: pred vs true│
      ├─────────────────────────┼─────────────────────────┤
      │ (c) Residuals vs truth  │ (d) TAC vs LCOM         │
      └─────────────────────────┴─────────────────────────┘
5. Prints a text report with std, IQR, skew, outlier count
   (|z| > 3 on LCOM), and correlation between TAC and LCOM.

Run:
    cd EURECHA/rl_dynamic_control/
    python scripts/diagnose_lcom.py

Output:
    outputs/gnn_sweep/lcom_diagnostics.png
    outputs/gnn_sweep/lcom_diagnostics.txt

Author: Pepe (Jose Maria Contreras Prada)
Project: EURECHA 2026 Process Design Contest
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

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
# Config — mirror the sweep defaults so the split is identical
# ─────────────────────────────────────────────────────────────────────

SEED         = 42
N_SAMPLES    = 2000
TRAIN_FRAC   = 0.70
VAL_FRAC     = 0.15
CKPT_PATH    = _PKG / "outputs" / "gnn_sweep" / "best_model.pt"
OUT_PNG      = _PKG / "outputs" / "gnn_sweep" / "lcom_diagnostics.png"
OUT_TXT      = _PKG / "outputs" / "gnn_sweep" / "lcom_diagnostics.txt"


def build_dataset(n_samples: int, seed: int) -> list:
    samples = generate_lhs_samples(n_samples, seed=seed)
    dataset = []
    for params in samples:
        res = evaluate_with_surrogates(params)
        graph = build_base_graph(
            node_params=params,
            edge_features=res["stream_features"],
            targets={"TAC": res["TAC"], "carbon_eff": res["carbon_eff"], "LCOM": res["LCOM"]},
        )
        dataset.append(graph)
    return dataset


def split_dataset(dataset: list, seed: int,
                  train_frac: float = TRAIN_FRAC, val_frac: float = VAL_FRAC):
    n = len(dataset)
    n_train = int(round(n * train_frac))
    n_val   = int(round(n * val_frac))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    idx_test = perm[n_train + n_val:]
    return [dataset[i] for i in idx_test]


def denormalise(y_norm: torch.Tensor, mu: torch.Tensor, sig: torch.Tensor) -> torch.Tensor:
    return y_norm * sig + mu


def infer_on_test(model, test_data, mu, sig) -> tuple[np.ndarray, np.ndarray]:
    from torch_geometric.loader import DataLoader
    model.eval()
    loader = DataLoader(test_data, batch_size=64)
    preds, truths = [], []
    with torch.no_grad():
        for batch in loader:
            preds.append(model(batch))
            truths.append(batch.y)
    y_pred_norm = torch.cat(preds, dim=0)
    y_true_norm = torch.cat(truths, dim=0)
    y_pred = denormalise(y_pred_norm, mu, sig).numpy()
    y_true = denormalise(y_true_norm, mu, sig).numpy()
    return y_pred, y_true


def summary_stats(y: np.ndarray, name: str) -> dict:
    from scipy.stats import skew
    q1, q3 = np.percentile(y, [25, 75])
    z = (y - y.mean()) / (y.std() + 1e-12)
    return {
        "name":         name,
        "n":            int(y.size),
        "mean":         float(y.mean()),
        "std":          float(y.std()),
        "min":          float(y.min()),
        "max":          float(y.max()),
        "range_pct":    float(100.0 * (y.max() - y.min()) / (abs(y.mean()) + 1e-12)),
        "IQR":          float(q3 - q1),
        "skew":         float(skew(y)),
        "outliers_3s":  int((np.abs(z) > 3).sum()),
    }


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def make_plot(y_true_all: np.ndarray, y_pred_all: np.ndarray,
              target_names: list[str], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Indices into the target tuple — match flowsheet_graph's order
    iT = target_names.index("TAC")
    iC = target_names.index("carbon_eff")
    iL = target_names.index("LCOM")

    lcom_true = y_true_all[:, iL]
    lcom_pred = y_pred_all[:, iL]
    tac_true  = y_true_all[:, iT]
    resid     = lcom_pred - lcom_true

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (a) Histogram of LCOM truth
    ax = axes[0, 0]
    ax.hist(lcom_true, bins=30, color="#2b7a78", edgecolor="white")
    ax.axvline(lcom_true.mean(), color="red", lw=1.5, label=f"mean = {lcom_true.mean():.1f}")
    ax.set_xlabel("LCOM (£/t) — ground truth")
    ax.set_ylabel("count")
    ax.set_title(f"(a) LCOM distribution — std={lcom_true.std():.1f}  range={lcom_true.max()-lcom_true.min():.0f}")
    ax.legend()
    ax.grid(ls=":", alpha=0.4)

    # (b) Parity plot pred vs truth
    ax = axes[0, 1]
    ax.scatter(lcom_true, lcom_pred, s=12, alpha=0.6, color="#2b7a78")
    lo, hi = min(lcom_true.min(), lcom_pred.min()), max(lcom_true.max(), lcom_pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    r2 = compute_r2(lcom_true, lcom_pred)
    mae = np.abs(lcom_pred - lcom_true).mean()
    ax.set_xlabel("LCOM truth (£/t)")
    ax.set_ylabel("LCOM predicted (£/t)")
    ax.set_title(f"(b) Parity — R²={r2:+.3f}  MAE={mae:.2f}")
    ax.legend()
    ax.grid(ls=":", alpha=0.4)

    # (c) Residuals vs truth
    ax = axes[1, 0]
    ax.scatter(lcom_true, resid, s=12, alpha=0.6, color="#d62728")
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("LCOM truth (£/t)")
    ax.set_ylabel("Residual (pred − truth)")
    ax.set_title("(c) Residuals — look for systematic bias or heteroscedasticity")
    ax.grid(ls=":", alpha=0.4)

    # (d) TAC vs LCOM — do they co-vary?
    ax = axes[1, 1]
    ax.scatter(tac_true / 1e6, lcom_true, s=12, alpha=0.6, color="#2b7a78")
    r_pearson = float(np.corrcoef(tac_true, lcom_true)[0, 1])
    ax.set_xlabel("TAC truth (M£/yr)")
    ax.set_ylabel("LCOM truth (£/t)")
    ax.set_title(f"(d) TAC vs LCOM  — pearson r={r_pearson:+.3f}")
    ax.grid(ls=":", alpha=0.4)

    fig.suptitle("LCOM diagnostic — best model (h64 l2 d0.0) on test set",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> int:
    # ─── 1. Load checkpoint ─────────────────────────────────────────
    if not CKPT_PATH.exists():
        print(f"ERROR: {CKPT_PATH} not found. Run the sweep first.", file=sys.stderr)
        return 1
    ck = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    mu, sig = ck["y_mean"], ck["y_std"]
    print(f"Loaded best_model.pt — config: {cfg}")
    print(f"  target means (train+val):  TAC={mu[0].item():.3e}  carb={mu[1].item():.4f}  LCOM={mu[2].item():.3f}")
    print(f"  target stds  (train+val):  TAC={sig[0].item():.3e}  carb={sig[1].item():.4f}  LCOM={sig[2].item():.3f}")

    # ─── 2. Rebuild dataset + split ─────────────────────────────────
    print(f"\nRebuilding {N_SAMPLES}-graph LHS dataset (seed={SEED})...")
    dataset = build_dataset(N_SAMPLES, SEED)
    test = split_dataset(dataset, SEED)
    print(f"  test set size: {len(test)}")

    # Apply the SAME z-score that the sweep used
    for d in test:
        d.y = (d.y - mu) / sig

    # ─── 3. Rebuild + load model, infer ─────────────────────────────
    model = FlowsheetGNN(
        hidden_dim=cfg["hidden_dim"],
        n_layers=cfg["n_layers"],
        dropout=cfg["dropout"],
        pool="mean+add",
    )
    model.load_state_dict(ck["state_dict"])

    y_pred, y_true = infer_on_test(model, test, mu, sig)
    target_names = ["TAC", "carbon_eff", "LCOM"]

    # ─── 4. Summary stats ────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("TEST-SET TARGET DISTRIBUTION")
    print("=" * 66)
    lines = []
    for i, name in enumerate(target_names):
        s = summary_stats(y_true[:, i], name)
        line = (f"{name:<11}  n={s['n']}  mean={s['mean']:+.4g}  "
                f"std={s['std']:.4g}  range={s['range_pct']:.1f}%  "
                f"skew={s['skew']:+.2f}  outliers(|z|>3)={s['outliers_3s']}")
        print(line); lines.append(line)

    print("\n" + "=" * 66)
    print("MODEL PERFORMANCE PER TARGET")
    print("=" * 66)
    for i, name in enumerate(target_names):
        r2  = compute_r2(y_true[:, i], y_pred[:, i])
        mae = np.abs(y_pred[:, i] - y_true[:, i]).mean()
        rel = 100.0 * mae / (abs(y_true[:, i].mean()) + 1e-12)
        line = f"{name:<11}  R²={r2:+.4f}  MAE={mae:.4g}  MAE%mean={rel:.2f}%"
        print(line); lines.append(line)

    r_tac_lcom = float(np.corrcoef(y_true[:, 0], y_true[:, 2])[0, 1])
    print(f"\nTAC ↔ LCOM correlation:       r = {r_tac_lcom:+.3f}")
    lines.append(f"TAC-LCOM correlation r={r_tac_lcom:+.3f}")

    # ─── 5. Plot + write text report ───────────────────────────────
    print(f"\nWriting diagnostic plot → {OUT_PNG}")
    make_plot(y_true, y_pred, target_names, OUT_PNG)
    OUT_TXT.write_text("\n".join(lines))
    print(f"Writing text report   → {OUT_TXT}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
