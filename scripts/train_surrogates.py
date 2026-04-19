#!/usr/bin/env python3
"""
Train Real GPR Surrogates and Save as .pkl
==========================================
Uses the existing surrogate_optimization.py pipeline to generate
training data, fits Gaussian Process regressors, validates them,
and saves as .pkl files for the RL environment to load.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "figures" / "scripts"))
sys.path.insert(0, str(_ROOT / "scripts"))

import numpy as np
import pickle
from surrogate_optimization import (
    BOUNDS, generate_training_data, simplified_process_model, TAU
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.metrics import r2_score


def main() -> None:
    save_dir = Path(__file__).resolve().parent.parent / "saved_models"
    save_dir.mkdir(exist_ok=True)

    # ── 1. Generate training data ──
    print("[1] Generating 500 LHS training samples...")
    X_train, lcom_train, util_train, prod_train = generate_training_data(n=500, seed=42)
    print(f"    LCOM: [{lcom_train.min():.0f}, {lcom_train.max():.0f}] GBP/t")
    print(f"    Util: [{util_train.min()*100:.1f}, {util_train.max()*100:.1f}]%")

    # ── 2. RL-specific training data ──
    print("[2] Generating RL-specific training data (1000 pts)...")
    rng = np.random.RandomState(123)
    n_rl = 1000

    # Electrolyser load → H2 production & energy
    loads = rng.uniform(0.05, 1.0, n_rl)
    full_h2 = 270e3 / 50.0  # 5400 kg/hr

    h2_prod = np.array([
        l * full_h2 * (1.0 - 0.05 * (1 - l) ** 2) + rng.normal(0, 20)
        for l in loads
    ])
    energy = np.array([
        l * 270.0 + 6.14 * max(l, 0.1) + rng.normal(0, 0.5)
        for l in loads
    ])

    # Reactor → MeOH output
    T_s = rng.uniform(210, 280, n_rl)
    P_s = rng.uniform(50, 100, n_rl)
    H2_s = rng.uniform(2515, 3147, n_rl)
    meoh_out = np.array([
        simplified_process_model([T, P, 0.005, H2, 1.5])[2] / TAU
        for T, P, H2 in zip(T_s, P_s, H2_s)
    ])

    # ── 3. Train GPs ──
    print("[3] Training GP surrogates...")

    k1d = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5
    )

    # GP: load → H2
    print("    load → H2 production...")
    gp_h2 = GaussianProcessRegressor(
        kernel=k1d, n_restarts_optimizer=5, alpha=1e-6,
        normalize_y=True, random_state=0
    )
    gp_h2.fit(loads.reshape(-1, 1), h2_prod)

    # GP: (H2, T, P) → MeOH
    print("    (H2, T, P) → MeOH output...")
    X_meoh = np.column_stack([H2_s, T_s, P_s])
    X_meoh_mu, X_meoh_sig = X_meoh.mean(0), X_meoh.std(0)

    k3d = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.ones(3), length_scale_bounds=(1e-2, 1e2), nu=2.5
    )
    gp_meoh = GaussianProcessRegressor(
        kernel=k3d, n_restarts_optimizer=5, alpha=1e-6,
        normalize_y=True, random_state=0
    )
    gp_meoh.fit((X_meoh - X_meoh_mu) / X_meoh_sig, meoh_out)

    # GP: load → energy
    print("    load → energy consumption...")
    gp_energy = GaussianProcessRegressor(
        kernel=k1d, n_restarts_optimizer=5, alpha=1e-6,
        normalize_y=True, random_state=0
    )
    gp_energy.fit(loads.reshape(-1, 1), energy)

    # GP: 5D → LCOM
    print("    5D → LCOM...")
    X_mu, X_sig = X_train.mean(0), X_train.std(0)
    X_train_n = (X_train - X_mu) / X_sig

    k5d = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.ones(5), length_scale_bounds=(1e-2, 1e2), nu=2.5
    )
    gp_lcom = GaussianProcessRegressor(
        kernel=k5d, n_restarts_optimizer=5, alpha=1e-6,
        normalize_y=True, random_state=0
    )
    gp_lcom.fit(X_train_n, lcom_train)

    # GP: 5D → utilisation
    print("    5D → CO2 utilisation...")
    gp_util = GaussianProcessRegressor(
        kernel=k5d, n_restarts_optimizer=5, alpha=1e-6,
        normalize_y=True, random_state=0
    )
    gp_util.fit(X_train_n, util_train)

    # ── 4. Validate ──
    print("[4] Validating on held-out test sets...")
    rng2 = np.random.RandomState(77)
    n_test = 100

    test_loads = rng2.uniform(0.05, 1.0, n_test)
    true_h2 = np.array([l * full_h2 * (1.0 - 0.05 * (1 - l) ** 2) for l in test_loads])
    r2_h2 = r2_score(true_h2, gp_h2.predict(test_loads.reshape(-1, 1)))
    print(f"    H2 production:  R2 = {r2_h2:.4f}")

    T_t, P_t, H2_t = rng2.uniform(210, 280, n_test), rng2.uniform(50, 100, n_test), rng2.uniform(2515, 3147, n_test)
    true_meoh = np.array([simplified_process_model([T, P, 0.005, H2, 1.5])[2] / TAU for T, P, H2 in zip(T_t, P_t, H2_t)])
    X_tm = np.column_stack([H2_t, T_t, P_t])
    r2_meoh = r2_score(true_meoh, gp_meoh.predict((X_tm - X_meoh_mu) / X_meoh_sig))
    print(f"    MeOH output:    R2 = {r2_meoh:.4f}")

    true_en = np.array([l * 270 + 6.14 * max(l, 0.1) for l in test_loads])
    r2_en = r2_score(true_en, gp_energy.predict(test_loads.reshape(-1, 1)))
    print(f"    Energy consump: R2 = {r2_en:.4f}")

    U_t = rng2.rand(n_test, 5)
    X_t5 = BOUNDS[:, 0] + U_t * (BOUNDS[:, 1] - BOUNDS[:, 0])
    true_l = np.array([simplified_process_model(x)[0] for x in X_t5])
    r2_l = r2_score(true_l, gp_lcom.predict((X_t5 - X_mu) / X_sig))
    print(f"    LCOM (5D):      R2 = {r2_l:.4f}")

    true_u = np.array([simplified_process_model(x)[1] for x in X_t5])
    r2_u = r2_score(true_u, gp_util.predict((X_t5 - X_mu) / X_sig))
    print(f"    CO2 util (5D):  R2 = {r2_u:.4f}")

    # ── 5. Save ──
    print("[5] Saving .pkl files...")
    models = {
        "gp_h2_production.pkl": {"model": gp_h2, "input_names": ["load_fraction"]},
        "gp_meoh_output.pkl": {"model": gp_meoh, "X_mu": X_meoh_mu, "X_sig": X_meoh_sig, "input_names": ["H2_feed_kmol", "T_C", "P_bar"]},
        "gp_energy_consumption.pkl": {"model": gp_energy, "input_names": ["load_fraction"]},
        "gp_lcom_5d.pkl": {"model": gp_lcom, "X_mu": X_mu, "X_sig": X_sig},
        "gp_util_5d.pkl": {"model": gp_util, "X_mu": X_mu, "X_sig": X_sig},
    }
    for fname, data in models.items():
        path = save_dir / fname
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"    {fname} ({path.stat().st_size / 1024:.1f} KB)")

    print("\nALL GPR SURROGATES TRAINED AND SAVED")


if __name__ == "__main__":
    main()
