#!/usr/bin/env python3
"""
Multi-Objective Pareto Sweep (Profit vs CO₂ Utilisation)
=========================================================
Trains a fresh SAC agent for each weight in
    λ_profit ∈ [0, 1]   (λ_co2 = 1 − λ_profit)
sweeping 8-10 steps.  Uses the multi-objective scalarisation already
built into the environment:

    R = λ_profit · profit_reward + λ_co2 · co2_utilisation

After training each weight vector, evaluates on 20 episodes and records
mean/std of profit and CO₂ utilisation.

Writes:
    outputs/pareto/pareto_results.csv        raw per-λ stats
    outputs/pareto/baselines.csv             constant full-load + rule-based
    (plots produced by scripts/plot_pareto.py)

The objective definitions (profit = reward from ``utils.reward`` with
λ_profit=1, utilisation = fraction of fresh CO₂ converted) are consistent
with the existing epsilon-constraint framework in
``EURECHA/scripts/surrogate_optimization.py``.

Usage::

    python -m rl_dynamic_control.scripts.pareto_sweep \\
        --n-weights 9 --steps 60000 --episodes 20

Run from the EURECHA root directory.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rl_dynamic_control.config import RL_CFG, T_REACTOR_NOMINAL, P_REACTOR_NOMINAL
from rl_dynamic_control.data.price_loader import load_prices
from rl_dynamic_control.environment.methanol_plant_env import MethanolPlantEnv
from rl_dynamic_control.models.surrogates import PlantSurrogates
from rl_dynamic_control.agents.sb3_agent import make_sb3_agent, train_sb3

LOGGER = logging.getLogger("pareto_sweep")


def evaluate_episode(env: MethanolPlantEnv, act_fn, seed: int) -> Dict[str, float]:
    obs, _ = env.reset(seed=seed)
    profit_sum = 0.0
    co2_sum = 0.0
    n = 0
    for _ in range(RL_CFG.episode_length):
        a = act_fn(obs)
        obs, _r, term, trunc, info = env.step(a)
        profit_sum += info["profit_reward"]
        co2_sum += info["co2_utilisation"]
        n += 1
        if term or trunc:
            break
    return {"profit": profit_sum, "co2_util": co2_sum / max(n, 1)}


def eval_agent(model, prices, surrogates, n_ep, seed) -> Dict[str, float]:
    profits, cos = [], []
    for ep in range(n_ep):
        # λ=(1,0) at eval to measure raw profit; CO₂ util is recorded separately
        env = MethanolPlantEnv(price_data=prices, surrogates=surrogates,
                               lambda_profit=1.0, lambda_co2=0.0)
        def act(o, _m=model):
            a, _ = _m.predict(o, deterministic=True)
            return a
        res = evaluate_episode(env, act, seed + ep)
        profits.append(res["profit"])
        cos.append(res["co2_util"])
    return {
        "profit_mean": float(np.mean(profits)),
        "profit_std": float(np.std(profits)),
        "co2_mean": float(np.mean(cos)),
        "co2_std": float(np.std(cos)),
    }


def eval_baseline(kind: str, prices, surrogates, n_ep, seed) -> Dict[str, float]:
    profits, cos = [], []
    act_full = np.array([1.0, T_REACTOR_NOMINAL, P_REACTOR_NOMINAL], dtype=np.float32)
    for ep in range(n_ep):
        env = MethanolPlantEnv(price_data=prices, surrogates=surrogates,
                               lambda_profit=1.0, lambda_co2=0.0)
        if kind == "full_load":
            def act(_o):
                return act_full
        else:  # rule-based price-threshold
            thr = float(np.median(prices))
            def act(o, _t=thr):
                load = 0.3 if o[0] > _t else 1.0
                return np.array([load, T_REACTOR_NOMINAL, P_REACTOR_NOMINAL],
                                dtype=np.float32)
        res = evaluate_episode(env, act, seed + ep)
        profits.append(res["profit"])
        cos.append(res["co2_util"])
    return {"kind": kind,
            "profit_mean": float(np.mean(profits)),
            "profit_std": float(np.std(profits)),
            "co2_mean": float(np.mean(cos)),
            "co2_std": float(np.std(cos))}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-weights", type=int, default=9,
                        help="Number of λ samples in [0, 1].")
    parser.add_argument("--steps", type=int, default=60_000,
                        help="SAC timesteps per weight (≈60-70%% of original run).")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Eval episodes per weight.")
    parser.add_argument("--price-dataset", default="auto")
    parser.add_argument("--seed", type=int, default=RL_CFG.seed)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--keep-models", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(__file__).resolve().parent.parent / "outputs" / "pareto"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    prices = load_prices(args.price_dataset)
    LOGGER.info("Using prices '%s' (n=%d, mean=%.2f)",
                args.price_dataset, len(prices), prices.mean())

    surrogates = PlantSurrogates(use_gpr=True)

    lambdas = np.linspace(0.0, 1.0, args.n_weights)
    rows: List[Dict] = []
    models_dir = out_dir / "models"
    models_dir.mkdir(exist_ok=True)

    for i, lam in enumerate(lambdas):
        lam_co2 = 1.0 - lam
        LOGGER.info("── [%d/%d] λ_profit=%.3f λ_co2=%.3f ──",
                    i + 1, args.n_weights, lam, lam_co2)
        env = MethanolPlantEnv(price_data=prices, surrogates=surrogates,
                               lambda_profit=float(lam), lambda_co2=float(lam_co2))
        model = make_sb3_agent(env, algo="SAC", seed=args.seed + i,
                               tb_log_dir=str(out_dir / "tb"))
        save_path = models_dir / f"sac_lambda_{lam:.3f}"
        train_sb3(model, total_timesteps=args.steps,
                  save_path=save_path if args.keep_models else None,
                  tb_log_name=f"SAC_lam{lam:.2f}")

        stats = eval_agent(model, prices, surrogates, args.episodes, args.seed + 1000 * i)
        stats.update(lambda_profit=float(lam), lambda_co2=float(lam_co2))
        rows.append(stats)
        LOGGER.info("λ=%.2f → profit=%.3f±%.3f  CO2util=%.3f±%.3f",
                    lam, stats["profit_mean"], stats["profit_std"],
                    stats["co2_mean"], stats["co2_std"])

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "pareto_results.csv", index=False)

    # Baselines
    base_rows = []
    for kind in ("full_load", "rule"):
        base_rows.append(eval_baseline(kind, prices, surrogates,
                                       args.episodes, args.seed))
    pd.DataFrame(base_rows).to_csv(out_dir / "baselines.csv", index=False)

    # ---- summary table ----
    print("\n" + "=" * 78)
    print(f"{'λ_profit':>8s} {'λ_co2':>8s} {'profit μ':>10s} {'σ':>8s} "
          f"{'CO₂ util μ':>11s} {'σ':>8s}")
    print("-" * 78)
    for r in rows:
        print(f"{r['lambda_profit']:8.3f} {r['lambda_co2']:8.3f} "
              f"{r['profit_mean']:10.3f} {r['profit_std']:8.3f} "
              f"{r['co2_mean']:11.4f} {r['co2_std']:8.4f}")
    print("-" * 78)
    for b in base_rows:
        print(f"[{b['kind']:10s}] profit={b['profit_mean']:+.3f}±{b['profit_std']:.3f} "
              f"CO2util={b['co2_mean']:.3f}±{b['co2_std']:.3f}")
    print("=" * 78)
    print(f"Results → {out_dir}/pareto_results.csv")
    print(f"Plot with: python -m rl_dynamic_control.scripts.plot_pareto")
    return 0


if __name__ == "__main__":
    sys.exit(main())
