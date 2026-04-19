#!/usr/bin/env python3
"""
Cross-Dataset Transfer Evaluation
==================================
Compares agents trained on the **GB synthetic** market to agents
trained on **ENTSO-E NL** and asks:

    * Does a GB-trained policy transfer to NL prices?
    * Does an NL-trained policy transfer back to GB?

Also evaluates the two rule-based baselines on each market for
reference.

Writes:
    outputs/transfer/transfer_matrix.csv
    outputs/transfer/transfer_bar.png

Usage::

    python -m rl_dynamic_control.scripts.transfer_evaluation \\
        --agents SAC PPO Q --episodes 20

Run from the EURECHA root.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rl_dynamic_control.config import RL_CFG, T_REACTOR_NOMINAL, P_REACTOR_NOMINAL
from rl_dynamic_control.data.price_loader import get_loader
from rl_dynamic_control.environment.methanol_plant_env import (
    DiscretizedMethanolEnv, MethanolPlantEnv,
)
from rl_dynamic_control.models.surrogates import PlantSurrogates

LOGGER = logging.getLogger("transfer_evaluation")


# --------------------------------------------------------------------------
# Evaluators
# --------------------------------------------------------------------------
def _run_episode(env, act_fn, ep_seed: int) -> Tuple[float, Dict[str, float]]:
    obs, _ = env.reset(seed=ep_seed)
    rews = []
    for _ in range(RL_CFG.episode_length):
        a = act_fn(obs)
        obs, r, term, trunc, info = env.step(a)
        rews.append(r)
        if term or trunc:
            break
    base = env.env if hasattr(env, "env") else env
    return float(np.sum(rews)), {
        "mean_reward": float(np.mean(rews)),
        "co2_util": base.mean_co2_utilisation() if hasattr(base, "mean_co2_utilisation") else 0.0,
        "extrap_frac": base.extrapolation_stats()["fraction"] if hasattr(base, "extrapolation_stats") else 0.0,
    }


def evaluate_agent(agent_kind: str, agent, prices: np.ndarray,
                   surrogates: PlantSurrogates, n_episodes: int, seed: int):
    """Returns (mean_total, std_total, diagnostics dict)"""
    totals, co2_utils, extrap_fracs = [], [], []
    for ep in range(n_episodes):
        if agent_kind == "q":
            base = MethanolPlantEnv(price_data=prices, surrogates=surrogates)
            env = DiscretizedMethanolEnv(base)
            def act(o, _a=agent):
                return _a.select_action(o, greedy=True)
        elif agent_kind in ("sac", "ppo"):
            env = MethanolPlantEnv(price_data=prices, surrogates=surrogates)
            def act(o, _a=agent):
                a, _ = _a.predict(o, deterministic=True)
                return a
        elif agent_kind == "full_load":
            env = MethanolPlantEnv(price_data=prices, surrogates=surrogates)
            _act_arr = np.array([1.0, T_REACTOR_NOMINAL, P_REACTOR_NOMINAL], dtype=np.float32)
            def act(_o):
                return _act_arr
        elif agent_kind == "rule":
            env = MethanolPlantEnv(price_data=prices, surrogates=surrogates)
            def act(o):
                load = 0.3 if o[0] > np.median(prices) else 1.0
                return np.array([load, T_REACTOR_NOMINAL, P_REACTOR_NOMINAL], dtype=np.float32)
        else:
            raise ValueError(agent_kind)
        total, diag = _run_episode(env, act, seed + ep)
        totals.append(total)
        co2_utils.append(diag["co2_util"])
        extrap_fracs.append(diag["extrap_frac"])
    return (float(np.mean(totals)), float(np.std(totals)),
            {"co2_util_mean": float(np.mean(co2_utils)),
             "extrap_frac_mean": float(np.mean(extrap_fracs))})


def load_agent(kind: str, market_tag: str, save_dir: Path):
    """Try to load a trained agent. Returns None if missing."""
    try:
        if kind == "q":
            from rl_dynamic_control.agents.q_learning import QLearningAgent
            p = save_dir / ("q_table.pkl" if market_tag == "gb" else "q_table_nl.pkl")
            return QLearningAgent.load(p) if p.exists() else None
        if kind == "sac":
            from stable_baselines3 import SAC
            p = save_dir / ("sac_methanol.zip" if market_tag == "gb" else "sac_methanol_nl.zip")
            return SAC.load(str(p)) if p.exists() else None
        if kind == "ppo":
            from stable_baselines3 import PPO
            p = save_dir / ("ppo_methanol.zip" if market_tag == "gb" else "ppo_methanol_nl.zip")
            return PPO.load(str(p)) if p.exists() else None
    except Exception as e:
        LOGGER.warning("Failed to load %s/%s: %s", kind, market_tag, e)
        return None
    return None


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", nargs="+", default=["SAC", "PPO", "Q"],
                        choices=["SAC", "PPO", "Q"])
    parser.add_argument("--gb-dataset", default="synthetic_gb")
    parser.add_argument("--nl-dataset", default="entso_nl_all")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=RL_CFG.seed)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(__file__).resolve().parent.parent / "outputs" / "transfer"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    save_dir = Path(__file__).resolve().parent.parent / "saved_models"
    loader = get_loader()
    markets = {"gb": args.gb_dataset, "nl": args.nl_dataset}
    for tag, name in list(markets.items()):
        if name not in loader.available():
            LOGGER.warning("Dataset %s missing; skipping market %s", name, tag)
            markets.pop(tag)

    surr = PlantSurrogates(use_gpr=True)

    rows: List[Dict[str, Any]] = []

    # Baselines per market
    for m_tag, m_name in markets.items():
        prices = loader.load(m_name)
        for base_kind in ("full_load", "rule"):
            mean_r, std_r, diag = evaluate_agent(
                base_kind, None, prices, surr, args.episodes, args.seed,
            )
            rows.append({
                "agent": base_kind, "trained_on": "-",
                "evaluated_on": m_tag, "mean_reward": mean_r,
                "std_reward": std_r, **diag,
            })
            LOGGER.info("%s on %s: %.3f ± %.3f", base_kind, m_tag, mean_r, std_r)

    # Cross-transfer matrix for each RL agent
    for algo in args.agents:
        kind = algo.lower()
        for train_tag in list(markets):
            agent = load_agent(kind, train_tag, save_dir)
            if agent is None:
                LOGGER.warning("No %s trained on %s — skipping", algo, train_tag)
                continue
            for eval_tag, eval_name in markets.items():
                prices = loader.load(eval_name)
                mean_r, std_r, diag = evaluate_agent(
                    kind, agent, prices, surr, args.episodes, args.seed,
                )
                rows.append({
                    "agent": algo, "trained_on": train_tag,
                    "evaluated_on": eval_tag, "mean_reward": mean_r,
                    "std_reward": std_r, **diag,
                })
                LOGGER.info("%s[train=%s] on %s: %.3f ± %.3f (CO2=%.2f, extrap=%.2f%%)",
                            algo, train_tag, eval_tag, mean_r, std_r,
                            diag["co2_util_mean"], 100 * diag["extrap_frac_mean"])

    df = pd.DataFrame(rows)
    csv_path = out_dir / "transfer_matrix.csv"
    df.to_csv(csv_path, index=False)

    # --- plot ---
    if not df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_df = df[df["agent"].isin(args.agents + ["full_load", "rule"])]
        labels, vals, errs, colors = [], [], [], []
        palette = {"gb": "#457b9d", "nl": "#e63946", "-": "#888"}
        for _, row in plot_df.iterrows():
            lbl = f"{row['agent']}\n({row['trained_on']}→{row['evaluated_on']})"
            labels.append(lbl)
            vals.append(row["mean_reward"])
            errs.append(row["std_reward"])
            colors.append(palette.get(row["trained_on"], "#888"))
        x = np.arange(len(labels))
        ax.bar(x, vals, yerr=errs, color=colors, alpha=0.85, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Mean episode reward (scaled £)")
        ax.set_title("Cross-Market Transfer — GB ↔ NL")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(out_dir / "transfer_bar.png", dpi=150)
        plt.close(fig)

    # --- summary ---
    print("\n" + "=" * 78)
    print(f"{'AGENT':10s} {'TRAIN':8s} {'EVAL':8s} {'MEAN':>10s} {'STD':>10s} "
          f"{'CO2 util':>10s} {'GPR extrap':>12s}")
    print("-" * 78)
    for _, r in df.iterrows():
        print(f"{r['agent']:10s} {str(r['trained_on']):8s} {r['evaluated_on']:8s} "
              f"{r['mean_reward']:10.3f} {r['std_reward']:10.3f} "
              f"{r['co2_util_mean']:10.3f} {100*r['extrap_frac_mean']:11.2f}%")
    print("=" * 78)
    print(f"CSV → {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
