# RL Dynamic Control — CO₂-to-Methanol Plant

Reinforcement-learning framework for dynamically operating a CO₂-to-methanol
synthesis plant in response to fluctuating electricity prices.  Part of the
[EURECHA 2026 Process Design Contest](../README.md) project.

## Context

The parent EURECHA project designs and optimises a cement-integrated
CO₂-to-methanol process (899 kmol/hr CO₂, 270 MW PEM electrolyser,
~213 kt/yr methanol).  The steady-state optimisation
(`scripts/surrogate_optimization.py`) and economic analysis
(`figures/scripts/constants.py`) provide all plant parameters.

This sub-project adds **dynamic operation**: an RL agent learns to
modulate the electrolyser load, reactor temperature, and reactor pressure
hour-by-hour to maximise profit under volatile electricity pricing.

## Architecture

```
rl_dynamic_control/
├── config.py                 # imports from ../figures/scripts/constants.py
├── environment/
│   └── methanol_plant_env.py # Gymnasium env (continuous + discretised)
├── agents/
│   ├── q_learning.py         # Tabular Q-learning (ε-greedy)
│   └── sb3_agent.py          # Stable-Baselines3 (PPO, SAC)
├── models/
│   └── surrogates.py         # Wraps existing simplified process model
├── utils/
│   └── reward.py             # Economic reward function
├── data/
│   └── electricity_prices.csv # 8760 h synthetic European day-ahead
└── scripts/
    ├── train_q_learning.py
    ├── train_sb3.py
    └── evaluate.py           # Compare RL vs constant-load & rule-based
```

## Quick Start

```bash
# From the EURECHA root directory:
pip install -r rl_dynamic_control/requirements.txt

# 1. Train tabular Q-learning (fast, good for debugging)
python -m rl_dynamic_control.scripts.train_q_learning --episodes 2000

# 2. Train PPO (deep RL, needs more compute)
python -m rl_dynamic_control.scripts.train_sb3 --algo PPO --timesteps 500000

# 3. Evaluate and compare against baselines
python -m rl_dynamic_control.scripts.evaluate --agent q_learning
python -m rl_dynamic_control.scripts.evaluate --agent ppo

# TensorBoard (for SB3 runs)
tensorboard --logdir rl_dynamic_control/tb_logs
```

## Observation & Action Spaces

**Observations** (8-dim continuous):
electricity price, electrolyser load, H₂ buffer level,
reactor T, reactor P, methanol rate, hour of day, day of week.

**Actions** (3-dim continuous / 250-dim discrete):
electrolyser load setpoint [0–1], reactor T [210–280 °C],
reactor P [50–100 bar].

## Reward

```
reward = methanol_revenue + CO₂_credits − electricity_cost
         − utility_cost − ramp_penalty − constraint_penalties
```

All economic values come from the project's `constants.py`:
e-methanol price £800/t, CO₂ credits £80/t, electrolyser 50 kWh/kg H₂,
CAPEX £503 M, utilities from Aspen heat integration.

## Extensions

Three extensions build on the base framework, reusing the existing
surrogates, reward, and environment with **no duplicated constants**:

### 1. Real ENTSO-E Dutch prices

* `data/fetch_entso_prices.py` — downloads 2023-2024 NL day-ahead prices
  from the ENTSO-E Transparency Platform (requires an API token set via
  `--token` or `ENTSOE_API_TOKEN`).  If no token is available, falls
  back to a deterministic synthetic series calibrated to the 2023-24
  NL statistics (mean ≈ €65/MWh, ~2–3 % negative-price hours, 12–48 h
  sustained wind troughs < €20, bank-holiday dips, winter/summer
  seasonality).
* `data/price_loader.py` — auto-discovers all CSVs in `data/` and
  exposes them as named datasets (`synthetic_gb`, `real_gb`,
  `entso_nl_all`, `entso_nl_2023`, `entso_nl_2024`) with
  season/half/date/month splits for robustness testing.
* The environment now accepts `price_dataset="entso_nl_all"` on init —
  no other code changes are needed to switch markets.
* `scripts/train_on_entso.py` — retrains SAC, PPO, and Q-learning on
  the NL data (saves `*_methanol_nl.*` separately from the originals).
* `scripts/transfer_evaluation.py` — cross-market evaluation: does a
  GB-trained agent transfer to NL, and vice versa?

```bash
python -m rl_dynamic_control.data.fetch_entso_prices --synthetic
python -m rl_dynamic_control.scripts.train_on_entso --algos SAC PPO Q
python -m rl_dynamic_control.scripts.transfer_evaluation --episodes 20
```

### 2. Multi-objective Pareto sweep

The reward has been generalised to
`R = λ_profit · profit + λ_co2 · co2_utilisation`, with
`co2_utilisation` = fraction of fresh CO₂ (`CO2_FEED_TPH` from
`constants.py`) converted to methanol per step — reusing the same
conversion logic the epsilon-constraint sweep in
`EURECHA/scripts/surrogate_optimization.py` uses.

* `scripts/pareto_sweep.py` — trains a fresh SAC agent for each
  λ ∈ {0, …, 1} (8–10 points), evaluates 20 episodes per point, records
  mean/std of profit and CO₂ utilisation.
* `scripts/plot_pareto.py` — produces the Pareto front and
  trade-off curves with the rule-based / full-load baselines overlaid.
  Output: `outputs/pareto/*.png`.

```bash
python -m rl_dynamic_control.scripts.pareto_sweep --n-weights 9 --steps 60000
python -m rl_dynamic_control.scripts.plot_pareto
```

### 3. Robustness testing

* `scripts/robustness_test.py` — temporal folds built via
  `PriceLoader.split()`:
  * H1 ↔ H2 (Q1+Q2 vs Q3+Q4, both directions),
  * leave-one-season-out (4 folds).

  For each fold we record retrained-SAC reward, **zero-shot transfer**
  reward from the original GB SAC, and the constant full-load
  baseline.  Each evaluation also reports the fraction of steps where
  the **GPR predictive std exceeds 2× the mean training-set std** —
  the new extrapolation-warning system built into
  `environment.methanol_plant_env` via
  `PlantSurrogates.flag_extrapolation()`.  A high fraction flags that
  the policy is exploiting regions where the surrogates are
  unreliable.
* `scripts/plot_robustness.py` — bar/box/heatmap plots +
  seasonal-breakdown chart in `outputs/robustness/`.

```bash
python -m rl_dynamic_control.scripts.robustness_test --steps 60000 --episodes 20
python -m rl_dynamic_control.scripts.plot_robustness
```

### 4. GPR Extrapolation Fix

The trained SAC agent operates in regions where the original 500-point
LHS surrogates are unreliable: 72-98% of evaluation steps trigger GPR
extrapolation warnings (predictive sigma > 2x training-set mean).
Absolute profit numbers are optimistic; relative rankings are robust.

Two complementary fixes address this:

**Approach A — Wider-domain surrogates (v2)**

* `models/retrain_surrogates.py` — collects agent trajectories, analyses
  the domain gap, generates 200-500 new LHS points biased toward the
  agent's operating region, and retrains GPR surrogates on the combined
  dataset.
* `models/surrogate_manager.py` — version-aware loader:
  `load_surrogates(version="v1")` or `load_surrogates(version="v2")`.
  The environment auto-selects v2 when available.
* **Limitation**: new points are pseudo-labelled by the simplified
  process model, not Aspen.  The surface is smoothed but systematic
  bias cannot be corrected without new Aspen runs.

**Approach B — Variance-penalised reward**

* `utils/variance_penalty.py` — modifies the reward:
  `R_safe = R_base - alpha * max(0, sigma_gpr - sigma_threshold)`
  where alpha is configurable and sigma_threshold = 2x mean training
  sigma.  The penalty steers the agent away from high-uncertainty
  regions during training.
* The environment accepts `use_variance_penalty=True` (off by default
  for backward compatibility).

**Retraining experiment**

Three SAC variants are trained to isolate each fix:

| Variant | Surrogates | Penalty | Purpose                    |
|---------|-----------|---------|----------------------------|
| (a)     | v2        | No      | Wider domain alone         |
| (b)     | v1        | Yes     | Penalty alone              |
| (c)     | v2        | Yes     | Both fixes combined        |

```bash
# Step 1: Retrain surrogates (Approach A)
python -m rl_dynamic_control.models.retrain_surrogates --episodes 50 --new-points 300

# Step 2: Train fix variants
python -m rl_dynamic_control.scripts.retrain_with_fixes --timesteps 500000

# Step 3: Plot results
python -m rl_dynamic_control.scripts.plot_extrapolation_fix

# Step 4: Full summary across all agents and price datasets
python -m rl_dynamic_control.scripts.final_summary --episodes 20
```

Output: `outputs/extrapolation_fix/` (CSVs, PNGs) and
`outputs/final_summary.csv` (definitive results table).

## Roadmap

1. ✅ Tabular Q-learning on discretised env
2. ✅ PPO / SAC via Stable-Baselines3
3. ✅ GPR extrapolation fix (wider surrogates + variance penalty)
4. 🔲 Plug in real Aspen-generated training data for v3 surrogates
5. 🔲 Aspen COM automation loop (live simulation in the RL step)
6. 🔲 Multi-agent: separate agents for electrolyser and reactor
7. 🔲 Stochastic price forecasting as additional observation
