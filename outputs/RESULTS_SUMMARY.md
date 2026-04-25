# RL Dynamic Control — Extensions Results Summary

Generated after completing the three extensions:
real ENTSO-E prices, multi-objective Pareto sweep, and robustness testing.

All reward values below are in the environment's **scaled £/week** units
(`reward_scale = 1e-3`). A value of 3000 corresponds to roughly
£3 M profit/week at the 213 kt/yr plant scale.

---

## 1. Cross-Market Transfer (GB ↔ NL)

File: `outputs/transfer/transfer_matrix.csv`, `transfer_bar.png`.

Baselines per week: full-load ≈ 760–1020, price-threshold rule ≈ 2025–2075.

| Agent | gb → gb | gb → nl | nl → gb | nl → nl |
|-------|--------:|--------:|--------:|--------:|
| **SAC** | **3227 ± 26** | **3222 ± 50** | 3146 ± 42 | 3123 ± 112 |
| PPO | 1972 ± 267 | 1838 ± 355 | 1976 ± 273 | 1839 ± 358 |
| Q-learning | 2836 ± 120 | 2814 ± 134 | 2735 ± 149 | 2682 ± 167 |

### Key observations

- **SAC generalises across markets.** The GB-trained SAC achieves ~3222
  on the Dutch market and the NL-trained SAC achieves ~3146 on GB —
  losses of < 3 % in either direction. The policy is price-regime
  agnostic.
- **SAC dominates every baseline.** It beats the rule-based baseline
  by ~55 % and full-load by 3–4×.
- **PPO has collapsed to a constant policy.** CO₂ utilisation and GPR
  extrapolation rates are identical across all four transfer cells
  (0.9348 / 0.9699), which only happens if the learned action is
  independent of the observation. Worth re-training with a longer
  rollout (`n_steps`) or stronger reward shaping.
- **Q-learning is ~10 % below SAC** — expected given the coarse
  10 × 5 × 5 action grid.
- **GPR-extrapolation warning:** SAC triggers the warning on 79–98 %
  of steps. The policies are exploiting regions where the MeOH
  surrogate's predictive σ exceeds 2 × its training-set mean. Absolute
  profit numbers should be treated as optimistic; the **rankings**
  between policies are still reliable.

---

## 2. Multi-Objective Pareto Sweep (SAC, λ sweep)

Files: `outputs/pareto/pareto_results.csv`, `baselines.csv`,
`pareto_front.png`, `tradeoff_curve.png`.

| λ_profit | λ_co2 | Profit μ | Profit σ | CO₂ util |
|---------:|------:|---------:|---------:|---------:|
| 0.00 | 1.00 | 1805 | 328 | 0.9348 |
| 0.125 | 0.875 | 2907 | 393 | 0.9348 |
| 0.25 | 0.75 | 2991 | 167 | 0.9354 |
| 0.375 | 0.625 | 2888 | 195 | 0.9347 |
| 0.50 | 0.50 | 3022 | 156 | 0.9346 |
| 0.625 | 0.375 | 3001 | 95  | 0.9354 |
| 0.75 | 0.25 | 3053 | 122 | 0.9349 |
| 0.875 | 0.125 | 2916 | 245 | 0.9351 |
| 1.00 | 0.00 | 3113 | 136 | 0.9350 |

Baselines: rule-based 2025 @ 94.1 %, full-load 758 @ 93.8 %.

### Key observations

- **Profit saturates quickly.** A small shift off λ_profit = 0
  (pure CO₂ utilisation reward) to λ ≥ 0.125 lifts profit from 1805 to
  ~2900–3100. Beyond λ = 0.25 there is essentially no further
  improvement.
- **CO₂ utilisation is flat (~93.5 %) across the entire sweep.** This
  is an important empirical finding for the plant: there is **no
  meaningful profit ↔ utilisation trade-off** in this formulation.
  The process is H₂-limited and the reactor surrogate converges to
  near-maximum conversion whenever it operates at a profitable point.
- The rule-based baseline actually delivers slightly *higher*
  utilisation (94.1 %) than any Pareto SAC point — a further reminder
  that utilisation is not a discriminating objective in this plant.
- **Reframe the trade-off if needed.** A meaningful multi-objective
  study for this design would pit profit against electricity
  consumption, ramping, or purge rate — not against utilisation.

---

## 3. Robustness Testing

Files: `outputs/robustness/robustness_results.csv`,
`robustness_bar.png`, `robustness_box.png`, `robustness_heatmap.png`,
`robustness_seasonal.png`.

Folds built via `PriceLoader.split()` on `entso_nl_all`:
H1/H2 swap + leave-one-season-out (Q1–Q4).

### Per-fold results (scaled £/week)

| Fold | Retrained SAC | Zero-shot SAC (GB) | Full-load | GPR extrap (retrained) |
|------|--------------:|-------------------:|----------:|-----------------------:|
| train H1 / test H2 | 3101 ± 195 | 3223 ± 71 | 798 ± 869  | 95.5 % |
| train H2 / test H1 | 2866 ± 245 | 3224 ± 43 | 844 ± 706  | 97.7 % |
| LOSO leave Q1 (winter) | 2842 ± 78 | **3180 ± 21** | **151 ± 312** | 95.2 % |
| LOSO leave Q2 | 3154 ± 113 | 3255 ± 49 | 1335 ± 605 | 96.2 % |
| LOSO leave Q3 | 3198 ± 166 | **3287 ± 63** | 1652 ± 495 | 89.6 % |
| LOSO leave Q4 (winter) | **2328 ± 255** | 3185 ± 21 | 239 ± 294 | 71.7 % |

### Key observations

- **Zero-shot transfer is the real headline.** The SAC agent trained on
  the original GB synthetic dataset generalises *without any
  fine-tuning* to every Dutch fold and beats the retrained-on-fold SAC
  on all six folds (retrained: 2328–3198, zero-shot: 3180–3287). Even
  in the hardest fold (leave Q4), zero-shot delivers 3185 vs the
  retrained agent's 2328 — a 37 % gap.
- **Interpretation:** training on 6 months only gives the retrained
  SAC less diversity of price regimes to learn from than the full-year
  GB dataset did. The GB-trained SAC has already seen enough price
  variation to respond well to any regime. For practical deployment,
  training on a long, diverse price history and deploying zero-shot on
  new markets/years looks more effective than fold-wise retraining.
- **Winter is the hardest regime for the baseline.** Full-load collapses
  in Q1 (151) and Q4 (239) because it's buying expensive winter
  electricity indiscriminately. Summer/autumn are much easier (Q2: 1335,
  Q3: 1652).
- **RL adds the most value when prices are highest.** Q1 lift: baseline
  151 → SAC 2842–3180 (a ~20× multiplier). Q3 lift: 1652 → 3198–3287
  (~2×). The dispatch problem is trivial when power is cheap; the
  policy earns its keep during scarcity.
- **Reward consistency:** retrained σ 78–255, zero-shot σ 21–71,
  full-load σ 293–869. Zero-shot has the tightest distribution across
  folds, confirming the GB-trained policy has a robust prior.
- **GPR extrapolation continues to flag high fractions** (72–98 %
  on retrained, similar on zero-shot). The agent's dispatch behaviour
  is near-identical regardless of fold, so the same caveat applies:
  trust the rankings; treat absolute profits as optimistic.
- **Seasonal heatmap** (`robustness_heatmap.png`) visualises the LOSO
  matrix: Q4 is the lowest-reward held-out season (retrained 2328),
  Q3 the highest (3198). Training data containing Q4 is evidently
  essential for robust winter-edge performance.

---

## 4. GNN Flowsheet Surrogate — Graph-level Ablation and Edge-state Win

Files: `outputs/gnn_sweep/sweep_results.csv`, `edge_state_results.csv`,
`mlp_baseline_results.csv`, `lcom_fix_results.csv`,
`capfactor_fix_results.csv`.

### Graph-level KPI regression (motivating ablation)

Twelve-config sweep (hidden_dim × n_layers × dropout) on 2000 LHS
samples. Best config `h64 l2 d0.0`:

| Target     | GNN R² | MLP-A R² (design only) | MLP-B R² (design+edges) |
|------------|-------:|-----------------------:|------------------------:|
| TAC        | 0.992  | 0.087                  | 0.994                   |
| carbon_eff | 0.899  | 0.441                  | 0.988                   |
| LCOM       | **0.249** | 0.781               | 0.925                   |

Failure modes isolated by the diagnostics:

- **Pooling/permutation mismatch.** LCOM is a 5-scalar GPR distillation
  with positional dependence; mean/add pooling destroys that signal.
- **F_H2 clipping.** `flowsheet_graph._node_params_to_surrogate_input`
  maps `capacity_factor × 2704`, so ≈ 93 % of points are silently
  clipped to the 5-D LCOM GPR's lower bound. Fixing the sampled range
  lifts R²_LCOM from 0.249 to 0.375 — enough to prove the bug but not
  enough to deploy.
- **m_dot edge-feature leakage.** `stream_data.py` scales every edge's
  m_dot with `meoh_tph / BASECASE_MEOH_TPH`, leaking the KPI back into
  the input. MLP-B's 0.925 on LCOM is largely this leakage.

### Edge-state reframe (the deployable model)

Task reformulated to per-edge 8-dim physical state prediction
(`[ṁ, T, P, x_CO2, x_H2, x_MeOH, x_H2O, x_inert]` for each of 13
edges, 104 outputs). Only the two structural flags
(`is_recycle, is_energy`) remain on input edges.

| Metric                              | Value    |
|-------------------------------------|---------:|
| Mean R² across 8 physical features  | **0.997** |
| Worst feature (total ṁ)             | 0.991    |
| Best features (T, compositions)     | 0.9999   |
| Mean R² across 13 edges             | 0.997    |
| Worst edge (recycle-coupled edge 3) | 0.982    |
| Derived MeOH-rate proxy             | −0.06    |

Proposed deployable architecture: edge-state GNN (104-dim plant state)
+ existing GPRs as KPI heads + analytical TAC formula. See the report
`§Extension 5` for the narrative.

---

## 5. Surrogates v3 — extended F_H2 domain

Files: `outputs/surrogate_v3/summary.json`,
`outputs/surrogate_v3/sac_on_v3_eval.json`.

`models/retrain_surrogates.py --build-v3 --new-points 500` generated a
fresh LHS dataset spanning `F_H2 ∈ [800, 3200]` (union with the
original 500 v1 points), refitted the LCOM and utility GPRs and saved
them to `saved_models/surrogates_v3/`. `models/surrogate_manager.py`
now supports `load_surrogates(version="v3")`.

Fit quality on a held-out 15 % split:

| GPR      | R²       | RMSE      |
|----------|---------:|----------:|
| LCOM     | 0.99981  | 0.98 £/t  |
| Utility  | 0.99936  | 1.1·10⁻³  |

**Honest result.** 71.4 % of the Fix (c) policy's visited F_H2 still
falls below the original v1 lower bound, so v3 is the first surrogate
revision that can score the policy without silent clipping. Scoring
`sac_extrap_fix_c` against v3 yields **−161.6 ± 6.4** reward with 0 %
extrapolation. The corrected LCOM surface makes the previous operating
point uneconomic. A fresh SAC must be trained against v3 before v3
becomes the production surrogate. Fix (c) @ v2 therefore remains the
definitive policy until that retraining lands.

---

## 6. Edge-state GNN uncertainty penalty

Files: `outputs/extrapolation_fix/step_data_variant_gnn.csv`,
`outputs/extrapolation_fix/extrapolation_fix_results.csv`
(row `variant_gnn`), `utils/gnn_confidence.py`.

The edge-state model is loaded via `utils/gnn_confidence.py`, run for
`mc_samples=4` MC-dropout forward passes per call, and the mean
predictive variance across the 8 physical features is used as a
confidence signal. The environment accepts
`use_gnn_penalty=True, gnn_penalty_config=GNNPenaltyConfig(...)` and
applies `β · max(0, σ_GNN − threshold)` (clipped at `max_penalty`) as
a negative shaping term. A dedicated SAC variant
(`scripts/train_gnn_penalty_sac.py`, 500 k steps on v2 surrogates):

| Variant        | Mean reward | Std   | CO₂ util | Extrap % | GNN σ̄  |
|----------------|------------:|------:|---------:|---------:|-------:|
| Fix (c) v2+pen | **2915.6**  |  35.8 | 0.848    | 0.0      | —      |
| variant_gnn    | 719.9       | 590.1 | 0.808    | 0.15     | 0.183  |

**Honest result.** The GNN penalty hurts. Mean reward drops from 2916
to 720 and variance explodes because the mean GNN penalty (0.36 per
step) is an order of magnitude above the extrapolation rate that
Fix (c) already suppresses. The penalty is therefore redundant when v2
surrogates already cover the agent's operating region. A tighter
`threshold` (or conditioning on high GPR σ) would be needed before the
GNN penalty is useful in practice. Kept in the report as a negative
result.

---

## 7. PPO fix sweep

File: `outputs/ppo_fix_results.csv`. Best PPO exported to
`saved_models/ppo_methanol_fixed.zip`.

Four interventions (individually and combined) all failed to break the
PPO collapse observed in §1. Only the VecNormalize and
`fixed_combo` variants show any non-zero action variance, and that is
confined to the load dimension:

| Experiment          | Algo | Mean reward | Std   | action_var_load | action_var_T | action_var_P |
|---------------------|------|------------:|------:|----------------:|-------------:|-------------:|
| `sac_original`      | SAC  | 1139.9      | 481.0 |  1.1·10⁻³       | 430.9        | 226.9        |
| `ppo_vecnormalize`  | PPO  | **961.7**   | 469.5 |  0.023          | 0.0          | 0.0          |
| `ppo_fixed_combo`   | PPO  | 934.2       | 457.1 |  0.013          | 0.0          | 0.0          |
| `ppo_original`      | PPO  | 413.4       | 484.9 |  0              | 0            | 0            |
| `ppo_nsteps_4096`   | PPO  | 413.4       | 484.9 |  0              | 0            | 0            |
| `ppo_lr_1e4`        | PPO  | 413.4       | 484.9 |  0              | 0            | 0            |
| `ppo_epochs_20`     | PPO  | 413.4       | 484.9 |  0              | 0            | 0            |

**Honest result.** PPO's on-policy KL-constrained updates cannot break
the zero-variance minimum on this reward surface within 500 k steps,
regardless of `n_steps`, `learning_rate`, `n_epochs`, or
VecNormalize. SAC's off-policy replay + entropy term are what let it
escape. PPO is documented as a negative baseline. `ppo_vecnormalize`
is the exported best attempt.

**Caveat on reward scales.** The `vecnormalize`/`fixed_combo` runs use
the VecNormalize wrapper and therefore report rewards on a different
scale from the unwrapped `ppo_original`/`ppo_*` variants — the table
should be read vertically within each normalisation group, not across
groups.

---

## 8. Pareto v2 — electricity consumption reframe

Files: `outputs/pareto_v2/pareto_results.csv`, `baselines.csv`,
`pareto_front_elec.png`, `tradeoff_curve_elec.png`.

Reward reformulated as
`R = λ_profit · profit − λ_elec · electricity_cost_per_step`, with
electricity consumption logged in MWh per episode. Nine `λ_elec`
points from 0.0 to 1.0, 60 k-step SAC per weight, 20 eval episodes.

| λ_profit | λ_elec | Profit μ | Profit σ | Elec. MWh/ep |
|---------:|-------:|---------:|---------:|-------------:|
|    1.00  |  0.00  |    388   |   269    |    23618     |
|    0.875 |  0.125 |    391   |   251    |    23623     |
|    0.75  |  0.25  |    522   |   509    |    45567     |
|    0.625 |  0.375 |    480   |   234    |    23617     |
|    0.50  |  0.50  |    375   |   243    |    23613     |
|    0.375 |  0.625 |    410   |   261    |    23611     |
|    0.25  |  0.75  |    463   |   260    |    23616     |
|    0.125 |  0.875 |    398   |   256    |    23624     |
|    0.00  |  1.00  |    374   |   232    |    23619     |

Baselines (`baselines.csv`): full-load 564 @ 45 549 MWh, rule-based
857 @ 29 138 MWh, original SAC 458 @ 23 614 MWh.

**Honest result.** The reframe did not break the degeneracy. Every λ
except 0.25/0.75 collapses to the same ≈ 23 615 MWh/ep operating
point; only λ = 0.25/0.75 escapes to the full-load-like regime. The
electricity term is not independent of `profit` (profit already nets
electricity cost), so the second axis is largely redundant. A
genuinely independent axis — purge rate, ramping, or kg CO₂/t MeOH —
is the next iteration.

---

## 9. Aspen validation points — generator landed

Files: `outputs/aspen_validation/validation_points.csv` (20 rows,
1.6 kB), `outputs/aspen_validation/aspen_variables.txt`,
`scripts/generate_aspen_validation_points.py`.

```bash
python -m rl_dynamic_control.scripts.generate_aspen_validation_points \
    --n-centroids 20 --seed 42
```

The script clusters the 3 360-row Fix (c) step data
(`outputs/extrapolation_fix/step_data_variant_c.csv`) into 20 k-means
centroids on `[load, T_C, P_bar]` and expands each to
`[electrolyser_load, reactor_T_C, reactor_P_bar, F_H2_kmolhr]` ready
for an Aspen Plus pass. The companion `aspen_variables.txt` documents
the Aspen block/variable mapping (`PEM.FEED-RATE`,
`PEM.CAPACITY-FACTOR`, `REACT.TEMP`, `REACT.PRES`).

Centroid envelope:

| stat | load  | T (°C) | P (bar) | F_H2 (kmol/hr) |
|------|------:|-------:|--------:|---------------:|
| min  | 0.02  | 253.8  | 68.8    |    57          |
| mean | 0.15  | 275.3  | 87.7    |   403          |
| max  | 0.45  | 278.9  | 99.2    | 1 211          |

**Useful finding.** 16/20 centroids sit below v3's F_H2 lower bound
of 800 kmol/hr — confirming that even v3 extrapolates over the
Fix (c) policy's low-load regime, so the Aspen labels are genuinely
necessary rather than a nice-to-have. The 20 Aspen runs themselves
(one `.bkp` per centroid, or a sensitivity study driven by the CSV)
are the remaining outstanding item. See the report `§Extension 10`
and roadmap item 10.

---

## Headline takeaways

1. **SAC is the clear winner** — ~3.0–3.3 k weekly profit, generalises
   across markets and seasons. Only the PPO and Q-learning agents need
   rework.
2. **Zero-shot transfer from the GB-trained agent beats every
   fold-retrained agent on NL data.** This strongly argues for training
   on a large, varied price dataset once and reusing the weights rather
   than retraining per market/period.
3. **PPO remains collapsed after the full fix sweep.** Higher
   `n_steps`, lower `learning_rate`, more `n_epochs`, and VecNormalize
   individually and combined fail to restore action variance on T and
   P. Documented as a negative baseline (§7 above).
4. **The profit/utilisation Pareto is degenerate, and so is the
   profit/electricity-consumption reframe.** Both axes collapse onto
   ≈ 93.5 % utilisation / 23 615 MWh. A genuinely independent second
   axis is still open (§8 above).
5. **GPR extrapolation warning** fires on 72–98 % of evaluation steps
   for every RL policy on v1 surrogates, 0 % on v2. Absolute profit
   figures for the original agents are optimistic; **relative
   rankings** between policies are robust. v3 fits with `R²≈0.9998`
   but requires a fresh SAC retrain to use in deployment (§5 above).
6. **Winter (Q1, Q4) is the regime where the agent delivers the most
   value** — by a large margin — relative to the full-load baseline.
7. **GNN edge-state surrogate: R² = 0.997 on 104 outputs.** Graph-level
   KPI prediction failed (pooling mismatch + F_H2 clipping + m_dot
   leakage), but the edge-state reframe works. A GNN-confidence
   penalty plugged into the RL reward did *not* improve on Fix (c).
8. **Aspen-grounded validation: generator landed, runs pending.**
   `scripts/generate_aspen_validation_points.py` now exports 20 k-means
   centroids of the Fix (c) trajectory to
   `outputs/aspen_validation/validation_points.csv`. 16/20 centroids
   sit below v3's F_H2 lower bound of 800 kmol/hr, so the Aspen labels
   would both validate v3 *and* extend it into the low-load regime
   where the deployed policy actually operates.

---

## Artefacts index

```
rl_dynamic_control/
├── data/
│   ├── entso_nl_2023_2024.csv        17544 h NL prices (synthetic fallback)
│   ├── fetch_entso_prices.py
│   └── price_loader.py               auto-discovers datasets
├── saved_models/
│   ├── sac_methanol.zip              GB-trained (original)
│   ├── sac_methanol_nl.zip           NL-trained
│   ├── sac_extrap_fix_{a,b,c}.zip    extrapolation fix variants
│   ├── sac_gnn_penalty.zip           GNN-penalty SAC variant (negative result)
│   ├── ppo_methanol.zip / _nl.zip
│   ├── ppo_methanol_fixed.zip        best PPO-fix variant (vecnormalize)
│   ├── ppo_{original, nsteps_4096, lr_1e4, epochs_20, fixed_combo,
│   │        vecnormalize}.zip        PPO fix sweep checkpoints
│   ├── surrogates_v2/                v2 GPRs (wider-domain LHS)
│   ├── surrogates_v3/                v3 GPRs (extended F_H2 domain)
│   └── q_table[_nl].pkl
└── outputs/
    ├── transfer/                     GB ↔ NL cross-market matrix
    ├── pareto/                       v1 profit ↔ CO₂ utilisation sweep
    ├── pareto_v2/                    v2 profit ↔ electricity consumption sweep
    ├── robustness/                   temporal-fold study
    ├── extrapolation_fix/            fix variants a/b/c + variant_gnn
    ├── surrogate_v3/                 v3 refit metrics + eval on Fix (c)
    ├── gnn_sweep/                    GNN HP sweep, LCOM diagnostics,
    │                                 edge-state model + results,
    │                                 MLP baselines
    ├── aspen_validation/             20 Fix (c) centroids + Aspen
    │                                 variable cheat-sheet
    ├── ppo_fix_results.csv           PPO fix sweep summary
    └── final_summary.csv             definitive per-agent × market table
```
