# RL Dynamic Control for CO2-to-Methanol

This repository contains the dynamic-control work for a cement-integrated
CO2-to-methanol process. The base EURECHA model gives the plant design and
economics; this folder adds reinforcement learning, electricity-price data,
surrogate checks, and evaluation scripts for hour-by-hour operation.

The core question is simple:

> Can an RL policy operate the electrolyser and reactor more profitably under
> volatile power prices without wandering into surrogate-model blind spots?

The current answer is: SAC works best, PPO is a useful negative baseline, and
uncertainty checks matter a lot.

## What is included

- A Gymnasium environment for hourly methanol-plant operation.
- Tabular Q-learning, PPO, and SAC agents.
- Synthetic and ENTSO-E-style electricity price loaders.
- GPR surrogate management, including wider-domain v2/v3 variants.
- Variance-penalised rewards for safer surrogate use.
- A flowsheet GNN experiment for edge-state prediction.
- Evaluation scripts, plots, trained model artifacts, and the project report.

## Repository Layout

```text
rl_dynamic_control/
  agents/                  RL agent wrappers
  data/                    price datasets and price-loading helpers
  environment/             Gymnasium methanol plant environment
  models/                  process surrogate loading/retraining
  scripts/                 training, evaluation, plotting, diagnostics
  utils/                   reward and uncertainty utilities
  outputs/                 result tables and generated figures
  report/                  paper/report source, figures, and PDF
  saved_models/            trained policies and surrogate checkpoints
  flowsheet_graph.py       flowsheet graph and GNN model definitions
  stream_data.py           stream/edge data used by the GNN experiments
```

## Setup

This folder was developed inside the larger EURECHA project, where the plant
constants live in:

```text
EURECHA/figures/scripts/constants.py
EURECHA/scripts/surrogate_optimization.py
```

For a fresh checkout, keep this folder at `EURECHA/rl_dynamic_control` and run
commands from the parent `EURECHA` directory.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r rl_dynamic_control/requirements.txt
```

## Common Commands

Train the smaller tabular baseline:

```bash
python -m rl_dynamic_control.scripts.train_q_learning --episodes 2000
```

Train SAC or PPO:

```bash
python -m rl_dynamic_control.scripts.train_sb3 --algo SAC --timesteps 500000
python -m rl_dynamic_control.scripts.train_sb3 --algo PPO --timesteps 500000
```

Evaluate policies:

```bash
python -m rl_dynamic_control.scripts.evaluate --agent sac
python -m rl_dynamic_control.scripts.final_summary --episodes 20
```

Run the extrapolation-fix comparison:

```bash
python -m rl_dynamic_control.models.retrain_surrogates --episodes 50 --new-points 300
python -m rl_dynamic_control.scripts.retrain_with_fixes --timesteps 500000
python -m rl_dynamic_control.scripts.plot_extrapolation_fix
```

Run the GNN diagnostics:

```bash
cd rl_dynamic_control
python scripts/train_gnn_sweep.py --smoke
python scripts/train_gnn_edge_state.py --smoke
python scripts/diagnose_lcom.py
```

Generate Aspen validation centroids:

```bash
python -m rl_dynamic_control.scripts.generate_aspen_validation_points --n-centroids 20 --seed 42
```

## Main Results

The detailed tables live in `outputs/` and the write-up is in `report/`. The
short version:

- SAC is the strongest RL policy in this environment.
- The v2 surrogate plus variance-penalised SAC policy is the current best
  deployable result: mean reward around 2916, CO2 utilisation around 0.85, and
  near-zero GPR extrapolation warnings in the evaluation runs.
- PPO remained close to a constant-action policy after several tuning attempts
  (`n_steps`, lower learning rate, more epochs, VecNormalize, and a combined
  setup). It is kept as a negative baseline rather than presented as a final
  controller.
- A graph-level GNN was not a good fit for direct KPI prediction, especially
  LCOM. Reframing the problem as per-edge physical-state prediction worked much
  better: the edge-state model reached mean R2 about 0.997 across the 13 edges
  and 8 physical stream features.
- Adding a GNN uncertainty penalty to the RL reward did not improve on the
  variance-penalised SAC controller. The penalty made rewards lower and noisier
  in the tested configuration.
- The profit-vs-CO2-utilisation Pareto sweep is nearly flat because CO2
  utilisation saturates. A second sweep using electricity consumption was also
  mostly degenerate because electricity cost is already embedded in profit.
- The Aspen validation point generator is in place. The actual Aspen runs and a
  future Aspen-labelled v4 refit are still open work.

## Important Files

```text
outputs/extrapolation_fix/extrapolation_fix_results.csv
outputs/ppo_fix_results.csv
outputs/pareto_v2/pareto_results.csv
outputs/gnn_sweep/edge_state_results.csv
outputs/surrogate_v3/summary.json
outputs/aspen_validation/validation_points.csv
report/rl_dynamic_control_report.pdf
```

## Notes on Reproducibility

- The committed outputs are the record of the runs used in the report.
- TensorBoard logs, raw checkpoint folders, and local cache files are ignored.
- Large final model artifacts are kept only when they are needed to reproduce a
  reported result.
- The v3 surrogate extends the H2-flow domain, but the existing v2-trained SAC
  policy should not be treated as validated outside the regions checked in the
  report.

## Roadmap

- Run the 20 Aspen validation cases exported in
  `outputs/aspen_validation/validation_points.csv`.
- Refit the next surrogate version on Aspen-labelled low-load points.
- Retrain SAC directly against the wider validated surrogate surface.
- Add a tighter uncertainty gate if the GNN confidence term is revisited.
- Keep PPO in the report as a baseline unless a different policy formulation
  breaks the collapse.
