#!/usr/bin/env bash
set -euo pipefail

TIMESTEPS="${TIMESTEPS:-500000}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
SEED="${SEED:-42}"
GNN_BETA="${GNN_BETA:-2.0}"
GNN_THRESHOLD="${GNN_THRESHOLD:-0.001}"
GNN_MC_SAMPLES="${GNN_MC_SAMPLES:-2}"
GNN_DROPOUT_P="${GNN_DROPOUT_P:-0.05}"
GNN_MAX_PENALTY="${GNN_MAX_PENALTY:-5}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
ENT_COEF="${ENT_COEF:-auto_0.1}"
DEVICE="${DEVICE:-auto}"
GNN_DEVICE="${GNN_DEVICE:-auto}"

cd "$(dirname "$0")/../.."

mkdir -p rl_dynamic_control/outputs/overnight_logs
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="rl_dynamic_control/outputs/overnight_logs/cloud_gnn_penalty_${STAMP}.log"

echo "Logging to ${LOG}"

python - <<'PY'
import importlib, sys
print("python:", sys.executable)
for name in ("torch", "torch_geometric", "stable_baselines3"):
    mod = importlib.import_module(name)
    print(f"{name}: {getattr(mod, '__version__', 'unknown')}")
try:
    import torch
    print("cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda_device:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("cuda_check_failed:", exc)
PY

{
  python -m rl_dynamic_control.scripts.train_gnn_penalty_sac \
    --timesteps "${TIMESTEPS}" \
    --eval-episodes "${EVAL_EPISODES}" \
    --seed "${SEED}" \
    --gnn-beta "${GNN_BETA}" \
    --gnn-threshold "${GNN_THRESHOLD}" \
    --gnn-mc-samples "${GNN_MC_SAMPLES}" \
    --gnn-dropout-p "${GNN_DROPOUT_P}" \
    --gnn-max-penalty "${GNN_MAX_PENALTY}" \
    --learning-rate "${LEARNING_RATE}" \
    --ent-coef "${ENT_COEF}" \
    --device "${DEVICE}" \
    --gnn-device "${GNN_DEVICE}"

  python -m rl_dynamic_control.scripts.plot_extrapolation_fix
} 2>&1 | tee "${LOG}"

echo "Done."
echo "Summary: rl_dynamic_control/outputs/extrapolation_fix/extrapolation_fix_results.csv"
echo "Plots:   rl_dynamic_control/outputs/extrapolation_fix/"
