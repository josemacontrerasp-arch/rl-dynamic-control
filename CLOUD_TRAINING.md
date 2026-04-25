# Cloud Training Quickstart

Use this when a local laptop run is too slow.

## Recommended Flow

1. Start a GPU instance with a normal Ubuntu + CUDA PyTorch image.
   - Easiest: RunPod Pod, Lambda GPU instance, Paperspace, or Google Cloud VM.
   - A small/cheap GPU is fine for this project. RTX 3090/4090, L4, A10, or T4 class hardware is enough.

2. Upload or clone the repo.

3. From the repo root, install dependencies:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r rl_dynamic_control/requirements.txt torch-geometric
   ```

4. Run the cloud script:

   ```bash
   bash rl_dynamic_control/scripts/cloud_gnn_penalty_train.sh
   ```

5. Download:

   ```text
   rl_dynamic_control/outputs/extrapolation_fix/extrapolation_fix_results.csv
   rl_dynamic_control/outputs/extrapolation_fix/*.png
   rl_dynamic_control/outputs/overnight_logs/*.log
   rl_dynamic_control/saved_models/sac_gnn_penalty*
   ```

## Useful Overrides

Faster, cheaper run:

```bash
GNN_MC_SAMPLES=1 EVAL_EPISODES=10 bash rl_dynamic_control/scripts/cloud_gnn_penalty_train.sh
```

Full run:

```bash
TIMESTEPS=500000 EVAL_EPISODES=20 GNN_MC_SAMPLES=2 bash rl_dynamic_control/scripts/cloud_gnn_penalty_train.sh
```

Force CUDA:

```bash
DEVICE=cuda GNN_DEVICE=cuda bash rl_dynamic_control/scripts/cloud_gnn_penalty_train.sh
```

## Practical Notes

- Do not use free notebook runtimes for this unless you are okay with disconnects.
- Prefer a persistent disk/volume so checkpoints survive interruptions.
- Stop or destroy the instance when finished so billing stops.
- Keep `GNN_MC_SAMPLES=2` unless you specifically need stronger uncertainty estimates.
