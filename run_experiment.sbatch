#!/bin/bash
#SBATCH -p normal
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -J mean-behavior-exp

# Experiment: 100 networks × 3 LRs × 100k steps
# Don't specify -o and -e, let Slurm use default (slurm-<jobid>.out)

set -euo pipefail

# Conda for non-interactive shells
export PS1=${PS1:-"slurm"}
source /om2/user/jefffrey/anaconda/etc/profile.d/conda.sh
conda activate mean-behavior
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# Paths & env
export DATASETS=/om2/user/jefffrey/mean-behavior-is-differentiable/datasets
export RESULTS=/om2/user/jefffrey/mean-behavior-is-differentiable/results
mkdir -p "$DATASETS" "$RESULTS"

# Fix MKL threading layer compatibility
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU

cd /om2/user/jefffrey/mean-behavior-is-differentiable

# wandb (offline-first)
export WANDB_MODE=offline
export WANDB_PROJECT=mean-behavior-is-differentiable
export WANDB_DIR="$RESULTS"

# Print info
echo "Job: $SLURM_JOB_ID on $(hostname) at $(date)"
echo "Config: 100 nets × 3 LRs × 100k steps | Checkpoints: 0,100,1k,5k,10k,50k,100k"

# Test GPU
nvidia-smi || echo "GPU check failed"

# Run the experiment pipeline
echo "[1/4] Training..."
python experiment/main.py --mode train --device cuda

echo "Syncing offline runs to wandb"
wandb sync "$RESULTS/wandb" --sync-all || echo "Sync failed, continuing..."

echo "[2/4] Collecting logits..."
python experiment/main.py --mode collect --device cuda

echo "[3/4] Aggregating..."
python experiment/main.py --mode aggregate --device cuda

echo "[4/4] Generating plots..."
python experiment/main.py --mode analyze --device cuda

echo "Done at $(date) | Results: $RESULTS"