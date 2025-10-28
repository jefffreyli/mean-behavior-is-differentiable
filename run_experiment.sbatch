#!/bin/bash
#SBATCH -p normal
#SBATCH -t 2:00:00
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -J mean-behavior-exp

# Don't specify -o and -e, let Slurm use default (slurm-<jobid>.out)

set -euo pipefail

# --- Conda for non-interactive shells ---
export PS1=${PS1:-"slurm"}
source /om2/user/jefffrey/anaconda/etc/profile.d/conda.sh
conda activate mean-behavior
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# --- Paths & env ---
export DATASETS=/om2/user/jefffrey/mean-behavior-is-differentiable/datasets
export RESULTS=/om2/user/jefffrey/mean-behavior-is-differentiable/results
mkdir -p "$DATASETS" "$RESULTS"

# --- Fix MKL threading layer compatibility ---
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU

cd /om2/user/jefffrey/mean-behavior-is-differentiable

# --- wandb (offline-first) ---
export WANDB_MODE=offline
export WANDB_PROJECT=mean-behavior-is-differentiable
export WANDB_DIR="$RESULTS"

# --- Print info ---
echo "Starting experiment on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"

# Test GPU
nvidia-smi || echo "GPU check failed"

# --- Run the experiment ---
echo "Running training..."
python experiment/main.py --mode train --device cuda

echo "Running collect..."
python experiment/main.py --mode collect --device cuda

echo "Running analyze..."
python experiment/main.py --mode analyze --device cuda

# --- Sync wandb offline runs ---
echo "Syncing wandb runs..."
wandb sync "$RESULTS" || echo "Wandb sync failed, continuing..."

echo "Experiment complete!"