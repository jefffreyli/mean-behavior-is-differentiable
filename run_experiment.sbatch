#!/bin/bash
#SBATCH -p normal
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:4
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -J mean-behavior-exp

# Experiment: 100 runs (networks) × 3 LRs × 100k steps
# Using 4 GPUs to parallelize training runs
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
echo "Config: Multiple nets × 3 LRs × 100k steps | Checkpoints: 0,100,1k,5k,10k,50k,100k"
echo "GPUs allocated: $SLURM_GPUS_ON_NODE"
echo "CPU cores: $SLURM_CPUS_ON_NODE"

# Test GPUs
echo "Available GPUs:"
nvidia-smi --list-gpus || echo "GPU check failed"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"

# Export number of GPUs for parallel training
export NUM_GPUS=${SLURM_GPUS_ON_NODE:-4}
echo "NUM_GPUS (from SLURM): $NUM_GPUS"
echo "CUDA_VISIBLE_DEVICES (from SLURM): ${CUDA_VISIBLE_DEVICES:-not set}"

# Set PyTorch memory management to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

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