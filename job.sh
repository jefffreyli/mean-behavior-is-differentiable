#!/bin/bash                        
#SBATCH -t 2:00:00        
#SBATCH --gres=gpu:1  # Request exactly 1 GPU
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=/om2/user/jefffrey/edge-of-stochastic-stability/logs/output_keller_jordan_%A.log
#SBATCH --error=/om2/user/jefffrey/edge-of-stochastic-stability/logs/error_keller_jordan_%A.log
#SBATCH --job-name keller-jordan-exp

# Print the hostname of the compute node
hostname

# Create the logs directory
mkdir -p /om2/user/jefffrey/mean-behavior-is-differentiable/logs
cd /om2/user/jefffrey/mean-behavior-is-differentiable

# Create and activate the environment
python3 -m venv mean-behavior-is-differentiable
source mean-behavior-is-differentiable/bin/activate

# Install dependencies
pip install -r requirements.txt

export DATASETS="/om2/user/jefffrey/mean-behavior-is-differentiable/datasets"
export RESULTS="/om2/user/jefffrey/mean-behavior-is-differentiable/results"
export WANDB_MODE=offline
export WANDB_PROJECT="mean-behavior-is-differentiable"
export WANDB_DIR="$RESULTS"
export WANDB_API_KEY="40a5d32af2c7d39165d366ec3c29c9858e6d1c12"

# Ensure datasets are downloaded
python setup/download_datasets.py

echo "Setup complete! Running experiment..."

# Run the experiment
python experiment/main.py --mode train --device cuda
python experiment/main.py --mode collect --device cuda
python experiment/main.py --mode analyze --device cuda

# Sync wandb to cloud
cd $WANDB_DIR
wandb sync --sync-all --include-offline --mark-synced --no-include-synced

# Print completion message
echo "Experiment complete! Check plots in experiment/plots/"