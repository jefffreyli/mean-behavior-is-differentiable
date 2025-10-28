#!/usr/bin/env python3
"""
Keller Jordan Experiment: Mean Behavior is Differentiable in the Learning Rate

This script reproduces the experiment showing that mean neural network behavior
is differentiable with respect to learning rate, while also tracking batch
sharpness and lambda_max during training.

Usage:
    python experiment/main.py --mode train    # Run training
    python experiment/main.py --mode analyze  # Generate plots
    python experiment/main.py --mode all      # Run both
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

# Add parent directory to path to import from utils
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from utils (after sys.path is modified)
# fmt: off
from utils.data import prepare_dataset, get_dataset_presets  # noqa: E402
from utils.nets import prepare_net, get_model_presets  # noqa: E402
from utils.wandb_utils import find_closest_checkpoint_wandb, get_checkpoint_dir_for_run  # noqa: E402
from utils.naming import compose_run_name  # noqa: E402
# fmt: on

# Check if wandb is available
try:
    import wandb
    from visualization.vis_utils import RunCollection
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Visualization features will be limited.")


# ============================================================================
# Configuration
# ============================================================================

class ExperimentConfig:
    """Configuration for the Keller Jordan experiment."""

    # Learning rates to test
    LEARNING_RATES = [0.4, 0.5, 0.6]

    # Number of runs per learning rate (1 for proof of concept)
    N_RUNS_PER_LR = 1

    # Training hyperparameters
    DATASET = "cifar10"
    MODEL = "cnn"
    BATCH_SIZE = 128
    NUM_STEPS = 5
    NUM_DATA = 2048
    CLASSES = [1, 9]

    # Experiment tracking
    WANDB_TAG = "keller-jordan-exp"

    # Paths
    EXPERIMENT_DIR = Path(__file__).parent
    DATA_DIR = EXPERIMENT_DIR / "data"
    PLOTS_DIR = EXPERIMENT_DIR / "plots"
    CHECKPOINT_DIR = Path(os.environ.get(
        "WANDB_DIR", ".")) / "wandb_checkpoints"

    @classmethod
    def get_logits_dir(cls, lr: float) -> Path:
        """Get directory for storing logits for a given learning rate."""
        lr_str = f"lr{int(lr*10):02d}"
        return cls.DATA_DIR / f"logits_{lr_str}"

    @classmethod
    def setup_directories(cls):
        """Create necessary directories."""
        cls.DATA_DIR.mkdir(exist_ok=True, parents=True)
        cls.PLOTS_DIR.mkdir(exist_ok=True, parents=True)
        for lr in cls.LEARNING_RATES:
            cls.get_logits_dir(lr).mkdir(exist_ok=True, parents=True)


# ============================================================================
# Training
# ============================================================================

def run_single_training(lr: float, run_idx: int, config: ExperimentConfig) -> Dict:
    """
    Launch a single training run.

    Args:
        lr: Learning rate
        run_idx: Run index (0-9)
        config: Experiment configuration

    Returns:
        Dictionary with run information
    """
    # Use different seeds for each run
    seed = 1000 + int(lr * 100) + run_idx

    # Build command
    cmd = [
        sys.executable, "training.py",
        "--dataset", config.DATASET,
        "--model", config.MODEL,
        "--batch", str(config.BATCH_SIZE),
        "--lr", str(lr),
        "--steps", str(config.NUM_STEPS),
        "--num-data", str(config.NUM_DATA),
        "--classes", str(config.CLASSES[0]), str(config.CLASSES[1]),
        "--init-seed", str(seed),
        "--dataset-seed", str(seed),
        "--batch-sharpness",
        "--lambdamax",
        "--wandb-tag", config.WANDB_TAG,
        "--wandb-name", f"exp_lr{lr}_run{run_idx}",
        "--checkpoint-every", "1",  # Save checkpoint every step for logit collection
    ]

    print(f"\n{'='*60}")
    print(f"Training: LR={lr}, Run={run_idx}/{config.N_RUNS_PER_LR-1}")
    print(f"Seed: {seed}")
    print(f"{'='*60}")

    # Run training
    result = subprocess.run(cmd, cwd=config.EXPERIMENT_DIR.parent, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Warning: Training failed with return code {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return {"lr": lr, "run_idx": run_idx, "seed": seed, "success": False}

    return {"lr": lr, "run_idx": run_idx, "seed": seed, "success": True}


def run_all_training(config: ExperimentConfig):
    """Run all training experiments."""
    print("\n" + "="*60)
    print("STARTING KELLER JORDAN EXPERIMENT")
    print("="*60)
    print(f"Learning rates: {config.LEARNING_RATES}")
    print(f"Runs per LR: {config.N_RUNS_PER_LR}")
    print(f"Total runs: {len(config.LEARNING_RATES) * config.N_RUNS_PER_LR}")
    print("="*60 + "\n")

    config.setup_directories()

    results = []
    for lr in config.LEARNING_RATES:
        for run_idx in range(config.N_RUNS_PER_LR):
            result = run_single_training(lr, run_idx, config)
            results.append(result)

    # Save results
    results_file = config.DATA_DIR / "training_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    successful = sum(1 for r in results if r.get("success", False))
    print(f"Successful runs: {successful}/{len(results)}")
    print(f"Results saved to: {results_file}")


# ============================================================================
# Logit Collection
# ============================================================================

def load_model_from_checkpoint(checkpoint_path: Path, config: ExperimentConfig, device: str = "cpu") -> nn.Module:
    """Load a trained model from checkpoint."""
    # Get model architecture
    model_presets = get_model_presets()
    dataset_presets = get_dataset_presets()

    params = model_presets[config.MODEL]['params'].copy()
    params['input_dim'] = dataset_presets[config.DATASET]['input_dim']
    params['output_dim'] = dataset_presets[config.DATASET]['output_dim']

    model = prepare_net(
        model_type=model_presets[config.MODEL]['type'],
        params=params
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)

    return model


def extract_logits(model: nn.Module, X_test: torch.Tensor, batch_size: int = 500, device: str = "cpu") -> np.ndarray:
    """Extract logits from a model on test data."""
    model.eval()
    logits_list = []

    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch = X_test[i:i+batch_size].to(device)
            outputs = model(batch)
            logits_list.append(outputs.cpu().numpy())

    return np.concatenate(logits_list, axis=0)


def collect_logits_for_run(lr: float, run_idx: int, config: ExperimentConfig,
                           X_test: torch.Tensor, device: str = "cpu") -> np.ndarray:
    """
    Collect logits for a single run.

    Args:
        lr: Learning rate
        run_idx: Run index
        config: Experiment configuration
        X_test: Test data
        device: Device to use

    Returns:
        Logits array of shape (n_test, n_classes)
    """
    # Construct the full run name using the same logic as training
    # Create a mock args object with the necessary attributes
    class MockArgs:
        def __init__(self, dataset, model, batch, lr, wandb_name):
            self.dataset = dataset
            self.model = model
            self.batch = batch
            self.lr = lr
            self.wandb_name = wandb_name
    
    mock_args = MockArgs(
        dataset=config.DATASET,
        model=config.MODEL,
        batch=config.BATCH_SIZE,
        lr=lr,
        wandb_name=f"exp_lr{lr}_run{run_idx}"
    )
    run_name = compose_run_name(mock_args)
    
    print(f"Searching for run: {run_name}")
    
    # Use the same approach as sharpness plots - query by tag and find matching LR
    if WANDB_AVAILABLE:
        try:
            from visualization.vis_utils import RunCollection
            
            # Load all runs with the experiment tag
            collection = RunCollection.from_tag(
                tag=config.WANDB_TAG,
                load_dataframes=False  # Don't need dataframes for checkpoint loading
            )
            
            print(f"Found {len(collection.runs)} runs with tag {config.WANDB_TAG}")
            
            # Debug: Print all runs and their LRs
            print(f"Looking for LR: {lr}")
            for run in collection.runs:
                print(f"  Run: {run.run_name}, LR: {run.lr}, ID: {run.run_id}")
            
            # Find the run with matching LR
            for run in collection.runs:
                if abs(run.lr - lr) < 0.001:  # Float comparison with tolerance
                    print(f"Found matching run: {run.run_name} (ID: {run.run_id}, LR: {run.lr})")
                    
                    # Look for checkpoints using the run ID
                    wandb_dir = Path(os.environ.get("WANDB_DIR", "."))
                    checkpoint_locations = [
                        wandb_dir / "wandb_checkpoints" / run.run_id,
                        Path(os.environ.get("RESULTS", ".")) / "wandb_checkpoints" / run.run_id,
                    ]
                    
                    for checkpoint_dir in checkpoint_locations:
                        print(f"  Checking: {checkpoint_dir}")
                        if checkpoint_dir.exists():
                            print(f"  Found checkpoints!")
                            return _load_logits_from_checkpoint_dir(checkpoint_dir, config, X_test, device)
                    
                    print(f"  Warning: Run found but no checkpoints at any location")
        except Exception as e:
            print(f"Error using RunCollection: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"Warning: No run found for {run_name}")
    return None


def _load_logits_from_checkpoint_dir(checkpoint_dir: Path, config: ExperimentConfig, 
                                   X_test: torch.Tensor, device: str) -> np.ndarray:
    """Helper function to load logits from a checkpoint directory."""
    # Find the final checkpoint
    metadata_file = checkpoint_dir / "checkpoint_metadata.json"
    if not metadata_file.exists():
        print(f"Warning: No metadata file in {checkpoint_dir}")
        return None

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    checkpoints = metadata.get('checkpoints', [])
    if not checkpoints:
        print(f"Warning: No checkpoints in metadata in {checkpoint_dir}")
        return None

    # Get the last checkpoint
    final_checkpoint = max(checkpoints, key=lambda x: x['step'])
    checkpoint_path = Path(final_checkpoint['path'])

    print(f"Loading checkpoint from {checkpoint_path}")

    # Load model and extract logits
    model = load_model_from_checkpoint(checkpoint_path, config, device)
    logits = extract_logits(model, X_test, device=device)

    return logits


def collect_all_logits(config: ExperimentConfig, device: str = "cpu"):
    """Collect logits from all trained models."""
    print("\n" + "="*60)
    print("COLLECTING LOGITS")
    print("="*60)

    # Load test dataset
    DATASET_FOLDER = Path(os.environ.get('DATASETS', './datasets'))
    data = prepare_dataset(
        config.DATASET,
        DATASET_FOLDER,
        config.NUM_DATA,
        config.CLASSES,
        dataset_seed=888,
        loss_type='mse'
    )
    X_train, Y_train, X_test, Y_test = data

    print(f"Test set size: {len(X_test)}")

    for lr in config.LEARNING_RATES:
        print(f"\nProcessing LR={lr}")
        logits_dir = config.get_logits_dir(lr)

        for run_idx in range(config.N_RUNS_PER_LR):
            output_file = logits_dir / f"run_{run_idx}.npy"

            if output_file.exists():
                print(f"  Run {run_idx}: Already exists, skipping")
                continue

            print(f"  Run {run_idx}: Collecting logits...")
            logits = collect_logits_for_run(
                lr, run_idx, config, X_test, device)

            if logits is not None:
                np.save(output_file, logits)
                print(f"  Run {run_idx}: Saved to {output_file}")
            else:
                print(f"  Run {run_idx}: Failed to collect logits")

    print("\n" + "="*60)
    print("LOGIT COLLECTION COMPLETE")
    print("="*60)


# ============================================================================
# Visualization
# ============================================================================

def generate_correlation_plot(config: ExperimentConfig):
    """Generate the correlation plot showing linear relationship between LR deltas."""
    print("\n" + "="*60)
    print("GENERATING CORRELATION PLOT")
    print("="*60)

    # Load all logits
    logits_by_lr = {}
    for lr in config.LEARNING_RATES:
        logits_dir = config.get_logits_dir(lr)
        logits_list = []

        for run_idx in range(config.N_RUNS_PER_LR):
            logit_file = logits_dir / f"run_{run_idx}.npy"
            if logit_file.exists():
                logits = np.load(logit_file)
                logits_list.append(logits)
            else:
                print(f"Warning: Missing logits for LR={lr}, run={run_idx}")

        if logits_list:
            # Stack to shape (n_runs, n_test, n_classes)
            logits_by_lr[lr] = np.stack(logits_list, axis=0)
            print(
                f"LR={lr}: Loaded {len(logits_list)} runs, shape={logits_by_lr[lr].shape}")

    if len(logits_by_lr) < 3:
        print("Error: Need logits for all 3 learning rates")
        return

    # Compute mean logits
    out4 = logits_by_lr[0.4].mean(axis=0)  # Shape: (n_test, n_classes)
    out5 = logits_by_lr[0.5].mean(axis=0)
    out6 = logits_by_lr[0.6].mean(axis=0)

    print(f"Mean logit shapes: {out4.shape}")

    # Compute deltas and flatten
    xx = (out5 - out4).flatten()
    yy = (out6 - out4).flatten()

    # Compute correlation
    correlation = np.corrcoef(xx, yy)[0, 1]
    r_squared = correlation ** 2

    # Compute the ratio (should be ~2)
    slope, intercept, _, _, _ = stats.linregress(xx, yy)

    print(f"Correlation: {correlation:.4f}")
    print(f"R²: {r_squared:.4f}")
    print(f"Slope: {slope:.4f} (expected ~2.0)")

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter plot with small points
    ax.scatter(xx, yy, s=1, alpha=0.5, color='blue')

    # Add regression line
    x_line = np.array([xx.min(), xx.max()])
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r-', linewidth=2,
            label=f'y = {slope:.2f}x + {intercept:.2f}')

    # Labels and title
    ax.set_xlabel('(out_0.5 - out_0.4).flatten()', fontsize=12)
    ax.set_ylabel('(out_0.6 - out_0.4).flatten()', fontsize=12)
    ax.set_title(f'Mean Behavior is Differentiable in Learning Rate\nR² = {r_squared:.4f}, N = {config.N_RUNS_PER_LR}',
                 fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save plot
    output_file = config.PLOTS_DIR / "correlation_plot.png"
    print(f"Saving plot to: {output_file}")
    print(f"Plots directory exists: {config.PLOTS_DIR.exists()}")
    if not config.PLOTS_DIR.exists():
        print(f"Creating plots directory: {config.PLOTS_DIR}")
        config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to: {output_file}")
    plt.close()


def generate_sharpness_plots(config: ExperimentConfig):
    """Generate batch sharpness and lambda_max comparison plots."""
    if not WANDB_AVAILABLE:
        print("Error: wandb required for sharpness plots")
        return

    print("\n" + "="*60)
    print("GENERATING SHARPNESS PLOTS")
    print("="*60)

    # Load runs from wandb
    collection = RunCollection.from_tag(
        tag=config.WANDB_TAG,
        load_dataframes=True
    )

    if len(collection.runs) == 0:
        print("Error: No runs found with tag:", config.WANDB_TAG)
        return

    print(f"Loaded {len(collection.runs)} runs")

    # Organize runs by learning rate
    runs_by_lr = {lr: [] for lr in config.LEARNING_RATES}
    for run in collection.runs:
        lr = run.lr
        if lr in runs_by_lr:
            runs_by_lr[lr].append(run)

    # Create plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    colors = {'0.4': '#1f77b4', '0.5': '#ff7f0e', '0.6': '#2ca02c'}

    for lr in config.LEARNING_RATES:
        lr_str = str(lr)
        runs = runs_by_lr[lr]

        if not runs:
            continue

        print(f"\nLR={lr}: {len(runs)} runs")

        # Collect batch sharpness data
        bs_data_list = []
        lmax_data_list = []

        for run in runs:
            if run.has_metric('batch_sharpn'):
                bs_data = run.get_metric_data('batch_sharpn')
                if not bs_data.empty:
                    bs_data_list.append(
                        bs_data.set_index('_step')['batch_sharpn'])

            if run.has_metric('lambda_max'):
                lmax_data = run.get_metric_data('lambda_max')
                if not lmax_data.empty:
                    lmax_data_list.append(
                        lmax_data.set_index('_step')['lambda_max'])

        # Plot batch sharpness
        if bs_data_list:
            # Combine all runs
            bs_df = pd.concat(bs_data_list, axis=1)
            bs_mean = bs_df.mean(axis=1)
            bs_std = bs_df.std(axis=1)

            steps = bs_mean.index
            ax1.plot(steps, bs_mean,
                     label=f'LR={lr}', color=colors[lr_str], linewidth=2)
            ax1.fill_between(steps, bs_mean - bs_std, bs_mean + bs_std,
                             alpha=0.2, color=colors[lr_str])

            # Add 2/eta reference line
            ax1.axhline(y=2/lr, color=colors[lr_str],
                        linestyle='--', alpha=0.5, linewidth=1)

        # Plot lambda_max
        if lmax_data_list:
            lmax_df = pd.concat(lmax_data_list, axis=1)
            lmax_mean = lmax_df.mean(axis=1)
            lmax_std = lmax_df.std(axis=1)

            steps = lmax_mean.index
            ax2.plot(steps, lmax_mean,
                     label=f'LR={lr}', color=colors[lr_str], linewidth=2)
            ax2.fill_between(steps, lmax_mean - lmax_std, lmax_mean + lmax_std,
                             alpha=0.2, color=colors[lr_str])

    # Format batch sharpness plot
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Batch Sharpness', fontsize=12)
    ax1.set_title('Batch Sharpness vs Learning Rate', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Format lambda_max plot
    ax2.set_xlabel('Training Steps', fontsize=12)
    ax2.set_ylabel('Lambda Max', fontsize=12)
    ax2.set_title('Lambda Max vs Learning Rate', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_file = config.PLOTS_DIR / "sharpness_comparison.png"
    print(f"Saving sharpness plot to: {output_file}")
    print(f"Plots directory exists: {config.PLOTS_DIR.exists()}")
    if not config.PLOTS_DIR.exists():
        print(f"Creating plots directory: {config.PLOTS_DIR}")
        config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Sharpness plot saved successfully to: {output_file}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Keller Jordan Experiment: Mean Behavior Differentiability"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "collect", "analyze", "all"],
        default="all",
        help="Experiment mode: train, collect logits, analyze, or all"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for logit collection"
    )
    parser.add_argument(
        "--local-mode",
        action="store_true",
        help="Run in local mode without wandb (for testing)"
    )

    args = parser.parse_args()
    config = ExperimentConfig()

    if args.local_mode:
        print("Running in local mode (no wandb)")
        # In local mode, skip training and collect, just test plot generation
        if args.mode in ["analyze", "all"]:
            # Create dummy data for testing
            print("Creating dummy logits for testing...")
            for lr in config.LEARNING_RATES:
                logits_dir = config.get_logits_dir(lr)
                logits_dir.mkdir(exist_ok=True, parents=True)
                
                # Create dummy logits (1000 test samples, 2 classes)
                dummy_logits = np.random.randn(1000, 2)
                np.save(logits_dir / "run_0.npy", dummy_logits)
                print(f"Created dummy logits for LR={lr}")
            
            generate_correlation_plot(config)
            print("Local mode test completed")
    else:
        if args.mode in ["train", "all"]:
            run_all_training(config)

        if args.mode in ["collect", "all"]:
            collect_all_logits(config, device=args.device)

        if args.mode in ["analyze", "all"]:
            generate_correlation_plot(config)
            generate_sharpness_plots(config)

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print(f"Plots saved to: {config.PLOTS_DIR}")


if __name__ == "__main__":
    main()
