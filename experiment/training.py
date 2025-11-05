"""
Training functions for the Keller Jordan experiment.
"""

import sys
import json
import subprocess
from pathlib import Path
from typing import Dict

# Try relative import first (when used as package), fall back to direct import
try:
    from .config import ExperimentConfig
except ImportError:
    from config import ExperimentConfig


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

    # Calculate checkpoint every 100 steps for saving logits
    checkpoint_every = 100

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
        "--batch-sharpness",  # Track batch sharpness at each step
        "--lambdamax",        # Track λ_max through time
        "--wandb-tag", config.WANDB_TAG,
        "--wandb-name", f"exp_lr{lr}_run{run_idx}",
        "--checkpoint-every", str(checkpoint_every),  # Save every 100 steps
    ]

    print(f"\n{'='*60}")
    print(f"Training: LR={lr}, Run={run_idx}/{config.N_RUNS_PER_LR-1}")
    print(f"Seed: {seed}")
    print(f"{'='*60}")

    # Run training (show output to see checkpoint messages)
    result = subprocess.run(cmd, cwd=config.EXPERIMENT_DIR.parent)

    if result.returncode != 0:
        print(f"Warning: Training failed with return code {result.returncode}")
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
