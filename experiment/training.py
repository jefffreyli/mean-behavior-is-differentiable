"""
Training functions for the Keller Jordan experiment.
"""

import sys
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Try relative import first (when used as package), fall back to direct import
try:
    from .config import ExperimentConfig
except ImportError:
    from config import ExperimentConfig


def run_single_training(lr: float, run_idx: int, config: ExperimentConfig, gpu_id: int = None) -> Dict:
    """
    Launch a single training run.

    Args:
        lr: Learning rate
        run_idx: Run index (0-9)
        config: Experiment configuration
        gpu_id: GPU ID to use (if None, uses default)

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

    # Set CUDA_VISIBLE_DEVICES to use a specific GPU
    # gpu_id can be either a GPU index (int) or a GPU device ID (str like "0" or "2")
    env = os.environ.copy()
    if gpu_id is not None:
        # Use only the specified GPU by setting CUDA_VISIBLE_DEVICES to that GPU ID
        # This makes the GPU appear as device 0 in the subprocess
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"\n{'='*60}")
        print(
            f"Training: LR={lr}, Run={run_idx}/{config.N_RUNS_PER_LR-1} on GPU {gpu_id}")
    else:
        print(f"\n{'='*60}")
        print(f"Training: LR={lr}, Run={run_idx}/{config.N_RUNS_PER_LR-1}")
    print(f"Seed: {seed}")
    print(f"{'='*60}")

    # Run training (show output to see checkpoint messages)
    result = subprocess.run(cmd, cwd=config.EXPERIMENT_DIR.parent, env=env)

    if result.returncode != 0:
        print(
            f"Warning: Training failed with return code {result.returncode} (LR={lr}, Run={run_idx}, GPU={gpu_id})")
        return {"lr": lr, "run_idx": run_idx, "seed": seed, "gpu_id": gpu_id, "success": False}

    return {"lr": lr, "run_idx": run_idx, "seed": seed, "gpu_id": gpu_id, "success": True}


def _run_training_wrapper(args: Tuple) -> Dict:
    """Wrapper function for multiprocessing."""
    lr, run_idx, config, gpu_id = args
    return run_single_training(lr, run_idx, config, gpu_id)


def run_all_training(config: ExperimentConfig):
    """Run all training experiments with parallelization across multiple GPUs."""
    print("\n" + "="*60)
    print("STARTING KELLER JORDAN EXPERIMENT")
    print("="*60)
    print(f"Learning rates: {config.LEARNING_RATES}")
    print(f"Runs per LR: {config.N_RUNS_PER_LR}")
    total_runs = len(config.LEARNING_RATES) * config.N_RUNS_PER_LR
    print(f"Total runs: {total_runs}")

    # Get number of GPUs available and their device IDs
    # Parse CUDA_VISIBLE_DEVICES to get actual allocated GPU device IDs
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    allocated_gpu_ids = []

    if cuda_visible:
        # Parse CUDA_VISIBLE_DEVICES to get list of GPU device IDs
        # Format is typically "0,1,2,3" or "2,3,4,5" etc.
        allocated_gpu_ids = [gpu.strip()
                             for gpu in cuda_visible.split(",") if gpu.strip()]
        num_gpus = len(allocated_gpu_ids) if allocated_gpu_ids else 1
        print(
            f"Detected {num_gpus} GPU(s) from CUDA_VISIBLE_DEVICES: {cuda_visible}")
        print(f"Allocated GPU device IDs: {allocated_gpu_ids}")
    else:
        # Try to detect GPUs using nvidia-smi as fallback
        try:
            result = subprocess.run(
                ["nvidia-smi", "--list-gpus"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                lines = [line.strip()
                         for line in result.stdout.split("\n") if line.strip()]
                detected_gpus = len(lines)
                if detected_gpus > 0:
                    num_gpus = detected_gpus
                    # Create GPU IDs as strings "0", "1", "2", etc.
                    allocated_gpu_ids = [str(i) for i in range(num_gpus)]
                    print(f"Detected {num_gpus} GPU(s) using nvidia-smi")
                    print(f"Using GPU device IDs: {allocated_gpu_ids}")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            num_gpus = int(os.environ.get("NUM_GPUS", "1"))
            allocated_gpu_ids = [str(i) for i in range(num_gpus)]
            print(f"Could not detect GPUs, using NUM_GPUS={num_gpus}")

    # Use minimum of available GPUs and total runs
    num_workers = min(num_gpus, total_runs)
    print(f"Using {num_workers} GPU(s) for parallel training (out of {num_gpus} available, {total_runs} total runs)")
    print("="*60 + "\n")

    config.setup_directories()

    # Prepare all training tasks
    # Each task gets assigned a GPU device ID from the allocated GPUs
    # We use round-robin assignment across the allocated GPU IDs
    tasks = []
    for lr in config.LEARNING_RATES:
        for run_idx in range(config.N_RUNS_PER_LR):
            # Assign GPU in round-robin fashion using actual GPU device IDs
            gpu_idx = len(tasks) % num_workers
            gpu_device_id = allocated_gpu_ids[gpu_idx] if allocated_gpu_ids else str(
                gpu_idx)
            tasks.append((lr, run_idx, config, gpu_device_id))

    # Run training in parallel
    results = []
    if num_workers > 1:
        print(
            f"Running {total_runs} training runs in parallel on {num_workers} GPU(s)...")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(_run_training_wrapper, task): task
                for task in tasks
            }

            # Collect results as they complete
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                    lr, run_idx, _, gpu_id = task
                    status = "✓" if result.get("success", False) else "✗"
                    print(
                        f"{status} Completed: LR={lr}, Run={run_idx}, GPU={gpu_id}")
                except Exception as exc:
                    lr, run_idx, _, gpu_id = task
                    print(
                        f"✗ Exception for LR={lr}, Run={run_idx}, GPU={gpu_id}: {exc}")
                    results.append({
                        "lr": lr,
                        "run_idx": run_idx,
                        "gpu_id": gpu_id,
                        "success": False,
                        "error": str(exc)
                    })
    else:
        # Sequential execution (single GPU or no parallelization)
        print(f"Running {total_runs} training runs sequentially...")
        for task in tasks:
            lr, run_idx, config, gpu_id = task
            result = run_single_training(lr, run_idx, config, gpu_id)
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

    # Print GPU utilization summary
    if num_workers > 1:
        gpu_usage = {}
        for r in results:
            gpu_id = r.get("gpu_id", "unknown")
            if gpu_id not in gpu_usage:
                gpu_usage[gpu_id] = {"total": 0, "success": 0}
            gpu_usage[gpu_id]["total"] += 1
            if r.get("success", False):
                gpu_usage[gpu_id]["success"] += 1
        print("\nGPU Utilization Summary:")
        for gpu_id in sorted(gpu_usage.keys()):
            usage = gpu_usage[gpu_id]
            print(
                f"  GPU {gpu_id}: {usage['success']}/{usage['total']} runs successful")
