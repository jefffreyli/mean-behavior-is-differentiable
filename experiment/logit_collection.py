"""
Logit collection functions for the Keller Jordan experiment.
"""

import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import os
import sys
import json
from pathlib import Path
from typing import Optional

# Add parent directory to path before importing utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data import prepare_dataset, get_dataset_presets
from utils.nets import prepare_net, get_model_presets
from utils.naming import compose_run_name


# Try relative import first (when used as package), fall back to direct import
try:
    from .config import ExperimentConfig
except ImportError:
    from config import ExperimentConfig


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
                           X_test: torch.Tensor, step: int, device: str = "cpu") -> Optional[np.ndarray]:
    """
    Collect logits for a single run at a specific checkpoint step.

    Args:
        lr: Learning rate
        run_idx: Run index
        config: Experiment configuration
        X_test: Test data
        step: Checkpoint step to load
        device: Device to use

    Returns:
        Logits array of shape (n_test, n_classes) or None if collection fails
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
    expected_wandb_name = f"exp_lr{lr}_run{run_idx}"

    print(f"Searching for run: {run_name} (wandb_name: {expected_wandb_name})")

    results_dir = Path(os.environ.get("RESULTS", "."))
    checkpoint_base = results_dir / "wandb_checkpoints"
    wandb_runs_dir = results_dir / "wandb"

    print(f"Looking in checkpoint directory: {checkpoint_base}")
    print(f"Searching wandb runs directory: {wandb_runs_dir}")

    if not checkpoint_base.exists():
        print(f"Checkpoint directory does not exist: {checkpoint_base}")
        return None

    # Strategy 1: Match by wandb run name (most reliable)
    matched_run_id = None
    
    if wandb_runs_dir.exists():
        # Search through all offline-run directories
        offline_run_dirs = [d for d in wandb_runs_dir.iterdir() 
                           if d.is_dir() and d.name.startswith("offline-run-")]
        
        print(f"Found {len(offline_run_dirs)} wandb offline-run directories")
        
        for offline_run_dir in offline_run_dirs:
            # Try to read config.json first (preferred)
            config_json = offline_run_dir / "files" / "config.json"
            config_yaml = offline_run_dir / "files" / "config.yaml"
            
            wandb_name = None
            
            # Read config.json
            if config_json.exists():
                try:
                    with open(config_json, 'r') as f:
                        config_data = json.load(f)
                        # Wandb stores config values as {"value": actual_value, ...}
                        wandb_name_obj = config_data.get('wandb_name', {})
                        if isinstance(wandb_name_obj, dict) and 'value' in wandb_name_obj:
                            wandb_name = wandb_name_obj['value']
                        elif isinstance(wandb_name_obj, str):
                            wandb_name = wandb_name_obj
                except Exception as e:
                    print(f"  Warning: Could not read config.json from {offline_run_dir.name}: {e}")
            
            # Fallback to config.yaml
            if wandb_name is None and config_yaml.exists():
                try:
                    import yaml
                    with open(config_yaml, 'r') as f:
                        config_data = yaml.safe_load(f)
                        wandb_name_obj = config_data.get('wandb_name', {})
                        if isinstance(wandb_name_obj, dict) and 'value' in wandb_name_obj:
                            wandb_name = wandb_name_obj['value']
                        elif isinstance(wandb_name_obj, str):
                            wandb_name = wandb_name_obj
                except Exception as e:
                    print(f"  Warning: Could not read config.yaml from {offline_run_dir.name}: {e}")
            
            # Check if this run matches what we're looking for
            if wandb_name == expected_wandb_name:
                # Extract run_id from directory name: offline-run-{timestamp}-{run_id}
                # The run_id is the part after the last dash
                run_id = offline_run_dir.name.split('-')[-1]
                
                # Verify checkpoint directory exists
                checkpoint_dir = checkpoint_base / run_id
                if checkpoint_dir.exists():
                    matched_run_id = run_id
                    print(f"  ✓ Matched by wandb_name '{wandb_name}' to run_id: {run_id}")
                    print(f"  Checkpoint dir: {checkpoint_dir}")
                    break
                else:
                    print(f"  ⚠ Found matching wandb run '{wandb_name}' but checkpoint directory missing: {checkpoint_dir}")
    
    # Strategy 2: Fallback to position-based matching (only if name-based failed)
    if matched_run_id is None:
        print(f"  Name-based matching failed, falling back to position-based matching...")
        
        try:
            run_dirs = [d for d in checkpoint_base.iterdir() if d.is_dir()]
            print(f"Found {len(run_dirs)} checkpoint directories")
            
            # Filter to only directories with valid checkpoints
            valid_run_dirs = []
            for d in run_dirs:
                metadata_file = d / "checkpoint_metadata.json"
                if metadata_file.exists():
                    valid_run_dirs.append(d)
            
            # Sort by modification time (most recent first)
            valid_run_dirs_sorted = sorted(
                valid_run_dirs, key=lambda d: d.stat().st_mtime, reverse=True)
            
            # Calculate position based on successful runs only
            learning_rates = config.LEARNING_RATES
            runs_per_lr = config.N_RUNS_PER_LR
            
            try:
                lr_index = learning_rates.index(lr)
            except ValueError:
                print(f"LR {lr} not found in config learning rates: {learning_rates}")
                return None
            
            overall_run_num = lr_index * runs_per_lr + run_idx
            
            print(f"Looking for LR={lr}, run_idx={run_idx}")
            print(f"  Overall run number: {overall_run_num} out of {len(learning_rates) * runs_per_lr}")
            print(f"  Valid checkpoint directories: {len(valid_run_dirs_sorted)}")
            
            # Count successful runs before this one
            # This is approximate - we assume runs complete in order
            if overall_run_num < len(valid_run_dirs_sorted):
                matched_dir = valid_run_dirs_sorted[overall_run_num]
                matched_run_id = matched_dir.name
                print(f"  Matched by position (index {overall_run_num}) to run_id: {matched_run_id}")
            else:
                print(f"  Position-based matching also failed (need index {overall_run_num}, have {len(valid_run_dirs_sorted)})")
                return None
        
        except Exception as e:
            print(f"Error in position-based matching: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # Load logits from matched checkpoint directory
    if matched_run_id:
        checkpoint_dir = checkpoint_base / matched_run_id
        metadata_file = checkpoint_dir / "checkpoint_metadata.json"
        
        if metadata_file.exists():
            return _load_logits_from_checkpoint_dir(checkpoint_dir, config, X_test, step, device)
        else:
            print(f"  Checkpoint directory {checkpoint_dir} exists but has no metadata.json")
            return None
    
    return None


def _load_logits_from_checkpoint_dir(checkpoint_dir: Path, config: ExperimentConfig,
                                     X_test: torch.Tensor, step: int, device: str) -> Optional[np.ndarray]:
    """Helper function to load logits from a checkpoint directory at a specific step."""
    # Find the checkpoint at the specified step
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

    # Find the checkpoint at the specified step
    target_checkpoint = None
    for ckpt in checkpoints:
        if ckpt['step'] == step:
            target_checkpoint = ckpt
            break

    if target_checkpoint is None:
        print(
            f"Warning: No checkpoint found at step {step} in {checkpoint_dir}")
        return None

    checkpoint_path = Path(target_checkpoint['path'])
    print(f"Loading checkpoint from {checkpoint_path} (step {step})")

    # Load model and extract logits
    model = load_model_from_checkpoint(checkpoint_path, config, device)
    logits = extract_logits(model, X_test, device=device)

    return logits


def collect_all_logits(config: ExperimentConfig, device: str = "cpu"):
    """Collect logits from all trained models at all checkpoint steps."""
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
    print(f"Number of checkpoints: {len(config.CHECKPOINT_STEPS)}")
    print(f"Checkpoint steps: {config.CHECKPOINT_STEPS}")

    # Iterate through all checkpoint steps
    for step in config.CHECKPOINT_STEPS:
        print(f"\n{'='*60}")
        print(f"CHECKPOINT STEP: {step}")
        print(f"{'='*60}")

        for lr in config.LEARNING_RATES:
            print(f"\n  Processing LR={lr} at step {step}")
            logits_dir = config.get_logits_dir(lr, step)

            for run_idx in range(config.N_RUNS_PER_LR):
                output_file = logits_dir / f"run_{run_idx}.npy"

                if output_file.exists():
                    print(f"    Run {run_idx}: Already exists, skipping")
                    continue

                print(f"    Run {run_idx}: Collecting logits...")
                logits = collect_logits_for_run(
                    lr, run_idx, config, X_test, step, device)

                if logits is not None:
                    np.save(output_file, logits)
                    print(f"    Run {run_idx}: Saved to {output_file}")
                else:
                    print(f"    Run {run_idx}: Failed to collect logits")

    print("\n" + "="*60)
    print("LOGIT COLLECTION COMPLETE")
    print("="*60)
    print(f"Total checkpoints processed: {len(config.CHECKPOINT_STEPS)}")
    print(
        f"Total runs per checkpoint: {len(config.LEARNING_RATES) * config.N_RUNS_PER_LR}")


def aggregate_logits_for_step(config: ExperimentConfig, lr: float, step: int) -> Optional[np.ndarray]:
    """
    Load and aggregate all logits for a given learning rate and checkpoint step.

    Args:
        config: Experiment configuration
        lr: Learning rate
        step: Checkpoint step

    Returns:
        Array of shape (n_runs, n_test_samples, n_classes) or None if data missing
    """
    logits_dir = config.get_logits_dir(lr, step)
    logits_list = []

    for run_idx in range(config.N_RUNS_PER_LR):
        logit_file = logits_dir / f"run_{run_idx}.npy"
        if logit_file.exists():
            logits = np.load(logit_file)
            logits_list.append(logits)
        else:
            print(
                f"Warning: Missing logits for LR={lr}, run={run_idx}, step={step}")
            return None

    # Stack to shape (n_runs, n_test, n_classes)
    aggregated = np.stack(logits_list, axis=0)
    print(
        f"Aggregated logits for LR={lr}, step={step}: shape={aggregated.shape}")
    return aggregated


def save_logits_to_dataframe(config: ExperimentConfig):
    """
    Save all logits to a pandas DataFrame for easy viewing.

    Creates a DataFrame with columns: lr, step, run_idx, test_sample_idx, class_idx, logit_value
    Also saves a summary CSV per (lr, step) combination.
    """
    print("\n" + "="*60)
    print("SAVING LOGITS TO DATAFRAME")
    print("="*60)

    all_logits_data = []

    for step in config.CHECKPOINT_STEPS:
        for lr in config.LEARNING_RATES:
            print(f"\nProcessing LR={lr}, step={step}")
            logits_dir = config.get_logits_dir(lr, step)

            for run_idx in range(config.N_RUNS_PER_LR):
                logit_file = logits_dir / f"run_{run_idx}.npy"
                if logit_file.exists():
                    # Shape: (n_test_samples, n_classes)
                    logits = np.load(logit_file)

                    # Flatten and add metadata
                    n_samples, n_classes = logits.shape
                    for sample_idx in range(n_samples):
                        for class_idx in range(n_classes):
                            all_logits_data.append({
                                'lr': lr,
                                'step': step,
                                'run_idx': run_idx,
                                'test_sample_idx': sample_idx,
                                'class_idx': class_idx,
                                'logit_value': logits[sample_idx, class_idx]
                            })
                else:
                    print(f"  Warning: Missing file for run {run_idx}")

    # Create DataFrame
    df = pd.DataFrame(all_logits_data)

    if len(df) == 0:
        print("\nWarning: No logits data found. DataFrame will be empty.")
        return None

    # Ensure results directory exists
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save full DataFrame
    # Try parquet first (more efficient), fallback to CSV if parquet not available
    parquet_file = config.DATA_DIR / "logits_dataframe.parquet"
    csv_file = config.RESULTS_DIR / "logits_dataframe.csv.gz"

    try:
        df.to_parquet(parquet_file, index=False)
        print(f"\nSaved full DataFrame (parquet) to: {parquet_file}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"\nWarning: Could not save as parquet ({e}), using CSV instead")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")

    # Save CSV to results directory (compressed, may be large)
    df.to_csv(csv_file, index=False, compression='gzip')
    print(f"Saved CSV to results directory: {csv_file}")

    # Save summary statistics per (lr, step)
    summary_data = []
    for (lr, step), group in df.groupby(['lr', 'step']):
        summary_data.append({
            'lr': lr,
            'step': step,
            'n_runs': group['run_idx'].nunique(),
            'n_samples': group['test_sample_idx'].nunique(),
            'n_classes': group['class_idx'].nunique(),
            'mean_logit': group['logit_value'].mean(),
            'std_logit': group['logit_value'].std(),
            'min_logit': group['logit_value'].min(),
            'max_logit': group['logit_value'].max()
        })

    summary_df = pd.DataFrame(summary_data)
    summary_file = config.RESULTS_DIR / "logits_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Saved summary to: {summary_file}")

    return df


def save_aggregated_logits(config: ExperimentConfig):
    """
    Create and save aggregated logit arrays for each (LR, step) combination.
    Also saves logits to DataFrame format.

    Creates arrays of shape (100 runs, 10000 test samples, 10 classes) for each
    learning rate and checkpoint step.
    """
    print("\n" + "="*60)
    print("AGGREGATING LOGITS")
    print("="*60)

    for step in config.CHECKPOINT_STEPS:
        for lr in config.LEARNING_RATES:
            print(f"\nAggregating LR={lr}, step={step}")
            aggregated = aggregate_logits_for_step(config, lr, step)

            if aggregated is not None:
                # Save aggregated file
                output_file = config.get_logits_dir(
                    lr, step) / "aggregated.npy"
                np.save(output_file, aggregated)
                print(f"  Saved aggregated logits to: {output_file}")
                print(f"  Shape: {aggregated.shape}")

    print("\n" + "="*60)
    print("AGGREGATION COMPLETE")
    print("="*60)

    # Also save to DataFrame and return summary
    logits_df = save_logits_to_dataframe(config)
    return logits_df
