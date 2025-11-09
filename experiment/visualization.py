"""
Visualization functions for the Keller Jordan experiment.
"""

import sys
import os
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

# Add parent directory to path to import from utils
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try relative import first (when used as package), fall back to direct import
try:
    from .config import ExperimentConfig
except ImportError:
    from config import ExperimentConfig

# Check if wandb is available
try:
    from visualization.vis_utils import RunCollection
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Visualization features will be limited.")


def generate_correlation_plot(config: ExperimentConfig, step: int = None):
    """Generate the correlation plot showing linear relationship between LR deltas.

    Args:
        config: Experiment configuration
        step: Checkpoint step to use (defaults to final step)
    """
    if step is None:
        step = config.CHECKPOINT_STEPS[-1]  # Use final checkpoint

    print("\n" + "="*60)
    print(f"GENERATING CORRELATION PLOT (step {step})")
    print("="*60)

    # Load all logits from the specified checkpoint step
    logits_by_lr = {}
    for lr in config.LEARNING_RATES:
        logits_dir = config.get_logits_dir(lr, step)
        logits_list = []

        for run_idx in range(config.N_RUNS_PER_LR):
            logit_file = logits_dir / f"run_{run_idx}.npy"
            if logit_file.exists():
                logits = np.load(logit_file)
                logits_list.append(logits)
            else:
                print(
                    f"Warning: Missing logits for LR={lr}, run={run_idx} at step {step}")

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
    ax.set_title(f'Mean Behavior is Differentiable in Learning Rate (Step {step})\nR² = {r_squared:.4f}, N = {config.N_RUNS_PER_LR}',
                 fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save plot to results directory
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = config.RESULTS_DIR / f"correlation_plot_step_{step}.png"
    print(f"Saving plot to: {output_file}")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to: {output_file}")
    plt.close()


def load_metrics_from_results_files(config: ExperimentConfig):
    """Load metrics from results.txt files in RESULTS directory.
    
    Returns:
        Dict mapping learning rate to list of DataFrames with metrics
    """
    
    results_dir = Path(os.environ.get("RESULTS", "."))
    plaintext_dir = results_dir / "plaintext" / f"{config.DATASET}_{config.MODEL}"
    
    if not plaintext_dir.exists():
        print(f"Warning: Results directory not found: {plaintext_dir}")
        return {}
    
    metrics_by_lr = {lr: {'batch_sharpness': [], 'lambda_max': []} for lr in config.LEARNING_RATES}
    
    # Find all run directories
    for run_dir in plaintext_dir.iterdir():
        if not run_dir.is_dir():
            continue
            
        # Extract learning rate from directory name (format: timestamp_lr{lr:.5f}_b{batch})
        # Example: 20250101_120000_lr0.40000_b256
        match = re.search(r'_lr([\d.]+)_b(\d+)', run_dir.name)
        if not match:
            continue
            
        try:
            lr = float(match.group(1))
            batch = int(match.group(2))
        except (ValueError, IndexError):
            continue
        
        # Only process runs with matching learning rate and batch size
        # Map to exact learning rate from config (handles floating point precision)
        matched_lr = None
        for target_lr in config.LEARNING_RATES:
            if abs(lr - target_lr) < 1e-5:
                matched_lr = target_lr
                break
        
        if matched_lr is None or batch != config.BATCH_SIZE:
            continue
        
        results_file = run_dir / "results.txt"
        if not results_file.exists():
            continue
        
        try:
            # Parse results.txt file
            # Format: epoch, step, batch_loss, full_loss, lambda_max, step_sharpness, batch_sharpness, gni, accuracy
            data = []
            with open(results_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse line (comma-separated values)
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) < 9:
                        continue
                    
                    try:
                        epoch = int(parts[0])
                        step = int(parts[1])
                        batch_loss = float(parts[2])
                        full_loss = float(parts[3]) if parts[3] != 'nan' else np.nan
                        lmax = float(parts[4]) if parts[4] != 'nan' else np.nan
                        step_sharpness = float(parts[5]) if parts[5] != 'nan' else np.nan
                        batch_sharpness = float(parts[6]) if parts[6] != 'nan' else np.nan
                        gni = float(parts[7]) if parts[7] != 'nan' else np.nan
                        accuracy = float(parts[8]) if parts[8] != 'nan' else np.nan
                        
                        data.append({
                            'step': step,
                            'epoch': epoch,
                            'batch_loss': batch_loss,
                            'full_loss': full_loss,
                            'lambda_max': lmax,
                            'step_sharpness': step_sharpness,
                            'batch_sharpness': batch_sharpness,
                            'gni': gni,
                            'accuracy': accuracy
                        })
                    except (ValueError, IndexError):
                        continue
            
            if data:
                df = pd.DataFrame(data)
                # Filter out NaN values for metrics we care about
                if 'batch_sharpness' in df.columns:
                    bs_df = df[['step', 'batch_sharpness']].dropna(subset=['batch_sharpness'])
                    if not bs_df.empty:
                        metrics_by_lr[matched_lr]['batch_sharpness'].append(bs_df.set_index('step')['batch_sharpness'])
                
                if 'lambda_max' in df.columns:
                    lmax_df = df[['step', 'lambda_max']].dropna(subset=['lambda_max'])
                    if not lmax_df.empty:
                        metrics_by_lr[matched_lr]['lambda_max'].append(lmax_df.set_index('step')['lambda_max'])
                
                print(f"Loaded metrics from {run_dir.name} (LR={matched_lr}): {len(data)} data points, {len([d for d in data if not np.isnan(d['batch_sharpness'])])} batch_sharpness, {len([d for d in data if not np.isnan(d['lambda_max'])])} lambda_max")
        
        except Exception as e:
            print(f"Error loading {results_file}: {e}")
            continue
    
    return metrics_by_lr


def generate_sharpness_plots(config: ExperimentConfig):
    """Generate batch sharpness and lambda_max comparison plots."""
    print("\n" + "="*60)
    print("GENERATING SHARPNESS PLOTS")
    print("="*60)

    # Try to load from wandb first, fall back to results.txt files
    metrics_by_lr = None
    
    if WANDB_AVAILABLE:
        try:
            print("Attempting to load metrics from wandb...")
            collection = RunCollection.from_tag(
                tag=config.WANDB_TAG,
                load_dataframes=True
            )
            
            if len(collection.runs) > 0:
                print(f"Loaded {len(collection.runs)} runs from wandb")
                metrics_by_lr = {lr: {'batch_sharpness': [], 'lambda_max': []} for lr in config.LEARNING_RATES}
                
                # Organize runs by learning rate
                runs_by_lr = {lr: [] for lr in config.LEARNING_RATES}
                for run in collection.runs:
                    lr = run.lr
                    if lr in runs_by_lr:
                        runs_by_lr[lr].append(run)
                
                # Extract metrics from wandb runs
                for lr in config.LEARNING_RATES:
                    runs = runs_by_lr[lr]
                    for run in runs:
                        if run.has_metric('batch_sharpn'):
                            bs_data = run.get_metric_data('batch_sharpn')
                            if not bs_data.empty:
                                metrics_by_lr[lr]['batch_sharpness'].append(
                                    bs_data.set_index('_step')['batch_sharpn'])
                        
                        if run.has_metric('lambda_max'):
                            lmax_data = run.get_metric_data('lambda_max')
                            if not lmax_data.empty:
                                metrics_by_lr[lr]['lambda_max'].append(
                                    lmax_data.set_index('_step')['lambda_max'])
        
        except Exception as e:
            print(f"Error loading from wandb: {e}")
            print("Falling back to results.txt files...")
            metrics_by_lr = None
    
    # Fall back to loading from results.txt files
    if metrics_by_lr is None:
        print("Loading metrics from results.txt files...")
        metrics_by_lr = load_metrics_from_results_files(config)
    
    # Initialize empty dict if None or empty
    if not metrics_by_lr:
        metrics_by_lr = {lr: {'batch_sharpness': [], 'lambda_max': []} for lr in config.LEARNING_RATES}
    
    # Check if we have any data
    has_data = False
    for lr in config.LEARNING_RATES:
        bs_list = metrics_by_lr.get(lr, {}).get('batch_sharpness', [])
        lmax_list = metrics_by_lr.get(lr, {}).get('lambda_max', [])
        if bs_list or lmax_list:
            has_data = True
            break
    
    if not has_data:
        print("Error: No metric data found. Make sure training has completed and results.txt files exist.")
        results_dir = Path(os.environ.get("RESULTS", "."))
        plaintext_dir = results_dir / "plaintext" / f"{config.DATASET}_{config.MODEL}"
        print(f"Looking in: {plaintext_dir}")
        print("Expected format: RESULTS/plaintext/{dataset}_{model}/timestamp_lr{lr}_b{batch}/results.txt")
        return
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    colors = {'0.4': '#1f77b4', '0.5': '#ff7f0e', '0.6': '#2ca02c'}
    
    for lr in config.LEARNING_RATES:
        lr_str = str(lr)
        bs_data_list = metrics_by_lr.get(lr, {}).get('batch_sharpness', [])
        lmax_data_list = metrics_by_lr.get(lr, {}).get('lambda_max', [])
        
        print(f"\nLR={lr}: {len(bs_data_list)} batch_sharpness runs, {len(lmax_data_list)} lambda_max runs")
        
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
    
    # Check if we have any data to plot
    has_bs_data = any(metrics_by_lr.get(lr, {}).get('batch_sharpness', []) for lr in config.LEARNING_RATES)
    has_lmax_data = any(metrics_by_lr.get(lr, {}).get('lambda_max', []) for lr in config.LEARNING_RATES)
    
    if not has_bs_data and not has_lmax_data:
        print("Error: No data to plot. Both batch_sharpness and lambda_max are empty.")
        return
    
    # Format batch sharpness plot
    if has_bs_data:
        ax1.set_xlabel('Training Steps', fontsize=12)
        ax1.set_ylabel('Batch Sharpness', fontsize=12)
        ax1.set_title('Batch Sharpness vs Learning Rate', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')  # Log scale for steps
    else:
        ax1.text(0.5, 0.5, 'No batch sharpness data available', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        ax1.set_title('Batch Sharpness vs Learning Rate (No Data)', fontsize=14)
    
    # Format lambda_max plot
    if has_lmax_data:
        ax2.set_xlabel('Training Steps', fontsize=12)
        ax2.set_ylabel('Lambda Max', fontsize=12)
        ax2.set_title('Lambda Max vs Learning Rate', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')  # Log scale for steps
    else:
        ax2.text(0.5, 0.5, 'No lambda_max data available', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Lambda Max vs Learning Rate (No Data)', fontsize=14)
    
    plt.tight_layout()
    
    # Save plot to results directory
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = config.RESULTS_DIR / "sharpness_comparison.png"
    print(f"\nSaving sharpness plot to: {output_file}")
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Sharpness plot saved successfully to: {output_file}")
    plt.close()
