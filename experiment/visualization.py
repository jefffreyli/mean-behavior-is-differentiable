"""
Visualization functions for the Keller Jordan experiment.
"""

import sys
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

    # Save plot
    output_file = config.PLOTS_DIR / f"correlation_plot_step_{step}.png"
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
