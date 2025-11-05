"""
Keller Jordan Experiment: Mean Behavior is Differentiable in the Learning Rate

This script reproduces the experiment showing that mean neural network behavior
is differentiable with respect to learning rate, while also tracking batch
sharpness and lambda_max during training.

Experiment Configuration:
- 100 networks trained on 10k CIFAR-10 samples for 100k steps
- 3 learning rates: [0.4, 0.5, 0.6]
- Batch sharpness and λ_max tracked at each step
- Checkpoints saved at: 0, 100, 1k, 5k, 10k, 50k, 100k steps
- Logits collected at each checkpoint (shape: 100 runs × 10000 test samples × 10 classes)

Usage:
    python experiment/main.py --mode train      # Run training (100 networks × 3 LRs)
    python experiment/main.py --mode collect    # Collect logits from checkpoints
    python experiment/main.py --mode aggregate  # Aggregate logits into single arrays
    python experiment/main.py --mode analyze    # Generate plots
    python experiment/main.py --mode all        # Run complete pipeline
"""

import argparse
import numpy as np
import torch

from config import ExperimentConfig
from training import run_all_training
from logit_collection import collect_all_logits, save_aggregated_logits
from visualization import generate_correlation_plot, generate_sharpness_plots


def main():
    """Main entry point for the Keller Jordan experiment."""
    parser = argparse.ArgumentParser(
        description="Keller Jordan Experiment: Mean Behavior Differentiability"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "collect", "aggregate", "analyze", "all"],
        default="all",
        help="Experiment mode: train, collect logits, aggregate, analyze, or all"
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

        if args.mode in ["aggregate", "all"]:
            save_aggregated_logits(config)

        if args.mode in ["analyze", "all"]:
            generate_correlation_plot(config)
            generate_sharpness_plots(config)

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print(f"Plots saved to: {config.PLOTS_DIR}")


if __name__ == "__main__":
    main()
