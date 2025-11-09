"""
Configuration for the Keller Jordan experiment.
"""

import os
from pathlib import Path


class ExperimentConfig:
    """Configuration for the Keller Jordan experiment."""

    # Learning rates to test
    LEARNING_RATES = [0.4, 0.5, 0.6]

    # Number of runs per learning rate (100 for full experiment)
    N_RUNS_PER_LR = 2

    # Training hyperparameters
    DATASET = "cifar10"
    MODEL = "cnn"
    BATCH_SIZE = 128  # Reduced from 256 to reduce memory usage for eigenvalue computation
    NUM_STEPS = 100000  # 100k steps for full experiment
    NUM_DATA = 10000    # 10k CIFAR-10 samples
    CLASSES = [1, 10]   # Binary classification (classes 1 and 10)

    # Checkpoint schedule: initialization and key milestones
    CHECKPOINT_STEPS = [0, 100, 1000, 5000, 10000, 50000, 100000]

    # Experiment tracking
    WANDB_TAG = "keller-jordan-exp"

    # Paths
    EXPERIMENT_DIR = Path(__file__).parent
    DATA_DIR = EXPERIMENT_DIR / "data"
    PLOTS_DIR = EXPERIMENT_DIR / "plots"
    RESULTS_DIR = Path(os.environ.get("RESULTS", "."))
    CHECKPOINT_DIR = Path(os.environ.get(
        "WANDB_DIR", ".")) / "wandb_checkpoints"

    @classmethod
    def get_logits_dir(cls, lr: float, step: int = None) -> Path:
        """Get directory for storing logits for a given learning rate and step."""
        lr_str = f"lr{int(lr*10):02d}"
        if step is None:
            return cls.DATA_DIR / f"logits_{lr_str}"
        else:
            return cls.DATA_DIR / f"logits_{lr_str}" / f"step_{step}"

    @classmethod
    def setup_directories(cls):
        """Create necessary directories."""
        cls.DATA_DIR.mkdir(exist_ok=True, parents=True)
        cls.PLOTS_DIR.mkdir(exist_ok=True, parents=True)
        # Create directories for logits at each checkpoint step
        for lr in cls.LEARNING_RATES:
            for step in cls.CHECKPOINT_STEPS:
                cls.get_logits_dir(lr, step).mkdir(exist_ok=True, parents=True)
