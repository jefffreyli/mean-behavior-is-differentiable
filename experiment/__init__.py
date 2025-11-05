"""
Keller Jordan Experiment: Mean Behavior is Differentiable in the Learning Rate

This package contains the experiment code for reproducing the experiment showing
that mean neural network behavior is differentiable with respect to learning rate.
"""

from .config import ExperimentConfig
from .training import run_single_training, run_all_training
from .logit_collection import (
    load_model_from_checkpoint,
    extract_logits,
    collect_logits_for_run,
    collect_all_logits,
    aggregate_logits_for_step,
    save_aggregated_logits
)
from .visualization import (
    generate_correlation_plot,
    generate_sharpness_plots
)

__all__ = [
    'ExperimentConfig',
    'run_single_training',
    'run_all_training',
    'load_model_from_checkpoint',
    'extract_logits',
    'collect_logits_for_run',
    'collect_all_logits',
    'aggregate_logits_for_step',
    'save_aggregated_logits',
    'generate_correlation_plot',
    'generate_sharpness_plots',
]
