import wandb
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from collections import OrderedDict
import os
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Project configuration
PROJECT_NAME = os.getenv("WANDB_PROJECT")
ENTITY = None  # Leave None for default entity

USE_PARALLEL_FETCH = True  # Set to False to disable parallel fetching
MAX_WORKERS = 8  # Number of parallel workers for fetching data


api = wandb.Api()


class Run:
    """
    A class to handle individual wandb runs with lazy loading and built-in utilities.
    """

    def __init__(self, run: wandb.Run):
        self.run = run
        self.config = run.config
        self.keys_to_ignore = ['_runtime',
                               '_timestamp', 'power_iteration_iterations']
        self._df = None  # Private DataFrame, loaded on demand

        # Extract metadata from config
        self._metadata = {
            'lr': self.config.get('lr', np.nan),
            'batch_size': self.config.get('batch', np.nan),
            'dataset': self.config.get('dataset', 'unknown'),
            'model': self.config.get('model', 'unknown'),
            'run_id': self.run.id,
            'run_name': self.run.name,
            'created_at': self.run.created_at,
            'dataset_size': self.config.get('num_data', 8192)
        }

    @property
    def metadata(self) -> Dict:
        """Get run metadata."""
        return self._metadata

    @property
    def df(self) -> pd.DataFrame:
        """Get DataFrame, loading it if necessary."""
        if self._df is None:
            self.load_dataframe()
        return self._df

    @property
    def lr(self) -> float:
        """Get learning rate."""
        return self._metadata['lr']

    @property
    def batch_size(self) -> int:
        """Get batch size."""
        return self._metadata['batch_size']

    @property
    def dataset_size(self) -> int:
        """Get dataset size."""
        return self._metadata['dataset_size']

    @property
    def run_id(self) -> str:
        """Get run ID."""
        return self._metadata['run_id']

    @property
    def run_name(self) -> str:
        """Get run name."""
        return self._metadata['run_name']

    def load_dataframe(self, columns: List[str] = None) -> pd.DataFrame:
        """
        Load DataFrame from wandb run history.

        Args:
            columns: List of column names to load (currently ignored due to wandb API limitations)

        Returns:
            DataFrame with run history
        """
        if self._df is not None:
            return self._df

        try:
            # Try to get specific keys first, but fallback to all data
            rows = self.run.scan_history(page_size=10_000)
            df = pd.DataFrame(list(rows))
        except Exception:
            # Fallback in case of any issues
            rows = self.run.scan_history(page_size=10_000)
            df = pd.DataFrame(list(rows))

        if '_step' in df.columns:
            df = df.sort_values("_step")
            # Keep _step as both index and column
            df = df.set_index("_step", drop=False)

        # Remove unwanted columns
        df = df.drop(columns=[
                     col for col in self.keys_to_ignore if col in df.columns], errors='ignore')

        self._df = df
        return self._df

    def get_available_metrics(self) -> List[str]:
        """Get list of available metrics in this run."""
        return list(self.df.columns)

    def has_metric(self, metric: str) -> bool:
        """Check if the run has a specific metric."""
        return metric in self.df.columns and not self.df[metric].isna().all()

    def get_metric_data(self, metric: str) -> pd.DataFrame:
        """Get non-NaN data for a specific metric."""
        if not self.has_metric(metric):
            return pd.DataFrame()
        return self.df[['_step', metric]].dropna()

    def get_steps_per_epoch(self) -> int:
        """Calculate steps per epoch based on dataset size and batch size."""
        return self.dataset_size // self.batch_size

    def get_smoothed_metric(self, metric: str, window_size: Optional[int] = None) -> pd.Series:
        """
        Get smoothed version of a metric.

        Args:
            metric: Metric name
            window_size: Window size for smoothing. If None, uses sqrt(batch_size) * steps_per_epoch

        Returns:
            Smoothed series
        """
        if not self.has_metric(metric):
            return pd.Series()

        if window_size is None:
            steps_per_epoch = self.get_steps_per_epoch()
            window_size = steps_per_epoch * int(np.sqrt(self.batch_size))

        return (self.df[metric]
                .rolling(window=window_size, min_periods=1, center=True)
                .mean())

    def make_quick_plot(self, metrics: List[str] = ['batch_sharpness', 'lambda_max', 'loss']):
        """
        Make a quick plot of specified metrics.

        Args:
            metrics: List of metrics to plot
        """
        num_metrics = len(metrics)
        fig, axes = plt.subplots(num_metrics, 1, figsize=(12, 4*num_metrics))
        if num_metrics == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            if self.has_metric(metric):
                data = self.get_metric_data(metric)
                ax.plot(data['_step'], data[metric], label=metric, alpha=0.5)

                smoothed = self.get_smoothed_metric(metric)
                ax.plot(self.df['_step'], smoothed,
                        label=f"{metric} (smoothed)", color='orange')

                ax.set_title(f'Run: {self.run_name} - {metric}')
                ax.set_xlabel('Steps')
                ax.set_ylabel(metric)
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'Metric "{metric}" not found',
                        horizontalalignment='center', verticalalignment='center')
                ax.axis('off')

        plt.tight_layout()
        plt.show()

    @classmethod
    def from_id(cls, run_id: str, entity: Optional[str] = None, project: str = PROJECT_NAME) -> 'Run':
        """
        Create Run instance from wandb run ID.

        Args:
            run_id: Wandb run ID
            entity: Wandb entity (None for default)
            project: Wandb project name

        Returns:
            Run instance
        """
        if entity:
            run_path = f"{entity}/{project}/{run_id}"
        else:
            run_path = f"{project}/{run_id}"
        run = api.run(run_path)
        return cls(run)

    def __repr__(self) -> str:
        return (f"Run(id='{self.run_id}', name='{self.run_name}', "
                f"lr={self.lr}, batch_size={self.batch_size}, "
                f"dataset='{self.metadata['dataset']}', model='{self.metadata['model']}')")


class RunCollection:
    """
    A collection of Run objects with batch operations and utilities.
    """

    def __init__(self, runs: List[Run] = None):
        self.runs = runs or []

    def __len__(self) -> int:
        return len(self.runs)

    def __getitem__(self, index) -> Run:
        return self.runs[index]

    def __iter__(self):
        return iter(self.runs)

    def add_run(self, run: Run):
        """Add a run to the collection."""
        self.runs.append(run)

    def add_run_by_id(self, run_id: str, entity: Optional[str] = None, project: str = PROJECT_NAME):
        """Add a run by ID to the collection."""
        run = Run.from_id(run_id, entity, project)
        self.add_run(run)

    def load_all_dataframes(self, use_parallel: bool = USE_PARALLEL_FETCH, max_workers: int = MAX_WORKERS):
        """
        Load DataFrames for all runs in the collection.

        Args:
            use_parallel: Whether to use parallel loading
            max_workers: Number of parallel workers
        """
        if not self.runs:
            return

        if use_parallel and len(self.runs) > 1:
            print(
                f"Loading dataframes for {len(self.runs)} runs using {max_workers} workers...")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(
                    run.load_dataframe): run for run in self.runs}

                for future in as_completed(futures):
                    run = futures[future]
                    try:
                        future.result()  # This will raise any exceptions
                    except Exception as e:
                        print(
                            f"Error loading dataframe for run {run.run_id}: {e}")
        else:
            print(
                f"Loading dataframes for {len(self.runs)} runs sequentially...")
            for run in self.runs:
                try:
                    run.load_dataframe()
                except Exception as e:
                    print(f"Error loading dataframe for run {run.run_id}: {e}")

    def sort_by_lr(self, reversed: bool = True):
        """Sort runs by batch size (ascending)."""
        self.runs.sort(key=lambda r: r.lr)

    def sort_by_batch_size(self, reversed: bool = False):
        """Sort runs by batch size (ascending)."""
        self.runs.sort(key=lambda r: r.batch_size, reverse=reversed)

    @classmethod
    def from_run_ids(cls, run_ids: List[str], entity: Optional[str] = None,
                     project: str = PROJECT_NAME, load_dataframes: bool = False,
                     use_parallel: bool = USE_PARALLEL_FETCH, max_workers: int = MAX_WORKERS) -> 'RunCollection':
        """
        Create RunCollection from a list of run IDs.

        Args:
            run_ids: List of wandb run IDs
            entity: Wandb entity (None for default)
            project: Wandb project name
            load_dataframes: Whether to load DataFrames immediately
            use_parallel: Whether to use parallel loading
            max_workers: Number of parallel workers

        Returns:
            RunCollection instance
        """
        runs = []
        for run_id in run_ids:
            try:
                run = Run.from_id(run_id, entity, project)
                runs.append(run)
            except Exception as e:
                print(f"Warning: Could not retrieve run {run_id}: {e}")

        collection = cls(runs)

        if load_dataframes and runs:
            collection.load_all_dataframes(use_parallel, max_workers)

        return collection

    @classmethod
    def from_tag(cls, tag: str, entity: Optional[str] = None,
                 project: str = PROJECT_NAME, load_dataframes: bool = False,
                 use_parallel: bool = USE_PARALLEL_FETCH, max_workers: int = MAX_WORKERS) -> 'RunCollection':
        """
        Create RunCollection from runs with a specific tag.

        Args:
            tag: Tag to filter runs
            entity: Wandb entity (None for default)
            project: Wandb project name
            load_dataframes: Whether to load DataFrames immediately
            use_parallel: Whether to use parallel loading
            max_workers: Number of parallel workers

        Returns:
            RunCollection instance
        """
        if entity:
            project_path = f"{entity}/{project}"
        else:
            project_path = project

        # Filter runs by tag
        filters = {"tags": {"$in": [tag]}}

        try:
            runs = api.runs(project_path, filters=filters)
            runs_list = list(runs)
            print(f"Found {len(runs_list)} runs with tag '{tag}'")

        except Exception as e:
            print(f"Error retrieving runs with tag '{tag}': {e}")
            return []

        collection = cls([Run(run) for run in runs_list])

        if load_dataframes and runs:
            collection.load_all_dataframes(use_parallel, max_workers)

        return collection

    @classmethod
    def get_latest_runs(cls, n_latest: int = 5, filters: Optional[Dict] = None,
                        entity: Optional[str] = None, project: str = PROJECT_NAME) -> List['Run']:
        """
        Get the latest n runs as Run instances (metadata only, no DataFrame loading).

        Args:
            n_latest: Number of latest runs to retrieve
            filters: Dictionary of filters to apply
            entity: Wandb entity (None for default)
            project: Wandb project name

        Returns:
            List of Run instances (DataFrames not loaded yet)
        """
        if entity:
            project_path = f"{entity}/{project}"
        else:
            project_path = project

        wandb_runs = api.runs(
            project_path, filters=filters, order="-created_at")
        wandb_runs = list(wandb_runs)[:n_latest]

        return [Run(run) for run in wandb_runs]

    @classmethod
    def from_latest_runs(cls, n_latest: int = 5, filters: Optional[Dict] = None,
                         entity: Optional[str] = None, project: str = PROJECT_NAME,
                         load_dataframes: bool = True, use_parallel: bool = USE_PARALLEL_FETCH,
                         max_workers: int = MAX_WORKERS) -> 'RunCollection':
        """
        Create RunCollection from the latest runs.

        Args:
            n_latest: Number of latest runs to retrieve
            filters: Dictionary of filters to apply
            entity: Wandb entity (None for default)
            project: Wandb project name
            load_dataframes: Whether to load DataFrames immediately
            use_parallel: Whether to use parallel loading
            max_workers: Number of parallel workers

        Returns:
            RunCollection instance
        """
        runs = RunCollection.get_latest_runs(
            n_latest, filters, entity, project)
        collection = cls(runs)

        if load_dataframes and runs:
            collection.load_all_dataframes(use_parallel, max_workers)

        return collection

    def print_summary(self):
        """Print a summary of all runs in the collection."""
        print(f"RunCollection with {len(self.runs)} runs:")
        print("-" * 80)

        for i, run in enumerate(self.runs):
            print(f"{i+1}. {run.run_name} (ID: {run.run_id})")
            print(
                f"   Dataset: {run.metadata['dataset']}, Model: {run.metadata['model']}")
            print(f"   Batch: {run.batch_size}, LR: {run.lr}")
            print(f"   Created: {run.metadata['created_at']}")

            # Add other metadata if present
            extra_info = []
            if 'momentum' in run.config and run.config['momentum'] is not None:
                extra_info.append(f"Momentum: {run.config['momentum']}")
            if run.config.get('sde', False):
                extra_info.append("SDE: True")
            if run.config.get('gd_noise', False):
                extra_info.append("GD Noise: On")
            if extra_info:
                print(f"   {', '.join(extra_info)}")
            print()

    def compare_metrics(self, metrics: List[str] = ['batch_sharpness', 'lambda_max']):
        """
        Compare specific metrics across all runs in the collection.

        Args:
            metrics: List of metrics to compare
        """
        if not self.runs:
            print("No runs in collection")
            return

        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4*len(metrics)))
        if len(metrics) == 1:
            axes = [axes]

        for i, metric in enumerate(metrics):
            ax = axes[i]

            for run in self.runs:
                if run.has_metric(metric):
                    data = run.get_metric_data(metric)
                    label = f"{run.run_name} (b={run.batch_size}, lr={run.lr})"
                    ax.plot(data['_step'], data[metric],
                            label=label, alpha=0.7)

            ax.set_title(f'Comparison: {metric}')
            ax.set_xlabel('Steps')
            ax.set_ylabel(metric)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
