"""LabWars Part 4 — experiment runners and analysis."""

from .aggregate import aggregate_experiment
from .batch import run_batch
from .conditions import EXPERIMENT_MATRIX, build_sim_config, get_condition
from .report import generate_report
from .runner import run_single

__all__ = [
    "EXPERIMENT_MATRIX",
    "aggregate_experiment",
    "build_sim_config",
    "generate_report",
    "get_condition",
    "run_batch",
    "run_single",
]
