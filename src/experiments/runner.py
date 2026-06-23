"""Single experiment run orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.engine.run_log import RunLog
from src.engine.simulation import run_simulation
from src.experiments.conditions import ExperimentCondition, build_sim_config, get_condition
from src.experiments.metrics import compute_run_metrics


def run_single(
    experiment_id: str,
    seed: int,
    condition_id: str | None = None,
    *,
    max_rounds: int = 60,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    condition = get_condition(experiment_id, condition_id)
    cfg = build_sim_config(condition, seed, max_rounds=max_rounds, output_dir=str(output_dir) if output_dir else None)
    log = run_simulation(cfg)
    metrics = compute_run_metrics(log)
    return {"log": log, "metrics": metrics, "condition": condition}


def run_experiment_a(seed: int, condition_id: str = "A1", **kwargs: Any) -> RunLog:
    return run_single("A", seed, condition_id, **kwargs)["log"]


def run_experiment_b(seed: int, condition_id: str = "B1", **kwargs: Any) -> RunLog:
    return run_single("B", seed, condition_id, **kwargs)["log"]


def run_experiment_c(seed: int, condition_id: str = "C1", **kwargs: Any) -> RunLog:
    return run_single("C", seed, condition_id, **kwargs)["log"]


def run_experiment_d(seed: int, condition_id: str = "D1", **kwargs: Any) -> RunLog:
    return run_single("D", seed, condition_id, **kwargs)["log"]


def run_experiment_validity(seed: int, condition_id: str = "V1", **kwargs: Any) -> RunLog:
    return run_single("V", seed, condition_id, **kwargs)["log"]
