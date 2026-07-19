"""Benchmark task runner tests."""

from __future__ import annotations

from src.engine import SimConfig
from src.experiments.benchmark_tasks import list_benchmark_tasks, run_benchmark_task, run_single_task_condition


def test_list_benchmark_tasks_contains_core_tasks():
    tasks = list_benchmark_tasks()

    assert "conflict_emergence" in tasks
    assert "authority_compliance" in tasks


def test_run_benchmark_task_returns_standard_summary():
    result = run_benchmark_task("conflict_emergence", SimConfig(max_rounds=3), seeds=[0])

    assert result.task == "conflict_emergence"
    assert result.n_per_condition == 1
    assert "full" in result.summary
    assert "organization_fragility_index" in result.summary["full"]


def test_run_single_task_condition_reports_metrics():
    row = run_single_task_condition("credit_attribution", "full", seed=0, max_rounds=3)

    assert row["task"] == "credit_attribution"
    assert "credit_attribution_gap" in row["metrics"]
