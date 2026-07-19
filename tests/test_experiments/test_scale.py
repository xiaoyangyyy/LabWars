"""Scale benchmark tests."""

from __future__ import annotations

from src.experiments.scale import run_scale_experiment


def test_run_scale_experiment_returns_summary():
    result = run_scale_experiment(population_sizes=[14, 20], rounds=3, seeds=[0], llm_provider="scripted")

    assert result.population_sizes == [14, 20]
    assert "14" in result.summary
    assert "20" in result.summary
    assert "organization_fragility_index" in result.summary["20"]
    assert len(result.per_run) == 2
