"""Tests for LLM candidate-scoring mix ablation."""

from __future__ import annotations

from src.engine.simulation import SimConfig, run_simulation
from src.experiments.llm_mix_ablation import run_dual_engine_ablation, run_llm_mix_ablation
from src.experiments.metrics import compute_run_metrics


def test_dual_engine_ablation_runs_and_reports_outcomes():
    result = run_dual_engine_ablation(
        SimConfig(max_rounds=4, interventions=[]),
        lambda_values=[0.0, 0.35],
        seeds=[0, 1],
        outcomes=["authorship_dispute_index", "llm_override_pressure", "integrity_risk"],
    )
    data = result.to_dict()
    assert result.n_per_lambda == 2
    assert len(result.per_seed) == 4
    assert "lambda_0.00" in result.summary
    assert "lambda_0.35" in result.summary
    assert "llm_override_pressure" in data["summary"]["lambda_0.35"]


def test_metrics_include_llm_scoring_influence():
    log = run_simulation(SimConfig(max_rounds=4, seed=3, interventions=[]))
    metrics = compute_run_metrics(log)
    influence = metrics["llm_scoring_influence"]
    assert "mean_override_pressure" in influence
    assert "examples" in influence


def test_legacy_mix_wrapper_keeps_old_api():
    result = run_llm_mix_ablation(
        SimConfig(max_rounds=3, interventions=[]),
        mix_values=[0.0],
        seeds=[0],
        outcomes=["llm_override_pressure"],
    )
    assert result.mix_values == [0.0]
    assert result.n_per_mix == 1
    assert "lambda_0.00" in result.summary
