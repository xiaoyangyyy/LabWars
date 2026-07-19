"""Repeatable experimental protocol tests."""

from __future__ import annotations

from src.experiments.scientific_protocol import run_scientific_protocol


def test_scientific_protocol_runs_conditions_and_sizes():
    result = run_scientific_protocol(
        population_sizes=[10, 20],
        rounds=3,
        seeds=[0],
        conditions=["baseline", "no_memory", "no_status", "no_trust"],
        llm_provider="scripted",
    )

    assert result.rounds == 3
    assert "N10:baseline" in result.summary
    assert "N20:no_trust" in result.summary
    assert "power_concentration_gini" in result.summary["N20:no_trust"]
    assert len(result.per_run) == 8