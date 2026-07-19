"""Sampling frontier tests."""

from __future__ import annotations

from src.experiments.sampling_frontier import run_sampling_frontier


def test_sampling_frontier_sweeps_k_values():
    result = run_sampling_frontier(population_size=20, rounds=3, seeds=[0], k_values=[0, 4, "full"], llm_provider="scripted")

    assert "0" in result.summary
    assert "4" in result.summary
    assert "full" in result.summary
    assert result.summary["0"]["llm_sampled_action_fraction"] == 0.0
    assert result.summary["4"]["llm_sampled_action_fraction"] > 0.0