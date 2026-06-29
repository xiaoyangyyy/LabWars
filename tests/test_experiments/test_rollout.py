"""Tests for stochastic rollout utilities."""

from __future__ import annotations

from src.engine.simulation import SimConfig, run_simulation
from src.experiments.rollout import run_state_event_counterfactual, run_stochastic_rollouts


def test_anchor_only_counterfactual_disables_state_events():
    log = run_simulation(SimConfig(max_rounds=5, seed=1, interventions=[], disable_state_events=True))
    generators = [e.get("payload", {}).get("generator") for e in log.events]
    assert "state_event_field" not in generators
    assert "anchor_only" in generators


def test_rollout_summary_runs_multiple_seeds():
    summary = run_stochastic_rollouts(
        SimConfig(max_rounds=6, interventions=[]),
        seeds=[0, 1],
        outcomes=["authorship_dispute_index"],
    )
    assert summary["n"] == 2
    assert len(summary["rows"]) == 2
    assert "authorship_dispute_index" in summary["summary"]
    assert summary["state_generated_event_count"]["mean"] >= 0.0


def test_state_event_counterfactual_returns_delta_summary():
    result = run_state_event_counterfactual(
        SimConfig(max_rounds=6, interventions=[]),
        seeds=[0, 1],
        outcome="authorship_dispute_index",
    )
    assert result["n"] == 2
    assert "delta_summary" in result
    assert len(result["per_seed"]) == 2
