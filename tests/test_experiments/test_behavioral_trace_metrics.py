"""Tests for behavioral trajectory metrics."""

from __future__ import annotations

from src.engine.simulation import SimConfig, run_simulation
from src.experiments.metrics import compute_run_metrics


def test_behavioral_trace_metrics_present_and_bounded():
    log = run_simulation(SimConfig(max_rounds=8, seed=9, interventions=[]))
    metrics = compute_run_metrics(log)
    trace = metrics["behavioral_trace_metrics"]

    assert 0.0 <= trace["action_entropy"] <= 1.0
    assert 0.0 <= trace["mean_motive_diversity"] <= 1.0
    assert trace["mean_candidate_count"] > 0.0
    assert 0.0 <= trace["state_generated_event_fraction"] <= 1.0
    assert trace["delayed_reaction_lag"] >= 0.0
