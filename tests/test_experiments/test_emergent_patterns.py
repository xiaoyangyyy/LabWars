"""Emergent-pattern metric tests."""

from __future__ import annotations

from src.engine import SimConfig, run_simulation
from src.experiments.social_metrics import compute_social_emergence_metrics


def test_emergent_pattern_metrics_are_reported():
    log = run_simulation(SimConfig(max_rounds=5, seed=5, population_size=30, llm_provider="scripted", policy_mode="social_physics", enable_llm_action_scoring=False))
    metrics = compute_social_emergence_metrics(log)

    assert "power_law_alpha" in metrics
    assert "network_modularity_q" in metrics
    assert "cascade_tail_alpha" in metrics
    assert "emergent_pattern_score" in metrics