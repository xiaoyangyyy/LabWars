"""Social emergence metric tests."""

from __future__ import annotations

from src.engine import SimConfig, run_simulation
from src.experiments.metrics import compute_run_metrics
from src.experiments.social_metrics import compute_social_emergence_metrics


def test_social_emergence_metrics_are_reported():
    log = run_simulation(SimConfig(max_rounds=4, seed=4))
    social = compute_social_emergence_metrics(log)

    assert "trust_entropy" in social
    assert "power_concentration_gini" in social
    assert "organization_fragility_index" in social
    assert social["conflict_cascade_length"] >= 0.0


def test_compute_run_metrics_includes_social_emergence_metrics():
    log = run_simulation(SimConfig(max_rounds=4, seed=5))
    metrics = compute_run_metrics(log)

    assert "social_emergence_metrics" in metrics
    assert "credit_attribution_gap" in metrics["social_emergence_metrics"]
