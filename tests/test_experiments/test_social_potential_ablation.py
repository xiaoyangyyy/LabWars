"""Post-hoc social-potential ablation tests."""

from __future__ import annotations

from src.engine import SimConfig, run_simulation
from src.experiments.social_potential_ablation import summarize_social_potential_ablation


def test_social_potential_ablation_summarizes_logged_actions():
    log = run_simulation(SimConfig(max_rounds=3, seed=3))

    summary = summarize_social_potential_ablation(log)

    assert "baseline_mean_pressure" in summary
    assert "lesion_mean_pressure" in summary
    assert "memory_pressure" in summary["lesion_delta"]
