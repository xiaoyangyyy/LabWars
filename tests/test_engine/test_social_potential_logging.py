"""Simulation logging for Social Potential Field."""

from __future__ import annotations

from src.engine import SimConfig, run_simulation


def test_simulation_records_social_potential_on_actions():
    log = run_simulation(SimConfig(max_rounds=3, seed=2))

    assert log.actions
    action = log.actions[0]
    assert "social_potential" in action
    assert "selected_social_pressure" in action
    assert "selected_social_pressure_decomposition" in action
    assert "social_potential_ablation" in action
    assert "selected_social_pressure_mean" in log.outcomes
