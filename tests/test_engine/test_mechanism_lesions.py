"""Mechanism lesion tests for experimental-science protocols."""

from __future__ import annotations

from src.engine import SimConfig, run_simulation


def test_status_lesion_is_recorded_and_runs():
    log = run_simulation(SimConfig(max_rounds=3, seed=3, status_lesion=True, llm_provider="scripted", policy_mode="social_physics", enable_llm_action_scoring=False))

    assert log.config["status_lesion"] is True
    assert log.actions
    assert "credit_attribution_gap" in log.outcomes


def test_trust_lesion_is_recorded_and_flattens_trust_outputs():
    log = run_simulation(SimConfig(max_rounds=3, seed=3, trust_lesion=True, llm_provider="scripted", policy_mode="social_physics", enable_llm_action_scoring=False))

    assert log.config["trust_lesion"] is True
    assert log.actions
    assert log.round_records[-1]["metrics"]["trust_phd_a_pi"] == 0.5