"""Hierarchy lesion simulation tests."""

from __future__ import annotations

from src.engine import SimConfig, run_simulation


def test_hierarchy_lesion_is_recorded_and_runs():
    log = run_simulation(SimConfig(max_rounds=3, seed=1, hierarchy_lesion=True))

    assert log.config["hierarchy_lesion"] is True
    assert log.actions
    assert "career_hostage_index" in log.outcomes
