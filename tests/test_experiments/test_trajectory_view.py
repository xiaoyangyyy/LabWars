"""Tests for trajectory view export."""

from __future__ import annotations

import json

from src.engine.simulation import SimConfig, run_simulation
from src.experiments.trajectory_view import build_trajectory_view, write_trajectory_html, write_trajectory_view


def test_trajectory_view_contains_round_actions_and_event_candidates(tmp_path):
    log = run_simulation(SimConfig(max_rounds=5, seed=3, interventions=[]))
    view = build_trajectory_view(log)
    assert view["rounds"]
    assert "event" in view["rounds"][0]
    assert "actions" in view["rounds"][0]
    assert "career_hostage_index" in view["rounds"][0]["metrics"]

    path = write_trajectory_view(log, tmp_path / "trajectory.json")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["run_id"] == log.run_id


def test_trajectory_html_is_self_contained(tmp_path):
    log = run_simulation(SimConfig(max_rounds=3, seed=4, interventions=[]))
    path = write_trajectory_html(log, tmp_path / "trajectory.html")
    text = path.read_text(encoding="utf-8")
    assert "LabWars Trajectory" in text
    assert "const data =" in text
    assert log.run_id in text
