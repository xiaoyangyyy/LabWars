"""Tests for calibratable action-field configuration."""

from __future__ import annotations

from src.engine.action_selection import generate_action_candidates, load_action_field_config
from src.world.loader import load_events, load_world


def test_action_field_parameters_loaded_from_yaml():
    cfg = load_action_field_config()
    assert cfg["source"].endswith("config\\action_field.yaml") or cfg["source"].endswith("config/action_field.yaml")
    assert "ask_for_authorship" in cfg["motive_weights"]
    assert cfg["action_field"]["baseline_tendency"] == 0.10


def test_candidate_records_parameter_source():
    world = load_world()
    agent = world.agents["phd_a"]
    event = load_events()[2]
    candidates = generate_action_candidates(agent, event, world, None, ["ask_for_authorship", "comply"], seed=5)
    assert candidates
    assert all("action_field.yaml" in c.parameter_source for c in candidates)
