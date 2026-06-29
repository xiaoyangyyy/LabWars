"""Tests for action-field ablation and runtime overrides."""

from __future__ import annotations

from src.engine.action_selection import (
    clear_action_field_override,
    generate_action_candidates,
    set_action_field_override,
)
from src.engine.simulation import SimConfig
from src.experiments.action_field_ablation import run_action_field_ablation
from src.world.loader import load_events, load_world


def test_runtime_override_changes_parameter_source():
    world = load_world()
    agent = world.agents["phd_a"]
    event = load_events()[2]
    set_action_field_override({"motive_weights": {"ask_for_authorship": {"authorship_anxiety": 0.9}}})
    try:
        candidates = generate_action_candidates(agent, event, world, None, ["ask_for_authorship", "comply"], seed=5)
    finally:
        clear_action_field_override()
    assert candidates
    assert all(c.parameter_source == "runtime_override" for c in candidates)


def test_action_field_ablation_runs():
    result = run_action_field_ablation(
        SimConfig(max_rounds=5, interventions=[]),
        {"motive_weights": {"confront": {"resentment_drive": 0.0}}},
        seeds=[0, 1],
        outcome="authorship_dispute_index",
    )
    assert result.n == 2
    assert len(result.per_seed) == 2
    assert result.outcome == "authorship_dispute_index"
