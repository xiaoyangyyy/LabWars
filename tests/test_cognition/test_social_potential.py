"""Tests for Social Potential Field."""

from __future__ import annotations

from src.cognition.social_potential import (
    SOCIAL_POTENTIAL_DIMENSIONS,
    compute_social_potential,
    summarize_action_social_potential,
)
from src.engine.event_agent import EventAgent
from src.world.loader import load_world


def test_social_potential_has_expected_dimensions():
    world = load_world()
    agent = world.agents["phd_a"]
    event = EventAgent().generate(52, world)

    field = compute_social_potential(world, agent, event)

    assert set(field.dimensions) == set(SOCIAL_POTENTIAL_DIMENSIONS)
    assert all(0.0 <= v <= 1.0 for v in field.dimensions.values())
    assert 0.0 <= field.total_pressure <= 1.0
    assert field.evidence["relationship_target"]


def test_social_potential_lesion_removes_dimension_and_changes_pressure():
    world = load_world()
    agent = world.agents["phd_a"]
    agent.beliefs.my_first_author_probability = 0.05
    agent.beliefs.my_contribution_recognized = 0.10
    event = EventAgent().generate(52, world)

    field = compute_social_potential(world, agent, event)
    lesioned = compute_social_potential(world, agent, event, lesions=["contribution_entitlement"])

    assert field.dimensions["contribution_entitlement"] > 0.0
    assert lesioned.dimensions["contribution_entitlement"] == 0.0
    assert lesioned.pressure_for_action("ask_for_authorship") <= field.pressure_for_action("ask_for_authorship")


def test_social_potential_summary_reads_action_logs():
    actions = [
        {
            "selected_social_pressure": 0.6,
            "social_potential": {"dimensions": {dim: 0.5 for dim in SOCIAL_POTENTIAL_DIMENSIONS}},
        },
        {
            "selected_social_pressure": 0.2,
            "social_potential": {"dimensions": {dim: 0.1 for dim in SOCIAL_POTENTIAL_DIMENSIONS}},
        },
    ]

    summary = summarize_action_social_potential(actions)

    assert summary["selected_social_pressure_mean"] == 0.4
    assert summary["selected_social_pressure_max"] == 0.6
    assert summary["social_potential_trust_deficit_mean"] == 0.3
