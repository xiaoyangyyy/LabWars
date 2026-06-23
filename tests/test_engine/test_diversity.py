"""Tests for action diversity guards."""

from __future__ import annotations

from src.engine.diversity import (
    action_usage_counts,
    avoid_actions,
    filter_allowed_actions,
    is_repetitive_choice,
)
from src.world.loader import load_world


def test_avoid_after_streak():
    world = load_world()
    agent = world.agents["phd_a"]
    agent.action_history = [
        {"round": 1, "action": {"type": "document_contribution"}},
        {"round": 2, "action": {"type": "document_contribution"}},
    ]
    assert "document_contribution" in avoid_actions(agent)


def test_filter_keeps_minimum():
    allowed = ["document_contribution", "comply", "share_result", "write_section"]
    filtered, applied = filter_allowed_actions(allowed, ["document_contribution"], min_keep=3)
    assert len(filtered) >= 3
    assert applied


def test_is_repetitive():
    world = load_world()
    agent = world.agents["phd_a"]
    agent.action_history = [{"round": 1, "action": {"type": "comply"}}]
    assert is_repetitive_choice(agent, "comply")


def test_usage_counts():
    world = load_world()
    agent = world.agents["phd_a"]
    agent.action_history = [
        {"round": i, "action": {"type": "document_contribution" if i % 2 else "comply"}}
        for i in range(8)
    ]
    counts = action_usage_counts(agent, window=8)
    assert counts["document_contribution"] == 4
