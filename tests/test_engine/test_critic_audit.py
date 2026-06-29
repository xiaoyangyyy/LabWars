"""Tests for LLM wording/action drift audit."""

from __future__ import annotations

from src.engine.critic import CriticAgent
from src.world.loader import load_world


def test_critic_flags_public_action_drift():
    world = load_world()
    agent = world.agents["phd_a"]
    action = {
        "agent": "phd_a",
        "type": "confront",
        "target": "pi",
        "intensity": 0.7,
        "public_position": {"statement_type": "team_support"},
        "private_intent": {"strategy": "comply"},
        "selected_action": {"type": "confront"},
    }
    violations = CriticAgent().check(action, agent, world)
    codes = {v.code for v in violations}
    assert "llm_public_action_drift" in codes
    assert "llm_private_strategy_drift" in codes
