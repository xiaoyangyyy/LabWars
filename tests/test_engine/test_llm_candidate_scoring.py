"""Tests for LLM candidate scoring in RolePolicyAgent."""

from __future__ import annotations

from src.engine.event_agent import EventAgent
from src.engine.role_policy import RolePolicyAgent
from src.world.loader import load_world


def test_role_policy_fuses_field_and_llm_candidate_scores(llm_adapter):
    world = load_world()
    agent = world.agents["phd_a"]
    event = EventAgent(seed=11).generate(30, world)
    policy = RolePolicyAgent(llm_adapter)

    action = policy.decide(agent, event, world, None, {"seed": 11})

    assert action is not None
    assert action["llm_action_scoring"]["source"] == "dual_engine_fused"
    assert action["selected_action"]["scoring_source"] == "dual_engine_fused"
    assert "llm_score" in action["selected_action"]
    assert "cognitive_policy_lambda" in action["selected_action"]
    assert "field_probability" in action["selected_action"]
    assert abs(sum(c["probability"] for c in action["action_candidates"]) - 1.0) < 1e-4


def test_role_policy_can_disable_llm_candidate_scoring(llm_adapter):
    world = load_world()
    agent = world.agents["phd_a"]
    event = EventAgent(seed=12).generate(30, world)
    policy = RolePolicyAgent(llm_adapter)

    action = policy.decide(
        agent,
        event,
        world,
        None,
        {"seed": 12, "enable_llm_action_scoring": False},
    )

    assert action is not None
    assert action["llm_action_scoring"]["enabled"] is False
    assert action["llm_action_scoring"]["source"] == "field_only"
    assert "llm_score" not in action["selected_action"]
