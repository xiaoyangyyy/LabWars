"""Tests for policy_mode contrast tracks."""

from __future__ import annotations

from src.engine.event_agent import EventAgent
from src.engine.role_policy import RolePolicyAgent
from src.engine.simulation import SimConfig, run_simulation
from src.experiments.policy_mode_comparison import run_policy_mode_comparison
from src.world.loader import load_world


def test_llm_native_policy_generates_candidate_space(llm_adapter):
    world = load_world()
    agent = world.agents["phd_a"]
    event = EventAgent(seed=31).generate(30, world)
    policy = RolePolicyAgent(llm_adapter)

    action = policy.decide(agent, event, world, None, {"seed": 31, "policy_mode": "llm_native"})

    assert action is not None
    assert action["llm_action_scoring"]["source"] == "llm_native_generated"
    assert action["selected_action"]["scoring_source"] == "llm_native_generated"
    assert action["selected_action"]["parameter_source"] == "llm_native_policy"
    assert action["action_candidates"]


def test_social_physics_policy_is_field_only():
    log = run_simulation(SimConfig(max_rounds=3, seed=32, interventions=[], policy_mode="social_physics"))
    assert log.actions
    assert all(a.get("llm_action_scoring", {}).get("source") == "field_only" for a in log.actions)


def test_policy_mode_comparison_runs():
    result = run_policy_mode_comparison(
        SimConfig(max_rounds=3, interventions=[]),
        policy_modes=["social_physics", "dual_engine", "llm_native"],
        seeds=[0],
        outcomes=["authorship_dispute_index", "llm_native_candidate_fraction"],
    )
    assert result.n_per_mode == 1
    assert set(result.summary) == {"social_physics", "dual_engine", "llm_native"}
    assert result.summary["llm_native"]["llm_native_candidate_fraction"] > 0.0
