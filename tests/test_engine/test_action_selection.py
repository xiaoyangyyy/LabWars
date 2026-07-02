"""Tests for continuous action candidate generation."""

from __future__ import annotations

from src.engine.action_selection import generate_action_candidates, sample_action_candidate_legacy
from src.engine.event_agent import EventAgent
from src.world.actions import get_allowed_actions
from src.world.loader import load_world


def test_candidates_have_probabilities_and_motives():
    world = load_world()
    agent = world.agents["phd_a"]
    event = EventAgent().generate(52, world)
    allowed = [a.value for a in get_allowed_actions(agent.id)]

    candidates = generate_action_candidates(agent, event, world, None, allowed, seed=42)

    assert candidates
    assert abs(sum(c.probability for c in candidates) - 1.0) < 1e-6
    assert all(c.motives for c in candidates)
    assert all(c.field_decomposition for c in candidates)
    assert all("motive_contributions" in c.field_decomposition for c in candidates)
    assert all(0.0 <= c.intensity <= 1.0 for c in candidates)


def test_authorship_pressure_lifts_authorship_actions():
    world = load_world()
    agent = world.agents["phd_a"]
    agent.beliefs.pi_fairness = 0.20
    agent.beliefs.my_first_author_probability = 0.15
    agent.emotion.resentment = 0.80
    event = EventAgent().generate(52, world)
    allowed = [a.value for a in get_allowed_actions(agent.id)]

    candidates = generate_action_candidates(agent, event, world, None, allowed, seed=7)
    top_types = {c.type for c in candidates[:5]}

    assert top_types & {"ask_for_authorship", "challenge_claim", "confront", "document_contribution"}


def test_legacy_sampling_is_seed_stable():
    world = load_world()
    agent = world.agents["phd_a"]
    event = EventAgent().generate(40, world)
    allowed = [a.value for a in get_allowed_actions(agent.id)]
    candidates = generate_action_candidates(agent, event, world, None, allowed, seed=3)

    a = sample_action_candidate_legacy(candidates, seed=3, round_num=40, agent_id="phd_a")
    b = sample_action_candidate_legacy(candidates, seed=3, round_num=40, agent_id="phd_a")

    assert a.type == b.type
