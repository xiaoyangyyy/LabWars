"""Tests for triadic private influence."""

from __future__ import annotations

from src.cognition.relationship import update_relationships
from src.world.loader import load_events, load_world


def _edge(world, src, tgt):
    return next(e for e in world.relationships if e.source == src and e.target == tgt)


def test_private_talk_contaminates_third_party_view():
    world = load_world()
    event = load_events()[0]
    # Make phd_a strongly suspicious of phd_b, then let phd_a lobby PI.
    a_b = _edge(world, "phd_a", "phd_b")
    a_b.trust = 0.15
    a_b.resentment = 0.80
    a_b.perceived_credit_threat = 0.90
    before = _edge(world, "pi", "phd_b")
    before_trust = before.trust
    before_resentment = before.resentment

    update_relationships(
        world,
        event,
        recalls={},
        actions=[{"agent": "phd_a", "type": "privately_lobby_pi", "target": "pi", "intensity": 0.9}],
    )

    after = _edge(world, "pi", "phd_b")
    assert after.trust < before_trust
    assert after.resentment > before_resentment
