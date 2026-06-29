"""Tests for private-channel relationship effects."""

from __future__ import annotations

from src.cognition.relationship import update_relationships
from src.world.loader import load_events, load_world


def _edge(world, src, tgt):
    return next(e for e in world.relationships if e.source == src and e.target == tgt)


def test_private_lobbying_changes_pi_access_relationship():
    world = load_world()
    event = load_events()[0]
    before = _edge(world, "phd_a", "pi")
    before_trust = before.trust
    before_alliance = before.alliance

    update_relationships(
        world,
        event,
        recalls={},
        actions=[{"agent": "phd_a", "type": "privately_lobby_pi", "target": "pi", "intensity": 0.8}],
    )

    after = _edge(world, "phd_a", "pi")
    assert after.trust > before_trust
    assert after.alliance > before_alliance


def test_confrontation_raises_credit_threat_and_resentment():
    world = load_world()
    event = load_events()[0]
    before = _edge(world, "phd_a", "phd_b")
    before_threat = before.perceived_credit_threat
    before_resentment = before.resentment

    update_relationships(
        world,
        event,
        recalls={},
        actions=[{"agent": "phd_a", "type": "challenge_claim", "target": "phd_b", "intensity": 0.7}],
    )

    after = _edge(world, "phd_a", "phd_b")
    assert after.perceived_credit_threat > before_threat
    assert after.resentment > before_resentment
