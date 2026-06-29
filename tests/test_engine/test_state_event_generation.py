"""Tests for state-driven event generation."""

from __future__ import annotations

from src.engine.event_agent import EventAgent
from src.world.loader import load_world


def test_state_event_field_records_candidates():
    world = load_world()
    ea = EventAgent(seed=11)
    event = ea.generate(1, world)

    assert event is not None
    assert event.payload.get("generator") == "state_event_field"
    candidates = event.payload.get("event_candidates")
    assert candidates
    assert all("probability" in c for c in candidates)
    assert abs(sum(c["probability"] for c in candidates) - 1.0) < 1e-4


def test_state_event_generation_is_seed_reproducible():
    world = load_world()
    e1 = EventAgent(seed=13).generate(8, world)
    e2 = EventAgent(seed=13).generate(8, world)

    assert e1 is not None and e2 is not None
    assert e1.event_id == e2.event_id
    assert e1.payload.get("event_candidates") == e2.payload.get("event_candidates")


def test_event_candidates_respond_to_world_state():
    calm = load_world()
    tense = load_world()
    tense.project.project.authorship_conflict = 0.95
    tense.project.project.deadline_pressure = 0.90
    tense.agents["phd_a"].memory.append({
        "memory_id": "M999",
        "owner": "phd_a",
        "round": 2,
        "event_ref": "E003",
        "content_type": "promise_broken",
        "target": "pi",
        "valence": -0.9,
        "strength": 0.95,
        "strength_0": 0.95,
        "decay": 0.03,
        "rehearsal_count": 1.0,
        "evidence_quality": 0.8,
        "interpretation": "PI may be walking back the first-author promise.",
        "behavioral_hooks": ["ask_for_authorship", "document_contribution"],
        "was_recalled": [],
        "objective_fact_ref": "E003.objective_fact",
    })

    ea = EventAgent(seed=17)
    calm_candidates = {c.type: c.tendency for c in ea._state_candidates(12, calm)}
    tense_candidates = {c.type: c.tendency for c in ea._state_candidates(12, tense)}

    assert tense_candidates["authorship_ambiguity"] > calm_candidates["authorship_ambiguity"]
