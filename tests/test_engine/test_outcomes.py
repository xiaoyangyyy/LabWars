"""Tests for outcome extraction fixes."""

from __future__ import annotations

from src.engine.run_log import RunLog, extract_outcome, finalize_outcomes
from src.world.models import Agent, AgentRole, Beliefs, Emotion, Personality, RelationshipEdge, Resources


def _minimal_agent(**belief_kw) -> Agent:
    defaults = dict(
        pi_fairness=0.5, project_publishability=0.5, rival_lab_threat=0.3,
        my_first_author_probability=0.4, team_trust=0.5, deadline_feasibility=0.5,
    )
    defaults.update(belief_kw)
    return Agent(
        id="phd_a",
        role=AgentRole.IDEA_ORIGINATOR,
        personality=Personality(
            ambition=0.5, cooperation=0.5, risk_taking=0.4, conflict_avoidance=0.4,
            credit_sensitivity=0.7, authority_dependence=0.5, deceptiveness=0.2,
            reciprocity=0.6, resentment_sensitivity=0.6,
        ),
        beliefs=Beliefs(**defaults),
        emotion=Emotion(
            confidence=0.5, anxiety=0.3, anger=0.2, resentment=0.2, guilt=0.1,
            hope=0.5, burnout=0.2,
        ),
        resources=Resources(
            code_control=0.5, data_control=0.5, writing_control=0.5,
            pi_access=0.6, external_network=0.3,
        ),
    )


def _log() -> RunLog:
    return RunLog(run_id="t", config={"experiment_id": "A", "condition_id": "A1", "seed": 1})


class TestOutcomeFixes:
    def test_trust_pi_final_uses_relationship_edge(self):
        log = _log()
        agent = _minimal_agent(pi_fairness=0.2, team_trust=0.3)
        rel = [RelationshipEdge(
            source="phd_a", target="pi", trust=0.72, resentment=0.1, dependency=0.6,
            obligation=0.1, perceived_credit_threat=0.2, communication_frequency=0.5,
            alliance=0.0, information_access=0.4, last_interaction_valence=0.0,
        )]
        finalize_outcomes(log, {"phd_a": agent}, rel)
        assert log.outcomes["trust_pi_final"] == 0.72
        assert log.outcomes["pi_trust_belief_final"] == 0.2

    def test_promise_broken_only_at_r52(self):
        log = _log()
        log.round_records = [
            {"round": 20, "agent_deltas": {"phd_a": {"memory_written": {
                "content_type": "promise_broken", "strength": 0.9, "event_ref": "E020",
            }}}},
            {"round": 52, "event_id": "E052", "agent_deltas": {"phd_a": {"memory_written": {
                "content_type": "promise_fulfilled", "strength": 0.8, "event_ref": "E052",
            }}}},
        ]
        assert extract_outcome(log, "promise_broken_strength_r52") == 0.0
        assert extract_outcome(log, "promise_honored_strength_r52") == 0.8

    def test_protest_zero_when_compliance(self):
        log = _log()
        log.actions = [{"agent": "phd_a", "round": 52, "type": "comply", "intensity": 0.9}]
        assert extract_outcome(log, "protest_authorship") == 0.0

    def test_protest_requires_high_escalation(self):
        log = _log()
        log.actions = [{"agent": "phd_a", "round": 52, "type": "confront", "intensity": 0.4}]
        log.round_records = [{"round": 52, "agent_deltas": {"phd_a": {
            "beliefs": {"pi_fairness": 0.15},
            "emotion": {"resentment": 0.7, "anger": 0.6},
        }}}]
        assert extract_outcome(log, "protest_authorship") in (0.0, 1.0)
