"""Soft power authorship game — continuous ranking, no discrete overrides."""

from __future__ import annotations

import math
from typing import Any

from src.world.models import Agent, ProjectState, WorldState

from .math_utils import clamp, entropy, logistic_gate, normalize_simplex, softplus


DIM_WEIGHTS = {
    "idea": 0.25,
    "experiments": 0.30,
    "writing": 0.20,
    "data": 0.10,
    "supervision": 0.15,
}


def merit_score(agent_id: str, ledger: dict[str, dict[str, float]]) -> float:
    total = 0.0
    for dim, weight in DIM_WEIGHTS.items():
        bucket = ledger.get(dim, {})
        share = bucket.get(agent_id, 0.0)
        total += weight * share
    return total


def _leverage_signal(agent: Agent, threat_events: int = 0) -> float:
    """Continuous leverage from resources and protest history."""
    code = softplus(agent.resources.code_control - 0.5)
    writing = softplus(agent.resources.writing_control - 0.4)
    pi_access = softplus(agent.resources.pi_access - 0.5) * 0.5
    protest = softplus(threat_events * 0.35)
    return code * 0.4 + writing * 0.25 + pi_access * 0.2 + protest * 0.15


def _pi_power_weight(project: ProjectState, pi: Agent | None) -> float:
    """PI power rises smoothly with deadline pressure — no step at 0.8."""
    pressure = project.project.deadline_pressure
    control = pi.personality.ambition if pi else 0.7
    return clamp(logistic_gate(pressure, center=0.55, steepness=5.5) * (0.45 + 0.55 * control))


def compute_authorship_scores(
    world: WorldState,
    candidate_ids: list[str] | None = None,
    threat_counts: dict[str, int] | None = None,
) -> dict[str, float]:
    ledger = world.project.contribution_ledger
    agents = world.agents
    candidates = candidate_ids or [
        aid for aid, a in agents.items()
        if a.role.value not in ("reviewer", "program_officer", "alumni", "rival_lab")
    ]
    threat_counts = threat_counts or {}
    pi = agents.get("pi")
    pi_weight = _pi_power_weight(world.project, pi)

    merit = {aid: merit_score(aid, ledger) for aid in candidates}
    merit_norm = normalize_simplex(merit)

    pi_pref: dict[str, float] = {}
    for aid in candidates:
        agent = agents[aid]
        pref = (
            0.25 * agent.personality.cooperation
            + 0.20 * (1.0 - agent.personality.risk_taking)
            + 0.15 * agent.resources.pi_access
        )
        if aid == "phd_b":
            pref += 0.12
        if aid == "pi":
            pref += 0.20
        pi_pref[aid] = pref
    pi_pref = normalize_simplex(pi_pref)

    blended: dict[str, float] = {}
    for aid in candidates:
        agent = agents[aid]
        leverage = _leverage_signal(agent, threat_counts.get(aid, 0))
        merit_part = merit_norm.get(aid, 0.0)
        pi_part = pi_pref.get(aid, 0.0)
        score = (1.0 - pi_weight) * merit_part + pi_weight * pi_part
        score *= 1.0 + leverage
        blended[aid] = score

    return normalize_simplex(blended)


def rank_authors(
    world: WorldState,
    candidate_ids: list[str] | None = None,
    threat_counts: dict[str, int] | None = None,
) -> list[str]:
    scores = compute_authorship_scores(world, candidate_ids, threat_counts)
    return sorted(scores.keys(), key=lambda a: scores[a], reverse=True)


def update_ledger_from_event(world: WorldState, event_type: str, payload: dict[str, Any]) -> None:
    ledger = world.project.contribution_ledger
    for key, value in payload.items():
        if not key.startswith("ledger_"):
            continue
        rest = key[len("ledger_") :]
        matched_dim = None
        matched_agent = None
        for dim in DIM_WEIGHTS:
            prefix = f"{dim}_"
            if rest.startswith(prefix):
                matched_dim = dim
                matched_agent = rest[len(prefix) :]
                break
        if matched_dim and matched_agent:
            bucket = ledger.setdefault(matched_dim, {})
            bucket[matched_agent] = clamp(float(value))
            ledger[matched_dim] = normalize_simplex(bucket)


def update_ledger_from_action(
    world: WorldState,
    agent_id: str,
    intensity: float = 0.5,
) -> None:
    """document_contribution: soft increment across agent's strongest dimension."""
    agent = world.agents[agent_id]
    ledger = world.project.contribution_ledger
    dim_strength = {
        "idea": agent.resources.writing_control * agent.personality.credit_sensitivity,
        "experiments": agent.resources.code_control,
        "writing": agent.resources.writing_control,
        "data": agent.resources.data_control,
    }
    best_dim = max(dim_strength, key=lambda d: dim_strength[d])
    bucket = ledger.setdefault(best_dim, {})
    current = bucket.get(agent_id, 0.0)
    bucket[agent_id] = current + 0.04 * intensity
    ledger[best_dim] = normalize_simplex(bucket)


def authorship_dispute_index(world: WorldState) -> float:
    """Variance + entropy of authorship beliefs — fully continuous."""
    internal = world.world_config.get("internal_agents", [])
    probs = [
        world.agents[aid].beliefs.my_first_author_probability
        for aid in internal
        if aid in world.agents
    ]
    if len(probs) < 2:
        return world.project.project.authorship_conflict

    mean = sum(probs) / len(probs)
    var = sum((p - mean) ** 2 for p in probs) / len(probs)
    ent = entropy([max(p, 1e-6) for p in probs])
    protest_proxy = sum(
        world.agents[aid].emotion.resentment + world.agents[aid].emotion.anger
        for aid in internal
        if aid in world.agents
    ) / len(probs)

    return round(
        clamp(math.sqrt(var) * 1.2 + ent * 0.15 + world.project.project.authorship_conflict * 0.5 + protest_proxy * 0.08),
        4,
    )
