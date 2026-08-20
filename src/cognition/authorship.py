"""Soft power authorship game 鈥?continuous ranking, no discrete overrides."""

from __future__ import annotations

import math
from typing import Any

from src.world.models import Agent, AgentRole, ProjectState, WorldState
from src.world.organization import authority_ids, primary_authority

from .math_utils import clamp, entropy, logistic_gate, normalize_simplex, softplus
from .power import pi_control_pressure, pi_preference_distribution


_RAW_DIM_WEIGHTS = {
    "idea": 0.25,
    "experiments": 0.30,
    "writing": 0.20,
    "data": 0.10,
    "code": 0.10,
    "rebuttal": 0.08,
    "funding": 0.05,
    "supervision": 0.12,
}
_DIM_TOTAL = sum(_RAW_DIM_WEIGHTS.values())
DIM_WEIGHTS = {dim: weight / _DIM_TOTAL for dim, weight in _RAW_DIM_WEIGHTS.items()}


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
    """PI power rises smoothly with deadline pressure 鈥?no step at 0.8."""
    pressure = project.project.deadline_pressure
    control = pi.personality.ambition if pi else 0.7
    legacy_pressure = logistic_gate(pressure, center=0.55, steepness=5.5) * (0.45 + 0.55 * control)
    return clamp(legacy_pressure)


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
    authority = primary_authority(world)
    pi = agents.get(authority) if authority else None
    pi_weight = clamp(0.45 * _pi_power_weight(world.project, pi) + 0.55 * pi_control_pressure(world))

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
        if agent.role == AgentRole.EXPERIMENTER or agent.extra_traits.get("archetype") == "phd_b":
            pref += 0.12
        if aid in authority_ids(world):
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


ACTION_LEDGER_EFFECTS: dict[str, dict[str, float]] = {
    "run_experiment": {"experiments": 0.045, "data": 0.020},
    "improve_baseline": {"experiments": 0.025, "code": 0.035},
    "debug_code": {"code": 0.050, "experiments": 0.015},
    "write_section": {"writing": 0.050, "idea": 0.012},
    "analyze_failure": {"experiments": 0.018, "data": 0.020, "code": 0.010},
    "prepare_rebuttal": {"rebuttal": 0.050, "writing": 0.020},
    "share_result": {"data": 0.020, "experiments": 0.012},
    "open_source_code": {"code": 0.035},
    "document_contribution": {"idea": 0.020, "experiments": 0.020, "code": 0.015, "data": 0.015, "writing": 0.020},
    "cite_prior_memory": {"idea": 0.025, "writing": 0.010},
    "contact_collaborator": {"writing": 0.012, "rebuttal": 0.010},
    "notify_program_officer": {"funding": 0.025},
}


def _agent_dim_fit(agent: Agent, dim: str) -> float:
    fit = {
        "idea": 0.35 * agent.personality.ambition + 0.35 * agent.resources.writing_control + 0.30 * agent.personality.credit_sensitivity,
        "experiments": 0.50 * agent.resources.code_control + 0.30 * agent.resources.data_control + 0.20 * agent.personality.ambition,
        "code": 0.70 * agent.resources.code_control + 0.30 * agent.personality.cooperation,
        "data": 0.70 * agent.resources.data_control + 0.30 * agent.personality.cooperation,
        "writing": 0.70 * agent.resources.writing_control + 0.30 * agent.personality.ambition,
        "rebuttal": 0.40 * agent.resources.writing_control + 0.30 * agent.resources.pi_access + 0.30 * agent.personality.cooperation,
        "funding": 0.55 * agent.resources.pi_access + 0.25 * agent.personality.authority_dependence + 0.20 * agent.personality.cooperation,
        "supervision": 0.50 * agent.resources.pi_access + 0.30 * agent.personality.cooperation + 0.20 * agent.personality.authority_dependence,
    }
    return clamp(fit.get(dim, 0.5))


def update_ledger_from_action(
    world: WorldState,
    agent_id: str,
    intensity: float = 0.5,
    action_type: str = "document_contribution",
) -> None:
    """Soft contribution ledger updates from the actual work/political action."""
    agent = world.agents[agent_id]
    ledger = world.project.contribution_ledger
    effects = ACTION_LEDGER_EFFECTS.get(action_type, {})
    if not effects:
        return
    for dim, base_delta in effects.items():
        bucket = ledger.setdefault(dim, {})
        current = bucket.get(agent_id, 0.0)
        fit = _agent_dim_fit(agent, dim)
        bucket[agent_id] = current + base_delta * intensity * (0.55 + 0.45 * fit)
        ledger[dim] = normalize_simplex(bucket)

def authorship_dispute_index(world: WorldState) -> float:
    """Variance + entropy of authorship beliefs 鈥?fully continuous."""
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




