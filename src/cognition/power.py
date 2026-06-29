"""Institutional power surfaces for PI-mediated academic conflict."""

from __future__ import annotations

from typing import Any

from src.world.models import Agent, WorldState

from .math_utils import clamp, logistic_gate, normalize_simplex, softplus


CONTROL_WEIGHTS = {
    "recommendation_letter": 0.30,
    "funding_access": 0.22,
    "authorship_veto": 0.28,
    "resource_allocation": 0.20,
}


def pi_control_surface(world: WorldState) -> dict[str, float]:
    """Continuous PI control over career, resources, and authorship arbitration."""
    pi = world.agents.get("pi")
    project = world.project.project
    authority = pi.personality.ambition if pi else 0.65
    cooperation = pi.personality.cooperation if pi else 0.55

    deadline = project.deadline_pressure
    funding = project.funding_pressure
    conflict = project.authorship_conflict
    morale_loss = 1.0 - project.team_morale

    return {
        "recommendation_letter": clamp(0.35 + 0.35 * authority + 0.20 * funding + 0.10 * conflict),
        "funding_access": clamp(0.25 + 0.30 * authority + 0.30 * funding + 0.15 * deadline),
        "authorship_veto": clamp(0.30 + 0.30 * authority + 0.25 * conflict + 0.15 * deadline),
        "resource_allocation": clamp(0.25 + 0.25 * authority + 0.25 * deadline + 0.15 * morale_loss + 0.10 * cooperation),
    }


def pi_control_pressure(world: WorldState, agent: Agent | None = None) -> float:
    """Agent-specific dependence on PI's institutional control."""
    surface = pi_control_surface(world)
    base = sum(surface[k] * w for k, w in CONTROL_WEIGHTS.items())
    if agent is None:
        return round(clamp(base), 4)
    dependence = 0.45 * agent.personality.authority_dependence + 0.35 * agent.resources.pi_access
    outside = 1.0 - agent.resources.external_network
    return round(clamp(base * (0.55 + 0.30 * dependence + 0.15 * outside)), 4)


def pi_preference_distribution(world: WorldState, candidate_ids: list[str]) -> dict[str, float]:
    """How PI control softly biases author order toward manageable agents."""
    surface = pi_control_surface(world)
    veto = surface["authorship_veto"]
    resource = surface["resource_allocation"]
    funding = surface["funding_access"]
    prefs: dict[str, float] = {}
    for aid in candidate_ids:
        agent = world.agents[aid]
        manageability = (
            0.34 * agent.personality.cooperation
            + 0.26 * (1.0 - agent.personality.risk_taking)
            + 0.20 * agent.resources.pi_access
            + 0.20 * agent.personality.authority_dependence
        )
        delivery = (
            0.30 * agent.resources.code_control
            + 0.25 * agent.resources.data_control
            + 0.25 * agent.resources.writing_control
            + 0.20 * agent.personality.ambition
        )
        pressure_fit = softplus(manageability * veto + delivery * resource + funding * agent.resources.pi_access)
        if aid == "pi":
            pressure_fit += 0.20 * veto
        prefs[aid] = pressure_fit
    return normalize_simplex(prefs)


def career_hostage_index(world: WorldState) -> float:
    """How much internal agents depend on PI-controlled career surfaces."""
    internal = [aid for aid in world.world_config.get("internal_agents", []) if aid in world.agents and aid != "pi"]
    if not internal:
        return 0.0
    vals = [pi_control_pressure(world, world.agents[aid]) for aid in internal]
    return round(sum(vals) / len(vals), 4)
