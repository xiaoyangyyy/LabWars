"""Evolving public reputation as a first-class social state."""

from __future__ import annotations

from typing import Any

from src.world.models import Agent, WorldState, clamp

REPUTATION_KEY = "reputation"

POSITIVE_REPUTATION_ACTIONS = {
    "support_teammate": 0.045,
    "share_result": 0.030,
    "open_source_code": 0.040,
    "apologize": 0.025,
    "document_contribution": 0.015,
    "analyze_failure": 0.020,
}

NEGATIVE_REPUTATION_ACTIONS = {
    "undermine_teammate": 0.055,
    "blame": 0.040,
    "withhold_code": 0.035,
    "hide_negative_result": 0.060,
    "selectively_report": 0.045,
    "leak_concern": 0.020,
}


def current_reputation(agent: Agent) -> float:
    return clamp(float(agent.extra_traits.get(REPUTATION_KEY, 0.5)))


def set_reputation(agent: Agent, value: float) -> float:
    agent.extra_traits[REPUTATION_KEY] = round(clamp(value), 4)
    return agent.extra_traits[REPUTATION_KEY]


def update_reputation_from_action(world: WorldState, action: dict[str, Any]) -> None:
    agent_id = action.get("agent")
    if not agent_id or agent_id not in world.agents:
        return
    agent = world.agents[agent_id]
    atype = str(action.get("type", ""))
    intensity = float(action.get("intensity", 0.5))
    current = current_reputation(agent)
    delta = 0.0
    if atype in POSITIVE_REPUTATION_ACTIONS:
        delta += POSITIVE_REPUTATION_ACTIONS[atype] * intensity
    if atype in NEGATIVE_REPUTATION_ACTIONS:
        delta -= NEGATIVE_REPUTATION_ACTIONS[atype] * intensity
    if delta:
        set_reputation(agent, current + delta)

    target_id = action.get("target")
    if target_id in world.agents and target_id != agent_id:
        target = world.agents[target_id]
        if atype == "support_teammate":
            set_reputation(target, current_reputation(target) + 0.012 * intensity)
        elif atype in {"undermine_teammate", "blame"}:
            set_reputation(target, current_reputation(target) - 0.018 * intensity)
