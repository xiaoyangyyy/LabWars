"""Role Policy Agent — LLM-driven action selection with diversity guards."""

from __future__ import annotations

from typing import Any

from src.cognition.memory import RecallResult
from src.engine.diversity import (
    avoid_actions,
    filter_allowed_actions,
    is_repetitive_choice,
)
from src.engine.llm_adapter import LLMAdapter, LLMError
from src.engine.prompts import ROLE_POLICY_SYSTEM, build_role_policy_prompt
from src.world.actions import ActionType, get_allowed_actions
from src.world.models import Agent, EventAtom, WorldState

from .event_agent import is_agent_active


def _pick_target(agent: Agent, world: WorldState, event: EventAtom, suggested: str | None) -> str:
    if suggested and suggested in world.agents:
        return suggested
    if event.source != agent.id and event.source in world.agents:
        return event.source
    internal = world.world_config.get("internal_agents", [])
    if "pi" in internal and agent.id != "pi":
        return "pi"
    for t in event.targets:
        if t in world.agents and t != agent.id:
            return t
    return "project"


def _normalize_action_response(
    raw: dict[str, Any],
    agent: Agent,
    event: EventAtom,
    world: WorldState,
    allowed: list[ActionType],
) -> dict[str, Any]:
    allowed_values = {a.value for a in allowed}
    primary = raw.get("primary_action") or raw
    atype = str(primary.get("type", ""))
    if atype not in allowed_values:
        atype = allowed[0].value

    intensity = float(primary.get("intensity", 0.5))
    intensity = max(0.0, min(1.0, intensity))
    target = _pick_target(agent, world, event, primary.get("target"))

    comm = raw.get("communication_action") or {}
    comm_type = str(comm.get("type", "share_result"))
    comm_target = _pick_target(agent, world, event, comm.get("target"))

    public = raw.get("public_position") or {"statement_type": "neutral", "authorship_claim": "any_authorship"}
    private = raw.get("private_intent") or {
        "goal": "secure_first_author",
        "strategy": atype,
        "trust_pi": agent.beliefs.pi_fairness,
    }

    return {
        "agent": agent.id,
        "type": atype,
        "target": target,
        "intensity": round(intensity, 4),
        "communication_action": {
            "type": comm_type,
            "target": comm_target,
            "content_summary": str(comm.get("content_summary", f"{agent.id} re {event.type}")),
        },
        "public_position": public,
        "private_intent": private,
        "llm_raw": raw,
    }


class RolePolicyAgent:
    def __init__(self, llm: LLMAdapter, max_retries: int = 3) -> None:
        self.llm = llm
        self.max_retries = max_retries

    def decide(
        self,
        agent: Agent,
        event: EventAtom,
        world: WorldState,
        recall: RecallResult | None,
        config: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not is_agent_active(agent.id, event.round, config):
            return None

        allowed = get_allowed_actions(agent.id, burnout=agent.emotion.burnout)
        if not allowed:
            return None

        avoid = avoid_actions(agent)
        dynamic_avoid = list(avoid)
        allowed_str, applied_avoid = filter_allowed_actions(
            [a.value for a in allowed],
            dynamic_avoid,
        )

        last_error = ""
        retry_note = ""

        for attempt in range(self.max_retries + 1):
            user_prompt = build_role_policy_prompt(
                agent,
                event,
                world,
                recall,
                allowed_str,
                avoid_actions=dynamic_avoid,
                retry_note=retry_note,
                validation_error=last_error,
            )
            try:
                raw = self.llm.complete_json(ROLE_POLICY_SYSTEM, user_prompt)
                act = _normalize_action_response(raw, agent, event, world, allowed)
                if is_repetitive_choice(agent, act["type"]) or act["type"] in dynamic_avoid:
                    if act["type"] not in dynamic_avoid:
                        dynamic_avoid.append(act["type"])
                    allowed_str, _ = filter_allowed_actions(
                        [a.value for a in allowed],
                        dynamic_avoid,
                    )
                    retry_note = (
                        f"You chose '{act['type']}' which is blocked. "
                        f"Pick a DIFFERENT type from {allowed_str}."
                    )
                    last_error = retry_note
                    continue
                return act
            except (LLMError, KeyError, TypeError, ValueError) as exc:
                last_error = str(exc)

        raise LLMError(f"RolePolicyAgent failed for {agent.id} after retries: {last_error}")

    def decide_all(
        self,
        world: WorldState,
        event: EventAtom,
        recalls: dict[str, RecallResult],
        config: dict[str, Any],
    ) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        for agent_id, agent in world.agents.items():
            act = self.decide(agent, event, world, recalls.get(agent_id), config)
            if act:
                actions.append(act)
        return actions
