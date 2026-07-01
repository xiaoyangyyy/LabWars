"""Role Policy Agent - LLM candidate scoring plus constrained stance rendering."""

from __future__ import annotations

import hashlib
import math
import random
from typing import Any

from src.cognition.memory import RecallResult
from src.engine.action_selection import generate_action_candidates
from src.engine.diversity import (
    avoid_actions,
    filter_allowed_actions,
)
from src.engine.llm_adapter import LLMAdapter, LLMError
from src.engine.prompts import (
    ACTION_SCORING_SYSTEM,
    ROLE_POLICY_SYSTEM,
    build_action_scoring_prompt,
    build_role_policy_prompt,
)
from src.world.actions import ActionType, get_allowed_actions
from src.world.models import Agent, EventAtom, WorldState

from .event_agent import is_agent_active


def _pick_target(agent: Agent, world: WorldState, event: EventAtom, suggested: str | None) -> str:
    if suggested and (suggested in world.agents or suggested in {"project", "shared_doc"}):
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


def _softmax(values: list[float], temperature: float = 0.22) -> list[float]:
    if not values:
        return []
    temp = max(temperature, 1e-4)
    peak = max(values)
    exps = [math.exp((v - peak) / temp) for v in values]
    total = sum(exps) or 1.0
    return [e / total for e in exps]


def _stable_rng(seed: int, round_num: int, agent_id: str, suffix: str) -> random.Random:
    h = hashlib.sha256(f"{seed}:{round_num}:{agent_id}:{suffix}".encode("utf-8")).hexdigest()
    return random.Random(int(h[:16], 16))


def _sample_payload(payloads: list[dict[str, Any]], *, seed: int, round_num: int, agent_id: str) -> dict[str, Any]:
    if not payloads:
        raise ValueError("No action payloads to sample")
    rng = _stable_rng(seed, round_num, agent_id, "llm_fused")
    needle = rng.random()
    total = 0.0
    for payload in payloads:
        total += float(payload.get("probability", 0.0))
        if needle <= total:
            return payload
    return payloads[-1]


def _coerce_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, score))


class RolePolicyAgent:
    def __init__(self, llm: LLMAdapter, max_retries: int = 3) -> None:
        self.llm = llm
        self.max_retries = max_retries

    def _score_candidates(
        self,
        agent: Agent,
        event: EventAtom,
        world: WorldState,
        recall: RecallResult | None,
        candidate_payload: list[dict[str, Any]],
        config: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Ask the LLM for subjective candidate plausibility, then fuse with field scores."""
        if not config.get("enable_llm_action_scoring", True):
            return candidate_payload, {"enabled": False, "source": "field_only"}

        raw_lambda = config.get("cognitive_policy_lambda")
        if raw_lambda is None:
            raw_lambda = config.get("llm_action_score_mix", 0.35)
        cognitive_lambda = max(0.0, min(1.0, float(raw_lambda)))
        social_lambda = 1.0 - cognitive_lambda
        try:
            prompt = build_action_scoring_prompt(agent, event, world, recall, candidate_payload)
            raw = self.llm.complete_json(ACTION_SCORING_SYSTEM, prompt)
        except (LLMError, KeyError, TypeError, ValueError) as exc:
            return candidate_payload, {"enabled": True, "source": "field_only_fallback", "error": str(exc)}

        raw_scores = raw.get("candidate_scores", [])
        score_map: dict[str, dict[str, Any]] = {}
        if isinstance(raw_scores, list):
            for item in raw_scores:
                if not isinstance(item, dict):
                    continue
                action_type = str(item.get("type", ""))
                if action_type:
                    score_map[action_type] = item

        fused: list[dict[str, Any]] = []
        for candidate in candidate_payload:
            item = dict(candidate)
            score_item = score_map.get(str(candidate.get("type", "")), {})
            llm_score = _coerce_score(score_item.get("plausibility", 0.5))
            field_tendency = float(candidate.get("tendency", 0.0))
            llm_cognitive_tendency = (llm_score - 0.5) * 2.0
            fused_tendency = social_lambda * field_tendency + cognitive_lambda * llm_cognitive_tendency
            item["field_probability"] = candidate.get("probability", 0.0)
            item["social_physics_tendency"] = round(field_tendency, 4)
            item["llm_score"] = round(llm_score, 4)
            item["llm_cognitive_tendency"] = round(llm_cognitive_tendency, 4)
            item["llm_score_reason"] = str(score_item.get("reason", ""))[:160]
            item["fused_tendency"] = round(fused_tendency, 4)
            item["cognitive_policy_lambda"] = round(cognitive_lambda, 4)
            item["social_physics_weight"] = round(social_lambda, 4)
            item["scoring_source"] = "dual_engine_fused"
            fused.append(item)

        probs = _softmax([float(item["fused_tendency"]) for item in fused], temperature=0.22)
        for item, prob in zip(fused, probs):
            item["probability"] = round(prob, 5)
        return fused, {
            "enabled": True,
            "source": "dual_engine_fused",
            "cognitive_policy_lambda": cognitive_lambda,
            "social_physics_weight": social_lambda,
            "raw": raw,
        }

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

        seed = int(config.get("seed", 0) or 0)
        candidates = generate_action_candidates(
            agent,
            event,
            world,
            recall,
            allowed_str,
            avoid_actions=dynamic_avoid,
            seed=seed,
        )
        base_payload = [c.to_dict() for c in candidates]
        candidate_payload, scoring_audit = self._score_candidates(
            agent,
            event,
            world,
            recall,
            base_payload,
            config,
        )
        selected_payload = _sample_payload(
            candidate_payload,
            seed=seed,
            round_num=event.round,
            agent_id=agent.id,
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
                action_candidates=candidate_payload,
                sampled_action=selected_payload,
                avoid_actions=dynamic_avoid,
                retry_note=retry_note,
                validation_error=last_error,
            )
            try:
                raw = self.llm.complete_json(ROLE_POLICY_SYSTEM, user_prompt)
                raw["primary_action"] = {
                    "type": selected_payload["type"],
                    "target": selected_payload["target"],
                    "intensity": selected_payload["intensity"],
                }
                act = _normalize_action_response(raw, agent, event, world, allowed)
                private = act.setdefault("private_intent", {})
                private.setdefault("private_motives", selected_payload.get("motives", {}))
                act["action_candidates"] = candidate_payload
                act["selected_action"] = selected_payload
                act["private_motives"] = selected_payload.get("motives", {})
                act["llm_action_scoring"] = scoring_audit
                return act
            except (LLMError, KeyError, TypeError, ValueError) as exc:
                last_error = str(exc)
                retry_note = "Keep the sampled action fixed and repair only JSON/public/private fields."

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
