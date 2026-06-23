"""Critic Agent — action consistency audit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.world.actions import ActionType, get_allowed_actions
from src.world.models import Agent, WorldState


@dataclass
class Violation:
    code: str
    severity: str  # hard | soft
    message: str


class CriticAgent:
    def check(self, action: dict[str, Any], agent: Agent, world: WorldState) -> list[Violation]:
        violations: list[Violation] = []
        atype = action.get("type")
        intensity = float(action.get("intensity", 0.5))

        allowed = {a.value for a in get_allowed_actions(agent.id, agent.emotion.burnout)}
        if atype not in allowed:
            violations.append(Violation("illegal_action", "hard", f"{atype} not in allowed set for {agent.id}"))

        if not (0.0 <= intensity <= 1.0):
            violations.append(Violation("intensity_range", "hard", f"intensity {intensity} out of [0,1]"))

        if agent.id == "engineer_e" and atype == "ask_for_authorship":
            violations.append(Violation("role_consistency", "hard", "engineer_e should not ask_for_authorship"))

        if atype == "cite_prior_memory" and not agent.memory:
            violations.append(Violation("memory_consistency", "hard", "cite_prior_memory with empty memory"))

        public = action.get("public_position", {})
        if (
            public.get("statement_type") == "team_support"
            and action.get("type") == "undermine_teammate"
            and agent.personality.deceptiveness < 0.6
        ):
            violations.append(Violation("public_private_conflict", "soft", "team_support public with undermine action"))

        recent = [h.get("action", {}).get("type") for h in agent.action_history[-2:]]
        if len(recent) == 2 and all(r == atype for r in recent):
            violations.append(Violation("action_repetition", "soft", f"{atype} repeated 3 rounds"))

        return violations

    def fix_or_reject(
        self,
        action: dict[str, Any],
        agent: Agent,
        violations: list[Violation],
    ) -> tuple[dict[str, Any], list[Violation]]:
        hard = [v for v in violations if v.severity == "hard"]
        if not hard:
            return action, violations

        fallback_pool = ["comply", "document_contribution", "share_result", "write_section", "seek_validation"]
        allowed = {a.value for a in get_allowed_actions(agent.id, agent.emotion.burnout)}
        fallback = next((f for f in fallback_pool if f in allowed), "comply" if "comply" in allowed else list(allowed)[0])

        fixed = dict(action)
        fixed["type"] = fallback
        fixed["intensity"] = min(float(action.get("intensity", 0.5)), 0.6)
        fixed["_critic_fallback"] = True
        return fixed, violations
