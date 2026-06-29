"""Simulation run log 鈥?JSONL persistence."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.world.models import EventAtom

from src.cognition.dynamics import (
    COMPLIANCE_ACTIONS,
    action_escalation_impulse,
    combine_escalation_score,
    escalation_potential_from_state,
    saturating_memory_cluster_from_records,
)

PROTEST_SOFT_ACTIONS = {"ask_for_authorship", "privately_lobby_pi"}
PROTEST_ESCALATED_ACTIONS = {"confront", "rebel", "challenge_claim", "withdraw", "leak_concern"}
PROTEST_ACTIONS = PROTEST_SOFT_ACTIONS | PROTEST_ESCALATED_ACTIONS
HELP_REBUTTAL_ACTIONS = {"prepare_rebuttal", "support_teammate", "write_section"}
PASSIVE_ACTIONS = {"delay_response", "lay_low", "comply"}
COMPLY_ACTIONS = {"comply", "support_teammate", "document_contribution"}
REBEL_ACTIONS = {"rebel", "confront", "withdraw", "challenge_claim", "leak_concern"}
R52_COMPLIANCE_ACTIONS = COMPLIANCE_ACTIONS


@dataclass
class RunLog:
    run_id: str
    config: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    round_records: list[dict[str, Any]] = field(default_factory=list)
    outcomes: dict[str, Any] = field(default_factory=dict)
    critic_violations: list[dict[str, Any]] = field(default_factory=list)
    interventions_applied: list[dict[str, Any]] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def record_event(self, event: EventAtom, intervention_id: str | None = None) -> None:
        self.events.append({
            "event_id": event.event_id,
            "round": event.round,
            "type": event.type,
            "source": event.source,
            "intervention_id": intervention_id,
            "payload": event.payload,
        })

    def record_action(self, agent_id: str, action: dict[str, Any], round_num: int) -> None:
        self.actions.append({"round": round_num, "agent": agent_id, **action})

    def record_round(
        self,
        round_num: int,
        event_id: str,
        metrics: dict[str, float],
        agent_deltas: dict[str, Any],
        intervention_id: str | None = None,
    ) -> None:
        self.round_records.append({
            "round": round_num,
            "event_id": event_id,
            "metrics": metrics,
            "agent_deltas": agent_deltas,
            "intervention_id": intervention_id,
        })

    def write_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"type": "run_meta", "run_id": self.run_id, "config": self.config, "started_at": self.started_at}, ensure_ascii=False) + "\n")
            for rec in self.round_records:
                f.write(json.dumps({"type": "round", **rec}, ensure_ascii=False) + "\n")
            for act in self.actions:
                f.write(json.dumps({"type": "action", **act}, ensure_ascii=False) + "\n")
            f.write(json.dumps({"type": "outcomes", **self.outcomes}, ensure_ascii=False) + "\n")

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "config": self.config,
            "events": self.events,
            "actions": self.actions,
            "round_records": self.round_records,
            "outcomes": self.outcomes,
            "critic_violations": self.critic_violations,
            "interventions_applied": self.interventions_applied,
        }


def _trust_at_round(log: RunLog, source: str, target: str, round_num: int) -> float | None:
    for rec in log.round_records:
        if rec.get("round") == round_num:
            return rec.get("metrics", {}).get(f"trust_{source}_{target}")
    return None


def _memory_cluster_strength(
    log: RunLog,
    agent_id: str = "phd_a",
    content_types: tuple[str, ...] = ("authorship_signal", "promise_fulfilled", "promise_broken"),
    round_min: int = 0,
    round_max: int = 999,
) -> float:
    memories: list[dict[str, Any]] = []
    for rec in log.round_records:
        rnd = rec.get("round", 0)
        if rnd < round_min or rnd > round_max:
            continue
        mem = rec.get("agent_deltas", {}).get(agent_id, {}).get("memory_written")
        if mem and mem.get("content_type") in content_types:
            memories.append(mem)
    return saturating_memory_cluster_from_records(
        memories, round_min=round_min, round_max=round_max, current_round=round_max,
    )


def _memory_cluster_from_agent(
    agent: Any,
    content_types: tuple[str, ...] = ("authorship_signal", "promise_fulfilled", "promise_broken"),
    round_min: int = 0,
    round_max: int = 999,
) -> float:
    memories = [
        m for m in agent.memory
        if round_min <= m.get("round", 0) <= round_max and m.get("content_type") in content_types
    ]
    return saturating_memory_cluster_from_records(
        memories, round_min=round_min, round_max=round_max, current_round=round_max,
    )


def _resolve_memory_cluster_strength(
    log: RunLog,
    world_agents: dict | None = None,
    *,
    round_min: int = 3,
    round_max: int = 40,
) -> float:
    """Prefer live agent memory (reflects R45 delete); fall back to round log sum."""
    if world_agents and "phd_a" in world_agents:
        return _memory_cluster_from_agent(world_agents["phd_a"], round_min=round_min, round_max=round_max)
    return _memory_cluster_strength(log, round_min=round_min, round_max=round_max)


def _agent_state_at_round(
    log: RunLog,
    round_num: int,
    agent_id: str = "phd_a",
) -> tuple[dict[str, float], dict[str, float]]:
    for rec in log.round_records:
        if rec.get("round") == round_num:
            deltas = rec.get("agent_deltas", {}).get(agent_id, {})
            return deltas.get("beliefs", {}), deltas.get("emotion", {})
    return {}, {}


def _authorship_escalation_potential(log: RunLog, agent_id: str = "phd_a") -> float:
    beliefs, emotion = _agent_state_at_round(log, 52, agent_id)
    if not beliefs:
        beliefs, emotion = _agent_state_at_round(log, 51, agent_id)
    cluster = _memory_cluster_strength(log, agent_id, round_min=3, round_max=52)
    broken = _promise_broken_strength(log, agent_id)
    return escalation_potential_from_state(
        beliefs, emotion, promise_broken=broken, promise_cluster=cluster,
    )


def _action_escalation_impulse_sum(
    log: RunLog,
    agent: str = "phd_a",
    r_min: int = 52,
    r_max: int = 53,
) -> float:
    total = 0.0
    for act in log.actions:
        if act.get("agent") != agent:
            continue
        rnd = act.get("round", 0)
        if rnd < r_min or rnd > r_max:
            continue
        total += action_escalation_impulse(
            str(act.get("type", "")),
            float(act.get("intensity", 0.5)),
        )
    return total


def _authorship_escalation_score(log: RunLog, agent_id: str = "phd_a") -> float:
    potential = _authorship_escalation_potential(log, agent_id)
    impulse = _action_escalation_impulse_sum(log, agent_id, r_min=52, r_max=53)
    return combine_escalation_score(potential, impulse)


def _protest_stats(
    log: RunLog,
    agent: str = "phd_a",
    r_min: int = 52,
    r_max: int = 55,
) -> tuple[int, int, float]:
    escalated = soft = 0
    intensity = 0.0
    for act in log.actions:
        if act.get("agent") != agent:
            continue
        rnd = act.get("round", 0)
        if rnd < r_min or rnd > r_max:
            continue
        atype = act.get("type")
        inten = float(act.get("intensity", 0.5))
        if atype in PROTEST_ESCALATED_ACTIONS:
            escalated += 1
            intensity += inten
        elif atype in PROTEST_SOFT_ACTIONS:
            soft += 1
            intensity += inten * 0.35
    return escalated, soft, intensity


def _trust_pi_from_relationship(
    world_agents: dict | None,
    relationships: list | None,
    agent_id: str = "phd_a",
    target: str = "pi",
) -> float | None:
    if relationships:
        for edge in relationships:
            src = edge.get("source") if isinstance(edge, dict) else edge.source
            tgt = edge.get("target") if isinstance(edge, dict) else edge.target
            if src == agent_id and tgt == target:
                return float(edge.get("trust") if isinstance(edge, dict) else edge.trust)
    return None


def _promise_broken_strength(log: RunLog, agent_id: str = "phd_a", at_round: int = 52) -> float:
    """Max promise_broken memory strength written at R52 (E052 draft), excluding honored drafts."""
    strength = 0.0
    for rec in log.round_records:
        rnd = rec.get("round", 0)
        if rnd != at_round:
            continue
        mem = rec.get("agent_deltas", {}).get(agent_id, {}).get("memory_written")
        if not mem or mem.get("content_type") != "promise_broken":
            continue
        strength = max(strength, float(mem.get("strength", 0)))
    return strength


def _promise_honored_strength_r52(log: RunLog, agent_id: str = "phd_a") -> float:
    strength = 0.0
    for rec in log.round_records:
        if rec.get("round") != 52:
            continue
        mem = rec.get("agent_deltas", {}).get(agent_id, {}).get("memory_written")
        if mem and mem.get("content_type") == "promise_fulfilled":
            strength = max(strength, float(mem.get("strength", 0)))
    return strength


def _escalated_action_pressure(
    log: RunLog,
    agent: str = "phd_a",
    r_min: int = 52,
    r_max: int = 53,
) -> float:
    pressure = 0.0
    for act in log.actions:
        if act.get("agent") != agent:
            continue
        rnd = act.get("round", 0)
        if rnd < r_min or rnd > r_max:
            continue
        pressure += action_escalation_impulse(
            str(act.get("type", "")),
            float(act.get("intensity", 0.5)),
        )
    return 1.0 - math.exp(-pressure)

def _interpretation_valence_at_event(log: RunLog, event_id: str, agent_id: str = "phd_a") -> float:
    for rec in log.round_records:
        if rec.get("event_id") != event_id:
            continue
        mem = rec.get("agent_deltas", {}).get(agent_id, {}).get("memory_written")
        if mem:
            return float(mem.get("valence", 0.0))
    return 0.0


def _action_in_window(log: RunLog, agent: str, action_types: set[str], r_min: int, r_max: int) -> float:
    for act in log.actions:
        rnd = act.get("round", 0)
        if r_min <= rnd <= r_max and act.get("agent") == agent and act.get("type") in action_types:
            return 1.0
    return 0.0


def _authority_compliance(log: RunLog, agent: str = "phd_a") -> float:
    comply = rebel = 0
    for act in log.actions:
        if act.get("agent") != agent:
            continue
        if act.get("type") in COMPLY_ACTIONS:
            comply += 1
        elif act.get("type") in REBEL_ACTIONS:
            rebel += 1
    total = comply + rebel
    return comply / total if total else 0.0


def extract_outcome(log: RunLog, outcome: str) -> float:
    if outcome == "protest_authorship":
        compliance = _action_in_window(log, "phd_a", R52_COMPLIANCE_ACTIONS, 52, 53)
        score = _authorship_escalation_score(log)
        action_pressure = _escalated_action_pressure(log, r_min=52, r_max=53)
        return max(0.0, min(1.0, score * (0.65 + 0.35 * action_pressure) * (1.0 - 0.65 * compliance)))
    if outcome == "protest_intensity":
        return _authorship_escalation_score(log)
    if outcome == "authorship_escalation_potential":
        return _authorship_escalation_potential(log)
    if outcome == "authorship_escalation_score":
        return _authorship_escalation_score(log)
    if outcome == "protest_action_count":
        escalated, soft, _ = _protest_stats(log)
        return float(escalated + soft)
    if outcome == "post_r52_compliance":
        return _action_in_window(log, "phd_a", R52_COMPLIANCE_ACTIONS, 52, 53)
    if outcome == "withdraw_threat":
        return _action_in_window(log, "phd_a", {"withdraw"}, 47, 48)
    if outcome == "withdraw_threat_event":
        for ev in log.events:
            if ev.get("round") == 47 and ev.get("event_id") == "E047" and ev.get("source") == "phd_a":
                return 1.0
        return 0.0
    if outcome == "document_contribution_count":
        seen: set[int] = set()
        count = 0
        for a in log.actions:
            rnd = a.get("round", 0)
            if a.get("type") == "document_contribution" and 18 <= rnd <= 25 and rnd not in seen:
                seen.add(rnd)
                count += 1
        return float(count)
    if outcome == "authorship_dispute_index":
        for rec in reversed(log.round_records):
            if rec.get("round", 0) >= 52:
                return rec.get("metrics", {}).get("authorship_dispute_index", 0.0)
        return log.round_records[-1].get("metrics", {}).get("authorship_dispute_index", 0.0) if log.round_records else 0.0
    if outcome == "memory_authorship_cluster_strength":
        return _memory_cluster_strength(log, round_min=3, round_max=40)
    if outcome == "memory_authorship_cluster_live":
        return _memory_cluster_strength(log, round_min=3, round_max=40)
    if outcome == "promise_broken_strength_r52":
        return _promise_broken_strength(log)
    if outcome == "promise_honored_strength_r52":
        return _promise_honored_strength_r52(log)
    if outcome == "help_rebuttal":
        return _action_in_window(log, "phd_a", HELP_REBUTTAL_ACTIONS, 57, 60)
    if outcome == "demand_authorship_exchange":
        return _action_in_window(log, "phd_a", {"ask_for_authorship", "privately_lobby_pi"}, 47, 52)
    if outcome == "passive_cooperation":
        return _action_in_window(log, "phd_a", PASSIVE_ACTIONS, 52, 55)
    if outcome == "trust_phd_b_r25":
        return _trust_at_round(log, "phd_a", "phd_b", 25) or 0.0
    if outcome == "trust_phd_b_r44":
        return _trust_at_round(log, "phd_a", "phd_b", 44) or 0.0
    if outcome == "trust_phd_b_r60":
        return _trust_at_round(log, "phd_a", "phd_b", 60) or 0.0
    if outcome == "trust_recovery_rate":
        t25 = _trust_at_round(log, "phd_a", "phd_b", 25)
        t44 = _trust_at_round(log, "phd_a", "phd_b", 44)
        t60 = _trust_at_round(log, "phd_a", "phd_b", 60)
        if t25 is None or t44 is None or t60 is None:
            return 0.0
        denom = t25 - t44
        if abs(denom) < 1e-6:
            return 0.0
        return (t60 - t44) / denom
    if outcome == "pi_fairness_r35":
        for rec in log.round_records:
            if rec.get("round") == 35:
                return rec.get("agent_deltas", {}).get("phd_a", {}).get("beliefs", {}).get("pi_fairness", 0.0)
        return 0.0
    if outcome == "pi_fairness_r52":
        for rec in log.round_records:
            if rec.get("round") == 52:
                return rec.get("agent_deltas", {}).get("phd_a", {}).get("beliefs", {}).get("pi_fairness", 0.0)
        return 0.0
    if outcome == "interpretation_of_E030":
        return _interpretation_valence_at_event(log, "E030")
    if outcome == "authority_compliance":
        return _authority_compliance(log)
    if outcome == "public_private_divergence_mean":
        if not log.round_records:
            return 0.0
        vals = [r.get("metrics", {}).get("public_private_divergence", 0.0) for r in log.round_records]
        return sum(vals) / len(vals)
    return 0.0


def finalize_outcomes(log: RunLog, world_agents: dict | None = None, relationships: list | None = None) -> None:
    log.outcomes["protest_authorship"] = extract_outcome(log, "protest_authorship")
    log.outcomes["authorship_escalation_potential"] = extract_outcome(log, "authorship_escalation_potential")
    log.outcomes["authorship_escalation_score"] = extract_outcome(log, "authorship_escalation_score")
    log.outcomes["protest_intensity"] = extract_outcome(log, "protest_intensity")
    log.outcomes["protest_action_count"] = extract_outcome(log, "protest_action_count")
    log.outcomes["post_r52_compliance"] = extract_outcome(log, "post_r52_compliance")
    log.outcomes["withdraw_threat"] = extract_outcome(log, "withdraw_threat")
    log.outcomes["withdraw_threat_event"] = extract_outcome(log, "withdraw_threat_event")
    log.outcomes["document_contribution_count"] = extract_outcome(log, "document_contribution_count")
    log.outcomes["authorship_dispute_index"] = extract_outcome(log, "authorship_dispute_index")
    cluster = _resolve_memory_cluster_strength(log, world_agents, round_min=3, round_max=40)
    log.outcomes["memory_authorship_cluster_strength"] = cluster
    log.outcomes["memory_authorship_cluster_live"] = cluster
    log.outcomes["promise_broken_strength_r52"] = _promise_broken_strength(log)
    log.outcomes["promise_honored_strength_r52"] = _promise_honored_strength_r52(log)
    log.outcomes["help_rebuttal"] = extract_outcome(log, "help_rebuttal")
    log.outcomes["demand_authorship_exchange"] = extract_outcome(log, "demand_authorship_exchange")
    log.outcomes["passive_cooperation"] = extract_outcome(log, "passive_cooperation")
    log.outcomes["trust_phd_b_r25"] = extract_outcome(log, "trust_phd_b_r25")
    log.outcomes["trust_phd_b_r44"] = extract_outcome(log, "trust_phd_b_r44")
    log.outcomes["trust_phd_b_r60"] = extract_outcome(log, "trust_phd_b_r60")
    log.outcomes["trust_recovery_rate"] = extract_outcome(log, "trust_recovery_rate")
    log.outcomes["pi_fairness_r35"] = extract_outcome(log, "pi_fairness_r35")
    log.outcomes["pi_fairness_r52"] = extract_outcome(log, "pi_fairness_r52")
    log.outcomes["interpretation_of_E030"] = extract_outcome(log, "interpretation_of_E030")
    log.outcomes["authority_compliance"] = extract_outcome(log, "authority_compliance")
    log.outcomes["public_private_divergence_mean"] = extract_outcome(log, "public_private_divergence_mean")
    if world_agents and "phd_a" in world_agents:
        rel_trust = _trust_pi_from_relationship(world_agents, relationships, "phd_a", "pi")
        agent = world_agents["phd_a"]
        if rel_trust is not None:
            log.outcomes["trust_pi_final"] = rel_trust
        else:
            log.outcomes["trust_pi_final"] = round(
                0.55 * agent.beliefs.pi_fairness + 0.45 * agent.beliefs.team_trust, 4,
            )
        log.outcomes["pi_fairness_r60"] = agent.beliefs.pi_fairness
        log.outcomes["pi_trust_belief_final"] = agent.beliefs.pi_fairness
        if relationships is not None:
            class _WorldProxy:
                pass
            proxy = _WorldProxy()
            proxy.agents = world_agents
            proxy.relationships = relationships
            # project is unavailable here; use final logged metrics proxy if full world was not passed.
        log.outcomes.setdefault("career_hostage_index", 0.0)


