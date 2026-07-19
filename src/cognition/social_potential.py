"""Social Potential Field for interpretable agent social pressure.

The field is a derived, auditable state vector. It does not replace memory,
belief, relationship, or action-field logic; it exposes a compact theory object
for Agent MRI reports and lesion-style ablations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from src.cognition.memory import RecallResult
from src.world.models import Agent, EventAtom, RelationshipEdge, WorldState, clamp

SOCIAL_POTENTIAL_DIMENSIONS = (
    "reputation_pressure",
    "trust_deficit",
    "power_constraint",
    "contribution_entitlement",
    "uncertainty",
    "memory_pressure",
)

ACTION_PRESSURE_WEIGHTS: dict[str, dict[str, float]] = {
    "ask_for_authorship": {"contribution_entitlement": 0.38, "memory_pressure": 0.22, "trust_deficit": 0.20, "power_constraint": -0.10, "uncertainty": 0.10},
    "privately_lobby_pi": {"contribution_entitlement": 0.24, "power_constraint": 0.20, "uncertainty": 0.18, "memory_pressure": 0.12, "trust_deficit": 0.08},
    "document_contribution": {"contribution_entitlement": 0.32, "uncertainty": 0.18, "reputation_pressure": 0.16, "memory_pressure": 0.12},
    "challenge_claim": {"contribution_entitlement": 0.30, "trust_deficit": 0.25, "memory_pressure": 0.20, "power_constraint": -0.12},
    "confront": {"trust_deficit": 0.26, "memory_pressure": 0.22, "contribution_entitlement": 0.20, "power_constraint": -0.14},
    "rebel": {"trust_deficit": 0.24, "memory_pressure": 0.20, "power_constraint": -0.18, "contribution_entitlement": 0.14},
    "withdraw": {"trust_deficit": 0.24, "memory_pressure": 0.18, "uncertainty": 0.14, "power_constraint": -0.16},
    "form_alliance": {"trust_deficit": 0.18, "power_constraint": 0.18, "contribution_entitlement": 0.16, "uncertainty": 0.12},
    "support_teammate": {"trust_deficit": -0.18, "uncertainty": -0.08, "power_constraint": 0.06},
    "comply": {"power_constraint": 0.30, "uncertainty": 0.14, "trust_deficit": -0.10, "memory_pressure": -0.08},
    "lay_low": {"power_constraint": 0.24, "uncertainty": 0.18, "memory_pressure": 0.08},
    "delay_response": {"power_constraint": 0.16, "uncertainty": 0.18, "memory_pressure": 0.10},
    "leak_concern": {"trust_deficit": 0.22, "memory_pressure": 0.18, "power_constraint": -0.12, "uncertainty": 0.10},
    "hide_negative_result": {"power_constraint": 0.20, "reputation_pressure": 0.20, "uncertainty": 0.16, "trust_deficit": 0.08},
    "selectively_report": {"reputation_pressure": 0.20, "power_constraint": 0.18, "uncertainty": 0.12},
}
DEFAULT_ACTION_WEIGHTS = {
    "reputation_pressure": 0.12,
    "trust_deficit": 0.08,
    "power_constraint": 0.08,
    "contribution_entitlement": 0.10,
    "uncertainty": 0.08,
    "memory_pressure": 0.08,
}


@dataclass(frozen=True)
class SocialPotential:
    agent_id: str
    round: int
    event_id: str
    target: str
    dimensions: dict[str, float]
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def total_pressure(self) -> float:
        return clamp(sum(self.dimensions.values()) / len(SOCIAL_POTENTIAL_DIMENSIONS))

    def pressure_for_action(self, action_type: str, lesions: Iterable[str] | None = None) -> float:
        disabled = set(lesions or [])
        weights = ACTION_PRESSURE_WEIGHTS.get(action_type, DEFAULT_ACTION_WEIGHTS)
        score = 0.0
        positive_weight = 0.0
        for dim in SOCIAL_POTENTIAL_DIMENSIONS:
            if dim in disabled:
                continue
            weight = float(weights.get(dim, 0.0))
            if weight > 0:
                positive_weight += weight
            score += self.dimensions.get(dim, 0.0) * weight
        # Negative weights represent inhibitory social pressure; keep the scale stable.
        norm = max(0.35, positive_weight)
        return clamp(score / norm)

    def action_decomposition(self, action_type: str, lesions: Iterable[str] | None = None) -> dict[str, Any]:
        disabled = set(lesions or [])
        weights = ACTION_PRESSURE_WEIGHTS.get(action_type, DEFAULT_ACTION_WEIGHTS)
        contributions = {
            dim: round(self.dimensions.get(dim, 0.0) * float(weights.get(dim, 0.0)), 4)
            for dim in SOCIAL_POTENTIAL_DIMENSIONS
            if dim not in disabled
        }
        return {
            "action_type": action_type,
            "lesions": sorted(disabled),
            "weights": {k: round(float(v), 4) for k, v in weights.items()},
            "contributions": contributions,
            "pressure": round(self.pressure_for_action(action_type, disabled), 4),
        }

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["total_pressure"] = round(self.total_pressure, 4)
        return data


def _edge(world: WorldState, source: str, target: str) -> RelationshipEdge | None:
    for edge in world.relationships:
        if edge.source == source and edge.target == target:
            return edge
    return None


def _target(agent: Agent, world: WorldState, event: EventAtom) -> str:
    if event.source in world.agents and event.source != agent.id:
        return event.source
    if "pi" in world.agents and agent.id != "pi":
        return "pi"
    for target in event.targets:
        if target in world.agents and target != agent.id:
            return target
    return "project"


def _contribution_share(agent: Agent, world: WorldState) -> float:
    ledger = world.project.contribution_ledger.get(agent.id, {})
    if not ledger:
        controls = agent.resources.code_control + agent.resources.data_control + agent.resources.writing_control
        return clamp(controls / 3.0)
    vals = [float(v) for v in ledger.values() if isinstance(v, (int, float))]
    return clamp(sum(vals) / max(1, len(vals)))


def _memory_pressure(agent: Agent, recall: RecallResult | None) -> float:
    if recall:
        return clamp(abs(recall.recall_field_valence) * recall.recall_field_strength)
    if not agent.memory:
        return 0.0
    recent = agent.memory[-8:]
    weighted = [abs(float(m.get("valence", 0.0))) * float(m.get("strength", 0.0)) for m in recent]
    return clamp(sum(weighted) / max(1, len(weighted)))


def compute_social_potential(
    world: WorldState,
    agent: Agent,
    event: EventAtom,
    recall: RecallResult | None = None,
    *,
    target: str | None = None,
    lesions: Iterable[str] | None = None,
) -> SocialPotential:
    """Compute a compact social-potential vector for an agent at one event.

    Lesions zero out selected dimensions for counterfactual diagnostics while
    preserving the evidence fields.
    """
    disabled = set(lesions or [])
    tgt = target or _target(agent, world, event)
    edge = _edge(world, agent.id, tgt)
    pi_edge = _edge(world, agent.id, "pi")
    project = world.project.project
    contribution_share = _contribution_share(agent, world)
    trust = edge.trust if edge else agent.beliefs.team_trust
    pi_dependency = pi_edge.dependency if pi_edge else agent.personality.authority_dependence

    dimensions = {
        "reputation_pressure": clamp(
            0.35 * (1.0 - agent.beliefs.my_contribution_recognized)
            + 0.25 * project.authorship_conflict
            + 0.20 * agent.personality.credit_sensitivity
            + 0.20 * (1.0 - agent.resources.external_network)
        ),
        "trust_deficit": clamp(
            0.40 * (1.0 - trust)
            + 0.25 * (edge.resentment if edge else agent.emotion.resentment)
            + 0.20 * agent.emotion.anger
            + 0.15 * (1.0 - project.team_morale)
        ),
        "power_constraint": clamp(
            0.35 * agent.personality.authority_dependence
            + 0.25 * pi_dependency
            + 0.20 * (1.0 - agent.resources.pi_access)
            + 0.20 * project.funding_pressure
        ),
        "contribution_entitlement": clamp(
            0.35 * contribution_share
            + 0.25 * (1.0 - agent.beliefs.my_first_author_probability)
            + 0.25 * (1.0 - agent.beliefs.my_contribution_recognized)
            + 0.15 * agent.personality.ambition
        ),
        "uncertainty": clamp(
            0.25 * (1.0 - agent.beliefs.deadline_feasibility)
            + 0.20 * project.deadline_pressure
            + 0.20 * project.rival_threat
            + 0.20 * agent.emotion.anxiety
            + 0.15 * (1.0 - agent.beliefs.project_publishability)
        ),
        "memory_pressure": _memory_pressure(agent, recall),
    }
    for dim in disabled:
        if dim in dimensions:
            dimensions[dim] = 0.0

    evidence = {
        "lesions": sorted(disabled),
        "relationship_target": tgt,
        "target_trust": round(float(trust), 4),
        "target_resentment": round(float(edge.resentment if edge else agent.emotion.resentment), 4),
        "pi_dependency": round(float(pi_dependency), 4),
        "contribution_share": round(float(contribution_share), 4),
        "authorship_conflict": round(float(project.authorship_conflict), 4),
        "deadline_pressure": round(float(project.deadline_pressure), 4),
        "rival_threat": round(float(project.rival_threat), 4),
    }
    return SocialPotential(
        agent_id=agent.id,
        round=event.round,
        event_id=event.event_id,
        target=tgt,
        dimensions={k: round(float(v), 4) for k, v in dimensions.items()},
        evidence=evidence,
    )


def summarize_action_social_potential(actions: list[dict[str, Any]]) -> dict[str, float]:
    """Summarize social-potential fields already recorded in action logs."""
    if not actions:
        return {}
    dims = {dim: [] for dim in SOCIAL_POTENTIAL_DIMENSIONS}
    selected_pressures: list[float] = []
    for action in actions:
        sp = action.get("social_potential") or {}
        for dim, value in (sp.get("dimensions") or {}).items():
            if dim in dims:
                dims[dim].append(float(value))
        if "selected_social_pressure" in action:
            selected_pressures.append(float(action["selected_social_pressure"]))
    summary = {
        f"social_potential_{dim}_mean": round(sum(vals) / len(vals), 4) if vals else 0.0
        for dim, vals in dims.items()
    }
    summary["selected_social_pressure_mean"] = round(sum(selected_pressures) / len(selected_pressures), 4) if selected_pressures else 0.0
    summary["selected_social_pressure_max"] = round(max(selected_pressures), 4) if selected_pressures else 0.0
    return summary
