"""Named social-pressure projections of the Agent Social State.

These fields do not replace SocialPotential. They are auditable, lesionable
projections used in Agent MRI reports:

- AuthorshipPressureField
- TrustCollapseField
- AuthorityComplianceField
- IntegrityRiskField
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.cognition.memory import RecallResult
from src.cognition.social_potential import SocialPotential, compute_social_potential
from src.world.models import Agent, EventAtom, WorldState, clamp
from src.world.organization import agent_contribution_share, primary_authority


@dataclass(frozen=True)
class NamedPressureField:
    name: str
    value: float
    terms: dict[str, float]
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["value"] = round(self.value, 4)
        data["terms"] = {k: round(float(v), 4) for k, v in self.terms.items()}
        return data


def _edge(world: WorldState, source: str, target: str):
    for edge in world.relationships:
        if edge.source == source and edge.target == target:
            return edge
    return None


def _promise_violation(recall: RecallResult | None) -> float:
    if not recall:
        return 0.0
    return clamp(max(0.0, -recall.recall_field_valence) * recall.recall_field_strength)


def authorship_pressure_field(
    world: WorldState,
    agent: Agent,
    event: EventAtom,
    recall: RecallResult | None = None,
    potential: SocialPotential | None = None,
) -> NamedPressureField:
    potential = potential or compute_social_potential(world, agent, event, recall)
    dims = potential.dimensions
    share = agent_contribution_share(world, agent.id)
    entitlement_gap = clamp(share - agent.beliefs.my_first_author_probability)
    fairness_gap = clamp(1.0 - agent.beliefs.my_contribution_recognized)
    coalition = 0.0
    for edge in world.relationships:
        if edge.source == agent.id:
            coalition = max(coalition, edge.alliance * edge.trust)
    terms = {
        "perceived_contribution_share": 0.16 * share,
        "authorship_entitlement_gap": 0.18 * entitlement_gap,
        "recognition_fairness_gap": 0.14 * fairness_gap,
        "promise_violation_memory": 0.14 * _promise_violation(recall),
        "contribution_entitlement": 0.16 * dims.get("contribution_entitlement", 0.0),
        "memory_pressure": 0.10 * dims.get("memory_pressure", 0.0),
        "coalition_support": 0.08 * coalition,
        "career_dependency_brake": -0.10 * dims.get("power_constraint", 0.0),
    }
    return NamedPressureField(
        name="AuthorshipPressureField",
        value=clamp(sum(terms.values())),
        terms=terms,
        evidence={
            "contribution_share": round(share, 4),
            "first_author_belief": round(agent.beliefs.my_first_author_probability, 4),
            "event_id": event.event_id,
        },
    )


def trust_collapse_field(
    world: WorldState,
    agent: Agent,
    event: EventAtom,
    recall: RecallResult | None = None,
    potential: SocialPotential | None = None,
) -> NamedPressureField:
    potential = potential or compute_social_potential(world, agent, event, recall)
    dims = potential.dimensions
    authority = primary_authority(world, agent)
    edge = _edge(world, agent.id, potential.target)
    pi_edge = _edge(world, agent.id, authority) if authority else None
    rumor = 1.0 if event.visibility == "bilateral" or event.truth_status in {"rumored", "disputed"} else 0.0
    terms = {
        "trust_deficit": 0.28 * dims.get("trust_deficit", 0.0),
        "betrayal_salience": 0.18 * _promise_violation(recall),
        "target_resentment": 0.14 * (edge.resentment if edge else agent.emotion.resentment),
        "repeated_ambiguity": 0.12 * (1.0 if event.framing == "ambiguous" else 0.35 * event.memory_salience),
        "third_party_rumor": 0.10 * rumor,
        "dependency_asymmetry": 0.10 * (pi_edge.dependency if pi_edge else agent.personality.authority_dependence),
        "team_morale_loss": 0.08 * (1.0 - world.project.project.team_morale),
    }
    return NamedPressureField(
        name="TrustCollapseField",
        value=clamp(sum(terms.values())),
        terms=terms,
        evidence={
            "target": potential.target,
            "truth_status": event.truth_status,
            "visibility": event.visibility,
        },
    )


def authority_compliance_field(
    world: WorldState,
    agent: Agent,
    event: EventAtom,
    recall: RecallResult | None = None,
    potential: SocialPotential | None = None,
) -> NamedPressureField:
    potential = potential or compute_social_potential(world, agent, event, recall)
    dims = potential.dimensions
    authority = primary_authority(world, agent)
    pi_edge = _edge(world, agent.id, authority) if authority else None
    coalition_protection = max((e.alliance for e in world.relationships if e.source == agent.id), default=0.0)
    terms = {
        "power_constraint": 0.28 * dims.get("power_constraint", 0.0),
        "career_hostage": 0.18 * agent.personality.authority_dependence,
        "low_alternative_access": 0.14 * (1.0 - agent.resources.external_network),
        "pi_dependency": 0.14 * (pi_edge.dependency if pi_edge else 0.35),
        "uncertainty": 0.10 * dims.get("uncertainty", 0.0),
        "private_resentment": 0.08 * agent.emotion.resentment,
        "coalition_protection_brake": -0.08 * coalition_protection,
        "loyalty": 0.08 * agent.emotion.loyalty,
    }
    return NamedPressureField(
        name="AuthorityComplianceField",
        value=clamp(sum(terms.values())),
        terms=terms,
        evidence={
            "authority": authority,
            "pi_access": round(agent.resources.pi_access, 4),
            "external_network": round(agent.resources.external_network, 4),
        },
    )


def integrity_risk_field(
    world: WorldState,
    agent: Agent,
    event: EventAtom,
    recall: RecallResult | None = None,
    potential: SocialPotential | None = None,
) -> NamedPressureField:
    potential = potential or compute_social_potential(world, agent, event, recall)
    dims = potential.dimensions
    project = world.project.project
    terms = {
        "deadline_pressure": 0.20 * project.deadline_pressure,
        "authorship_threat": 0.16 * dims.get("contribution_entitlement", 0.0),
        "pi_control_pressure": 0.14 * dims.get("power_constraint", 0.0),
        "external_competition": 0.12 * project.rival_threat,
        "low_reproducibility": 0.12 * (1.0 - project.code_stability),
        "uncertainty": 0.10 * dims.get("uncertainty", 0.0),
        "reputation_pressure": 0.10 * dims.get("reputation_pressure", 0.0),
        "belief_integrity_risk": 0.10 * agent.beliefs.academic_integrity_risk,
        "integrity_brake": -0.08 * agent.personality.cooperation,
    }
    return NamedPressureField(
        name="IntegrityRiskField",
        value=clamp(sum(terms.values())),
        terms=terms,
        evidence={
            "integrity_risk": round(project.integrity_risk, 4),
            "deadline_pressure": round(project.deadline_pressure, 4),
            "rival_threat": round(project.rival_threat, 4),
        },
    )


def compute_pressure_fields(
    world: WorldState,
    agent: Agent,
    event: EventAtom,
    recall: RecallResult | None = None,
    potential: SocialPotential | None = None,
) -> dict[str, Any]:
    potential = potential or compute_social_potential(world, agent, event, recall)
    fields = [
        authorship_pressure_field(world, agent, event, recall, potential),
        trust_collapse_field(world, agent, event, recall, potential),
        authority_compliance_field(world, agent, event, recall, potential),
        integrity_risk_field(world, agent, event, recall, potential),
    ]
    return {field.name: field.to_dict() for field in fields}


def summarize_pressure_fields(actions: list[dict[str, Any]]) -> dict[str, float]:
    names = (
        "AuthorshipPressureField",
        "TrustCollapseField",
        "AuthorityComplianceField",
        "IntegrityRiskField",
    )
    buckets = {name: [] for name in names}
    for action in actions:
        recorded = action.get("pressure_fields") or {}
        for name in names:
            item = recorded.get(name) or {}
            if "value" in item:
                buckets[name].append(float(item["value"]))
    summary: dict[str, float] = {}
    for name, values in buckets.items():
        key = name.replace("Field", "").lower()
        summary[f"{key}_mean"] = round(sum(values) / len(values), 4) if values else 0.0
        summary[f"{key}_max"] = round(max(values), 4) if values else 0.0
    return summary
