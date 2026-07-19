"""Scalable population synthesis for Agent Social Dynamics experiments."""

from __future__ import annotations

import copy
import hashlib
import random
from dataclasses import dataclass

from src.world.loader import build_initial_relationships
from src.world.models import Agent, AgentRole, RelationshipEdge, WorldState, clamp


INTERNAL_ARCHETYPES = ("phd_a", "phd_b", "master_c", "postdoc_d", "engineer_e", "visiting_f")
EXTERNAL_ARCHETYPES = ("collaborator_g", "reviewer_1", "reviewer_2", "reviewer_3", "program_officer", "lab_alumni")


@dataclass(frozen=True)
class PopulationSpec:
    target_size: int
    seed: int = 0
    hierarchy: bool = True
    labs: int | None = None
    external_fraction: float = 0.12
    egalitarian: bool = False


def _rng_for(seed: int, *parts: object) -> random.Random:
    raw = ":".join(str(p) for p in (seed, *parts))
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return random.Random(int(h[:16], 16))


def _jitter(value: float, rng: random.Random, width: float = 0.08) -> float:
    return round(clamp(float(value) + rng.uniform(-width, width)), 4)


def _clone_agent(template: Agent, *, new_id: str, display_name: str, seed: int, lab_id: int) -> Agent:
    rng = _rng_for(seed, new_id)
    agent = copy.deepcopy(template)
    agent.id = new_id
    agent.display_name = display_name
    agent.memory = []
    agent.memory_recall_log = []
    agent.action_history = []
    agent.public_position = {}
    agent.private_intent = {}
    agent.extra_traits = dict(agent.extra_traits)
    agent.extra_traits["synthetic_population"] = True
    agent.extra_traits["archetype"] = template.id
    agent.extra_traits["lab_id"] = lab_id

    for field in type(agent.personality).model_fields:
        setattr(agent.personality, field, _jitter(getattr(agent.personality, field), rng))
    for field in type(agent.beliefs).model_fields:
        setattr(agent.beliefs, field, _jitter(getattr(agent.beliefs, field), rng))
    for field in type(agent.emotion).model_fields:
        setattr(agent.emotion, field, _jitter(getattr(agent.emotion, field), rng))
    for field in type(agent.resources).model_fields:
        setattr(agent.resources, field, _jitter(getattr(agent.resources, field), rng))
    return agent


def _make_pi(template: Agent, *, lab_id: int, seed: int) -> Agent:
    if lab_id == 1:
        pi = copy.deepcopy(template)
        pi.extra_traits = dict(pi.extra_traits)
        pi.extra_traits.setdefault("lab_id", 1)
        return pi
    return _clone_agent(
        template,
        new_id=f"pi_lab_{lab_id}",
        display_name=f"PI Lab {lab_id}",
        seed=seed,
        lab_id=lab_id,
    )


def _role_prefix(role: AgentRole) -> str:
    return {
        AgentRole.IDEA_ORIGINATOR: "phd_idea",
        AgentRole.EXPERIMENTER: "phd_exp",
        AgentRole.MASTER_STUDENT: "master",
        AgentRole.POSTDOC: "postdoc",
        AgentRole.ENGINEER: "engineer",
        AgentRole.VISITING_STUDENT: "visiting",
        AgentRole.COLLABORATOR: "collab",
        AgentRole.REVIEWER: "reviewer",
        AgentRole.PROGRAM_OFFICER: "program",
        AgentRole.ALUMNI: "alumni",
        AgentRole.RIVAL_LAB: "rival",
        AgentRole.PI: "pi",
    }.get(role, "agent")


def _synthetic_id(template: Agent, idx: int, lab_id: int) -> str:
    return f"{_role_prefix(template.role)}_{idx:03d}_lab_{lab_id}"


def _relationship_with_labs(edge: RelationshipEdge, agents: dict[str, Agent]) -> RelationshipEdge:
    src_lab = int(agents[edge.source].extra_traits.get("lab_id", 1))
    tgt_lab = int(agents[edge.target].extra_traits.get("lab_id", 1))
    if src_lab == tgt_lab:
        return edge
    adjusted = copy.deepcopy(edge)
    adjusted.trust = round(max(0.15, adjusted.trust - 0.18), 4)
    adjusted.communication_frequency = round(max(0.08, adjusted.communication_frequency - 0.20), 4)
    adjusted.information_access = round(max(0.08, adjusted.information_access - 0.18), 4)
    adjusted.perceived_credit_threat = round(min(1.0, adjusted.perceived_credit_threat + 0.12), 4)
    adjusted.dependency = round(max(0.04, adjusted.dependency - 0.18), 4)
    return adjusted




def equalize_population(world: WorldState) -> WorldState:
    """Set all agents and internal edges to equal initial social conditions.

    This is an anti-script intervention: any later hierarchy, inequality, or
    coalition structure must come from endogenous dynamics rather than unequal
    initial status, resource, or network conditions.
    """
    w = copy.deepcopy(world)
    internal = [aid for aid in w.world_config.get("internal_agents", []) if aid in w.agents]
    for agent in w.agents.values():
        for field in type(agent.personality).model_fields:
            setattr(agent.personality, field, 0.50)
        agent.personality.cooperation = 0.55
        agent.personality.conflict_avoidance = 0.45
        for field in type(agent.beliefs).model_fields:
            setattr(agent.beliefs, field, 0.50)
        agent.beliefs.my_contribution_recognized = 0.50
        agent.beliefs.others_are_free_riding = 0.0
        for field in type(agent.emotion).model_fields:
            setattr(agent.emotion, field, 0.30)
        agent.emotion.confidence = 0.50
        agent.emotion.hope = 0.50
        agent.emotion.loyalty = 0.50
        for field in type(agent.resources).model_fields:
            setattr(agent.resources, field, 0.50)
        agent.memory = []
        agent.memory_recall_log = []
        agent.action_history = []
        agent.public_position = {}
        agent.private_intent = {}
        agent.extra_traits = dict(agent.extra_traits)
        agent.extra_traits["egalitarian_initialization"] = True
    for edge in w.relationships:
        edge.trust = 0.50
        edge.resentment = 0.0
        edge.dependency = 0.50
        edge.obligation = 0.0
        edge.perceived_credit_threat = 0.0
        edge.communication_frequency = 0.50
        edge.alliance = 0.0
        edge.information_access = 0.50
        edge.last_interaction_valence = 0.0
    for dimension in list(w.project.contribution_ledger):
        if internal:
            share = round(1.0 / len(internal), 6)
            w.project.contribution_ledger[dimension] = {aid: share for aid in internal}
    w.world_config["egalitarian_initialization"] = True
    return w

def expand_population(world: WorldState, spec: PopulationSpec) -> WorldState:
    """Return a deterministic larger organization from the canonical LabWars world.

    The canonical hand-authored 14-agent world remains the default. This function
    creates a larger, hierarchical population for scale tests without adding new
    narrative scripts: synthetic agents inherit role archetypes but receive
    deterministic trait jitter and lab membership.
    """
    if spec.target_size <= len(world.agents):
        w = copy.deepcopy(world)
        w.world_config["population_size"] = len(w.agents)
        w.world_config["population_synthesis"] = "canonical"
        if spec.egalitarian:
            w = equalize_population(w)
        return w

    w = copy.deepcopy(world)
    templates = w.agents
    labs = spec.labs or max(1, min(12, round(spec.target_size / 18))) if spec.hierarchy else 1
    labs = max(1, int(labs))

    agents: dict[str, Agent] = {}
    internal: list[str] = []
    external: list[str] = []

    for lab_id in range(1, labs + 1):
        pi = _make_pi(templates["pi"], lab_id=lab_id, seed=spec.seed)
        agents[pi.id] = pi
        internal.append(pi.id)

    internal_templates = [templates[k] for k in INTERNAL_ARCHETYPES if k in templates]
    external_templates = [templates[k] for k in EXTERNAL_ARCHETYPES if k in templates]
    next_idx = 1
    while len(agents) < spec.target_size:
        external_slot = len(external) < int(spec.target_size * spec.external_fraction)
        pool = external_templates if external_slot and external_templates else internal_templates
        template = pool[(next_idx - 1) % len(pool)]
        lab_id = ((next_idx - 1) % labs) + 1
        agent_id = _synthetic_id(template, next_idx, lab_id)
        if agent_id in agents:
            next_idx += 1
            continue
        agent = _clone_agent(
            template,
            new_id=agent_id,
            display_name=f"{template.role.value}-{next_idx:03d} / lab {lab_id}",
            seed=spec.seed,
            lab_id=lab_id,
        )
        agents[agent.id] = agent
        if agent.role in {AgentRole.REVIEWER, AgentRole.PROGRAM_OFFICER, AgentRole.COLLABORATOR, AgentRole.ALUMNI, AgentRole.RIVAL_LAB}:
            external.append(agent.id)
        else:
            internal.append(agent.id)
        next_idx += 1

    w.agents = agents
    w.world_config["internal_agents"] = internal
    w.world_config["external_agents"] = external
    w.world_config["population_size"] = len(agents)
    w.world_config["population_labs"] = labs
    w.world_config["population_synthesis"] = "deterministic_archetype_jitter"
    w.relationships = [
        _relationship_with_labs(edge, agents)
        for edge in build_initial_relationships(internal)
    ]

    for ledger in w.project.contribution_ledger.values():
        total = sum(float(v) for v in ledger.values()) or 1.0
        for key in list(ledger):
            ledger[key] = round(float(ledger[key]) / total, 4)
    if spec.egalitarian:
        w = equalize_population(w)
    return w

