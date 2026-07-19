"""LabWars simulation engine 鈥?main loop."""

from __future__ import annotations

import copy
import random
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.cognition.pipeline import commit_cognition_phase, pre_decision_recall
from src.cognition.power import career_hostage_index, pi_control_pressure, pi_control_surface
from src.engine.critic import CriticAgent
from src.engine.event_agent import EventAgent, is_agent_active
from src.engine.intervention import (
    Intervention,
    apply_event_override,
    apply_world_intervention,
    get_active_interventions,
    load_interventions,
)
from src.engine.llm_adapter import LLMAdapter, get_adapter, load_llm_config
from src.engine.probe import ProbeAgent
from src.engine.role_policy import RolePolicyAgent
from src.engine.run_log import RunLog, finalize_outcomes
from src.world.actions import ActionType, apply_project_effects
from src.world.loader import PROJECT_ROOT, load_world
from src.world.models import ProjectMetrics, WorldState
from src.world.population import PopulationSpec, expand_population

CONFIG_DIR = PROJECT_ROOT / "config"


@dataclass
class SimConfig:
    max_rounds: int = 60
    seed: int = 0
    active_agents: list[str] | None = None
    offstage_agents: list[str] | None = None
    interventions: list[Intervention] = field(default_factory=list)
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_temperature: float | None = None
    llm_adapter: LLMAdapter | None = None
    run_id: str | None = None
    output_dir: Path | None = None
    mvp: bool = False
    disable_memory: bool = False
    shuffle_memory: bool = False
    disable_state_events: bool = False
    experiment_id: str | None = None
    condition_id: str | None = None
    policy_mode: str = "dual_engine"
    enable_llm_action_scoring: bool = True
    cognitive_policy_lambda: float | None = 0.35
    llm_action_score_mix: float = 0.35
    hierarchy_lesion: bool = False
    status_lesion: bool = False
    trust_lesion: bool = False
    population_size: int | None = None
    population_labs: int | None = None
    cognitive_sampling_top_k: int | None = None
    cognitive_sampling_threshold: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        llm_cfg = load_llm_config()
        return {
            "max_rounds": self.max_rounds,
            "seed": self.seed,
            "active_agents": self.active_agents,
            "offstage_agents": self.offstage_agents,
            "interventions": [i.intervention_id for i in self.interventions],
            "mvp": self.mvp,
            "disable_memory": self.disable_memory,
            "shuffle_memory": self.shuffle_memory,
            "disable_state_events": self.disable_state_events,
            "experiment_id": self.experiment_id,
            "condition_id": self.condition_id,
            "policy_mode": self.policy_mode,
            "enable_llm_action_scoring": self.enable_llm_action_scoring,
            "cognitive_policy_lambda": self.cognitive_policy_lambda,
            "llm_action_score_mix": self.llm_action_score_mix,
            "hierarchy_lesion": self.hierarchy_lesion,
            "status_lesion": self.status_lesion,
            "trust_lesion": self.trust_lesion,
            "population_size": self.population_size,
            "population_labs": self.population_labs,
            "cognitive_sampling_top_k": self.cognitive_sampling_top_k,
            "cognitive_sampling_threshold": self.cognitive_sampling_threshold,
            "llm_provider": self.llm_provider or llm_cfg.get("provider"),
            "llm_model": self.llm_model or llm_cfg.get("model"),
        }


def load_mvp_config() -> SimConfig:
    path = CONFIG_DIR / "mvp.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    intervention_ids = data.get("default_interventions", [])
    all_interventions = {i.intervention_id: i for i in load_interventions()}
    return SimConfig(
        max_rounds=data.get("max_rounds", 20),
        active_agents=data.get("active_agents"),
        offstage_agents=data.get("offstage_agents", ["rival_lab_h"]),
        mvp=True,
        interventions=[all_interventions[iid] for iid in intervention_ids if iid in all_interventions],
    )


def _filter_world(world: WorldState, config: SimConfig) -> WorldState:
    if not config.active_agents:
        return world
    w = copy.deepcopy(world)
    keep = set(config.active_agents) | set(config.offstage_agents or [])
    w.agents = {k: v for k, v in w.agents.items() if k in keep}
    internal = [a for a in w.world_config.get("internal_agents", []) if a in keep]
    w.world_config["internal_agents"] = internal
    w.relationships = [e for e in w.relationships if e.source in keep and e.target in keep]
    return w



def _apply_hierarchy_lesion(world: WorldState) -> WorldState:
    """Flatten PI-centered authority while preserving the same event stream.

    This is a mechanism lesion for organization-level ablations: it lowers
    authority dependence, PI dependency, and funding pressure while increasing
    alternative access. It deliberately avoids deleting agents or events.
    """
    w = copy.deepcopy(world)
    for agent in w.agents.values():
        agent.personality.authority_dependence = min(agent.personality.authority_dependence, 0.18)
        agent.resources.pi_access = max(agent.resources.pi_access, 0.72)
        agent.resources.external_network = max(agent.resources.external_network, 0.68)
        agent.beliefs.pi_fairness = max(agent.beliefs.pi_fairness, 0.52)
    for edge in w.relationships:
        if edge.target == "pi" or edge.source == "pi":
            edge.dependency = min(edge.dependency, 0.18)
            edge.obligation = min(edge.obligation, 0.25)
            edge.information_access = max(edge.information_access, 0.58)
    w.project.project.funding_pressure = min(w.project.project.funding_pressure, 0.20)
    w.world_config["hierarchy_lesion"] = True
    return w



def _apply_status_lesion(world: WorldState) -> WorldState:
    """Remove status/credit-attribution incentives while preserving task pressure."""
    w = copy.deepcopy(world)
    internal = [aid for aid in w.world_config.get("internal_agents", []) if aid in w.agents]
    for agent in w.agents.values():
        agent.personality.credit_sensitivity = min(agent.personality.credit_sensitivity, 0.18)
        agent.personality.ambition = min(agent.personality.ambition, 0.48)
        agent.beliefs.my_first_author_probability = 0.35
        agent.beliefs.my_contribution_recognized = max(agent.beliefs.my_contribution_recognized, 0.78)
        agent.beliefs.others_are_free_riding = min(agent.beliefs.others_are_free_riding, 0.08)
    for edge in w.relationships:
        edge.perceived_credit_threat = min(edge.perceived_credit_threat, 0.06)
    for dimension, ledger in w.project.contribution_ledger.items():
        keys = [aid for aid in internal if aid in w.agents]
        if not keys:
            continue
        share = round(1.0 / len(keys), 6)
        w.project.contribution_ledger[dimension] = {aid: share for aid in keys}
    w.project.project.authorship_conflict = min(w.project.project.authorship_conflict, 0.06)
    w.world_config["status_lesion"] = True
    return w


def _apply_trust_lesion(world: WorldState) -> WorldState:
    """Cut trust/alliance as a state channel while leaving other social pressure intact."""
    w = copy.deepcopy(world)
    for agent in w.agents.values():
        agent.beliefs.team_trust = 0.50
    for edge in w.relationships:
        edge.trust = 0.50
        edge.resentment = 0.0
        edge.alliance = 0.0
        edge.last_interaction_valence = 0.0
    w.world_config["trust_lesion"] = True
    return w

def _shuffle_memory_refs(world: WorldState, seed: int) -> None:
    rng = random.Random(seed)
    for agent in world.agents.values():
        if len(agent.memory) < 2:
            continue
        refs = [m.get("event_ref", "E000") for m in agent.memory]
        shuffled = refs[:]
        rng.shuffle(shuffled)
        for mem, ref in zip(agent.memory, shuffled):
            mem["event_ref"] = ref


def _relationship_snapshot(world: WorldState) -> dict[str, float]:
    snap: dict[str, float] = {}
    for edge in world.relationships:
        snap[f"trust_{edge.source}_{edge.target}"] = edge.trust
        snap[f"resentment_{edge.source}_{edge.target}"] = edge.resentment
    return snap


def _resolve_llm(cfg: SimConfig) -> LLMAdapter:
    if cfg.llm_adapter is not None:
        return cfg.llm_adapter
    return get_adapter(
        provider=cfg.llm_provider,
        model=cfg.llm_model,
        temperature=cfg.llm_temperature,
    )


def run_simulation(config: SimConfig | None = None) -> RunLog:
    cfg = config or SimConfig()
    if cfg.mvp:
        mvp = load_mvp_config()
        cfg.active_agents = cfg.active_agents or mvp.active_agents
        cfg.offstage_agents = cfg.offstage_agents or mvp.offstage_agents
        if cfg.max_rounds == 60:
            cfg.max_rounds = mvp.max_rounds

    run_id = cfg.run_id or str(uuid.uuid4())[:8]
    log = RunLog(run_id=run_id, config=cfg.to_dict())

    world = _filter_world(load_world(), cfg)
    if cfg.population_size:
        world = expand_population(world, PopulationSpec(target_size=cfg.population_size, seed=cfg.seed, labs=cfg.population_labs))
    if cfg.hierarchy_lesion:
        world = _apply_hierarchy_lesion(world)
    if cfg.status_lesion:
        world = _apply_status_lesion(world)
    if cfg.trust_lesion:
        world = _apply_trust_lesion(world)
    llm = _resolve_llm(cfg)
    event_agent = EventAgent(seed=cfg.seed, state_events=not cfg.disable_state_events)
    policy = RolePolicyAgent(llm=llm)
    critic = CriticAgent()
    probe = ProbeAgent()

    sim_config_dict = {
        "seed": cfg.seed,
        "policy_mode": cfg.policy_mode,
        "enable_llm_action_scoring": cfg.enable_llm_action_scoring,
        "cognitive_policy_lambda": cfg.cognitive_policy_lambda,
        "llm_action_score_mix": cfg.llm_action_score_mix,
        "active_agents": cfg.active_agents,
        "offstage_agents": cfg.offstage_agents,
        "offstage_min_round": 21,
        "hierarchy_lesion": cfg.hierarchy_lesion,
        "status_lesion": cfg.status_lesion,
        "trust_lesion": cfg.trust_lesion,
        "population_size": cfg.population_size,
        "population_labs": cfg.population_labs,
        "cognitive_sampling_top_k": cfg.cognitive_sampling_top_k,
        "cognitive_sampling_threshold": cfg.cognitive_sampling_threshold,
    }

    for round_num in range(1, cfg.max_rounds + 1):
        active_inters = get_active_interventions(cfg.interventions, round_num)

        event = event_agent.generate(round_num, world)
        if event is None:
            continue

        intervention_id = None
        for inter in active_inters:
            if inter.skip_event:
                event = None
                break
            if inter.type == "memory_intervention":
                removed = apply_world_intervention(world, inter)
                intervention_id = inter.intervention_id
                entry = {"round": round_num, **asdict(inter)}
                if removed:
                    entry["memories_removed"] = removed
                log.interventions_applied.append(entry)
            elif inter.target_event is None or inter.target_event == event.event_id:
                event = apply_event_override(event, inter)
                intervention_id = inter.intervention_id
                log.interventions_applied.append({"round": round_num, **asdict(inter)})

        if event is None:
            continue

        if cfg.shuffle_memory:
            _shuffle_memory_refs(world, cfg.seed + round_num)

        log.record_event(event, intervention_id)

        recalls = pre_decision_recall(world, event, disable_memory=cfg.disable_memory)
        raw_actions = policy.decide_all(world, event, recalls, sim_config_dict)

        vetted_actions: list[dict[str, Any]] = []
        for act in raw_actions:
            agent = world.agents[act["agent"]]
            violations = critic.check(act, agent, world)
            if violations:
                log.critic_violations.extend([
                    {"round": round_num, "agent": act["agent"], **v.__dict__} for v in violations
                ])
                act, _ = critic.fix_or_reject(act, agent, violations)
            vetted_actions.append(act)
            log.record_action(act["agent"], act, round_num)

        cog = commit_cognition_phase(
            world,
            event,
            recalls,
            actions=vetted_actions,
            disable_memory=cfg.disable_memory,
            llm_adapter=llm,
        )
        if cfg.status_lesion:
            world = _apply_status_lesion(world)
        if cfg.trust_lesion:
            world = _apply_trust_lesion(world)

        for atype in (ActionType.RUN_EXPERIMENT, ActionType.DEBUG_CODE, ActionType.WRITE_SECTION):
            pass  # project effects applied below

        for act in vetted_actions:
            try:
                action_enum = ActionType(act["type"])
                pd = world.project.project.model_dump()
                pd = apply_project_effects(pd, action_enum)
                world.project.project = ProjectMetrics(**pd)
            except ValueError:
                continue

        if event.type == "submission_decision" and event.payload.get("submission_status") == "submitted":
            world.project.submission_status = "submitted"
        if event.type == "authorship_draft":
            world.project.author_order_draft = event.payload.get("author_order", [])

        metrics = {
            **cog.metrics,
            "career_hostage_index": career_hostage_index(world),
            "pi_control_pressure_phd_a": pi_control_pressure(world, world.agents.get("phd_a")),
            "integrity_risk": world.project.project.integrity_risk,
            **_relationship_snapshot(world),
        }
        log.record_round(round_num, event.event_id, metrics, cog.agent_deltas, intervention_id)

        if world.project.submission_status == "accepted":
            break

    finalize_outcomes(log, world.agents, world.relationships)
    log.outcomes["career_hostage_index"] = career_hostage_index(world)
    log.outcomes["pi_control_surface"] = pi_control_surface(world)
    from src.experiments.social_metrics import compute_social_emergence_metrics
    log.outcomes.update(compute_social_emergence_metrics(log))
    log.outcomes["probe_suggestions"] = probe.suggest(log.round_records)

    if cfg.output_dir:
        log.write_jsonl(Path(cfg.output_dir) / f"run_{run_id}.jsonl")

    return log
