"""Unified Agent Organization MRI ablation matrix."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.engine.run_log import RunLog, extract_outcome
from src.engine.simulation import SimConfig, run_simulation
from src.experiments.metrics import compute_run_metrics

ORGANIZATION_ABLATION_CONDITIONS = [
    "full",
    "memory_lesion",
    "hierarchy_lesion",
    "status_lesion",
    "trust_lesion",
    "observation_lesion",
    "social_physics_only",
    "llm_native",
    "llm_scoring_off",
]

DEFAULT_ORGANIZATION_OUTCOMES = [
    "authorship_dispute_index",
    "trust_fragmentation",
    "coalition_strength",
    "public_private_divergence_mean",
    "memory_authorship_cluster_strength",
    "selected_social_pressure_mean",
    "selected_social_pressure_max",
    "career_hostage_index",
    "integrity_risk",
    "llm_override_pressure",
    "trust_entropy",
    "power_concentration_gini",
    "alliance_modularity_proxy",
    "conflict_cascade_length",
    "reputation_volatility",
    "credit_attribution_gap",
    "social_state_volatility",
    "organization_fragility_index",
]


@dataclass
class OrganizationAblationResult:
    conditions: list[str]
    outcomes: list[str]
    n_per_condition: int
    summary: dict[str, dict[str, float]]
    per_seed: list[dict[str, Any]]
    research_question: str = "Which organizational mechanism sustains conflict, hierarchy, and social structure in LLM-agent organizations?"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clone_config(base: SimConfig, *, seed: int, condition: str) -> SimConfig:
    data = dict(base.__dict__)
    data.update({
        "seed": seed,
        "disable_memory": False,
        "hierarchy_lesion": False,
        "status_lesion": False,
        "trust_lesion": False,
        "observation_lesion": False,
        "policy_mode": "dual_engine",
        "enable_llm_action_scoring": True,
        "cognitive_policy_lambda": 0.35,
        "llm_action_score_mix": 0.35,
    })
    if condition == "memory_lesion":
        data["disable_memory"] = True
    elif condition == "hierarchy_lesion":
        data["hierarchy_lesion"] = True
    elif condition == "status_lesion":
        data["status_lesion"] = True
    elif condition == "trust_lesion":
        data["trust_lesion"] = True
    elif condition == "observation_lesion":
        data["observation_lesion"] = True
    elif condition == "social_physics_only":
        data["policy_mode"] = "social_physics"
        data["enable_llm_action_scoring"] = False
        data["cognitive_policy_lambda"] = 0.0
        data["llm_action_score_mix"] = 0.0
    elif condition == "llm_native":
        data["policy_mode"] = "llm_native"
        data["enable_llm_action_scoring"] = True
        data["cognitive_policy_lambda"] = 1.0
        data["llm_action_score_mix"] = 1.0
    elif condition == "llm_scoring_off":
        data["enable_llm_action_scoring"] = False
        data["cognitive_policy_lambda"] = 0.0
        data["llm_action_score_mix"] = 0.0
    elif condition != "full":
        raise ValueError(f"Unknown organization ablation condition: {condition}")
    return SimConfig(**data)


def _round_metric(log: RunLog, key: str, *, mode: str = "final") -> float:
    values = [float(r.get("metrics", {}).get(key, 0.0)) for r in log.round_records]
    if not values:
        return 0.0
    if mode == "mean":
        return sum(values) / len(values)
    return values[-1]


def _measure(log: RunLog, outcome: str, metrics: dict[str, Any]) -> float:
    if outcome == "trust_fragmentation":
        return _round_metric(log, "trust_fragmentation")
    if outcome == "coalition_strength":
        return _round_metric(log, "coalition_strength")
    if outcome == "integrity_risk":
        return _round_metric(log, "integrity_risk")
    if outcome == "llm_override_pressure":
        return float(metrics.get("llm_scoring_influence", {}).get("mean_override_pressure", 0.0))
    social = metrics.get("social_emergence_metrics", {})
    if outcome in social:
        return float(social.get(outcome, 0.0))
    if outcome in log.outcomes:
        return float(log.outcomes.get(outcome, 0.0))
    return float(extract_outcome(log, outcome))


def run_organization_ablation(
    base_config: SimConfig | None = None,
    *,
    conditions: list[str] | None = None,
    seeds: list[int] | None = None,
    n: int = 5,
    outcomes: list[str] | None = None,
) -> OrganizationAblationResult:
    """Run a unified lesion matrix for Agent Organization MRI.

    Conditions:
    - full: dual-engine baseline
    - memory_lesion: disable pre-decision recall and memory writes
    - hierarchy_lesion: flatten PI-centered dependency and authority pressure
    - status_lesion: remove status, authorship entitlement, and credit-threat incentives
    - trust_lesion: remove trust/alliance as a dynamic state channel
    - observation_lesion: remove information asymmetry (omniscient affect/belief/recall cueing)
    - social_physics_only: field-only policy mode
    - llm_native: LLM proposes candidates directly
    - llm_scoring_off: candidates come from the field, no LLM candidate scoring
    """
    base = base_config or SimConfig(max_rounds=60)
    condition_names = conditions or ORGANIZATION_ABLATION_CONDITIONS
    seed_values = seeds if seeds is not None else list(range(n))
    outcome_names = outcomes or DEFAULT_ORGANIZATION_OUTCOMES

    rows: list[dict[str, Any]] = []
    for condition in condition_names:
        for seed in seed_values:
            cfg = _clone_config(base, seed=seed, condition=condition)
            log = run_simulation(cfg)
            metrics = compute_run_metrics(log)
            rows.append({
                "condition": condition,
                "seed": seed,
                "run_id": log.run_id,
                "policy_mode": log.config.get("policy_mode"),
                "disable_memory": bool(log.config.get("disable_memory")),
                "hierarchy_lesion": bool(log.config.get("hierarchy_lesion")),
                "status_lesion": bool(log.config.get("status_lesion")),
                "trust_lesion": bool(log.config.get("trust_lesion")),
                "observation_lesion": bool(log.config.get("observation_lesion")),
                **{name: round(_measure(log, name, metrics), 6) for name in outcome_names},
            })

    summary: dict[str, dict[str, float]] = {}
    for condition in condition_names:
        condition_rows = [r for r in rows if r["condition"] == condition]
        summary[condition] = {}
        for outcome in outcome_names:
            vals = [float(r.get(outcome, 0.0)) for r in condition_rows]
            summary[condition][outcome] = round(sum(vals) / len(vals), 6) if vals else 0.0

    return OrganizationAblationResult(
        conditions=condition_names,
        outcomes=outcome_names,
        n_per_condition=len(seed_values),
        summary=summary,
        per_seed=rows,
    )
