"""Standard Agent Social Dynamics benchmark tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.engine.simulation import SimConfig, run_simulation
from src.experiments.metrics import compute_run_metrics
from src.experiments.organization_ablation import run_organization_ablation

BENCHMARK_TASKS = {
    "conflict_emergence": {
        "conditions": ["full", "hierarchy_lesion", "memory_lesion"],
        "outcomes": ["organization_fragility_index", "conflict_cascade_length", "authorship_dispute_index", "selected_social_pressure_mean"],
        "question": "When organizational pressure rises, do conflict cascades emerge?",
    },
    "alliance_formation": {
        "conditions": ["full", "memory_lesion", "social_physics_only"],
        "outcomes": ["alliance_modularity_proxy", "trust_entropy", "trust_fragmentation", "social_state_volatility"],
        "question": "Which mechanisms support stable alliance-like trust structure?",
    },
    "credit_attribution": {
        "conditions": ["full", "memory_lesion", "hierarchy_lesion"],
        "outcomes": ["credit_attribution_gap", "authorship_dispute_index", "reputation_volatility", "selected_social_pressure_mean"],
        "question": "How does contribution uncertainty become credit conflict?",
    },
    "memory_mediation": {
        "conditions": ["full", "memory_lesion", "llm_scoring_off"],
        "outcomes": ["memory_authorship_cluster_strength", "conflict_cascade_length", "organization_fragility_index"],
        "question": "Does long-term memory mediate social conflict trajectories?",
    },
    "authority_compliance": {
        "conditions": ["full", "hierarchy_lesion", "social_physics_only"],
        "outcomes": ["career_hostage_index", "power_concentration_gini", "public_private_divergence_mean", "organization_fragility_index"],
        "question": "Does authority pressure generate compliance and public-private splits?",
    },
    "integrity_stress": {
        "conditions": ["full", "hierarchy_lesion", "llm_native"],
        "outcomes": ["integrity_risk", "organization_fragility_index", "power_concentration_gini", "llm_override_pressure"],
        "question": "How does organizational stress affect integrity risk?",
    },
}


@dataclass
class BenchmarkTaskResult:
    task: str
    question: str
    conditions: list[str]
    outcomes: list[str]
    n_per_condition: int
    summary: dict[str, dict[str, float]]
    per_seed: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def list_benchmark_tasks() -> list[str]:
    return list(BENCHMARK_TASKS.keys())


def run_benchmark_task(
    task: str,
    base_config: SimConfig | None = None,
    *,
    seeds: list[int] | None = None,
    n: int = 5,
) -> BenchmarkTaskResult:
    if task not in BENCHMARK_TASKS:
        raise ValueError(f"Unknown benchmark task: {task}")
    spec = BENCHMARK_TASKS[task]
    ablation = run_organization_ablation(
        base_config or SimConfig(max_rounds=60),
        conditions=list(spec["conditions"]),
        seeds=seeds,
        n=n,
        outcomes=list(spec["outcomes"]),
    )
    return BenchmarkTaskResult(
        task=task,
        question=str(spec["question"]),
        conditions=list(spec["conditions"]),
        outcomes=list(spec["outcomes"]),
        n_per_condition=ablation.n_per_condition,
        summary=ablation.summary,
        per_seed=ablation.per_seed,
    )


def run_single_task_condition(
    task: str,
    condition: str = "full",
    *,
    seed: int = 0,
    max_rounds: int = 60,
) -> dict[str, Any]:
    """Small helper for smoke-testing a task/condition without the whole matrix."""
    if task not in BENCHMARK_TASKS:
        raise ValueError(f"Unknown benchmark task: {task}")
    if condition not in BENCHMARK_TASKS[task]["conditions"]:
        raise ValueError(f"Condition {condition} is not part of task {task}")
    cfg = SimConfig(max_rounds=max_rounds, seed=seed)
    if condition == "memory_lesion":
        cfg.disable_memory = True
    elif condition == "hierarchy_lesion":
        cfg.hierarchy_lesion = True
    elif condition == "social_physics_only":
        cfg.policy_mode = "social_physics"
        cfg.enable_llm_action_scoring = False
        cfg.cognitive_policy_lambda = 0.0
    elif condition == "llm_native":
        cfg.policy_mode = "llm_native"
        cfg.cognitive_policy_lambda = 1.0
    elif condition == "llm_scoring_off":
        cfg.enable_llm_action_scoring = False
        cfg.cognitive_policy_lambda = 0.0
    log = run_simulation(cfg)
    metrics = compute_run_metrics(log)
    social = metrics.get("social_emergence_metrics", {})
    outcomes = BENCHMARK_TASKS[task]["outcomes"]
    return {
        "task": task,
        "condition": condition,
        "seed": seed,
        "run_id": log.run_id,
        "metrics": {name: social.get(name, log.outcomes.get(name, 0.0)) for name in outcomes},
    }
