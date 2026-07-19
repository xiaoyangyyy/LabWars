"""Repeatable experimental-science protocol for Agent Social Dynamics."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.engine.simulation import SimConfig, run_simulation
from src.experiments.metrics import compute_run_metrics
from src.world.loader import PROJECT_ROOT

DEFAULT_PROTOCOL_OUT = PROJECT_ROOT / "output" / "protocol"

PROTOCOL_CONDITIONS = [
    "baseline",
    "no_memory",
    "no_status",
    "no_trust",
    "no_hierarchy",
]

PROTOCOL_METRICS = [
    "trust_entropy",
    "trust_fragmentation",
    "coalition_strength",
    "alliance_modularity_proxy",
    "power_concentration_gini",
    "credit_attribution_gap",
    "conflict_cascade_length",
    "reputation_volatility",
    "social_state_volatility",
    "organization_fragility_index",
    "power_law_alpha",
    "power_law_fit_r2",
    "network_modularity_q",
    "cascade_tail_alpha",
    "cascade_tail_r2",
    "emergent_pattern_score",
    "action_entropy",
    "coalition_persistence",
    "cascade_probability",
    "authorship_dispute_index",
    "career_hostage_index",
]


@dataclass
class ScientificProtocolResult:
    protocol_id: str
    population_sizes: list[int]
    rounds: int
    seeds: list[int]
    conditions: list[str]
    policy_mode: str
    llm_provider: str
    metrics: list[str]
    summary: dict[str, dict[str, float]]
    per_run: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _condition_config(base: SimConfig, condition: str) -> SimConfig:
    data = dict(base.__dict__)
    data.update({
        "disable_memory": False,
        "status_lesion": False,
        "trust_lesion": False,
        "hierarchy_lesion": False,
    })
    if condition == "baseline":
        pass
    elif condition == "no_memory":
        data["disable_memory"] = True
    elif condition == "no_status":
        data["status_lesion"] = True
    elif condition == "no_trust":
        data["trust_lesion"] = True
    elif condition == "no_hierarchy":
        data["hierarchy_lesion"] = True
    else:
        raise ValueError(f"Unknown protocol condition: {condition}")
    return SimConfig(**data)


def _extract_metric(log: Any, metrics: dict[str, Any], name: str) -> float:
    social = metrics.get("social_emergence_metrics", {})
    if name in social:
        return float(social[name])
    if name in log.outcomes:
        return float(log.outcomes[name])
    if log.round_records:
        return float(log.round_records[-1].get("metrics", {}).get(name, 0.0))
    return 0.0


def _summarize(rows: list[dict[str, Any]], metric_names: list[str]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = f"N{row['population_size']}:{row['condition']}"
        grouped.setdefault(key, []).append(row)
    summary: dict[str, dict[str, float]] = {}
    for key, items in grouped.items():
        summary[key] = {"n": float(len(items))}
        for metric in metric_names:
            vals = [float(item.get(metric, 0.0)) for item in items]
            summary[key][metric] = round(sum(vals) / len(vals), 6) if vals else 0.0
    return summary


def _format_markdown(result: ScientificProtocolResult) -> str:
    lines = [
        f"# {result.protocol_id}",
        "",
        "This report summarizes a repeatable Agent Social Dynamics protocol.",
        "",
        f"- Population sizes: {result.population_sizes}",
        f"- Rounds: {result.rounds}",
        f"- Seeds: {result.seeds}",
        f"- Conditions: {result.conditions}",
        f"- Policy mode: {result.policy_mode}",
        f"- LLM provider: {result.llm_provider}",
        "",
        "| cell | n | fragility | trust_entropy | coalition | power_gini | credit_gap | cascade |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in sorted(result.summary):
        item = result.summary[key]
        lines.append(
            f"| {key} | {item.get('n', 0):.0f} | "
            f"{item.get('organization_fragility_index', 0):.3f} | "
            f"{item.get('trust_entropy', 0):.3f} | "
            f"{item.get('coalition_strength', 0):.3f} | "
            f"{item.get('power_concentration_gini', 0):.3f} | "
            f"{item.get('credit_attribution_gap', 0):.3f} | "
            f"{item.get('conflict_cascade_length', 0):.3f} |"
        )
    lines.append("")
    lines.append("Interpretation rule: compare each lesion cell against the same-size baseline; do not infer mechanism from a single narrative trajectory.")
    return "\n".join(lines) + "\n"


def run_scientific_protocol(
    *,
    protocol_id: str = "agent_social_dynamics_protocol_v1",
    population_sizes: list[int] | None = None,
    rounds: int = 500,
    seeds: list[int] | None = None,
    conditions: list[str] | None = None,
    policy_mode: str = "social_physics",
    llm_provider: str = "scripted",
    output_dir: str | Path | None = None,
    write_output: bool = False,
) -> ScientificProtocolResult:
    sizes = population_sizes or [10, 50, 100, 500]
    run_seeds = seeds if seeds is not None else list(range(100))
    condition_names = conditions or list(PROTOCOL_CONDITIONS)
    rows: list[dict[str, Any]] = []

    for size in sizes:
        for condition in condition_names:
            for seed in run_seeds:
                base = SimConfig(
                    max_rounds=rounds,
                    seed=seed,
                    population_size=size if size > 14 else None,
                    policy_mode=policy_mode,
                    enable_llm_action_scoring=policy_mode != "social_physics",
                    cognitive_policy_lambda=0.0 if policy_mode == "social_physics" else 0.35,
                    llm_provider=llm_provider,
                )
                cfg = _condition_config(base, condition)
                log = run_simulation(cfg)
                metrics = compute_run_metrics(log)
                row = {
                    "protocol_id": protocol_id,
                    "population_size": size,
                    "condition": condition,
                    "seed": seed,
                    "rounds": rounds,
                    "run_id": log.run_id,
                    "action_count": len(log.actions),
                    "event_count": len(log.events),
                    "round_count": len(log.round_records),
                }
                row.update({name: round(_extract_metric(log, metrics, name), 6) for name in PROTOCOL_METRICS})
                rows.append(row)

    result = ScientificProtocolResult(
        protocol_id=protocol_id,
        population_sizes=sizes,
        rounds=rounds,
        seeds=run_seeds,
        conditions=condition_names,
        policy_mode=policy_mode,
        llm_provider=llm_provider,
        metrics=list(PROTOCOL_METRICS),
        summary=_summarize(rows, PROTOCOL_METRICS),
        per_run=rows,
    )
    if write_output:
        out = Path(output_dir) if output_dir else DEFAULT_PROTOCOL_OUT
        out.mkdir(parents=True, exist_ok=True)
        stem = f"{protocol_id}_{policy_mode}_{rounds}r"
        (out / f"{stem}.json").write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        (out / f"{stem}.md").write_text(_format_markdown(result), encoding="utf-8")
    return result