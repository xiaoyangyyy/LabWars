"""Scale experiments for Agent Social Dynamics Benchmark."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.engine.simulation import SimConfig, run_simulation
from src.experiments.metrics import compute_run_metrics
from src.world.loader import PROJECT_ROOT

DEFAULT_SCALE_OUT = PROJECT_ROOT / "output" / "scale"


DEFAULT_SCALE_METRICS = [
    "trust_entropy",
    "power_concentration_gini",
    "alliance_modularity_proxy",
    "conflict_cascade_length",
    "reputation_volatility",
    "credit_attribution_gap",
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
]


@dataclass
class ScaleExperimentResult:
    population_sizes: list[int]
    rounds: int
    seeds: list[int]
    policy_mode: str
    llm_provider: str
    metrics: list[str]
    summary: dict[str, dict[str, float]]
    per_run: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _summarize(rows: list[dict[str, Any]], metrics: list[str]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["population_size"]), []).append(row)
    summary: dict[str, dict[str, float]] = {}
    for size, items in grouped.items():
        summary[size] = {}
        for metric in metrics:
            values = [float(item.get(metric, 0.0)) for item in items]
            summary[size][metric] = round(sum(values) / len(values), 4) if values else 0.0
        summary[size]["n"] = float(len(items))
    return summary


def run_scale_experiment(
    *,
    population_sizes: list[int] | None = None,
    rounds: int = 100,
    seeds: list[int] | None = None,
    policy_mode: str = "social_physics",
    llm_provider: str = "scripted",
    population_labs: int | None = None,
    output_dir: str | Path | None = None,
    write_output: bool = False,
) -> ScaleExperimentResult:
    """Run deterministic scale sweeps over population size.

    Use this for the reviewer-facing question: do metrics remain measurable and
    non-degenerate as LabWars moves from a canonical 14-agent lab to larger
    50-200-agent hierarchical organizations?
    """
    sizes = population_sizes or [14, 50, 100, 200]
    run_seeds = seeds or [0, 1, 2]
    rows: list[dict[str, Any]] = []
    for size in sizes:
        for seed in run_seeds:
            cfg = SimConfig(
                max_rounds=rounds,
                seed=seed,
                population_size=size if size > 14 else None,
                population_labs=population_labs,
                policy_mode=policy_mode,
                enable_llm_action_scoring=policy_mode != "social_physics",
                cognitive_policy_lambda=0.0 if policy_mode == "social_physics" else 0.35,
                llm_provider=llm_provider,
            )
            log = run_simulation(cfg)
            metrics = compute_run_metrics(log).get("social_emergence_metrics", {})
            row = {
                "population_size": size,
                "rounds": rounds,
                "seed": seed,
                "run_id": log.run_id,
                "action_count": len(log.actions),
                "event_count": len(log.events),
                "round_count": len(log.round_records),
            }
            row.update({metric: float(metrics.get(metric, 0.0)) for metric in DEFAULT_SCALE_METRICS})
            rows.append(row)
    result = ScaleExperimentResult(
        population_sizes=sizes,
        rounds=rounds,
        seeds=run_seeds,
        policy_mode=policy_mode,
        llm_provider=llm_provider,
        metrics=list(DEFAULT_SCALE_METRICS),
        summary=_summarize(rows, DEFAULT_SCALE_METRICS),
        per_run=rows,
    )
    if write_output:
        out = Path(output_dir) if output_dir else DEFAULT_SCALE_OUT
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"scale_{policy_mode}_{rounds}r.json"
        path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return result
