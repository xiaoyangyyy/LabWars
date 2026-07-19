"""Anti-script emergence challenge protocols."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.engine.simulation import SimConfig, run_simulation
from src.experiments.metrics import compute_run_metrics
from src.experiments.scientific_protocol import PROTOCOL_METRICS, _extract_metric
from src.world.loader import PROJECT_ROOT

DEFAULT_CHALLENGE_OUT = PROJECT_ROOT / "output" / "emergence_challenge"


@dataclass
class EgalitarianEmergenceResult:
    population_size: int
    rounds: int
    seeds: list[int]
    conditions: list[str]
    summary: dict[str, dict[str, float]]
    per_run: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for condition in sorted({row["condition"] for row in rows}):
        items = [row for row in rows if row["condition"] == condition]
        summary[condition] = {"n": float(len(items))}
        for metric in [
            "power_concentration_gini",
            "credit_attribution_gap",
            "network_modularity_q",
            "coalition_strength",
            "coalition_persistence",
            "cascade_probability",
            "emergent_pattern_score",
            "organization_fragility_index",
        ]:
            vals = [float(item.get(metric, 0.0)) for item in items]
            summary[condition][metric] = round(sum(vals) / len(vals), 6) if vals else 0.0
    return summary


def run_egalitarian_emergence_challenge(
    *,
    population_size: int = 500,
    rounds: int = 500,
    seeds: list[int] | None = None,
    llm_provider: str = "scripted",
    policy_mode: str = "social_physics",
    output_dir: str | Path | None = None,
    write_output: bool = False,
) -> EgalitarianEmergenceResult:
    run_seeds = seeds if seeds is not None else list(range(10))
    rows: list[dict[str, Any]] = []
    for condition, egalitarian in [("canonical_initialization", False), ("egalitarian_initialization", True)]:
        for seed in run_seeds:
            cfg = SimConfig(
                max_rounds=rounds,
                seed=seed,
                population_size=population_size if population_size > 14 else None,
                llm_provider=llm_provider,
                policy_mode=policy_mode,
                enable_llm_action_scoring=policy_mode != "social_physics",
                cognitive_policy_lambda=0.0 if policy_mode == "social_physics" else 0.35,
                egalitarian_initialization=egalitarian,
            )
            log = run_simulation(cfg)
            metrics = compute_run_metrics(log)
            row = {
                "condition": condition,
                "seed": seed,
                "population_size": population_size,
                "rounds": rounds,
                "run_id": log.run_id,
                "action_count": len(log.actions),
                "egalitarian_initialization": egalitarian,
            }
            row.update({name: round(_extract_metric(log, metrics, name), 6) for name in PROTOCOL_METRICS})
            rows.append(row)
    result = EgalitarianEmergenceResult(
        population_size=population_size,
        rounds=rounds,
        seeds=run_seeds,
        conditions=["canonical_initialization", "egalitarian_initialization"],
        summary=_summarize(rows),
        per_run=rows,
    )
    if write_output:
        out = Path(output_dir) if output_dir else DEFAULT_CHALLENGE_OUT
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"egalitarian_emergence_N{population_size}_{rounds}r.json"
        path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return result