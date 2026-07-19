"""Cognitive-sampling performance/compute frontier."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.engine.simulation import SimConfig, run_simulation
from src.experiments.metrics import compute_run_metrics
from src.experiments.policy_protocol import _llm_sampled_fraction
from src.experiments.scientific_protocol import PROTOCOL_METRICS, _extract_metric
from src.world.loader import PROJECT_ROOT

DEFAULT_FRONTIER_OUT = PROJECT_ROOT / "output" / "sampling_frontier"


@dataclass
class SamplingFrontierResult:
    population_size: int
    rounds: int
    seeds: list[int]
    k_values: list[str]
    llm_provider: str
    summary: dict[str, dict[str, float]]
    per_run: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_k(k: int | str | None, population_size: int) -> int | None:
    if k is None:
        return None
    if isinstance(k, str) and k.lower() in {"full", "all", "none"}:
        return None
    return max(0, min(population_size, int(k)))


def _summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for k_label in sorted({row["k"] for row in rows}, key=lambda v: (v == "full", int(v) if str(v).isdigit() else 10**9)):
        items = [row for row in rows if row["k"] == k_label]
        summary[k_label] = {"n": float(len(items))}
        for name in ["llm_sampled_action_fraction", "emergent_pattern_score", "action_entropy", "coalition_persistence", "cascade_probability", "organization_fragility_index"]:
            vals = [float(item.get(name, 0.0)) for item in items]
            summary[k_label][name] = round(sum(vals) / len(vals), 6) if vals else 0.0
    return summary


def run_sampling_frontier(
    *,
    population_size: int = 100,
    rounds: int = 100,
    seeds: list[int] | None = None,
    k_values: list[int | str] | None = None,
    llm_provider: str = "scripted",
    output_dir: str | Path | None = None,
    write_output: bool = False,
) -> SamplingFrontierResult:
    run_seeds = seeds if seeds is not None else [0, 1, 2]
    ks = k_values or [0, 5, 10, 20, 50, 100, "full"]
    rows: list[dict[str, Any]] = []
    for raw_k in ks:
        top_k = _parse_k(raw_k, population_size)
        label = "full" if top_k is None else str(top_k)
        for seed in run_seeds:
            cfg = SimConfig(
                max_rounds=rounds,
                seed=seed,
                population_size=population_size if population_size > 14 else None,
                llm_provider=llm_provider,
                policy_mode="dual_engine",
                enable_llm_action_scoring=True,
                cognitive_policy_lambda=0.35,
                cognitive_sampling_top_k=top_k,
            )
            log = run_simulation(cfg)
            metrics = compute_run_metrics(log)
            row = {
                "k": label,
                "seed": seed,
                "population_size": population_size,
                "rounds": rounds,
                "run_id": log.run_id,
                "action_count": len(log.actions),
                "llm_sampled_action_fraction": _llm_sampled_fraction(log),
            }
            row.update({name: round(_extract_metric(log, metrics, name), 6) for name in PROTOCOL_METRICS})
            rows.append(row)
    result = SamplingFrontierResult(
        population_size=population_size,
        rounds=rounds,
        seeds=run_seeds,
        k_values=["full" if _parse_k(k, population_size) is None else str(_parse_k(k, population_size)) for k in ks],
        llm_provider=llm_provider,
        summary=_summarize(rows),
        per_run=rows,
    )
    if write_output:
        out = Path(output_dir) if output_dir else DEFAULT_FRONTIER_OUT
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"sampling_frontier_N{population_size}_{rounds}r.json"
        path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return result