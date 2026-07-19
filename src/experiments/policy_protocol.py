"""Policy-regime comparison for Agent Social Dynamics."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.engine.simulation import SimConfig, run_simulation
from src.experiments.metrics import compute_run_metrics
from src.experiments.scientific_protocol import PROTOCOL_METRICS, _extract_metric
from src.world.loader import PROJECT_ROOT

DEFAULT_POLICY_OUT = PROJECT_ROOT / "output" / "policy_protocol"

POLICY_REGIMES = {
    "rule_baseline": {
        "policy_mode": "social_physics",
        "enable_llm_action_scoring": False,
        "cognitive_policy_lambda": 0.0,
        "cognitive_sampling_top_k": None,
    },
    "llm_native": {
        "policy_mode": "llm_native",
        "enable_llm_action_scoring": True,
        "cognitive_policy_lambda": 1.0,
        "cognitive_sampling_top_k": None,
    },
    "hybrid": {
        "policy_mode": "dual_engine",
        "enable_llm_action_scoring": True,
        "cognitive_policy_lambda": 0.35,
        "cognitive_sampling_top_k": None,
    },
    "hybrid_sampled": {
        "policy_mode": "dual_engine",
        "enable_llm_action_scoring": True,
        "cognitive_policy_lambda": 0.35,
        "cognitive_sampling_top_k": 20,
    },
}


@dataclass
class PolicyComparisonResult:
    population_size: int
    rounds: int
    seeds: list[int]
    regimes: list[str]
    llm_provider: str
    summary: dict[str, dict[str, float]]
    per_run: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for regime in sorted({row["regime"] for row in rows}):
        items = [row for row in rows if row["regime"] == regime]
        summary[regime] = {"n": float(len(items))}
        for metric in PROTOCOL_METRICS:
            vals = [float(item.get(metric, 0.0)) for item in items]
            summary[regime][metric] = round(sum(vals) / len(vals), 6) if vals else 0.0
        calls = [float(item.get("llm_sampled_action_fraction", 0.0)) for item in items]
        summary[regime]["llm_sampled_action_fraction"] = round(sum(calls) / len(calls), 6) if calls else 0.0
    return summary


def _llm_sampled_fraction(log: Any) -> float:
    if not log.actions:
        return 0.0
    sampled = 0
    enabled = 0
    for action in log.actions:
        audit = action.get("cognitive_sampling") or action.get("llm_action_scoring", {}).get("cognitive_sampling") or {}
        if audit.get("enabled"):
            enabled += 1
            if audit.get("sampled"):
                sampled += 1
        elif action.get("llm_action_scoring", {}).get("source") not in {"field_only", "field_only_unsampled"}:
            sampled += 1
            enabled += 1
    return round(sampled / max(1, enabled or len(log.actions)), 6)


def run_policy_comparison_protocol(
    *,
    population_size: int = 50,
    rounds: int = 60,
    seeds: list[int] | None = None,
    regimes: list[str] | None = None,
    llm_provider: str = "scripted",
    sampled_top_k: int = 20,
    output_dir: str | Path | None = None,
    write_output: bool = False,
) -> PolicyComparisonResult:
    run_seeds = seeds if seeds is not None else [0, 1, 2]
    regime_names = regimes or ["rule_baseline", "llm_native", "hybrid", "hybrid_sampled"]
    rows: list[dict[str, Any]] = []
    for regime in regime_names:
        if regime not in POLICY_REGIMES:
            raise ValueError(f"Unknown policy regime: {regime}")
        spec = dict(POLICY_REGIMES[regime])
        if regime == "hybrid_sampled":
            spec["cognitive_sampling_top_k"] = sampled_top_k
        for seed in run_seeds:
            cfg = SimConfig(
                max_rounds=rounds,
                seed=seed,
                population_size=population_size if population_size > 14 else None,
                llm_provider=llm_provider,
                policy_mode=str(spec["policy_mode"]),
                enable_llm_action_scoring=bool(spec["enable_llm_action_scoring"]),
                cognitive_policy_lambda=float(spec["cognitive_policy_lambda"]),
                cognitive_sampling_top_k=spec["cognitive_sampling_top_k"],
            )
            log = run_simulation(cfg)
            metrics = compute_run_metrics(log)
            row = {
                "regime": regime,
                "seed": seed,
                "population_size": population_size,
                "rounds": rounds,
                "run_id": log.run_id,
                "action_count": len(log.actions),
                "llm_sampled_action_fraction": _llm_sampled_fraction(log),
            }
            row.update({metric: round(_extract_metric(log, metrics, metric), 6) for metric in PROTOCOL_METRICS})
            rows.append(row)
    result = PolicyComparisonResult(
        population_size=population_size,
        rounds=rounds,
        seeds=run_seeds,
        regimes=regime_names,
        llm_provider=llm_provider,
        summary=_summarize(rows),
        per_run=rows,
    )
    if write_output:
        out = Path(output_dir) if output_dir else DEFAULT_POLICY_OUT
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"policy_comparison_N{population_size}_{rounds}r.json"
        path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return result