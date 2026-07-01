"""Ablation over LLM candidate-scoring mix values."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.engine.run_log import RunLog, extract_outcome
from src.engine.simulation import SimConfig, run_simulation
from src.experiments.metrics import compute_run_metrics

DEFAULT_MIX_VALUES = [0.0, 0.2, 0.35, 0.6, 1.0]
DEFAULT_MIX_OUTCOMES = [
    "authorship_dispute_index",
    "trust_fragmentation",
    "public_private_divergence_mean",
    "memory_authorship_cluster_strength",
    "protest_authorship",
    "integrity_risk",
    "llm_override_pressure",
]


@dataclass
class LLMMixAblationResult:
    mix_values: list[float]
    outcomes: list[str]
    n_per_mix: int
    summary: dict[str, dict[str, float]]
    per_seed: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clone_config(base: SimConfig, *, seed: int, mix: float) -> SimConfig:
    data = dict(base.__dict__)
    data.update({
        "seed": seed,
        "enable_llm_action_scoring": mix > 0.0,
        "llm_action_score_mix": mix,
    })
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
    if outcome == "integrity_risk":
        return _round_metric(log, "integrity_risk")
    if outcome == "llm_override_pressure":
        return float(metrics.get("llm_scoring_influence", {}).get("mean_override_pressure", 0.0))
    if outcome == "llm_selected_rank_lift":
        return float(metrics.get("llm_scoring_influence", {}).get("mean_selected_rank_lift", 0.0))
    return float(extract_outcome(log, outcome))


def run_llm_mix_ablation(
    base_config: SimConfig,
    *,
    mix_values: list[float] | None = None,
    seeds: list[int] | None = None,
    n: int = 10,
    outcomes: list[str] | None = None,
) -> LLMMixAblationResult:
    """Compare long-horizon trajectories under different LLM scoring weights.

    mix=0.0 is treated as a field-only baseline by disabling candidate scoring.
    Larger values keep the same structural action field but let LLM plausibility
    scores exert more influence over fused action sampling.
    """
    mixes = mix_values if mix_values is not None else DEFAULT_MIX_VALUES
    seed_values = seeds if seeds is not None else list(range(n))
    outcome_names = outcomes if outcomes is not None else DEFAULT_MIX_OUTCOMES

    rows: list[dict[str, Any]] = []
    for mix in mixes:
        for seed in seed_values:
            cfg = _clone_config(base_config, seed=seed, mix=float(mix))
            log = run_simulation(cfg)
            metrics = compute_run_metrics(log)
            row = {
                "mix": float(mix),
                "seed": seed,
                "run_id": log.run_id,
                **{name: _measure(log, name, metrics) for name in outcome_names},
            }
            rows.append(row)

    summary: dict[str, dict[str, float]] = {}
    for mix in mixes:
        key = f"mix_{float(mix):.2f}"
        mix_rows = [r for r in rows if float(r["mix"]) == float(mix)]
        summary[key] = {}
        for outcome in outcome_names:
            vals = [float(r.get(outcome, 0.0)) for r in mix_rows]
            summary[key][outcome] = round(sum(vals) / len(vals), 6) if vals else 0.0

    return LLMMixAblationResult(
        mix_values=[float(m) for m in mixes],
        outcomes=outcome_names,
        n_per_mix=len(seed_values),
        summary=summary,
        per_seed=rows,
    )
