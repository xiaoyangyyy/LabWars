"""Dual-engine ablation over cognitive-policy lambda values."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.engine.run_log import RunLog, extract_outcome
from src.engine.simulation import SimConfig, run_simulation
from src.experiments.metrics import compute_run_metrics

DEFAULT_LAMBDA_VALUES = [0.0, 0.2, 0.35, 0.6, 1.0]
DEFAULT_MIX_VALUES = DEFAULT_LAMBDA_VALUES
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
class DualEngineAblationResult:
    lambda_values: list[float]
    outcomes: list[str]
    n_per_lambda: int
    summary: dict[str, dict[str, float]]
    per_seed: list[dict[str, Any]]
    research_question: str = "Are trajectories driven more by Social Physics or by the LLM Cognitive Policy Layer?"

    @property
    def mix_values(self) -> list[float]:
        return self.lambda_values

    @property
    def n_per_mix(self) -> int:
        return self.n_per_lambda

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {
            "mix_values": self.lambda_values,
            "n_per_mix": self.n_per_lambda,
        }


LLMMixAblationResult = DualEngineAblationResult


def _clone_config(base: SimConfig, *, seed: int, cognitive_lambda: float) -> SimConfig:
    data = dict(base.__dict__)
    data.update({
        "seed": seed,
        "enable_llm_action_scoring": cognitive_lambda > 0.0,
        "cognitive_policy_lambda": cognitive_lambda,
        "llm_action_score_mix": cognitive_lambda,
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


def run_dual_engine_ablation(
    base_config: SimConfig,
    *,
    lambda_values: list[float] | None = None,
    seeds: list[int] | None = None,
    n: int = 10,
    outcomes: list[str] | None = None,
) -> DualEngineAblationResult:
    """Compare trajectories across Social Physics / LLM Cognitive blends.

    lambda=0.0 is Social Physics only. lambda=1.0 lets the LLM Cognitive
    Policy Layer dominate candidate ranking while the physics engine still
    supplies candidate actions and state dynamics.
    """
    lambdas = lambda_values if lambda_values is not None else DEFAULT_LAMBDA_VALUES
    seed_values = seeds if seeds is not None else list(range(n))
    outcome_names = outcomes if outcomes is not None else DEFAULT_MIX_OUTCOMES

    rows: list[dict[str, Any]] = []
    for cognitive_lambda in lambdas:
        for seed in seed_values:
            cfg = _clone_config(base_config, seed=seed, cognitive_lambda=float(cognitive_lambda))
            log = run_simulation(cfg)
            metrics = compute_run_metrics(log)
            row = {
                "lambda": float(cognitive_lambda),
                "seed": seed,
                "run_id": log.run_id,
                **{name: _measure(log, name, metrics) for name in outcome_names},
            }
            rows.append(row)

    summary: dict[str, dict[str, float]] = {}
    for cognitive_lambda in lambdas:
        key = f"lambda_{float(cognitive_lambda):.2f}"
        lambda_rows = [r for r in rows if float(r["lambda"]) == float(cognitive_lambda)]
        summary[key] = {}
        for outcome in outcome_names:
            vals = [float(r.get(outcome, 0.0)) for r in lambda_rows]
            summary[key][outcome] = round(sum(vals) / len(vals), 6) if vals else 0.0

    return DualEngineAblationResult(
        lambda_values=[float(value) for value in lambdas],
        outcomes=outcome_names,
        n_per_lambda=len(seed_values),
        summary=summary,
        per_seed=rows,
    )


def run_llm_mix_ablation(
    base_config: SimConfig,
    *,
    mix_values: list[float] | None = None,
    seeds: list[int] | None = None,
    n: int = 10,
    outcomes: list[str] | None = None,
) -> DualEngineAblationResult:
    """Backward-compatible wrapper for the dual-engine lambda ablation."""
    return run_dual_engine_ablation(
        base_config,
        lambda_values=mix_values,
        seeds=seeds,
        n=n,
        outcomes=outcomes,
    )
