"""Compare Social Physics, Dual Engine, and LLM-native policy modes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.engine.run_log import RunLog, extract_outcome
from src.engine.simulation import SimConfig, run_simulation
from src.experiments.metrics import compute_run_metrics

DEFAULT_POLICY_MODES = ["social_physics", "dual_engine", "llm_native"]
DEFAULT_POLICY_OUTCOMES = [
    "authorship_dispute_index",
    "trust_fragmentation",
    "public_private_divergence_mean",
    "memory_authorship_cluster_strength",
    "protest_authorship",
    "integrity_risk",
    "llm_override_pressure",
    "llm_native_candidate_fraction",
]


@dataclass
class PolicyModeComparisonResult:
    policy_modes: list[str]
    outcomes: list[str]
    n_per_mode: int
    summary: dict[str, dict[str, float]]
    per_seed: list[dict[str, Any]]
    research_question: str = "Do conflict trajectories come from structural social physics, hybrid cognition, or LLM-native action-space generation?"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clone_config(base: SimConfig, *, seed: int, policy_mode: str) -> SimConfig:
    data = dict(base.__dict__)
    data.update({"seed": seed, "policy_mode": policy_mode})
    if policy_mode == "social_physics":
        data["enable_llm_action_scoring"] = False
        data["cognitive_policy_lambda"] = 0.0
    elif policy_mode == "llm_native":
        data["enable_llm_action_scoring"] = True
        data["cognitive_policy_lambda"] = 1.0
    return SimConfig(**data)


def _round_metric(log: RunLog, key: str) -> float:
    if not log.round_records:
        return 0.0
    return float(log.round_records[-1].get("metrics", {}).get(key, 0.0))


def _native_fraction(log: RunLog) -> float:
    if not log.actions:
        return 0.0
    native = 0
    for action in log.actions:
        selected = action.get("selected_action") or {}
        if selected.get("scoring_source") in {"llm_native_generated", "llm_native_fallback"}:
            native += 1
    return native / len(log.actions)


def _measure(log: RunLog, outcome: str, metrics: dict[str, Any]) -> float:
    if outcome == "trust_fragmentation":
        return _round_metric(log, "trust_fragmentation")
    if outcome == "integrity_risk":
        return _round_metric(log, "integrity_risk")
    if outcome == "llm_override_pressure":
        return float(metrics.get("llm_scoring_influence", {}).get("mean_override_pressure", 0.0))
    if outcome == "llm_native_candidate_fraction":
        return _native_fraction(log)
    return float(extract_outcome(log, outcome))


def run_policy_mode_comparison(
    base_config: SimConfig,
    *,
    policy_modes: list[str] | None = None,
    seeds: list[int] | None = None,
    n: int = 10,
    outcomes: list[str] | None = None,
) -> PolicyModeComparisonResult:
    modes = policy_modes if policy_modes is not None else DEFAULT_POLICY_MODES
    seed_values = seeds if seeds is not None else list(range(n))
    outcome_names = outcomes if outcomes is not None else DEFAULT_POLICY_OUTCOMES

    rows: list[dict[str, Any]] = []
    for mode in modes:
        for seed in seed_values:
            cfg = _clone_config(base_config, seed=seed, policy_mode=mode)
            log = run_simulation(cfg)
            metrics = compute_run_metrics(log)
            rows.append({
                "policy_mode": mode,
                "seed": seed,
                "run_id": log.run_id,
                **{name: _measure(log, name, metrics) for name in outcome_names},
            })

    summary: dict[str, dict[str, float]] = {}
    for mode in modes:
        mode_rows = [r for r in rows if r["policy_mode"] == mode]
        summary[mode] = {}
        for outcome in outcome_names:
            vals = [float(r.get(outcome, 0.0)) for r in mode_rows]
            summary[mode][outcome] = round(sum(vals) / len(vals), 6) if vals else 0.0

    return PolicyModeComparisonResult(
        policy_modes=modes,
        outcomes=outcome_names,
        n_per_mode=len(seed_values),
        summary=summary,
        per_seed=rows,
    )
