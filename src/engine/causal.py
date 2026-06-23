"""Causal experiment runner — ATE and outcome extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.engine.intervention import Intervention
from src.engine.run_log import RunLog, _memory_cluster_strength, extract_outcome
from src.engine.simulation import SimConfig, run_simulation


@dataclass
class CausalResult:
    intervention_id: str
    outcome: str
    ate: float
    control_mean: float
    treatment_mean: float
    n_seeds: int
    per_seed: list[dict[str, Any]]


def compute_ate(per_seed: list[dict[str, Any]]) -> tuple[float, float, float]:
    if not per_seed:
        return 0.0, 0.0, 0.0
    controls = [p["Y_control"] for p in per_seed]
    treatments = [p["Y_treatment"] for p in per_seed]
    c_mean = sum(controls) / len(controls)
    t_mean = sum(treatments) / len(treatments)
    return t_mean - c_mean, c_mean, t_mean


def run_causal_experiment(
    base_config: SimConfig,
    intervention: Intervention,
    outcome: str = "protest_authorship",
    n_seeds: int = 10,
) -> CausalResult:
    per_seed: list[dict[str, Any]] = []

    for seed in range(n_seeds):
        ctrl_cfg = SimConfig(**{**base_config.__dict__, "seed": seed, "interventions": []})
        treat_cfg = SimConfig(**{**base_config.__dict__, "seed": seed, "interventions": [intervention]})

        ctrl_log = run_simulation(ctrl_cfg)
        treat_log = run_simulation(treat_cfg)

        per_seed.append({
            "seed": seed,
            "Y_control": extract_outcome(ctrl_log, outcome),
            "Y_treatment": extract_outcome(treat_log, outcome),
            "M_control": ctrl_log.outcomes.get("memory_authorship_cluster_strength", _memory_cluster_strength(ctrl_log)),
            "M_treatment": treat_log.outcomes.get("memory_authorship_cluster_strength", _memory_cluster_strength(treat_log)),
        })

    ate, c_mean, t_mean = compute_ate(per_seed)
    return CausalResult(
        intervention_id=intervention.intervention_id,
        outcome=outcome,
        ate=ate,
        control_mean=c_mean,
        treatment_mean=t_mean,
        n_seeds=n_seeds,
        per_seed=per_seed,
    )
