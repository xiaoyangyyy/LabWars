"""Experiment A — promise memory conditions."""

from __future__ import annotations

from typing import Any

from src.engine.causal import run_causal_experiment
from src.experiments.conditions import EXPERIMENT_A, list_conditions
from src.experiments.runner import run_single

PRIMARY_OUTCOMES = [
    "trust_pi_final",
    "pi_fairness_r52",
    "authorship_escalation_score",
    "authorship_escalation_potential",
    "post_r52_compliance",
    "memory_authorship_cluster_strength",
]
MEDIATORS = ["memory_authorship_cluster_strength", "promise_broken_strength_r52"]


def run_condition(condition_id: str, seed: int, **kwargs: Any):
    return run_single("A", seed, condition_id, **kwargs)


def run_all_conditions(seed: int, **kwargs: Any) -> dict[str, Any]:
    return {cid: run_condition(cid, seed, **kwargs) for cid in list_conditions("A")}


def run_causal_delete_vs_explicit(n_seeds: int = 10, *, max_rounds: int = 60) -> dict[str, Any]:
    """Compare A2 (explicit promise) vs A5 (explicit + R45 memory delete)."""
    from src.engine.simulation import run_simulation
    from src.experiments.conditions import build_sim_config, get_condition

    explicit = get_condition("A", "A2")
    deleted = get_condition("A", "A5")
    per_seed: list[dict[str, Any]] = []

    for seed in range(n_seeds):
        ctrl_log = run_simulation(build_sim_config(explicit, seed, max_rounds=max_rounds))
        treat_log = run_simulation(build_sim_config(deleted, seed, max_rounds=max_rounds))
        per_seed.append({
            "seed": seed,
            "protest_intensity_control": ctrl_log.outcomes.get("protest_intensity", 0),
            "protest_intensity_treatment": treat_log.outcomes.get("protest_intensity", 0),
            "protest_authorship_control": ctrl_log.outcomes.get("protest_authorship", 0),
            "protest_authorship_treatment": treat_log.outcomes.get("protest_authorship", 0),
            "cluster_control": ctrl_log.outcomes.get("memory_authorship_cluster_strength", 0),
            "cluster_treatment": treat_log.outcomes.get("memory_authorship_cluster_strength", 0),
            "trust_pi_control": ctrl_log.outcomes.get("trust_pi_final", 0),
            "trust_pi_treatment": treat_log.outcomes.get("trust_pi_final", 0),
        })

    def _mean(key: str) -> float:
        return sum(p[key] for p in per_seed) / len(per_seed) if per_seed else 0.0

    return {
        "comparison": "A5 (explicit+delete) vs A2 (explicit)",
        "n_seeds": n_seeds,
        "ate_protest_intensity": _mean("protest_intensity_treatment") - _mean("protest_intensity_control"),
        "ate_protest_authorship": _mean("protest_authorship_treatment") - _mean("protest_authorship_control"),
        "ate_cluster": _mean("cluster_treatment") - _mean("cluster_control"),
        "ate_trust_pi": _mean("trust_pi_treatment") - _mean("trust_pi_control"),
        "per_seed": per_seed,
    }


def condition_table() -> list[dict[str, str]]:
    return [{"id": cid, "label": c.label, "interventions": ",".join(c.intervention_ids)} for cid, c in EXPERIMENT_A.items()]
