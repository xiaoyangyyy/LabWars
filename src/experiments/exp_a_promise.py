"""Experiment A — promise memory conditions."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from src.experiments.conditions import EXPERIMENT_A, list_conditions
from src.experiments.runner import run_single

PRIMARY_OUTCOMES = [
    "trust_pi_final",
    "pi_fairness_r52",
    "authorship_escalation_score",
    "authorship_escalation_potential",
    "post_r52_compliance",
    "memory_authorship_cluster_strength",
    "promise_broken_strength_r52",
    "promise_honored_strength_r52",
]
MEDIATORS = ["memory_authorship_cluster_strength", "promise_broken_strength_r52"]


def run_condition(condition_id: str, seed: int, **kwargs: Any):
    return run_single("A", seed, condition_id, **kwargs)


def run_all_conditions(seed: int, **kwargs: Any) -> dict[str, Any]:
    return {cid: run_condition(cid, seed, **kwargs) for cid in list_conditions("A")}


def _explicit_betrayal_condition():
    """Explicit promise at R3, default betrayal draft at R52 (no honor intervention)."""
    base = EXPERIMENT_A["A5"]
    return replace(
        base,
        condition_id="A_explicit_betrayal",
        label="explicit_betrayal",
        intervention_ids=["INT_AUTH_EXPLICIT"],
    )


def run_causal_delete_vs_explicit(n_seeds: int = 10, *, max_rounds: int = 60) -> dict[str, Any]:
    """Compare explicit+memory-delete vs explicit-only, both with betrayal draft at R52."""
    from src.engine.simulation import run_simulation
    from src.experiments.conditions import build_sim_config, get_condition

    explicit = _explicit_betrayal_condition()
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
            "promise_broken_control": ctrl_log.outcomes.get("promise_broken_strength_r52", 0),
            "promise_broken_treatment": treat_log.outcomes.get("promise_broken_strength_r52", 0),
        })

    def _mean(key: str) -> float:
        return sum(p[key] for p in per_seed) / len(per_seed) if per_seed else 0.0

    return {
        "comparison": "A5 (explicit+delete+betrayal) vs explicit-only+betrayal",
        "control_interventions": explicit.intervention_ids,
        "treatment_interventions": deleted.intervention_ids,
        "n_seeds": n_seeds,
        "ate_protest_intensity": _mean("protest_intensity_treatment") - _mean("protest_intensity_control"),
        "ate_protest_authorship": _mean("protest_authorship_treatment") - _mean("protest_authorship_control"),
        "ate_cluster": _mean("cluster_treatment") - _mean("cluster_control"),
        "ate_trust_pi": _mean("trust_pi_treatment") - _mean("trust_pi_control"),
        "ate_promise_broken_r52": _mean("promise_broken_treatment") - _mean("promise_broken_control"),
        "per_seed": per_seed,
    }


def run_paper(seed: int = 0, *, max_rounds: int = 60, **kwargs: Any) -> dict[str, Any]:
    from src.experiments.paper_contrasts import run_experiment_contrasts

    return run_experiment_contrasts("A", seeds=[seed], max_rounds=max_rounds, **kwargs)


def condition_table() -> list[dict[str, str]]:
    return [{"id": cid, "label": c.label, "interventions": ",".join(c.intervention_ids)} for cid, c in EXPERIMENT_A.items()]
