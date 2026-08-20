"""Validity control experiments."""

from __future__ import annotations

from typing import Any

from src.engine.run_log import extract_outcome
from src.experiments.conditions import EXPERIMENT_VALIDITY, list_conditions
from src.experiments.metrics import welch_t_stat
from src.experiments.runner import run_single

PRIMARY_OUTCOMES = ["protest_authorship", "memory_authorship_cluster_strength"]


def run_condition(condition_id: str, seed: int, **kwargs: Any):
    return run_single("V", seed, condition_id, **kwargs)


def run_all_conditions(seed: int, **kwargs: Any) -> dict[str, Any]:
    return {cid: run_condition(cid, seed, **kwargs) for cid in list_conditions("V")}


def shuffle_vs_full_test(seeds: list[int], outcome: str = "protest_authorship") -> dict[str, Any]:
    """Compare V6 (full memory) vs V2 (shuffled) — validity gate."""
    full_vals: list[float] = []
    shuf_vals: list[float] = []
    for seed in seeds:
        full = run_condition("V6", seed)
        shuf = run_condition("V2", seed)
        full_vals.append(extract_outcome(full["log"], outcome))
        shuf_vals.append(extract_outcome(shuf["log"], outcome))
    t = welch_t_stat(full_vals, shuf_vals)
    return {
        "outcome": outcome,
        "full_mean": sum(full_vals) / len(full_vals) if full_vals else 0.0,
        "shuffled_mean": sum(shuf_vals) / len(shuf_vals) if shuf_vals else 0.0,
        "welch_t": t,
        "n_seeds": len(seeds),
        "passes_directional_difference": abs(t) > 1.96,
    }


def run_paper(seed: int = 0, *, max_rounds: int = 60, **kwargs: Any) -> dict[str, Any]:
    from src.experiments.paper_contrasts import run_experiment_contrasts

    return run_experiment_contrasts("V", seeds=[seed], max_rounds=max_rounds, **kwargs)


def condition_table() -> list[dict[str, str]]:
    return [{"id": cid, "label": c.label} for cid, c in EXPERIMENT_VALIDITY.items()]
