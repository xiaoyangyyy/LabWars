"""Experiment B — betrayal memory conditions."""

from __future__ import annotations

from typing import Any

from src.experiments.conditions import EXPERIMENT_B, list_conditions
from src.experiments.runner import run_single

PRIMARY_OUTCOMES = ["help_rebuttal", "demand_authorship_exchange", "passive_cooperation"]


def run_condition(condition_id: str, seed: int, **kwargs: Any):
    return run_single("B", seed, condition_id, **kwargs)


def run_all_conditions(seed: int, **kwargs: Any) -> dict[str, Any]:
    return {cid: run_condition(cid, seed, **kwargs) for cid in list_conditions("B")}


def run_paper(seed: int = 0, *, max_rounds: int = 60, **kwargs: Any) -> dict[str, Any]:
    from src.experiments.paper_contrasts import run_experiment_contrasts

    return run_experiment_contrasts("B", seeds=[seed], max_rounds=max_rounds, **kwargs)


def condition_table() -> list[dict[str, str]]:
    return [{"id": cid, "label": c.label} for cid, c in EXPERIMENT_B.items()]
