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


def condition_table() -> list[dict[str, str]]:
    return [{"id": cid, "label": c.label} for cid, c in EXPERIMENT_B.items()]
