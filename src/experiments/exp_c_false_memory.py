"""Experiment C — false memory contamination."""

from __future__ import annotations

from typing import Any

from src.experiments.conditions import EXPERIMENT_C, list_conditions
from src.experiments.runner import run_single

PRIMARY_OUTCOMES = ["trust_phd_b_r25", "trust_phd_b_r44", "trust_phd_b_r60", "trust_recovery_rate"]


def run_condition(condition_id: str, seed: int, **kwargs: Any):
    return run_single("C", seed, condition_id, **kwargs)


def run_all_conditions(seed: int, **kwargs: Any) -> dict[str, Any]:
    return {cid: run_condition(cid, seed, **kwargs) for cid in list_conditions("C")}


def condition_table() -> list[dict[str, str]]:
    return [{"id": cid, "label": c.label} for cid, c in EXPERIMENT_C.items()]
