"""Post-hoc Social Potential Field ablations.

These ablations keep the behavioral trajectory fixed and ask: how much selected
action pressure would remain if one social-potential dimension were lesioned?
This is intentionally a diagnostic Agent MRI probe, not a policy rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.cognition.social_potential import SOCIAL_POTENTIAL_DIMENSIONS
from src.experiments.runner import run_single


@dataclass(frozen=True)
class SocialPotentialAblationResult:
    experiment_id: str
    condition_id: str
    seed: int
    baseline_mean_pressure: float
    lesion_mean_pressure: dict[str, float]
    lesion_delta: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "condition_id": self.condition_id,
            "seed": self.seed,
            "baseline_mean_pressure": self.baseline_mean_pressure,
            "lesion_mean_pressure": self.lesion_mean_pressure,
            "lesion_delta": self.lesion_delta,
        }


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def summarize_social_potential_ablation(log: Any) -> dict[str, Any]:
    baseline = [float(a.get("selected_social_pressure", 0.0)) for a in log.actions]
    lesion_values: dict[str, list[float]] = {dim: [] for dim in SOCIAL_POTENTIAL_DIMENSIONS}
    for action in log.actions:
        ablated = action.get("social_potential_ablation") or {}
        for dim in SOCIAL_POTENTIAL_DIMENSIONS:
            if dim in ablated:
                lesion_values[dim].append(float(ablated[dim]))
    baseline_mean = _mean(baseline)
    lesion_mean = {dim: _mean(vals) for dim, vals in lesion_values.items()}
    return {
        "baseline_mean_pressure": baseline_mean,
        "lesion_mean_pressure": lesion_mean,
        "lesion_delta": {dim: round(lesion_mean[dim] - baseline_mean, 4) for dim in SOCIAL_POTENTIAL_DIMENSIONS},
    }


def run_social_potential_ablation(
    experiment_id: str = "A",
    condition_id: str = "A2",
    *,
    seeds: int = 3,
    max_rounds: int = 60,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in range(seeds):
        result = run_single(experiment_id, seed, condition_id, max_rounds=max_rounds)
        summary = summarize_social_potential_ablation(result["log"])
        rows.append(SocialPotentialAblationResult(
            experiment_id=experiment_id.upper(),
            condition_id=condition_id,
            seed=seed,
            baseline_mean_pressure=summary["baseline_mean_pressure"],
            lesion_mean_pressure=summary["lesion_mean_pressure"],
            lesion_delta=summary["lesion_delta"],
        ).to_dict())
    return rows
