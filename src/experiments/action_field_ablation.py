"""Action-field calibration and ablation utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.engine.action_selection import clear_action_field_override, set_action_field_override
from src.engine.run_log import extract_outcome
from src.engine.simulation import SimConfig, run_simulation


@dataclass
class ActionFieldAblationResult:
    outcome: str
    n: int
    control_mean: float
    ablation_mean: float
    delta: float
    per_seed: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clone_config(base: SimConfig, seed: int) -> SimConfig:
    return SimConfig(**{**base.__dict__, "seed": seed})


def run_action_field_ablation(
    base_config: SimConfig,
    override: dict[str, Any],
    *,
    seeds: list[int] | None = None,
    n: int = 10,
    outcome: str = "authorship_escalation_score",
) -> ActionFieldAblationResult:
    """Compare baseline action-field parameters with a runtime override.

    This is deliberately small and deterministic: it lets researchers test
    whether a result depends on hand-set motive weights before claiming a
    social-dynamics interpretation.
    """
    seed_values = seeds if seeds is not None else list(range(n))
    rows: list[dict[str, Any]] = []

    for seed in seed_values:
        clear_action_field_override()
        control_log = run_simulation(_clone_config(base_config, seed))
        y_control = float(extract_outcome(control_log, outcome))

        set_action_field_override(override)
        ablation_log = run_simulation(_clone_config(base_config, seed))
        y_ablation = float(extract_outcome(ablation_log, outcome))
        clear_action_field_override()

        rows.append({
            "seed": seed,
            "Y_control": y_control,
            "Y_ablation": y_ablation,
            "delta": y_ablation - y_control,
        })

    c_mean = sum(r["Y_control"] for r in rows) / len(rows) if rows else 0.0
    a_mean = sum(r["Y_ablation"] for r in rows) / len(rows) if rows else 0.0
    return ActionFieldAblationResult(
        outcome=outcome,
        n=len(rows),
        control_mean=round(c_mean, 6),
        ablation_mean=round(a_mean, 6),
        delta=round(a_mean - c_mean, 6),
        per_seed=rows,
    )
