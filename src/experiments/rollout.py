"""Stochastic rollout and state-event counterfactual utilities."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from src.engine.run_log import extract_outcome
from src.engine.simulation import SimConfig, run_simulation


def _clone_config(base: SimConfig, **overrides: Any) -> SimConfig:
    data = {**base.__dict__, **overrides}
    return SimConfig(**data)


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "variance": 0.0, "min": 0.0, "max": 0.0}
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return {
        "mean": round(mean, 6),
        "variance": round(var, 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def run_stochastic_rollouts(
    base_config: SimConfig,
    *,
    seeds: list[int] | None = None,
    n: int = 10,
    outcomes: list[str] | None = None,
) -> dict[str, Any]:
    """Run repeated stochastic simulations and aggregate behavioral outcomes."""
    seed_values = seeds if seeds is not None else list(range(n))
    outcome_names = outcomes or [
        "protest_authorship",
        "authorship_escalation_score",
        "authorship_escalation_potential",
        "authorship_dispute_index",
        "public_private_divergence_mean",
        "memory_authorship_cluster_strength",
    ]
    rows: list[dict[str, Any]] = []
    by_outcome: dict[str, list[float]] = {name: [] for name in outcome_names}

    for seed in seed_values:
        cfg = _clone_config(base_config, seed=seed)
        log = run_simulation(cfg)
        row = {"seed": seed, "run_id": log.run_id}
        for name in outcome_names:
            value = float(extract_outcome(log, name))
            row[name] = value
            by_outcome[name].append(value)
        generated = [e for e in log.events if e.get("payload", {}).get("generator") == "state_event_field"]
        row["state_generated_event_count"] = len(generated)
        rows.append(row)

    return {
        "n": len(seed_values),
        "seeds": seed_values,
        "rows": rows,
        "summary": {name: _summary(values) for name, values in by_outcome.items()},
        "state_generated_event_count": _summary([float(r["state_generated_event_count"]) for r in rows]),
    }


def run_state_event_counterfactual(
    base_config: SimConfig,
    *,
    seeds: list[int] | None = None,
    n: int = 10,
    outcome: str = "authorship_escalation_score",
) -> dict[str, Any]:
    """Compare normal state-event simulation against anchor-only event replay."""
    seed_values = seeds if seeds is not None else list(range(n))
    per_seed: list[dict[str, float]] = []

    for seed in seed_values:
        state_cfg = _clone_config(base_config, seed=seed, disable_state_events=False)
        anchor_cfg = _clone_config(base_config, seed=seed, disable_state_events=True)
        state_log = run_simulation(state_cfg)
        anchor_log = run_simulation(anchor_cfg)
        y_state = float(extract_outcome(state_log, outcome))
        y_anchor = float(extract_outcome(anchor_log, outcome))
        per_seed.append({
            "seed": float(seed),
            "Y_state_event": y_state,
            "Y_anchor_only": y_anchor,
            "delta_state_minus_anchor": y_state - y_anchor,
        })

    deltas = [row["delta_state_minus_anchor"] for row in per_seed]
    return {
        "outcome": outcome,
        "n": len(seed_values),
        "per_seed": per_seed,
        "delta_summary": _summary(deltas),
        "state_event_mean": _summary([row["Y_state_event"] for row in per_seed])["mean"],
        "anchor_only_mean": _summary([row["Y_anchor_only"] for row in per_seed])["mean"],
    }
