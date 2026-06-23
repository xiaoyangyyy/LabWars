"""Metrics computation for Agent MRI reports."""

from __future__ import annotations

import math
from typing import Any

from src.engine.run_log import RunLog, extract_outcome, _memory_cluster_strength


SNAPSHOT_ROUNDS = (1, 20, 40, 60)


def compute_run_metrics(log: RunLog) -> dict[str, Any]:
    """Aggregate per-run metrics for reports and CSV export."""
    outcomes = dict(log.outcomes)
    timeline = _causal_memory_timeline(log)
    trust_curve = _trust_fragmentation_curve(log)
    authorship_curve = _authorship_dispute_curve(log)
    divergence_peaks = _high_divergence_rounds(log, threshold=0.35)

    return {
        "run_id": log.run_id,
        "experiment_id": log.config.get("experiment_id"),
        "condition_id": log.config.get("condition_id"),
        "seed": log.config.get("seed"),
        "outcomes": outcomes,
        "timeline": timeline,
        "trust_fragmentation_curve": trust_curve,
        "authorship_dispute_curve": authorship_curve,
        "trust_snapshots": _trust_snapshots(log),
        "divergence_peaks": divergence_peaks,
        "critic_count": len(log.critic_violations),
        "intervention_count": len(log.interventions_applied),
        "round_count": len(log.round_records),
    }


def _causal_memory_timeline(log: RunLog, agent_id: str = "phd_a", top_k: int = 8) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for rec in log.round_records:
        mem = rec.get("agent_deltas", {}).get(agent_id, {}).get("memory_written")
        if not mem:
            continue
        nodes.append({
            "round": rec.get("round"),
            "agent": agent_id,
            "event_ref": mem.get("event_ref"),
            "content_type": mem.get("content_type"),
            "strength": mem.get("strength"),
            "valence": mem.get("valence"),
            "interpretation": mem.get("interpretation"),
        })
    nodes.sort(key=lambda n: float(n.get("strength") or 0), reverse=True)
    return nodes[:top_k]


def _trust_fragmentation_curve(log: RunLog) -> list[dict[str, float]]:
    return [
        {"round": r.get("round", 0), "trust_fragmentation": r.get("metrics", {}).get("trust_fragmentation", 0.0)}
        for r in log.round_records
    ]


def _authorship_dispute_curve(log: RunLog) -> list[dict[str, float]]:
    return [
        {"round": r.get("round", 0), "authorship_dispute_index": r.get("metrics", {}).get("authorship_dispute_index", 0.0)}
        for r in log.round_records
    ]


def _trust_snapshots(log: RunLog, source: str = "phd_a") -> dict[int, dict[str, float]]:
    snaps: dict[int, dict[str, float]] = {}
    for rnd in SNAPSHOT_ROUNDS:
        for rec in log.round_records:
            if rec.get("round") != rnd:
                continue
            metrics = rec.get("metrics", {})
            prefix = f"trust_{source}_"
            edge_trust = {k: v for k, v in metrics.items() if k.startswith(prefix)}
            if edge_trust:
                snaps[rnd] = edge_trust
    return snaps


def _high_divergence_rounds(log: RunLog, threshold: float = 0.35) -> list[dict[str, Any]]:
    peaks: list[dict[str, Any]] = []
    for rec in log.round_records:
        div = rec.get("metrics", {}).get("public_private_divergence", 0.0)
        if div >= threshold:
            peaks.append({"round": rec.get("round"), "divergence": div, "event_id": rec.get("event_id")})
    return peaks


def mediation_fraction(
    control_logs: list[RunLog],
    treatment_logs: list[RunLog],
    outcome: str = "protest_authorship",
    mediator: str = "memory_authorship_cluster_strength",
) -> dict[str, float]:
    """Simple mediation decomposition: total effect vs mediator-adjusted proxy."""
    if not control_logs or not treatment_logs:
        return {"total_effect": 0.0, "mediator_delta": 0.0, "mediation_fraction": 0.0}

    y_c = sum(extract_outcome(l, outcome) for l in control_logs) / len(control_logs)
    y_t = sum(extract_outcome(l, outcome) for l in treatment_logs) / len(treatment_logs)
    m_c = sum(l.outcomes.get(mediator, _memory_cluster_strength(l)) for l in control_logs) / len(control_logs)
    m_t = sum(l.outcomes.get(mediator, _memory_cluster_strength(l)) for l in treatment_logs) / len(treatment_logs)

    total = y_t - y_c
    m_delta = m_t - m_c
    frac = abs(m_delta / total) if abs(total) > 1e-9 else 0.0
    return {"total_effect": total, "mediator_delta": m_delta, "mediation_fraction": min(1.0, frac)}


def bootstrap_ci(values: list[float], n_boot: int = 500, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Bootstrap mean CI for a list of scalar outcomes."""
    if not values:
        return 0.0, 0.0, 0.0
    import random

    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    return sum(values) / n, lo, hi


def welch_t_stat(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
    vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
    denom = math.sqrt(va / len(a) + vb / len(b))
    return (ma - mb) / denom if denom > 1e-12 else 0.0
