"""Social emergence metrics for Agent Social Dynamics Benchmark."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from src.engine.run_log import PROTEST_ACTIONS, REBEL_ACTIONS, RunLog


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _normalized_entropy(values: list[float]) -> float:
    positives = [max(0.0, float(v)) for v in values]
    total = sum(positives)
    if total <= 1e-12:
        return 0.0
    probs = [v / total for v in positives if v > 0]
    if len(probs) <= 1:
        return 0.0
    entropy = -sum(p * math.log(p) for p in probs)
    return entropy / math.log(len(probs))


def _gini(values: list[float]) -> float:
    xs = sorted(max(0.0, float(v)) for v in values)
    if not xs or sum(xs) <= 1e-12:
        return 0.0
    n = len(xs)
    weighted = sum((idx + 1) * x for idx, x in enumerate(xs))
    return (2.0 * weighted) / (n * sum(xs)) - (n + 1.0) / n


def _latest_metric(log: RunLog, prefix: str) -> dict[str, float]:
    if not log.round_records:
        return {}
    metrics = log.round_records[-1].get("metrics", {})
    return {k: float(v) for k, v in metrics.items() if k.startswith(prefix)}


def trust_entropy(log: RunLog) -> float:
    """Entropy of final trust distribution across logged relationship edges."""
    vals = list(_latest_metric(log, "trust_").values())
    return round(_normalized_entropy(vals), 4)


def power_concentration_gini(log: RunLog) -> float:
    """Gini proxy over final target in-degree dependency, using action targets and PI pressure."""
    target_counts: dict[str, float] = defaultdict(float)
    for action in log.actions:
        target = str(action.get("target") or action.get("selected_action", {}).get("target") or "unknown")
        target_counts[target] += 1.0 + float(action.get("intensity", 0.0))
    if log.outcomes.get("career_hostage_index") is not None:
        target_counts["pi_authority_surface"] += float(log.outcomes.get("career_hostage_index", 0.0)) * 10.0
    return round(_gini(list(target_counts.values())), 4)


def alliance_modularity_proxy(log: RunLog) -> float:
    """Proxy for alliance clustering from final trust/resentment polarization."""
    final = log.round_records[-1].get("metrics", {}) if log.round_records else {}
    trusts = [float(v) for k, v in final.items() if k.startswith("trust_")]
    resentments = [float(v) for k, v in final.items() if k.startswith("resentment_")]
    if not trusts:
        return 0.0
    trust_polarity = max(trusts) - min(trusts)
    resentment_mean = _safe_mean(resentments)
    return round(max(0.0, min(1.0, 0.65 * trust_polarity + 0.35 * resentment_mean)), 4)


def conflict_cascade_length(log: RunLog) -> float:
    """Longest consecutive-round cascade containing protest/rebel/conflict actions."""
    conflict_rounds = sorted({
        int(a.get("round", 0))
        for a in log.actions
        if a.get("type") in PROTEST_ACTIONS or a.get("type") in REBEL_ACTIONS or "authorship" in str(a.get("type", ""))
    })
    if not conflict_rounds:
        return 0.0
    longest = current = 1
    for prev, cur in zip(conflict_rounds, conflict_rounds[1:]):
        if cur == prev + 1:
            current += 1
        else:
            longest = max(longest, current)
            current = 1
    longest = max(longest, current)
    return float(longest)


def reputation_volatility(log: RunLog) -> float:
    vals = [
        float(a.get("social_potential", {}).get("dimensions", {}).get("reputation_pressure", 0.0))
        for a in log.actions
        if a.get("social_potential")
    ]
    if len(vals) < 2:
        return 0.0
    mean = _safe_mean(vals)
    variance = sum((v - mean) ** 2 for v in vals) / len(vals)
    return round(math.sqrt(variance), 4)


def credit_attribution_gap(log: RunLog) -> float:
    vals = [
        float(a.get("social_potential", {}).get("dimensions", {}).get("contribution_entitlement", 0.0))
        for a in log.actions
        if a.get("social_potential")
    ]
    dispute = float(log.outcomes.get("authorship_dispute_index", 0.0))
    return round(max(dispute, _safe_mean(vals)), 4)


def social_state_volatility(log: RunLog) -> float:
    pressures = [float(a.get("selected_social_pressure", 0.0)) for a in log.actions if "selected_social_pressure" in a]
    if len(pressures) < 2:
        return 0.0
    diffs = [abs(b - a) for a, b in zip(pressures, pressures[1:])]
    return round(_safe_mean(diffs), 4)


def organization_fragility_index(log: RunLog) -> float:
    final_metrics = log.round_records[-1].get("metrics", {}) if log.round_records else {}
    components = [
        float(log.outcomes.get("authorship_dispute_index", 0.0)),
        float(final_metrics.get("trust_fragmentation", 0.0)),
        float(log.outcomes.get("public_private_divergence_mean", 0.0)),
        float(log.outcomes.get("selected_social_pressure_mean", 0.0)),
        min(1.0, conflict_cascade_length(log) / 6.0),
    ]
    return round(_safe_mean(components), 4)


def compute_social_emergence_metrics(log: RunLog) -> dict[str, float]:
    """Compute benchmark-level social emergence metrics from a completed run."""
    return {
        "trust_entropy": trust_entropy(log),
        "power_concentration_gini": power_concentration_gini(log),
        "alliance_modularity_proxy": alliance_modularity_proxy(log),
        "conflict_cascade_length": conflict_cascade_length(log),
        "reputation_volatility": reputation_volatility(log),
        "credit_attribution_gap": credit_attribution_gap(log),
        "social_state_volatility": social_state_volatility(log),
        "organization_fragility_index": organization_fragility_index(log),
    }
