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



def _linear_regression(xs: list[float], ys: list[float]) -> tuple[float, float]:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0, 0.0
    mx = _safe_mean(xs)
    my = _safe_mean(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    if denom <= 1e-12:
        return 0.0, 0.0
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return slope, max(0.0, min(1.0, r2))


def _power_law_fit(values: list[float]) -> dict[str, float]:
    xs = sorted([float(v) for v in values if float(v) > 0], reverse=True)
    if len(xs) < 4:
        return {"alpha": 0.0, "r2": 0.0}
    ccdf_x: list[float] = []
    ccdf_y: list[float] = []
    n = len(xs)
    for idx, value in enumerate(sorted(xs)):
        tail = sum(1 for item in xs if item >= value) / n
        if value > 0 and tail > 0:
            ccdf_x.append(math.log(value))
            ccdf_y.append(math.log(tail))
    slope, r2 = _linear_regression(ccdf_x, ccdf_y)
    return {"alpha": round(max(0.0, -slope), 4), "r2": round(r2, 4)}


def _target_power_values(log: RunLog) -> list[float]:
    counts: dict[str, float] = defaultdict(float)
    for action in log.actions:
        target = str(action.get("target") or action.get("selected_action", {}).get("target") or "unknown")
        counts[target] += 1.0 + float(action.get("intensity", 0.0))
    return list(counts.values())


def power_law_alpha(log: RunLog) -> float:
    return _power_law_fit(_target_power_values(log))["alpha"]


def power_law_fit_r2(log: RunLog) -> float:
    return _power_law_fit(_target_power_values(log))["r2"]


def network_modularity_q(log: RunLog) -> float:
    final = log.round_records[-1].get("metrics", {}) if log.round_records else {}
    edges: list[tuple[str, str, float]] = []
    nodes: set[str] = set()
    for key, value in final.items():
        if not key.startswith("trust_"):
            continue
        rest = key[len("trust_"):]
        if "_" not in rest:
            continue
        src, tgt = rest.rsplit("_", 1)
        weight = max(0.0, float(value) - 0.5)
        if weight <= 0:
            continue
        edges.append((src, tgt, weight))
        nodes.update([src, tgt])
    if len(nodes) < 3 or not edges:
        return 0.0
    # Greedy proxy: partition by each node's strongest trusted target.
    strongest: dict[str, str] = {}
    for src, tgt, weight in edges:
        if weight > float(strongest.get(src + ":w", -1.0)):
            strongest[src] = tgt
            strongest[src + ":w"] = str(weight)
    communities = {node: strongest.get(node, node) for node in nodes}
    m = sum(w for _, _, w in edges)
    out_strength: dict[str, float] = defaultdict(float)
    in_strength: dict[str, float] = defaultdict(float)
    for src, tgt, weight in edges:
        out_strength[src] += weight
        in_strength[tgt] += weight
    q = 0.0
    for src, tgt, weight in edges:
        if communities.get(src) == communities.get(tgt):
            q += weight - (out_strength[src] * in_strength[tgt] / max(m, 1e-12))
    return round(max(0.0, min(1.0, q / max(m, 1e-12))), 4)


def _cascade_sizes(log: RunLog) -> list[int]:
    conflict_rounds = sorted({
        int(a.get("round", 0))
        for a in log.actions
        if a.get("type") in PROTEST_ACTIONS or a.get("type") in REBEL_ACTIONS or "authorship" in str(a.get("type", ""))
    })
    if not conflict_rounds:
        return []
    sizes: list[int] = []
    current = 1
    for prev, cur in zip(conflict_rounds, conflict_rounds[1:]):
        if cur == prev + 1:
            current += 1
        else:
            sizes.append(current)
            current = 1
    sizes.append(current)
    return sizes


def cascade_tail_alpha(log: RunLog) -> float:
    return _power_law_fit([float(v) for v in _cascade_sizes(log)])["alpha"]


def cascade_tail_r2(log: RunLog) -> float:
    return _power_law_fit([float(v) for v in _cascade_sizes(log)])["r2"]


def emergent_pattern_score(log: RunLog) -> float:
    parts = [
        power_law_fit_r2(log),
        network_modularity_q(log),
        cascade_tail_r2(log),
        min(1.0, conflict_cascade_length(log) / 10.0),
    ]
    return round(_safe_mean(parts), 4)
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
        "power_law_alpha": power_law_alpha(log),
        "power_law_fit_r2": power_law_fit_r2(log),
        "network_modularity_q": network_modularity_q(log),
        "cascade_tail_alpha": cascade_tail_alpha(log),
        "cascade_tail_r2": cascade_tail_r2(log),
        "emergent_pattern_score": emergent_pattern_score(log),
    }
