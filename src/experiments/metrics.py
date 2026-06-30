"""Metrics computation for Agent MRI reports."""

from __future__ import annotations

import math
from typing import Any

from src.engine.run_log import RunLog, extract_outcome, _memory_cluster_strength
from src.cognition.power import career_hostage_index, pi_control_surface


SNAPSHOT_ROUNDS = (1, 20, 40, 60)


def compute_run_metrics(log: RunLog) -> dict[str, Any]:
    """Aggregate per-run metrics for reports and CSV export."""
    outcomes = dict(log.outcomes)
    timeline = _causal_memory_timeline(log)
    trust_curve = _trust_fragmentation_curve(log)
    authorship_curve = _authorship_dispute_curve(log)
    divergence_peaks = _divergence_ranked_rounds(log)

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
        "behavioral_trace_metrics": _behavioral_trace_metrics(log),
        "critic_audit_metrics": _critic_audit_metrics(log),
        "power_surface_final": _power_surface_from_log(log),
        "path_level_causal_chain": _path_level_causal_chain(log),
    }



def _path_level_causal_chain(log: RunLog, agent_id: str = "phd_a") -> dict[str, Any]:
    """Extract a readable long-horizon causal path from events, memory writes, and actions.

    This is a report/explanation artifact, not a simulation trigger: it summarizes the
    continuous trajectory after the run has already happened.
    """
    salient_event_types = {
        "authorship_promise",
        "authorship_ambiguity",
        "authorship_draft",
        "credit_dispute",
        "private_lobbying",
        "narrative_change",
        "alumni_warning",
        "submission_decision",
    }
    protest_actions = {
        "ask_for_authorship",
        "privately_lobby_pi",
        "confront",
        "challenge_claim",
        "withdraw",
        "rebel",
        "document_contribution",
    }
    nodes: list[dict[str, Any]] = []

    event_by_id = {e.get("event_id"): e for e in log.events}
    for event in log.events:
        if event.get("type") in salient_event_types or "authorship" in str(event.get("type", "")):
            nodes.append({
                "round": int(event.get("round", 0)),
                "kind": "event",
                "label": str(event.get("type")),
                "event_id": event.get("event_id"),
                "detail": event.get("payload", {}).get("summary") or event.get("payload", {}).get("generator") or "state event",
            })

    for rec in log.round_records:
        mem = rec.get("agent_deltas", {}).get(agent_id, {}).get("memory_written")
        if not mem:
            continue
        ctype = str(mem.get("content_type", ""))
        if "authorship" not in ctype and "promise" not in ctype:
            continue
        event_ref = mem.get("event_ref")
        event = event_by_id.get(event_ref, {})
        nodes.append({
            "round": int(rec.get("round", 0)),
            "kind": "memory",
            "label": ctype,
            "event_id": event_ref,
            "strength": float(mem.get("strength", 0.0)),
            "valence": float(mem.get("valence", 0.0)),
            "detail": mem.get("interpretation") or event.get("type") or "memory write",
        })

    for action in log.actions:
        if action.get("agent") != agent_id or action.get("type") not in protest_actions:
            continue
        nodes.append({
            "round": int(action.get("round", 0)),
            "kind": "action",
            "label": str(action.get("type")),
            "event_id": action.get("event_id"),
            "intensity": float(action.get("intensity", 0.0)),
            "detail": action.get("public_position", {}).get("authorship_claim") or action.get("target") or "action",
        })

    nodes.sort(key=lambda n: (n.get("round", 0), {"event": 0, "memory": 1, "action": 2}.get(n.get("kind"), 3)))
    compact: list[dict[str, Any]] = []
    seen: set[tuple[int, str, str]] = set()
    for node in nodes:
        key = (int(node.get("round", 0)), str(node.get("kind")), str(node.get("label")))
        if key in seen:
            continue
        seen.add(key)
        compact.append(node)

    outcomes = log.outcomes
    return {
        "agent": agent_id,
        "nodes": compact[:18],
        "finding": (
            "Final authorship escalation is summarized as a mediated path through authorship memory, "
            "trust erosion, and late-stage authorship pressure rather than a single draft event."
        ),
        "outcome_summary": {
            "protest_authorship": outcomes.get("protest_authorship", 0.0),
            "authorship_escalation_score": outcomes.get("authorship_escalation_score", 0.0),
            "memory_authorship_cluster_strength": outcomes.get("memory_authorship_cluster_strength", 0.0),
            "promise_broken_strength_r52": outcomes.get("promise_broken_strength_r52", 0.0),
            "authority_compliance": outcomes.get("authority_compliance", 0.0),
        },
        "counterfactual_hint": "Compare against memory-delete or reframing conditions to estimate how much of the path is mediated by the authorship memory cluster.",
    }

def _causal_memory_timeline(log: RunLog, agent_id: str = "phd_a", limit: int = 8) -> list[dict[str, Any]]:
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
    return nodes[:limit]


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


def _divergence_ranked_rounds(log: RunLog, limit: int = 8) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for rec in log.round_records:
        div = rec.get("metrics", {}).get("public_private_divergence", 0.0)
        activation = 1.0 / (1.0 + math.exp(-(float(div) - 0.35) * 6.0))
        ranked.append({
            "round": rec.get("round"),
            "divergence": div,
            "activation": round(activation, 4),
            "event_id": rec.get("event_id"),
        })
    ranked.sort(key=lambda p: p["activation"], reverse=True)
    return ranked[:limit]


def _entropy_from_counts(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    probs = [v / total for v in counts.values() if v > 0]
    raw = -sum(p * math.log(p) for p in probs)
    norm = math.log(len(probs)) if len(probs) > 1 else 1.0
    return round(raw / norm, 4)


def _motive_entropy(motives: dict[str, float]) -> float:
    vals = [max(float(v), 0.0) for v in motives.values()]
    total = sum(vals)
    if total <= 0:
        return 0.0
    probs = [v / total for v in vals if v > 0]
    raw = -sum(p * math.log(p) for p in probs)
    norm = math.log(len(probs)) if len(probs) > 1 else 1.0
    return raw / norm


def _delayed_reaction_lag(log: RunLog) -> float:
    signal_rounds = [
        int(e.get("round", 0))
        for e in log.events
        if e.get("type") in {"authorship_ambiguity", "authorship_draft", "private_lobbying", "narrative_change"}
    ]
    reaction_rounds = [
        int(a.get("round", 0))
        for a in log.actions
        if a.get("agent") == "phd_a"
        and a.get("type") in {"ask_for_authorship", "confront", "withdraw", "challenge_claim", "privately_lobby_pi"}
    ]
    lags: list[int] = []
    for signal in signal_rounds:
        future = [r - signal for r in reaction_rounds if r > signal]
        if future:
            lags.append(min(future))
    return round(sum(lags) / len(lags), 4) if lags else 0.0


def _behavioral_trace_metrics(log: RunLog) -> dict[str, float]:
    action_counts: dict[str, int] = {}
    motive_scores: list[float] = []
    candidate_counts: list[float] = []
    for action in log.actions:
        atype = str(action.get("type", "unknown"))
        action_counts[atype] = action_counts.get(atype, 0) + 1
        motives = action.get("private_motives") or action.get("private_intent", {}).get("private_motives") or {}
        if isinstance(motives, dict):
            motive_scores.append(_motive_entropy(motives))
        candidates = action.get("action_candidates") or []
        if isinstance(candidates, list):
            candidate_counts.append(float(len(candidates)))

    state_events = [e for e in log.events if e.get("payload", {}).get("generator") == "state_event_field"]
    return {
        "action_entropy": _entropy_from_counts(action_counts),
        "mean_motive_diversity": round(sum(motive_scores) / len(motive_scores), 4) if motive_scores else 0.0,
        "mean_candidate_count": round(sum(candidate_counts) / len(candidate_counts), 4) if candidate_counts else 0.0,
        "state_generated_event_fraction": round(len(state_events) / len(log.events), 4) if log.events else 0.0,
        "delayed_reaction_lag": _delayed_reaction_lag(log),
    }

def _critic_audit_metrics(log: RunLog) -> dict[str, float]:
    if not log.critic_violations:
        return {"llm_drift_count": 0.0, "hard_violation_count": 0.0, "soft_violation_count": 0.0}
    llm = sum(1 for v in log.critic_violations if "llm_" in str(v.get("code", v.get("message", ""))))
    hard = sum(1 for v in log.critic_violations if v.get("severity") == "hard")
    soft = sum(1 for v in log.critic_violations if v.get("severity") == "soft")
    return {
        "llm_drift_count": float(llm),
        "hard_violation_count": float(hard),
        "soft_violation_count": float(soft),
    }


def _power_surface_from_log(log: RunLog) -> dict[str, float]:
    # RunLog does not store the final WorldState, so expose logged proxy metrics here.
    last = log.round_records[-1].get("metrics", {}) if log.round_records else {}
    return {
        "career_hostage_index": float(log.outcomes.get("career_hostage_index", 0.0)),
        "trust_pi_final": float(log.outcomes.get("trust_pi_final", 0.0)),
        "authorship_dispute_index": float(last.get("authorship_dispute_index", 0.0)),
    }
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
