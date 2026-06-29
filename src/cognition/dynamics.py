"""Nonlinear coupling primitives — smooth hysteresis and saturation."""

from __future__ import annotations

import math
from typing import Any

from src.world.models import Agent, EventAtom

from .math_utils import clamp, impulse_response, logistic_gate, recency_kernel

AUTHorship_MEMORY_TYPES = ("authorship_signal", "promise_fulfilled", "promise_broken")

ESCALATED_ACTIONS = frozenset({"confront", "rebel", "challenge_claim", "withdraw", "leak_concern"})
COMPLIANCE_ACTIONS = frozenset({"comply", "lay_low", "delay_response", "support_teammate"})


def authorship_memory_cluster(
    agent: Agent,
    *,
    round_min: int = 0,
    round_max: int = 999,
    current_round: int | None = None,
) -> float:
    """Saturating sum of authorship-related memories — sublinear in raw strength."""
    total = 0.0
    now = current_round or round_max
    for mem in agent.memory:
        rnd = mem.get("round", 0)
        if rnd < round_min or rnd > round_max:
            continue
        if mem.get("content_type") not in AUTHorship_MEMORY_TYPES:
            continue
        strength = float(mem.get("strength", 0))
        age = max(0, now - rnd)
        recency = recency_kernel(age, half_life=18.0)
        total += (1.0 - math.exp(-strength * recency))
    return total


def saturating_memory_cluster_from_records(
    memories: list[dict[str, Any]],
    *,
    round_min: int = 0,
    round_max: int = 999,
    current_round: int | None = None,
) -> float:
    total = 0.0
    now = current_round or round_max
    for mem in memories:
        rnd = mem.get("round", 0)
        if rnd < round_min or rnd > round_max:
            continue
        if mem.get("content_type") not in AUTHorship_MEMORY_TYPES:
            continue
        strength = float(mem.get("strength", 0))
        age = max(0, now - rnd)
        recency = recency_kernel(age, half_life=18.0)
        total += (1.0 - math.exp(-strength * recency))
    return total


def nonlinear_belief_target(
    prior: float,
    signed_shock: float,
    salience: float,
    *,
    cluster_amp: float = 0.0,
    positive_boost: float = 1.0,
) -> float:
    """
    Move belief with hysteresis: betrayal amplified by promise cluster;
    positive shocks (explicit promise, honored draft) use separate saturation.
    """
    negative_gate = logistic_gate(-signed_shock, center=0.0, steepness=12.0)
    positive_gate = 1.0 - negative_gate
    betrayal_amp = cluster_amp * (0.15 + 0.85 * negative_gate)
    amp = 1.0 + betrayal_amp
    magnitude = abs(signed_shock) * salience * amp
    pos_delta = impulse_response(magnitude, sensitivity=0.62 * positive_boost, saturation=2.8)
    neg_delta = -impulse_response(magnitude, sensitivity=0.58, saturation=2.4)
    return clamp(prior + positive_gate * pos_delta + negative_gate * neg_delta)
    return clamp(prior + delta)


def nonlinear_recall_shift(prior: float, valence: float, strength: float) -> float:
    magnitude = abs(valence) * strength
    delta = impulse_response(magnitude, sensitivity=0.28, saturation=3.0)
    positive_gate = 1.0 - logistic_gate(-valence, center=0.0, steepness=12.0)
    signed_delta = positive_gate * delta - (1.0 - positive_gate) * delta
    return clamp(prior + signed_delta)


def apply_saturating_emotion_delta(current: float, impulse: float) -> float:
    positive_gate = 1.0 - logistic_gate(-impulse, center=0.0, steepness=12.0)
    magnitude = abs(impulse)
    pos = clamp(current + impulse_response(magnitude, sensitivity=1.0, saturation=3.2))
    neg = clamp(current - impulse_response(magnitude, sensitivity=1.0, saturation=3.2))
    return clamp(positive_gate * pos + (1.0 - positive_gate) * neg)

def escalation_potential_from_state(
    beliefs: dict[str, float],
    emotion: dict[str, float],
    *,
    promise_broken: float = 0.0,
    promise_cluster: float = 0.0,
) -> float:
    """Latent protest potential before R52 actions — smooth phase-transition proxy."""
    unfairness = 1.0 - beliefs.get("pi_fairness", 0.5)
    resentment = emotion.get("resentment", 0.0)
    anger = emotion.get("anger", 0.0)
    memory_load = logistic_gate(promise_cluster, center=0.35, steepness=4.0)
    broken_load = logistic_gate(promise_broken, center=0.30, steepness=5.5)
    emotional_gate = logistic_gate(resentment, center=0.42, steepness=5.0)
    unfairness_gate = logistic_gate(unfairness, center=0.32, steepness=4.5)
    anger_gate = 0.55 + 0.45 * logistic_gate(anger, center=0.28, steepness=4.0)
    memory_term = 0.30 + 0.45 * memory_load + 0.35 * broken_load
    return clamp(emotional_gate * unfairness_gate * memory_term * anger_gate)


def action_escalation_impulse(
    action_type: str,
    intensity: float,
    *,
    escalated_weight: float = 0.55,
    soft_weight: float = 0.12,
) -> float:
    inten = max(0.0, min(1.0, intensity))
    if action_type in ESCALATED_ACTIONS:
        damp = inten * (0.65 + 0.35 * logistic_gate(inten, center=0.6, steepness=8.0))
        return escalated_weight * damp
    if action_type in {"ask_for_authorship", "privately_lobby_pi"}:
        return soft_weight * inten
    return 0.0


def combine_escalation_score(potential: float, action_impulse_sum: float) -> float:
    """Potential × saturating action perturbation — avoids linear ceiling at 1.0."""
    return clamp(potential * (1.0 - math.exp(-action_impulse_sum)))


def draft_rank_shock(agent: Agent, event: EventAtom, cluster: float) -> tuple[float, float]:
    """Return (fairness_shock, first_author_shock) for authorship_draft."""
    order = event.payload.get("author_order", [])
    if agent.id not in order:
        return 0.0, 0.0
    rank = order.index(agent.id)
    if event.payload.get("draft_severity") == "honored" or event.framing == "positive":
        if rank == 0:
            return 0.28, 0.32
        return 0.05, 0.08
    cluster_gate = logistic_gate(cluster, center=0.35, steepness=4.0)
    expected_rank = 1.0 - cluster_gate
    rank_gap = max(0.0, rank - expected_rank)
    honor_pull = logistic_gate(0.15 - rank_gap, center=0.0, steepness=8.0) * (1.0 - min(1.0, float(rank)))
    fairness_shock = 0.12 * honor_pull - 0.22 * (1.0 + cluster_gate * 0.8) * (1.0 + 0.4 * rank_gap) * (1.0 - honor_pull)
    author_shock = 0.08 * honor_pull - 0.18 * rank_gap * (1.0 - honor_pull)
    return fairness_shock, author_shock
