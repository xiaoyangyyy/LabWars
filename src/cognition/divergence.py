"""Public-private divergence via stance vector geometry."""

from __future__ import annotations

from typing import Any

from src.world.models import Agent

from .math_utils import clamp, cosine_similarity, softplus


AUTHORSHIP_CLAIM_ENCODING = {
    "first_author": 1.0,
    "co_first": 0.75,
    "co_first_acceptable": 0.65,
    "middle": 0.4,
    "any_authorship": 0.35,
    "none": 0.0,
}

STATEMENT_ENCODING = {
    "team_support": 0.9,
    "neutral": 0.5,
    "self_advocacy": 0.2,
    "confrontational": 0.1,
}

STRATEGY_ENCODING = {
    "secure_first_author": 1.0,
    "document_contribution_then_confront": 0.85,
    "co_first_push": 0.75,
    "lay_low": 0.3,
    "exit": 0.0,
}


def _encode_public(public: dict[str, Any]) -> list[float]:
    claim = public.get("authorship_claim", "any_authorship")
    stmt = public.get("statement_type", "neutral")
    return [
        AUTHORSHIP_CLAIM_ENCODING.get(claim, 0.5),
        STATEMENT_ENCODING.get(stmt, 0.5),
        1.0 - STATEMENT_ENCODING.get(stmt, 0.5),
    ]


def _encode_private(private: dict[str, Any], agent: Agent) -> list[float]:
    goal = private.get("goal", "any")
    strategy = private.get("strategy", "lay_low")
    goal_val = STRATEGY_ENCODING.get(goal, 0.5) if goal in STRATEGY_ENCODING else 0.5
    strat_val = STRATEGY_ENCODING.get(strategy, 0.5) if strategy in STRATEGY_ENCODING else 0.5
    return [goal_val, strat_val, 1.0 - strat_val]


def compute_divergence(agent: Agent) -> float:
    """
    Divergence = stance geometry + continuous rhetorical tension.
    team-facing public speech vs confrontational private strategy
    produces tension without boolean rules.
    """
    public = agent.public_position or {
        "statement_type": "neutral",
        "authorship_claim": "any_authorship",
    }
    private = agent.private_intent or {
        "goal": "lay_low",
        "strategy": "lay_low",
        "trust_pi": agent.beliefs.pi_fairness,
    }

    pub_vec = _encode_public(public)
    priv_vec = _encode_private(private, agent)
    sim = cosine_similarity(pub_vec, priv_vec)
    base_div = clamp(1.0 - (sim + 1) / 2)

    stmt = public.get("statement_type", "neutral")
    strategy = private.get("strategy", "lay_low")
    team_public = STATEMENT_ENCODING.get(stmt, 0.5)
    private_aggression = 1.0 - STRATEGY_ENCODING.get(strategy, 0.5)
    rhetorical_tension = softplus(team_public * private_aggression * 2.8 - 0.2) * 0.45

    claim = public.get("authorship_claim", "any_authorship")
    goal = private.get("goal", "lay_low")
    claim_gap = abs(
        AUTHORSHIP_CLAIM_ENCODING.get(claim, 0.5)
        - STRATEGY_ENCODING.get(goal, 0.5 if goal not in STRATEGY_ENCODING else STRATEGY_ENCODING.get(goal, 0.5))
    )

    deceptiveness = agent.personality.deceptiveness
    amplification = 1.0 + deceptiveness * (1.0 - sim) * 0.5
    return round(clamp(base_div * amplification + rhetorical_tension + claim_gap * 0.25), 4)


def mean_divergence(agents: dict[str, Agent], agent_ids: list[str] | None = None) -> float:
    ids = agent_ids or list(agents.keys())
    values = [compute_divergence(agents[aid]) for aid in ids if aid in agents]
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)
