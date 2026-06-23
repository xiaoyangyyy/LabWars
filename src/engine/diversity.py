"""Action diversity helpers — reduce repetitive LLM policy loops."""

from __future__ import annotations

from collections import Counter
from typing import Any

from src.world.models import Agent


def recent_action_history(agent: Agent, n: int = 5) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for entry in agent.action_history[-n:]:
        act = entry.get("action", {})
        history.append({
            "round": entry.get("round"),
            "type": act.get("type"),
            "target": act.get("target"),
            "intensity": act.get("intensity"),
        })
    return history


def action_usage_counts(agent: Agent, window: int = 8) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for entry in agent.action_history[-window:]:
        atype = entry.get("action", {}).get("type")
        if atype:
            counts[str(atype)] += 1
    return dict(counts)


def avoid_actions(agent: Agent, *, streak_limit: int = 2, saturation: int = 4, window: int = 8) -> list[str]:
    """Actions the LLM should not pick this round."""
    avoid: set[str] = set()
    recent = [h.get("action", {}).get("type") for h in agent.action_history[-streak_limit:]]
    recent = [t for t in recent if t]
    if len(recent) >= streak_limit and len(set(recent)) == 1:
        avoid.add(str(recent[-1]))

    counts = action_usage_counts(agent, window=window)
    for atype, count in counts.items():
        if count >= saturation:
            avoid.add(atype)

    return sorted(avoid)


def filter_allowed_actions(allowed: list[str], avoid: list[str], min_keep: int = 3) -> tuple[list[str], list[str]]:
    if not avoid:
        return allowed, []
    filtered = [a for a in allowed if a not in avoid]
    if len(filtered) >= min_keep:
        return filtered, avoid
    # Keep at least min_keep options — drop avoid for least-used saturated only
    if len(allowed) <= min_keep:
        return allowed, []
    trimmed = avoid[:-1] if len(avoid) > 1 else []
    filtered = [a for a in allowed if a not in trimmed]
    return (filtered if len(filtered) >= min_keep else allowed), trimmed


def is_repetitive_choice(agent: Agent, action_type: str, *, streak_limit: int = 2) -> bool:
    recent = [h.get("action", {}).get("type") for h in agent.action_history[-(streak_limit - 1):]]
    return len(recent) == streak_limit - 1 and all(t == action_type for t in recent)


def memory_action_hints(agent: Agent, recall: Any | None, max_hooks: int = 4) -> list[str]:
    """Soft hints from recalled memory — not mandatory actions."""
    if not recall or not recall.attention_weights:
        return []
    hooks: list[str] = []
    ranked = sorted(recall.attention_weights.items(), key=lambda x: x[1], reverse=True)[:3]
    for mem_id, _ in ranked:
        mem = next((m for m in agent.memory if m.get("memory_id") == mem_id), None)
        if not mem:
            continue
        for hook in mem.get("behavioral_hooks", [])[:2]:
            if hook not in hooks:
                hooks.append(hook)
            if len(hooks) >= max_hooks:
                return hooks
    return hooks


def diversity_metrics(actions: list[dict[str, Any]]) -> dict[str, Any]:
    by_agent: dict[str, Counter[str]] = {}
    for act in actions:
        aid = act.get("agent", "?")
        by_agent.setdefault(aid, Counter())[act.get("type", "?")] += 1

    agent_stats = {}
    for aid, counter in by_agent.items():
        total = sum(counter.values())
        unique = len(counter)
        top = counter.most_common(1)[0] if counter else ("", 0)
        agent_stats[aid] = {
            "unique_actions": unique,
            "total": total,
            "top_action": top[0],
            "top_share": round(top[1] / total, 3) if total else 0.0,
        }
    return {"by_agent": agent_stats}
