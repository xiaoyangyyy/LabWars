"""Action diversity helpers — expose recent history without hard action bans."""

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


def avoid_actions(agent: Agent) -> list[str]:
    """No hard action bans; repetition is handled as a continuous tendency penalty."""
    return []


def filter_allowed_actions(allowed: list[str], avoid: list[str], min_keep: int = 3) -> tuple[list[str], list[str]]:
    """Compatibility shim: keep the legal action set intact."""
    return allowed, []


def is_repetitive_choice(agent: Agent, action_type: str) -> bool:
    """Compatibility shim for old callers; repetition is no longer a hard violation."""
    return False


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