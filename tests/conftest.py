"""Shared test fixtures — inject deterministic fake LLM (not heuristic policy)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from src.engine.llm_adapter import LLMAdapter


class FakeLLMAdapter(LLMAdapter):
    """Test double: schema-valid JSON from prompt hash. Does not implement softmax/heuristic scoring."""

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        if "memory interpretation" in system.lower():
            payload = json.loads(user)
            return {
                "interpretation": (
                    f"I remember {payload.get('event_type')} as "
                    f"{'threatening' if payload.get('valence', 0) < 0 else 'reassuring'} to my credit."
                )
            }

        payload = json.loads(user)
        allowed = payload.get("allowed_actions") or ["document_contribution"]
        avoid = set(payload.get("avoid_actions") or [])
        candidates = [a for a in allowed if a not in avoid] or allowed
        agent_id = payload.get("state", {}).get("agent_id", "phd_a")
        event_type = payload.get("current_event", {}).get("type", "team_meeting")
        round_num = payload.get("state", {}).get("round", 0)
        salt = payload.get("retry_note", "") + payload.get("validation_error", "")
        h = int(hashlib.sha256(f"{user}{round_num}{salt}".encode()).hexdigest(), 16)
        action = candidates[h % len(candidates)]
        target = "pi" if agent_id != "pi" else "phd_a"


        return {
            "primary_action": {"type": action, "target": target, "intensity": 0.55 + (h % 40) / 100.0},
            "communication_action": {
                "type": "share_result" if action in ("run_experiment", "debug_code") else "seek_validation",
                "target": target,
                "content_summary": f"{agent_id} responds to {event_type}",
            },
            "public_position": {
                "statement_type": "team_support" if action == "comply" else "neutral",
                "authorship_claim": "co_first" if action == "ask_for_authorship" else "any_authorship",
            },
            "private_intent": {
                "goal": "secure_first_author" if agent_id == "phd_a" else "lay_low",
                "strategy": action,
                "trust_pi": payload.get("state", {}).get("beliefs", {}).get("pi_fairness", 0.5),
            },
        }


@pytest.fixture
def llm_adapter() -> FakeLLMAdapter:
    return FakeLLMAdapter()


@pytest.fixture(autouse=True)
def _patch_llm_for_tests(monkeypatch: pytest.MonkeyPatch, llm_adapter: FakeLLMAdapter) -> None:
    monkeypatch.setattr("src.engine.llm_adapter.get_adapter", lambda **kwargs: llm_adapter)
    monkeypatch.setattr("src.engine.simulation.get_adapter", lambda **kwargs: llm_adapter)
