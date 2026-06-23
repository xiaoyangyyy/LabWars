"""Event Agent — deterministic anchor schedule + state-triggered hooks."""

from __future__ import annotations

import copy
from typing import Any

from src.world.loader import load_events
from src.world.models import EventAtom, WorldState

REVIEWER_ACTIVE_FROM = 57


class EventAgent:
    def __init__(self, events: list[EventAtom] | None = None) -> None:
        self._events = events or load_events()
        self._by_round = {e.round: e for e in self._events}

    def generate(self, round_num: int, world: WorldState) -> EventAtom | None:
        event = self._by_round.get(round_num)
        if event is None:
            return None
        return copy.deepcopy(event)

    def get_schedule(self, max_round: int) -> list[EventAtom]:
        return [copy.deepcopy(self._by_round[r]) for r in range(1, max_round + 1) if r in self._by_round]


def is_agent_active(agent_id: str, round_num: int, config: dict[str, Any]) -> bool:
    active = config.get("active_agents")
    if active is not None:
        if agent_id not in active:
            offstage = config.get("offstage_agents", [])
            if agent_id in offstage and round_num >= config.get("offstage_min_round", 1):
                return agent_id in ("rival_lab_h",) and round_num >= 21
            return False
    if agent_id.startswith("reviewer_"):
        return round_num >= REVIEWER_ACTIVE_FROM
    if agent_id == "rival_lab_h":
        return round_num >= 21
    return True
