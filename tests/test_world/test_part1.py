"""Part 1 world layer tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.world.actions import ACTION_REGISTRY, ActionType, all_action_types, get_allowed_actions
from src.world.events import EVENT_TYPE_REGISTRY, all_event_types
from src.world.loader import (
    PROJECT_ROOT,
    load_agents,
    load_events,
    load_world,
    validate_events_schedule,
    validate_relationship_coverage,
)

SCHEMAS = PROJECT_ROOT / "schemas"
CONFIG = PROJECT_ROOT / "config"


class TestSchemas:
    def test_schema_files_exist(self):
        for name in [
            "agent.schema.json",
            "event.schema.json",
            "relationship.schema.json",
            "project.schema.json",
            "memory.schema.json",
        ]:
            assert (SCHEMAS / name).exists()

    def test_schema_files_are_valid_json(self):
        for path in SCHEMAS.glob("*.json"):
            json.loads(path.read_text(encoding="utf-8"))


class TestAgents:
    def test_load_14_agents(self):
        agents = load_agents()
        assert len(agents) == 14

    def test_agent_ids_unique(self):
        agents = load_agents()
        ids = [a.id for a in agents]
        assert len(ids) == len(set(ids))

    def test_personality_in_range(self):
        for agent in load_agents():
            for field, value in agent.personality.model_dump().items():
                assert 0.0 <= value <= 1.0, f"{agent.id}.personality.{field}={value}"

    def test_beliefs_in_range(self):
        for agent in load_agents():
            for field, value in agent.beliefs.model_dump().items():
                assert 0.0 <= value <= 1.0, f"{agent.id}.beliefs.{field}={value}"


class TestEvents:
    def test_load_60_events(self):
        events = load_events()
        assert len(events) == 60

    def test_event_ids_unique(self):
        events = load_events()
        ids = [e.event_id for e in events]
        assert len(ids) == len(set(ids))

    def test_rounds_monotonic_and_complete(self):
        events = sorted(load_events(), key=lambda e: e.round)
        assert events[0].round == 1
        assert events[-1].round == 60
        rounds = [e.round for e in events]
        assert rounds == list(range(1, 61))

    def test_mandatory_anchors_present(self):
        errors = validate_events_schedule(load_events())
        assert errors == []

    def test_all_event_types_registered(self):
        for event in load_events():
            assert event.type in EVENT_TYPE_REGISTRY, f"{event.event_id} type {event.type}"


class TestActions:
    def test_all_actions_registered(self):
        assert len(all_action_types()) == len(ActionType)
        assert len(ACTION_REGISTRY) == len(ActionType)

    def test_engineer_cannot_ask_authorship(self):
        allowed = get_allowed_actions("engineer_e")
        assert ActionType.ASK_FOR_AUTHORSHIP not in allowed

    def test_visiting_cannot_lobby_pi(self):
        allowed = get_allowed_actions("visiting_f")
        assert ActionType.PRIVATELY_LOBBY_PI not in allowed


class TestWorld:
    def test_load_world(self):
        world = load_world()
        assert len(world.agents) == 14
        assert world.project.current_round == 0
        assert world.project.target_conference == "NeurIPS"

    def test_internal_relationship_coverage(self):
        world = load_world()
        internal = world.world_config["internal_agents"]
        errors = validate_relationship_coverage(world.relationships, internal)
        assert errors == []

    def test_internal_edge_count(self):
        world = load_world()
        internal = world.world_config["internal_agents"]
        n = len(internal)
        assert len(world.relationships) == n * (n - 1)


class TestConfigIntegrity:
    def test_world_yaml(self):
        data = yaml.safe_load((CONFIG / "world.yaml").read_text(encoding="utf-8"))
        assert data["world"]["total_rounds"] == 60
        assert len(data["internal_agents"]) == 7
        assert len(data["external_agents"]) == 7
        assert len(data["mandatory_anchor_events"]) == 13
