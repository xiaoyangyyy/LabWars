"""Tests for detailed contribution ledger updates."""

from __future__ import annotations

from src.cognition.authorship import update_ledger_from_action
from src.world.loader import load_world


def test_research_actions_update_specific_ledger_dimensions():
    world = load_world()
    before = world.project.contribution_ledger.get("code", {}).get("engineer_e", 0.0)
    update_ledger_from_action(world, "engineer_e", intensity=0.8, action_type="debug_code")
    after = world.project.contribution_ledger.get("code", {}).get("engineer_e", 0.0)
    assert after > before


def test_writing_action_updates_writing_ledger():
    world = load_world()
    before = world.project.contribution_ledger.get("writing", {}).get("phd_a", 0.0)
    update_ledger_from_action(world, "phd_a", intensity=0.7, action_type="write_section")
    after = world.project.contribution_ledger.get("writing", {}).get("phd_a", 0.0)
    assert after > before
