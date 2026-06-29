"""Tests for institutional PI power surfaces."""

from __future__ import annotations

from src.cognition.power import career_hostage_index, pi_control_pressure, pi_control_surface
from src.world.loader import load_world


def test_pi_control_surface_is_bounded_and_state_sensitive():
    world = load_world()
    base = pi_control_pressure(world, world.agents["phd_a"])
    world.project.project.deadline_pressure = 0.95
    world.project.project.funding_pressure = 0.90
    world.project.project.authorship_conflict = 0.85
    tense = pi_control_pressure(world, world.agents["phd_a"])

    surface = pi_control_surface(world)
    assert set(surface) == {"recommendation_letter", "funding_access", "authorship_veto", "resource_allocation"}
    assert all(0.0 <= v <= 1.0 for v in surface.values())
    assert 0.0 <= career_hostage_index(world) <= 1.0
    assert tense > base
