"""Scalable population synthesis tests."""

from __future__ import annotations

from src.world.loader import load_world
from src.world.population import PopulationSpec, expand_population


def test_expand_population_builds_requested_hierarchical_size():
    world = load_world()
    expanded = expand_population(world, PopulationSpec(target_size=50, seed=7, labs=3))

    assert len(expanded.agents) == 50
    assert expanded.world_config["population_size"] == 50
    assert expanded.world_config["population_labs"] == 3
    assert len(expanded.world_config["internal_agents"]) > len(world.world_config["internal_agents"])
    internal = expanded.world_config["internal_agents"]
    assert len(expanded.relationships) == len(internal) * (len(internal) - 1)


def test_expand_population_keeps_canonical_when_target_not_larger():
    world = load_world()
    expanded = expand_population(world, PopulationSpec(target_size=14, seed=7))

    assert len(expanded.agents) == len(world.agents)
    assert expanded.world_config["population_synthesis"] == "canonical"
