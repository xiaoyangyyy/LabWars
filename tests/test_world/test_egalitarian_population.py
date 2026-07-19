"""Egalitarian population initialization tests."""

from __future__ import annotations

from src.world.loader import load_world
from src.world.population import PopulationSpec, expand_population


def test_egalitarian_population_equalizes_agents_and_edges():
    world = load_world()
    expanded = expand_population(world, PopulationSpec(target_size=20, seed=1, egalitarian=True))

    assert expanded.world_config["egalitarian_initialization"] is True
    assert len(expanded.agents) == 20
    ambitions = {agent.personality.ambition for agent in expanded.agents.values()}
    trusts = {edge.trust for edge in expanded.relationships}
    threats = {edge.perceived_credit_threat for edge in expanded.relationships}
    assert ambitions == {0.5}
    assert trusts == {0.5}
    assert threats == {0.0}