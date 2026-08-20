"""Long-horizon and scalable population simulation tests."""

from __future__ import annotations

from src.engine import SimConfig, run_simulation


def test_simulation_runs_after_anchor_schedule_with_state_events():
    log = run_simulation(SimConfig(max_rounds=63, seed=2, llm_provider="scripted", policy_mode="social_physics", enable_llm_action_scoring=False))

    assert len(log.round_records) == 63
    assert log.events[-1]["round"] == 63
    assert log.events[-1]["event_id"].startswith("S")


def test_population_size_is_recorded_and_runs():
    log = run_simulation(SimConfig(max_rounds=3, seed=2, population_size=30, llm_provider="scripted", policy_mode="social_physics", enable_llm_action_scoring=False))

    assert log.config["population_size"] == 30
    assert len(log.actions) >= 30
    assert "organization_fragility_index" in log.outcomes
    cast = log.config["event_cast"]
    actor_ids = {a["agent"] for a in log.actions}
    assert cast["idea"] in actor_ids
    assert cast["pi"]
    assert "story_beats" in log.config
