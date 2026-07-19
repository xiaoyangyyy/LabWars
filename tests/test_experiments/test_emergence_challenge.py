"""Anti-script emergence challenge tests."""

from __future__ import annotations

from src.experiments.emergence_challenge import run_egalitarian_emergence_challenge


def test_egalitarian_emergence_challenge_runs_conditions():
    result = run_egalitarian_emergence_challenge(population_size=20, rounds=3, seeds=[0])

    assert "canonical_initialization" in result.summary
    assert "egalitarian_initialization" in result.summary
    assert "emergent_pattern_score" in result.summary["egalitarian_initialization"]