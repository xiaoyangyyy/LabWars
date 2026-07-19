"""Unified organization ablation matrix tests."""

from __future__ import annotations

from src.engine.simulation import SimConfig
from src.experiments.organization_ablation import run_organization_ablation


def test_organization_ablation_runs_core_lesions():
    result = run_organization_ablation(
        SimConfig(max_rounds=3),
        conditions=["full", "memory_lesion", "hierarchy_lesion", "social_physics_only", "llm_scoring_off"],
        seeds=[0],
        outcomes=["authorship_dispute_index", "trust_fragmentation", "selected_social_pressure_mean", "llm_override_pressure"],
    )

    data = result.to_dict()
    assert result.n_per_condition == 1
    assert set(result.summary) == {"full", "memory_lesion", "hierarchy_lesion", "social_physics_only", "llm_scoring_off"}
    assert len(result.per_seed) == 5
    assert data["summary"]["llm_scoring_off"]["llm_override_pressure"] == 0.0
    assert result.per_seed[2]["hierarchy_lesion"] is True
