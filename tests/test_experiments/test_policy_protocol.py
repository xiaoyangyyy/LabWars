"""Policy-regime protocol tests."""

from __future__ import annotations

from src.experiments.policy_protocol import run_policy_comparison_protocol


def test_policy_comparison_protocol_runs_core_regimes():
    result = run_policy_comparison_protocol(
        population_size=20,
        rounds=3,
        seeds=[0],
        regimes=["rule_baseline", "hybrid_sampled"],
        llm_provider="scripted",
        sampled_top_k=4,
    )

    assert "rule_baseline" in result.summary
    assert "hybrid_sampled" in result.summary
    assert result.summary["hybrid_sampled"]["llm_sampled_action_fraction"] > 0
    assert result.summary["hybrid_sampled"]["llm_sampled_action_fraction"] < 1