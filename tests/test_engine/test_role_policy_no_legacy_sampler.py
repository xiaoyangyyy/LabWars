"""Regression tests for RolePolicyAgent action selection ownership."""

from __future__ import annotations

from pathlib import Path


def test_role_policy_does_not_import_or_call_legacy_sampler():
    source = Path("src/engine/role_policy.py").read_text(encoding="utf-8")
    assert "sample_action_candidate" not in source
    assert "sample_action_candidate_legacy" not in source
    assert "_score_candidates" in source
    assert "_sample_payload" in source
