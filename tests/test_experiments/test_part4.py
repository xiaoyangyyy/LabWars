"""Part 4 experiment integration tests."""

from __future__ import annotations

import pytest

from src.engine.intervention import load_interventions
from src.engine.simulation import SimConfig, run_simulation
from src.experiments.aggregate import aggregate_experiment
from src.experiments.batch import run_batch
from src.experiments.conditions import EXPERIMENT_MATRIX, build_sim_config, get_condition, list_conditions
from src.experiments.exp_validity import shuffle_vs_full_test
from src.experiments.metrics import compute_run_metrics, mediation_fraction
from src.experiments.report import generate_report, generate_report_from_log
from src.experiments.runner import run_single


class TestConditions:
    def test_matrix_size(self):
        total = sum(len(v) for v in EXPERIMENT_MATRIX.values())
        assert total == 21  # 5+4+3+3+6

    def test_build_config_tags(self):
        cond = get_condition("A", "A2")
        cfg = build_sim_config(cond, seed=7)
        assert cfg.experiment_id == "A"
        assert cfg.condition_id == "A2"
        assert len(cfg.interventions) == 2


class TestExperimentRuns:
    def test_run_a2_completes(self):
        result = run_single("A", seed=1, condition_id="A2", max_rounds=60)
        log = result["log"]
        assert len(log.round_records) == 60
        assert "protest_authorship" in log.outcomes
        assert log.config["event_cast"]["idea"] == "phd_a"
        assert log.config["story_beats"]["draft_round"] == 52

    def test_run_validity_no_memory(self):
        result = run_single("V", seed=2, condition_id="V1", max_rounds=20)
        log = result["log"]
        assert log.config["disable_memory"] is True
        assert all(
            rec.get("agent_deltas", {}).get("phd_a", {}).get("memory_written") is None
            for rec in log.round_records
        )

    def test_shuffle_memory_config(self):
        log = run_simulation(SimConfig(max_rounds=30, seed=3, shuffle_memory=True, interventions=[]))
        assert log.config["shuffle_memory"] is True
        assert len(log.round_records) == 30


class TestMetricsAndReport:
    def test_compute_run_metrics(self):
        result = run_single("A", seed=4, condition_id="A1", max_rounds=30)
        metrics = compute_run_metrics(result["log"])
        assert metrics["run_id"] == result["log"].run_id
        assert "timeline" in metrics
        assert "trust_fragmentation_curve" in metrics

    def test_report_has_thirteen_sections(self, tmp_path):
        result = run_single("A", seed=5, condition_id="A1", max_rounds=10)
        text = generate_report_from_log(result["log"])
        for i in range(1, 15):
            assert f"## {i}." in text
        path = generate_report(
            experiment_id="A",
            condition_id="A1",
            seed=5,
            output_dir=tmp_path,
            log=result["log"],
        )
        assert "LLM Scoring Influence" in text
        assert "Action Field Explanation" in text
        assert "Causal Decompiler MRI" in text
        assert path.exists()


class TestBatchAndAggregate:
    def test_mini_batch(self, tmp_path):
        rows = run_batch("A", seeds=2, condition_ids=["A1", "A2"], parallel=1, output_dir=tmp_path, max_rounds=10)
        assert len(rows) == 4
        summary = tmp_path / "batch_A_summary.json"
        assert summary.exists()
        assert "authorship_escalation_score" in rows[0]

    def test_aggregate_from_batch(self, tmp_path):
        run_batch("A", seeds=3, condition_ids=["A1", "A2"], output_dir=tmp_path, max_rounds=10)
        agg = aggregate_experiment("A", batch_path=tmp_path / "batch_A_summary.json")
        assert "A1" in agg["conditions"]
        assert "A2" in agg["conditions"]
        assert agg["outcome"] == "authorship_escalation_score"

    def test_batch_rows_include_experiment_specific_outcomes(self, tmp_path):
        from src.experiments.batch import ANALYSIS_OUTCOMES

        for key in (
            "help_rebuttal", "trust_phd_b_r60", "trust_recovery_rate",
            "interpretation_of_E030", "public_private_divergence_mean", "trust_pi_logged",
        ):
            assert key in ANALYSIS_OUTCOMES
        rows = run_batch("A", seeds=1, condition_ids=["A1"], output_dir=tmp_path, max_rounds=8)
        assert "interpretation_of_E030" in rows[0]
        assert "trust_phd_b_r60" in rows[0]
        assert "public_private_divergence_mean" in rows[0]

    def test_skip_existing_rebuilds_row_without_rerun(self, tmp_path):
        from src.experiments.batch import run_batch

        jsonl = tmp_path / "run_AA1_seed0.jsonl"
        jsonl.write_text(
            "\n".join([
                '{"type":"run_meta","run_id":"AA1_seed0","config":{"experiment_id":"A","condition_id":"A1","seed":0},"started_at":"t"}',
                '{"type":"round","round":1,"event_id":"E001","metrics":{"trust_phd_a_pi":0.61,"public_private_divergence":0.5},"agent_deltas":{}}',
                '{"type":"round","round":60,"event_id":"E060","metrics":{"trust_phd_a_pi":0.58,"public_private_divergence":0.4},"agent_deltas":{}}',
                '{"type":"outcomes","protest_authorship":0.02,"trust_pi_final":0.0}',
            ]),
            encoding="utf-8",
        )
        rows = run_batch("A", seed_list=[0], condition_ids=["A1"], output_dir=tmp_path, skip_existing=True)
        assert len(rows) == 1
        assert rows[0]["skipped_existing"] is True
        assert rows[0]["trust_pi_final"] == 0.58
        assert rows[0]["public_private_divergence_last"] == 0.4

    def test_aggregate_uses_experiment_primary_outcome(self, tmp_path):
        summary = tmp_path / "batch_C_summary.json"
        summary.write_text(
            """[
              {"experiment_id":"C","condition_id":"C1","seed":0,"trust_phd_b_r60":0.4,"trust_recovery_rate":0.2,"protest_authorship":0.01},
              {"experiment_id":"C","condition_id":"C3","seed":0,"trust_phd_b_r60":0.7,"trust_recovery_rate":0.0,"protest_authorship":0.01}
            ]""",
            encoding="utf-8",
        )
        agg = aggregate_experiment("C", batch_path=summary)
        assert agg["outcome"] == "trust_phd_b_r60"
        assert agg["conditions"]["C3"]["mean"] == 0.7

    def test_new_outcomes_present(self):
        result = run_single("A", seed=6, condition_id="A2", max_rounds=55)
        outcomes = result["log"].outcomes
        for key in ("authorship_escalation_score", "authorship_escalation_potential", "post_r52_compliance", "withdraw_threat_event"):
            assert key in outcomes


class TestInterventionsLoaded:
    def test_part4_interventions_exist(self):
        ids = {i.intervention_id for i in load_interventions()}
        for required in (
            "INT_SKIP_E003",
            "INT_SKIP_E031",
            "INT_SKIP_E035",
            "INT_FALSE_MEMORY_INSERT",
            "INT_ALUMNI_POSITIVE",
            "INT_DELAYED_MEMORY_INSERT",
            "INT_B4_REBUTTAL_REQUEST",
        ):
            assert required in ids


class TestMediation:
    def test_mediation_fraction_bounded(self):
        ctrl = run_single("V", 0, "V1", max_rounds=10)["log"]
        treat = run_single("V", 0, "V6", max_rounds=10)["log"]
        med = mediation_fraction([ctrl], [treat])
        assert 0.0 <= med["mediation_fraction"] <= 1.0
