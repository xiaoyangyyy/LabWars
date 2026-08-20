"""Paper Causal Decompiler battery: split-Y, Shapley vs skip, three-worlds, persist."""

from __future__ import annotations

from src.engine.causal import CausalDecompiler, load_factual, run_factual, skip_event
from src.engine.causal.algebra import lesion
from src.engine.causal.estimands import beat_event_ids, split_y, story_shapley, three_worlds
from src.engine.causal.toy import coalition_value, contrastive_leave_one_out, exact_shapley, planted_factors
from src.engine.simulation import SimConfig
from src.experiments.paper_protocol import run_paper_protocol
from src.experiments.paper_tables import render_paper_markdown, table_shapley


def _cfg(**kwargs) -> SimConfig:
    defaults = dict(mvp=True, seed=11, max_rounds=6, interventions=[], llm_provider="scripted")
    defaults.update(kwargs)
    return SimConfig(**defaults)


def test_story_shapley_adds_to_total_effect():
    cfg = _cfg()
    factual = run_factual(cfg)
    ids = beat_event_ids(factual, limit=2)
    result = story_shapley(cfg, factual, ids, "protest_authorship")
    if len(result.get("factors") or []) < 2:
        return
    phi = sum(result["shapley"].values())
    assert abs(phi - result["total_effect"]) < 1e-6


def test_three_worlds_returns_split_vector():
    cfg = _cfg()
    factual = run_factual(cfg)
    op = skip_event(int(factual.events[0]["round"]), str(factual.events[0]["event_id"]))
    worlds = three_worlds(cfg, factual, op, "protest_authorship")
    assert set(worlds["y"]) == {"w0", "w1", "w2"}
    assert "hypocrisy_index" in worlds
    assert "public_private_divergence_mean" in worlds["split"]["w0"]


def test_decompile_log_replays_persisted_factual(tmp_path):
    cfg = _cfg(output_dir=tmp_path)
    factual = run_factual(cfg)
    jsonl = tmp_path / f"run_{factual.run_id}.jsonl"
    assert jsonl.exists()
    loaded = load_factual(jsonl)
    report = CausalDecompiler().decompile_log(
        loaded,
        memory_rounds=[3],
        blame_limit=1,
        include_story_shapley=False,
        include_three_worlds=False,
    )
    assert report.identity_twin_ok
    assert report.llm_replay["identity_run_misses"] == 0
    assert "protest_authorship" in report.split_y


def test_paper_protocol_lite_emits_tables():
    result = run_paper_protocol(
        _cfg(),
        auto_battery=False,
        include_toy_shapley=True,
        write_output=False,
    )
    assert result.report.identity_twin_ok
    assert result.report.findings
    md = render_paper_markdown(result.report.to_dict())
    assert "Split-Y" in md
    assert "AND" in table_shapley(result.report.to_dict())
    assert result.report.shapley_toy["promise"] == 0.5


def test_paper_crn_pair_records_split_ates():
    from src.experiments.paper_contrasts import run_crn_pair

    row = run_crn_pair("A", "A1", "A2", seed=0, max_rounds=6)
    assert "public_private_divergence_mean" in row["ates"]
    assert "ate" in row["ates"]["protest_authorship"]


def test_lesion_observation_is_a_causal_op():
    op = lesion("observation")
    assert "observation" in op.factor_id()


def test_split_y_has_private_channel():
    log = run_factual(_cfg())
    y = split_y(log)
    assert "public_private_divergence_mean" in y
    assert "trust_pi_logged" in y


def test_planted_oracle_still_the_and_lie():
    factors = planted_factors()
    shapley = exact_shapley(lambda s: coalition_value(s), factors)
    knockout = contrastive_leave_one_out(factors, factors)
    assert shapley["promise"] == 0.5
    assert knockout["promise"] == 1.0
    assert sum(knockout.values()) == 2.0
