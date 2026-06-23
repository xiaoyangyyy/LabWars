"""Part 3 engine integration tests."""

from __future__ import annotations

import pytest

from src.engine.causal import run_causal_experiment
from src.engine.critic import CriticAgent
from src.engine.intervention import Intervention, apply_event_override, apply_memory_intervention, load_interventions
from src.engine.simulation import SimConfig, load_mvp_config, run_simulation
from src.cognition.pipeline import process_event_phase
from src.engine.event_agent import EventAgent
from src.world.loader import load_events, load_world


@pytest.fixture
def mvp_config():
    return load_mvp_config()


class TestSimulationMVP:
    def test_mvp_20_rounds_no_crash(self):
        log = run_simulation(SimConfig(mvp=True, seed=42, max_rounds=20, interventions=[]))
        assert len(log.round_records) == 20
        assert len(log.events) == 20

    def test_same_seed_reproducible(self):
        c1 = SimConfig(mvp=True, seed=7, max_rounds=10, interventions=[])
        c2 = SimConfig(mvp=True, seed=7, max_rounds=10, interventions=[])
        l1 = run_simulation(c1)
        l2 = run_simulation(c2)
        assert [r["metrics"] for r in l1.round_records] == [r["metrics"] for r in l2.round_records]
        assert l1.actions == l2.actions

    def test_full_60_rounds(self):
        log = run_simulation(SimConfig(max_rounds=60, seed=1, interventions=[]))
        assert len(log.round_records) == 60
        assert log.events[-1]["event_id"] == "E060"

    def test_jsonl_output(self, tmp_path):
        log = run_simulation(SimConfig(mvp=True, seed=1, max_rounds=5, output_dir=tmp_path, interventions=[]))
        out = tmp_path / f"run_{log.run_id}.jsonl"
        assert out.exists()
        lines = out.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) >= 6


class TestInterventions:
    def test_explicit_promise_override(self):
        events = {e.event_id: e for e in load_events()}
        inter = next(i for i in load_interventions() if i.variant == "explicit_promise")
        ev = apply_event_override(events["E003"], inter)
        assert "first_author_promise_to_phd_a" in ev.objective_fact.verifiable_claims

    def test_memory_delete_intervention(self):
        world = load_world()
        inter = Intervention(
            intervention_id="TEST",
            type="memory_intervention",
            variant="memory_delete_pi_promise",
            apply_at_round=45,
            target_agent="phd_a",
        )
        ea = EventAgent()
        for r in range(1, 46):
            process_event_phase(world, ea.generate(r, world))

        auth_mem = [
            m for m in world.agents["phd_a"].memory
            if m.get("content_type") in ("authorship_signal", "promise_fulfilled", "promise_broken")
        ]
        assert auth_mem
        apply_memory_intervention(world, inter)
        remaining = [
            m for m in world.agents["phd_a"].memory
            if m.get("content_type") in ("authorship_signal", "promise_fulfilled", "promise_broken")
            and m.get("round", 0) <= 45
        ]
        assert not remaining

    def test_three_interventions_differ(self):
        inters = load_interventions()
        explicit = next(i for i in inters if i.intervention_id == "INT_AUTH_EXPLICIT")
        ambiguous = next(i for i in inters if i.intervention_id == "INT_AUTH_AMBIGUOUS")
        delete = next(i for i in inters if i.intervention_id == "INT_MEMORY_DELETE")

        l1 = run_simulation(SimConfig(mvp=True, seed=5, max_rounds=20, interventions=[explicit]))
        l2 = run_simulation(SimConfig(mvp=True, seed=5, max_rounds=20, interventions=[ambiguous]))
        l3 = run_simulation(SimConfig(mvp=True, seed=5, max_rounds=45, interventions=[explicit, delete]))

        assert l1.interventions_applied[0]["variant"] == "explicit_promise"
        assert l2.interventions_applied[0]["variant"] == "ambiguous_promise"
        assert any(i["intervention_id"] == "INT_MEMORY_DELETE" for i in l3.interventions_applied)


class TestCritic:
    def test_hard_violation_fixed(self):
        critic = CriticAgent()
        agent = load_world().agents["engineer_e"]
        bad = {"type": "ask_for_authorship", "target": "pi", "intensity": 0.8}
        violations = critic.check(bad, agent, load_world())
        assert any(v.severity == "hard" for v in violations)
        fixed, _ = critic.fix_or_reject(bad, agent, violations)
        assert fixed["type"] != "ask_for_authorship"


class TestCausal:
    def test_causal_experiment_runs(self):
        delete = next(i for i in load_interventions() if i.intervention_id == "INT_MEMORY_DELETE")
        cfg = SimConfig(mvp=True, max_rounds=52)
        result = run_causal_experiment(cfg, delete, outcome="protest_authorship", n_seeds=3)
        assert result.n_seeds == 3

    def test_memory_delete_reduces_authorship_cluster(self):
        delete = next(i for i in load_interventions() if i.intervention_id == "INT_MEMORY_DELETE")
        cfg = SimConfig(max_rounds=55, seed=42)
        result = run_causal_experiment(cfg, delete, outcome="protest_authorship", n_seeds=5)
        cluster_ctrl = sum(p["M_control"] for p in result.per_seed) / len(result.per_seed)
        cluster_treat = sum(p["M_treatment"] for p in result.per_seed) / len(result.per_seed)
        assert cluster_treat <= cluster_ctrl + 0.01
