"""Run MVP 20 rounds + Experiment A2 (explicit_promise) via configured LLM."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.engine.intervention import load_interventions
from src.engine.simulation import SimConfig, run_simulation
from src.experiments.metrics import compute_run_metrics
from src.experiments.report import generate_report


def main() -> int:
    inters = {i.intervention_id: i for i in load_interventions()}
    explicit = inters["INT_AUTH_EXPLICIT"]
    out_dir = ROOT / "output" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = SimConfig(
        mvp=True,
        seed=42,
        max_rounds=20,
        interventions=[explicit],
        run_id="mvp_A2_seed42_v2",
        experiment_id="A",
        condition_id="A2",
        output_dir=out_dir,
    )

    print("Starting MVP 20 rounds + A2 (explicit_promise)...", flush=True)
    print(f"LLM: {cfg.to_dict().get('llm_provider')} / {cfg.to_dict().get('llm_model')}", flush=True)

    log = run_simulation(cfg)
    metrics = compute_run_metrics(log)

    from src.engine.diversity import diversity_metrics
    div = diversity_metrics(log.actions)

    summary = {
        "run_id": log.run_id,
        "rounds": len(log.round_records),
        "actions": len(log.actions),
        "interventions": log.interventions_applied,
        "outcomes": log.outcomes,
        "diversity": div,
        "critic_violations": len(log.critic_violations),
    }
    summary_path = out_dir / f"{log.run_id}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    report_path = generate_report(log=log, output_dir=ROOT / "output" / "reports")

    print(f"run_id={log.run_id} rounds={len(log.round_records)} actions={len(log.actions)}", flush=True)
    phd_a_div = div["by_agent"].get("phd_a", {})
    print(f"  diversity: phd_a_unique={phd_a_div.get('unique_actions')} top_share={phd_a_div.get('top_share')} critic={len(log.critic_violations)}", flush=True)
    for k in ("protest_authorship", "withdraw_threat", "trust_pi_final", "memory_authorship_cluster_strength", "document_contribution_count"):
        print(f"  {k}={log.outcomes.get(k)}", flush=True)
    print(f"jsonl={out_dir / f'run_{log.run_id}.jsonl'}", flush=True)
    print(f"summary={summary_path}", flush=True)
    print(f"report={report_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
