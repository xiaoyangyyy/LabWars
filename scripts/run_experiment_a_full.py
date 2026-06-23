"""Re-run Experiment A (A1–A5) full 60 rounds after diversity fix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.engine.diversity import diversity_metrics
from src.experiments.conditions import EXPERIMENT_A, build_sim_config
from src.experiments.report import generate_report
from src.engine.simulation import run_simulation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="start_from", default="A1", help="Start condition e.g. A3")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    args = parser.parse_args()

    out_dir = ROOT / "output" / "runs" / "exp_a_rerun"
    report_dir = ROOT / "output" / "reports" / "exp_a_rerun"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    seed = 42
    max_rounds = 60
    summary_path = out_dir / "experiment_a_rerun_summary.json"
    results: list[dict] = []
    if summary_path.exists():
        results = json.loads(summary_path.read_text(encoding="utf-8"))

    started = False
    for cid, condition in EXPERIMENT_A.items():
        if not started:
            if cid == args.start_from:
                started = True
            else:
                continue
        run_id = f"expA_{cid}_seed{seed}_v2"
        jsonl = out_dir / f"run_{run_id}.jsonl"
        if args.skip_existing and jsonl.exists():
            print(f"\n=== {cid} skip (exists) ===", flush=True)
            continue

        print(f"\n=== {cid} {condition.label} ===", flush=True)
        cfg = build_sim_config(condition, seed, max_rounds=max_rounds, output_dir=str(out_dir))
        cfg.run_id = run_id
        log = run_simulation(cfg)

        div = diversity_metrics(log.actions)
        phd_a = div["by_agent"].get("phd_a", {})
        row = {
            "condition": cid,
            "run_id": log.run_id,
            "rounds": len(log.round_records),
            "critic_violations": len(log.critic_violations),
            "phd_a_unique_actions": phd_a.get("unique_actions"),
            "phd_a_top_share": phd_a.get("top_share"),
            "outcomes": {k: log.outcomes.get(k) for k in set(condition.primary_outcomes) | {"protest_intensity", "post_r52_compliance", "memory_authorship_cluster_strength"}},
            "diversity": div,
        }
        results = [r for r in results if r.get("condition") != cid] + [row]
        generate_report(log=log, output_dir=report_dir)
        print(
            f"  rounds={row['rounds']} critic={row['critic_violations']} "
            f"phd_a_unique={row['phd_a_unique_actions']} top_share={row['phd_a_top_share']}",
            flush=True,
        )
        for k, v in row["outcomes"].items():
            print(f"  {k}={v}", flush=True)
        summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print(f"\nDone. Summary: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
