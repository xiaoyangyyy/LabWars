"""Experiment A multi-seed batch + aggregate (A1–A5)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments.aggregate import aggregate_experiment_multi, compare_conditions
from src.experiments.batch import ANALYSIS_OUTCOMES, run_batch
from src.experiments.conditions import EXPERIMENT_A, list_conditions
from src.experiments.exp_a_promise import run_causal_delete_vs_explicit


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Experiment A across multiple seeds and aggregate.")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds (0..N-1)")
    parser.add_argument("--seed-start", type=int, default=0, help="First seed value")
    parser.add_argument("--conditions", nargs="*", default=None, help="Condition ids, default all A1-A5")
    parser.add_argument("--max-rounds", type=int, default=60)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "output" / "runs" / "exp_a_v4"))
    parser.add_argument("--report-dir", type=str, default=str(ROOT / "output" / "reports" / "exp_a_v4"))
    parser.add_argument("--skip-batch", action="store_true", help="Only aggregate existing batch summary")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--force-rerun", action="store_true", help="Re-run even if jsonl exists")
    parser.add_argument("--skip-causal", action="store_true", help="Skip causal memory-delete pair")
    parser.add_argument("--from-condition", dest="from_condition", default=None)
    parser.add_argument("--causal-seeds", type=int, default=10, help="Seeds for A5 memory-delete causal pair")
    args = parser.parse_args()

    if args.force_rerun:
        args.skip_existing = False

    out_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    conditions = args.conditions or list_conditions("A")
    seed_list = list(range(args.seed_start, args.seed_start + args.seeds))
    summary_path = out_dir / "batch_A_summary.json"

    if args.from_condition and args.from_condition in conditions:
        conditions = conditions[conditions.index(args.from_condition):]

    if not args.skip_batch:
        print(f"Running Experiment A: conditions={conditions} seeds={seed_list} rounds={args.max_rounds}", flush=True)
        rows = run_batch(
            "A",
            seed_list=seed_list,
            condition_ids=conditions,
            parallel=args.parallel,
            output_dir=out_dir,
            max_rounds=args.max_rounds,
            skip_existing=args.skip_existing,
        )
        print(f"Batch complete: {len(rows)} rows in summary", flush=True)
    elif not summary_path.exists():
        print(f"Missing batch summary: {summary_path}", flush=True)
        return 1

    multi = aggregate_experiment_multi(
        "A",
        batch_path=summary_path,
        outcomes=ANALYSIS_OUTCOMES,
        baseline_condition="A1",
    )
    multi_path = report_dir / "aggregate_A_multi.json"
    multi_path.write_text(json.dumps(multi, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = [
        "# LabWars Experiment A — Multi-seed Aggregate",
        "",
        f"Batch: `{summary_path}`",
        f"Seeds: `{seed_list}` | Conditions: `{conditions}`",
        "",
    ]
    for outcome, stats in multi["outcomes"].items():
        md_lines.append(f"## Outcome: `{outcome}` (baseline `{stats['baseline']}`)")
        md_lines.append("")
        md_lines.append("| Condition | N | Mean | 95% CI | ATE vs baseline | t |")
        md_lines.append("|-----------|---|------|--------|-----------------|---|")
        for cid, row in stats["conditions"].items():
            md_lines.append(
                f"| {cid} | {row['n']} | {row['mean']:.3f} | "
                f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}] | "
                f"{row['ate_vs_baseline']:+.3f} | {row['welch_t_vs_baseline']:.2f} |"
            )
        md_lines.append("")

    comparisons = {}
    comparison_specs = [
        ("A3", "A5", "A5 vs A3 (explicit+delete vs ambiguous, both betrayal draft)"),
        ("A1", "A2", "A2 vs A1 (explicit+honor vs baseline)"),
    ]
    for cond_a, cond_b, label in comparison_specs:
        if cond_a in conditions and cond_b in conditions:
            block: dict[str, Any] = {"label": label}
            for outcome in (
                "authorship_escalation_score",
                "authorship_escalation_potential",
                "memory_authorship_cluster_strength",
                "protest_authorship",
                "trust_pi_final",
            ):
                block[outcome] = compare_conditions(summary_path, cond_a, cond_b, outcome=outcome)
            comparisons[f"{cond_a}_vs_{cond_b}"] = block
    if comparisons:
        md_lines.append("## Condition comparisons")
        md_lines.append("")
        for key, block in comparisons.items():
            md_lines.append(f"### {block['label']}")
            md_lines.append("")
            for outcome, cmp in block.items():
                if outcome == "label":
                    continue
                md_lines.append(
                    f"- **{outcome}**: {cmp['condition_a']}={cmp['a_mean']:.3f}, "
                    f"{cmp['condition_b']}={cmp['b_mean']:.3f}, "
                    f"Δ={cmp['ate_b_minus_a']:+.3f}, t={cmp['welch_t']:.2f}"
                )
            md_lines.append("")
        cmp_path = report_dir / "compare_conditions.json"
        cmp_path.write_text(json.dumps(comparisons, indent=2, ensure_ascii=False), encoding="utf-8")

    md_path = report_dir / "aggregate_A_multi.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print("Running causal memory-delete pair (explicit vs delete)...", flush=True)
    causal_path = report_dir / "causal_memory_delete.json"
    if args.skip_causal:
        print("Skipped causal analysis (--skip-causal)", flush=True)
    else:
        try:
            causal = run_causal_delete_vs_explicit(n_seeds=args.causal_seeds)
            causal_path.write_text(json.dumps(causal, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
            print(f"Causal delete report: {causal_path}", flush=True)
        except Exception as exc:
            print(f"WARNING: causal memory-delete analysis failed (batch results kept): {exc}", flush=True)

    print(f"Multi-outcome report: {md_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
