"""Aggregate batch results — ATE, CI, validity gates."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.experiments.metrics import bootstrap_ci, mediation_fraction, welch_t_stat
from src.experiments.runner import run_single
from src.world.loader import PROJECT_ROOT

DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "reports"


def load_batch_summary(path: Path | str) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".csv":
        with path.open(encoding="utf-8") as f:
            return list(csv.DictReader(f))
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate_experiment(
    experiment_id: str,
    *,
    batch_path: Path | str | None = None,
    outcome: str = "protest_authorship",
    baseline_condition: str | None = None,
) -> dict[str, Any]:
    exp = experiment_id.upper()
    if batch_path:
        rows = load_batch_summary(batch_path)
        rows = [r for r in rows if r.get("experiment_id", exp) == exp or r.get("experiment_id") is None]
    else:
        default = PROJECT_ROOT / "output" / "runs" / f"batch_{exp}_summary.json"
        if not default.exists():
            raise FileNotFoundError(f"No batch summary at {default}; run batch first.")
        rows = load_batch_summary(default)

    by_condition: dict[str, list[float]] = {}
    for row in rows:
        cid = row.get("condition_id", "")
        val = float(row.get(outcome, 0))
        by_condition.setdefault(cid, []).append(val)

    if not by_condition:
        raise ValueError(f"No rows to aggregate for experiment {exp}; run batch with at least one seed.")

    baseline = baseline_condition or sorted(by_condition.keys())[0]
    summary: dict[str, Any] = {
        "experiment_id": exp,
        "outcome": outcome,
        "baseline": baseline,
        "conditions": {},
    }

    base_vals = by_condition.get(baseline, [])
    base_mean, base_lo, base_hi = bootstrap_ci(base_vals)

    for cid, vals in sorted(by_condition.items()):
        mean, lo, hi = bootstrap_ci(vals)
        ate = mean - base_mean if cid != baseline else 0.0
        summary["conditions"][cid] = {
            "n": len(vals),
            "mean": mean,
            "ci_low": lo,
            "ci_high": hi,
            "ate_vs_baseline": ate,
            "welch_t_vs_baseline": welch_t_stat(vals, base_vals) if cid != baseline else 0.0,
        }

    summary["baseline_stats"] = {"mean": base_mean, "ci_low": base_lo, "ci_high": base_hi, "n": len(base_vals)}
    return summary


def aggregate_experiment_multi(
    experiment_id: str,
    *,
    batch_path: Path | str,
    outcomes: list[str] | None = None,
    baseline_condition: str | None = None,
) -> dict[str, Any]:
    outcomes = outcomes or [
        "trust_pi_final",
        "pi_fairness_r52",
        "authorship_escalation_score",
        "authorship_escalation_potential",
        "memory_authorship_cluster_strength",
        "post_r52_compliance",
    ]
    return {
        "experiment_id": experiment_id.upper(),
        "batch_path": str(batch_path),
        "outcomes": {
            outcome: aggregate_experiment(
                experiment_id,
                batch_path=batch_path,
                outcome=outcome,
                baseline_condition=baseline_condition,
            )
            for outcome in outcomes
        },
    }


def compare_conditions(
    batch_path: Path | str,
    condition_a: str,
    condition_b: str,
    outcome: str = "protest_intensity",
) -> dict[str, Any]:
    rows = load_batch_summary(batch_path)
    a_vals = [float(r[outcome]) for r in rows if r.get("condition_id") == condition_a]
    b_vals = [float(r[outcome]) for r in rows if r.get("condition_id") == condition_b]
    a_mean, a_lo, a_hi = bootstrap_ci(a_vals)
    b_mean, b_lo, b_hi = bootstrap_ci(b_vals)
    return {
        "outcome": outcome,
        "condition_a": condition_a,
        "condition_b": condition_b,
        "a_mean": a_mean,
        "a_ci": [a_lo, a_hi],
        "b_mean": b_mean,
        "b_ci": [b_lo, b_hi],
        "ate_b_minus_a": b_mean - a_mean,
        "welch_t": welch_t_stat(b_vals, a_vals),
        "n_a": len(a_vals),
        "n_b": len(b_vals),
    }


def aggregate_validity_gate(seeds: list[int] | None = None) -> dict[str, Any]:
    seeds = seeds or list(range(10))
    full_logs = [run_single("V", s, "V6")["log"] for s in seeds]
    shuf_logs = [run_single("V", s, "V2")["log"] for s in seeds]
    nomem_logs = [run_single("V", s, "V1")["log"] for s in seeds]

    full_y = [l.outcomes.get("protest_authorship", 0) for l in full_logs]
    shuf_y = [l.outcomes.get("protest_authorship", 0) for l in shuf_logs]
    nomem_y = [l.outcomes.get("protest_authorship", 0) for l in nomem_logs]

    med = mediation_fraction(nomem_logs, full_logs)
    return {
        "full_vs_shuffled_t": welch_t_stat(full_y, shuf_y),
        "full_vs_no_memory_t": welch_t_stat(full_y, nomem_y),
        "mediation_full_vs_nomem": med,
        "passes_shuffle_gate": abs(welch_t_stat(full_y, shuf_y)) > 1.96,
    }


def write_aggregate_report(
    experiment_id: str,
    output_dir: Path | str | None = None,
    batch_path: Path | str | None = None,
) -> Path:
    out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT
    out_dir.mkdir(parents=True, exist_ok=True)
    if batch_path is None and output_dir is not None:
        candidate = out_dir / f"batch_{experiment_id.upper()}_summary.json"
        if candidate.exists():
            batch_path = candidate
    if batch_path:
        summary = aggregate_experiment(experiment_id, batch_path=batch_path)
    else:
        summary = aggregate_experiment(experiment_id)
    path = out_dir / f"aggregate_{experiment_id.upper()}.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = [
        f"# LabWars Aggregate — Experiment {experiment_id.upper()}",
        "",
        f"Outcome: `{summary['outcome']}` | Baseline: `{summary['baseline']}`",
        "",
        "| Condition | N | Mean | 95% CI | ATE vs baseline | t |",
        "|-----------|---|------|--------|-----------------|---|",
    ]
    for cid, stats in summary["conditions"].items():
        md_lines.append(
            f"| {cid} | {stats['n']} | {stats['mean']:.3f} | "
            f"[{stats['ci_low']:.3f}, {stats['ci_high']:.3f}] | "
            f"{stats['ate_vs_baseline']:+.3f} | {stats['welch_t_vs_baseline']:.2f} |"
        )
    md_path = out_dir / f"aggregate_{experiment_id.upper()}.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return md_path
