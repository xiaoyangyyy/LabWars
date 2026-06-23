"""Batch experiment orchestration — 630-run matrix."""

from __future__ import annotations

import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from src.experiments.conditions import EXPERIMENT_MATRIX, list_conditions
from src.experiments.runner import run_single
from src.world.loader import PROJECT_ROOT

DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "runs"

ANALYSIS_OUTCOMES = [
    "trust_pi_final",
    "pi_fairness_r52",
    "authorship_escalation_score",
    "authorship_escalation_potential",
    "protest_authorship",
    "protest_intensity",
    "protest_action_count",
    "post_r52_compliance",
    "withdraw_threat",
    "document_contribution_count",
    "memory_authorship_cluster_strength",
    "authorship_dispute_index",
    "promise_broken_strength_r52",
    "promise_honored_strength_r52",
]


def _run_one_task(payload: dict[str, Any]) -> dict[str, Any]:
    result = run_single(
        payload["experiment_id"],
        payload["seed"],
        payload["condition_id"],
        max_rounds=payload.get("max_rounds", 60),
        output_dir=payload.get("output_dir"),
    )
    log = result["log"]
    row = {
        "run_id": log.run_id,
        "experiment_id": payload["experiment_id"],
        "condition_id": payload["condition_id"],
        "seed": payload["seed"],
        **{k: log.outcomes.get(k, 0) for k in ANALYSIS_OUTCOMES},
    }
    return row


def run_batch(
    experiment_id: str,
    *,
    seeds: int = 30,
    seed_list: list[int] | None = None,
    condition_ids: list[str] | None = None,
    parallel: int = 1,
    output_dir: Path | str | None = None,
    max_rounds: int = 60,
    skip_existing: bool = False,
) -> list[dict[str, Any]]:
    exp = experiment_id.upper()
    conditions = condition_ids or list_conditions(exp)
    out = Path(output_dir) if output_dir else DEFAULT_OUTPUT / f"batch_{exp}"
    out.mkdir(parents=True, exist_ok=True)

    seed_values = seed_list if seed_list is not None else list(range(seeds))
    tasks: list[dict[str, Any]] = []
    for cid in conditions:
        for seed in seed_values:
            tasks.append({
                "experiment_id": exp,
                "condition_id": cid,
                "seed": seed,
                "max_rounds": max_rounds,
                "output_dir": str(out),
            })

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if parallel <= 1:
        for task in tasks:
            run_id = f"{exp}{task['condition_id']}_seed{task['seed']}"
            jsonl = out / f"run_{run_id}.jsonl"
            if skip_existing and jsonl.exists():
                continue
            try:
                row = _run_one_task(task)
            except Exception as exc:
                failure = {
                    "condition_id": task["condition_id"],
                    "seed": task["seed"],
                    "error": str(exc),
                }
                failures.append(failure)
                print(
                    f"  FAILED {task['condition_id']} seed={task['seed']}: {exc}",
                    flush=True,
                )
                _append_failure(out / f"batch_{exp}_failures.jsonl", failure)
                continue
            rows.append(row)
            _append_batch_row(out / f"batch_{exp}_summary.json", row)
            print(
                f"  done {row['condition_id']} seed={row['seed']} "
                f"trust_pi={row.get('trust_pi_final', 0):.3f} "
                f"escalation={row.get('authorship_escalation_score', 0):.3f}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=parallel) as pool:
            futures = {pool.submit(_run_one_task, t): t for t in tasks}
            for fut in as_completed(futures):
                task = futures[fut]
                try:
                    row = fut.result()
                except Exception as exc:
                    failure = {
                        "condition_id": task["condition_id"],
                        "seed": task["seed"],
                        "error": str(exc),
                    }
                    failures.append(failure)
                    print(
                        f"  FAILED {task['condition_id']} seed={task['seed']}: {exc}",
                        flush=True,
                    )
                    _append_failure(out / f"batch_{exp}_failures.jsonl", failure)
                    continue
                rows.append(row)
                _append_batch_row(out / f"batch_{exp}_summary.json", row)

    summary_path = out / f"batch_{exp}_summary.json"
    if summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        seen = {(r.get("condition_id"), r.get("seed")) for r in existing}
        merged = existing + [r for r in rows if (r.get("condition_id"), r.get("seed")) not in seen]
    else:
        merged = rows
    summary_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(out / f"batch_{exp}_summary.csv", merged)
    return merged


def _append_failure(path: Path, failure: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(failure, ensure_ascii=False) + "\n")


def _append_batch_row(summary_path: Path, row: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    if summary_path.exists():
        rows = json.loads(summary_path.read_text(encoding="utf-8"))
    key = (row.get("condition_id"), row.get("seed"))
    rows = [r for r in rows if (r.get("condition_id"), r.get("seed")) != key] + [row]
    summary_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def run_full_matrix(seeds: int = 30, parallel: int = 1, output_dir: Path | str | None = None) -> dict[str, list[dict[str, Any]]]:
    results: dict[str, list[dict[str, Any]]] = {}
    for exp_id in EXPERIMENT_MATRIX:
        results[exp_id] = run_batch(exp_id, seeds=seeds, parallel=parallel, output_dir=output_dir)
    return results


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
