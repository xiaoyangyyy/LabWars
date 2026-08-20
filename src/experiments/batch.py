"""Batch experiment orchestration — 630-run matrix."""

from __future__ import annotations

import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from src.engine.llm_adapter import QuotaExhaustedError
from src.engine.run_log import EXTRACTABLE_OUTCOMES, RunLog, extract_outcome
from src.experiments.conditions import EXPERIMENT_MATRIX, list_conditions
from src.experiments.runner import run_single
from src.world.loader import PROJECT_ROOT

DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "runs"

ANALYSIS_OUTCOMES = list(dict.fromkeys([
    *EXTRACTABLE_OUTCOMES,
    "trust_pi_final",
    "trust_pi_logged",
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
    "help_rebuttal",
    "demand_authorship_exchange",
    "passive_cooperation",
    "trust_phd_b_r25",
    "trust_phd_b_r44",
    "trust_phd_b_r60",
    "trust_recovery_rate",
    "pi_fairness_r35",
    "interpretation_of_E030",
    "public_private_divergence_mean",
    "public_private_divergence_last",
    "authority_compliance",
]))


def batch_jsonl_path(output_dir: Path | str, experiment_id: str, condition_id: str, seed: int) -> Path:
    run_id = f"{experiment_id.upper()}{condition_id}_seed{seed}"
    return Path(output_dir) / f"run_{run_id}.jsonl"


def _outcome_value(log: RunLog, key: str) -> float | Any:
    if key not in log.outcomes or log.outcomes[key] in (None, ""):
        return extract_outcome(log, key)
    val = log.outcomes[key]
    if key in {"trust_pi_final", "trust_pi_logged"} and float(val or 0.0) == 0.0:
        extracted = extract_outcome(log, key)
        if extracted:
            return extracted
    return val


def row_from_run_log(
    log: RunLog,
    *,
    experiment_id: str,
    condition_id: str,
    seed: int,
    skipped_existing: bool = False,
) -> dict[str, Any]:
    row = {
        "run_id": log.run_id,
        "experiment_id": experiment_id,
        "condition_id": condition_id,
        "seed": seed,
        **{k: _outcome_value(log, k) for k in ANALYSIS_OUTCOMES},
    }
    if skipped_existing:
        row["skipped_existing"] = True
    return row


def _run_one_task(payload: dict[str, Any]) -> dict[str, Any]:
    result = run_single(
        payload["experiment_id"],
        payload["seed"],
        payload["condition_id"],
        max_rounds=payload.get("max_rounds", 60),
        output_dir=payload.get("output_dir"),
    )
    return row_from_run_log(
        result["log"],
        experiment_id=payload["experiment_id"],
        condition_id=payload["condition_id"],
        seed=payload["seed"],
    )


def row_from_existing_jsonl(path: Path | str, payload: dict[str, Any]) -> dict[str, Any]:
    log = RunLog.from_jsonl(path)
    return row_from_run_log(
        log,
        experiment_id=payload["experiment_id"],
        condition_id=payload["condition_id"],
        seed=payload["seed"],
        skipped_existing=True,
    )


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
    pending: list[dict[str, Any]] = []
    for task in tasks:
        jsonl = batch_jsonl_path(out, task["experiment_id"], task["condition_id"], task["seed"])
        if skip_existing and jsonl.exists():
            row = row_from_existing_jsonl(jsonl, task)
            rows.append(row)
            _append_batch_row(out / f"batch_{exp}_summary.json", row)
            print(
                f"  skip {row['condition_id']} seed={row['seed']} "
                f"(existing {jsonl.name})",
                flush=True,
            )
            continue
        pending.append(task)

    try:
        if parallel <= 1:
            for task in pending:
                row = _execute_task(out, exp, task, failures)
                if row is not None:
                    rows.append(row)
        else:
            with ProcessPoolExecutor(max_workers=parallel) as pool:
                futures = {pool.submit(_run_one_task, t): t for t in pending}
                for fut in as_completed(futures):
                    task = futures[fut]
                    try:
                        row = fut.result()
                    except QuotaExhaustedError:
                        failure = {
                            "condition_id": task["condition_id"],
                            "seed": task["seed"],
                            "error": "quota_exhausted",
                        }
                        failures.append(failure)
                        _append_failure(out / f"batch_{exp}_failures.jsonl", failure)
                        raise
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
    except QuotaExhaustedError:
        merged = _write_summary(out / f"batch_{exp}_summary.json", rows)
        _write_csv(out / f"batch_{exp}_summary.csv", merged)
        raise

    merged = _write_summary(out / f"batch_{exp}_summary.json", rows)
    _write_csv(out / f"batch_{exp}_summary.csv", merged)
    return merged


def _execute_task(
    out: Path,
    exp: str,
    task: dict[str, Any],
    failures: list[dict[str, Any]],
) -> dict[str, Any] | None:
    try:
        row = _run_one_task(task)
    except QuotaExhaustedError:
        failure = {
            "condition_id": task["condition_id"],
            "seed": task["seed"],
            "error": "quota_exhausted",
        }
        failures.append(failure)
        print(
            f"  STOP quota exhausted at {task['condition_id']} seed={task['seed']}",
            flush=True,
        )
        _append_failure(out / f"batch_{exp}_failures.jsonl", failure)
        raise
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
        return None
    _append_batch_row(out / f"batch_{exp}_summary.json", row)
    print(
        f"  done {row['condition_id']} seed={row['seed']} "
        f"trust_pi={float(row.get('trust_pi_final') or 0):.3f} "
        f"ppd={float(row.get('public_private_divergence_mean') or 0):.3f} "
        f"escalation={float(row.get('authorship_escalation_score') or 0):.3f}",
        flush=True,
    )
    return row


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


def _write_summary(summary_path: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    existing: list[dict[str, Any]] = []
    if summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
    seen = {(r.get("condition_id"), r.get("seed")) for r in rows}
    merged = [r for r in existing if (r.get("condition_id"), r.get("seed")) not in seen] + rows
    merged.sort(key=lambda r: (str(r.get("condition_id")), int(r.get("seed") or 0)))
    summary_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    return merged


def rebuild_batch_summary(experiment_id: str, output_dir: Path | str | None = None) -> list[dict[str, Any]]:
    """Rebuild summary rows from existing jsonl files without rerunning the simulator."""
    exp = experiment_id.upper()
    out = Path(output_dir) if output_dir else DEFAULT_OUTPUT / f"batch_{exp}"
    rows: list[dict[str, Any]] = []
    for jsonl in sorted(out.glob("run_*.jsonl")):
        stem = jsonl.stem
        if stem.startswith("run_"):
            stem = stem[4:]
        if "_seed" not in stem:
            continue
        prefix, seed_text = stem.rsplit("_seed", 1)
        if not prefix.startswith(exp):
            continue
        condition_id = prefix[len(exp):]
        payload = {
            "experiment_id": exp,
            "condition_id": condition_id,
            "seed": int(seed_text),
        }
        rows.append(row_from_existing_jsonl(jsonl, payload))
    rows.sort(key=lambda r: (str(r.get("condition_id")), int(r.get("seed") or 0)))
    if rows:
        summary_path = out / f"batch_{exp}_summary.json"
        summary_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        _write_csv(out / f"batch_{exp}_summary.csv", rows)
    return rows


def run_full_matrix(
    seeds: int = 30,
    parallel: int = 1,
    output_dir: Path | str | None = None,
    skip_existing: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    results: dict[str, list[dict[str, Any]]] = {}
    for exp_id in EXPERIMENT_MATRIX:
        results[exp_id] = run_batch(
            exp_id,
            seeds=seeds,
            parallel=parallel,
            output_dir=output_dir,
            skip_existing=skip_existing,
        )
    return results


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
