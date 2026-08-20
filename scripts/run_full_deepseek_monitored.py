"""Monitored DeepSeek full matrix: A/B/C/D + validity, 60 rounds, 1 seed.

Stops immediately on quota / insufficient balance. Resume-safe via skip_existing.
Never prints the API key.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engine.llm_adapter import QuotaExhaustedError, get_adapter  # noqa: E402
from src.experiments.aggregate import write_aggregate_report  # noqa: E402
from src.experiments.batch import run_batch  # noqa: E402
from src.world.loader import PROJECT_ROOT  # noqa: E402

STATUS_PATH = PROJECT_ROOT / "output" / "runs" / "full_deepseek_status.json"
PROGRESS_PATH = PROJECT_ROOT / "output" / "runs" / "full_deepseek_progress.jsonl"
RESULTS_PATH = PROJECT_ROOT / "output" / "reports" / "full_deepseek_results.md"

JOBS = [
    ("A", None),
    ("B", None),
    ("C", None),
    ("D", None),
    ("V", ["V1", "V2", "V3", "V6"]),
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_status(payload: dict) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _log_progress(event: dict) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    event = {"ts": _now(), **event}
    with PROGRESS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    print(json.dumps(event, ensure_ascii=False), flush=True)


def _summarize() -> dict:
    summary: dict[str, list] = {}
    for exp, _ in JOBS:
        path = PROJECT_ROOT / "output" / "runs" / f"batch_{exp}" / f"batch_{exp}_summary.json"
        if path.exists():
            summary[exp] = json.loads(path.read_text(encoding="utf-8"))
        else:
            summary[exp] = []
    return summary


def _write_results_md(status: str, reason: str | None = None) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = _summarize()
    lines = [
        "# LabWars DeepSeek full-run results",
        "",
        f"- Status: **{status}**",
        f"- Finished at: {_now()}",
    ]
    if reason:
        lines.append(f"- Stop reason: {reason}")
    lines += ["", "## Condition outcomes", ""]
    for exp, rows in data.items():
        lines.append(f"### Experiment {exp} ({len(rows)} completed)")
        if not rows:
            lines.append("_No completed rows._")
            lines.append("")
            continue
        extra_keys = {
            "A": "promise_honored_strength_r52",
            "B": "help_rebuttal",
            "C": "trust_phd_b_r60",
            "D": "interpretation_of_E030",
            "V": "pi_fairness_r52",
        }
        extra_key = extra_keys.get(exp, "pi_fairness_r52")
        primary_key = {
            "A": "authorship_escalation_score",
            "B": "help_rebuttal",
            "C": "trust_phd_b_r60",
            "D": "interpretation_of_E030",
            "V": "protest_authorship",
        }.get(exp, "protest_authorship")
        lines.append(
            f"| condition | seed | {primary_key} | protest | ppd | trust_logged | trust_pi | cluster | {extra_key} |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in rows:
            lines.append(
                "| {condition_id} | {seed} | {primary:.3f} | {protest:.3f} | {ppd:.3f} | {logged:.3f} | {trust:.3f} | {mem:.3f} | {extra:.3f} |".format(
                    condition_id=row.get("condition_id"),
                    seed=row.get("seed"),
                    primary=float(row.get(primary_key) or 0),
                    protest=float(row.get("protest_authorship") or 0),
                    ppd=float(row.get("public_private_divergence_mean") or 0),
                    logged=float(row.get("trust_pi_logged") or row.get("trust_pi_final") or 0),
                    trust=float(row.get("trust_pi_final") or 0),
                    mem=float(row.get("memory_authorship_cluster_strength") or 0),
                    extra=float(row.get(extra_key) or 0),
                )
            )
        lines.append("")
    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    if "--confirm-matrix" not in sys.argv:
        print(
            "This script burns the A/B/C/D/V LLM matrix. "
            "Paper default is scripts/run_causal_deepseek_full.py. "
            "Re-run with --confirm-matrix if you really want the grid.",
            flush=True,
        )
        return 2

    os.environ["LABWARS_LLM_CONFIG"] = "config/llm.deepseek.yaml"
    os.environ.setdefault("LABWARS_LLM_PROVIDER", "deepseek")
    os.environ.setdefault("LABWARS_REQUEST_DELAY_SEC", "0.15")
    os.environ.setdefault("LABWARS_LLM_MAX_RETRIES", "2")
    os.environ.setdefault("LABWARS_PROGRESS", "1")
    os.environ.setdefault("LABWARS_FAST_JSON", "1")
    os.environ.setdefault("LABWARS_POLICY_RETRIES", "0")
    os.environ.setdefault("LABWARS_LLM_MODEL", "deepseek-chat")
    os.environ.setdefault("LABWARS_COGNITIVE_TOP_K", "5")

    _write_status({"state": "starting", "ts": _now()})
    _log_progress({"event": "start", "jobs": [j[0] for j in JOBS], "seeds": 1, "rounds": 60})

    try:
        llm = get_adapter(provider="deepseek")
        ping = llm.complete_json("Output JSON only.", '{"task":"ping","reply_schema":{"ok":true}}')
        _log_progress({"event": "ping_ok", "payload": ping})
    except QuotaExhaustedError as exc:
        _log_progress({"event": "quota_exhausted", "phase": "ping", "error": str(exc)[:400]})
        _write_status({"state": "stopped_quota", "phase": "ping", "ts": _now()})
        _write_results_md("stopped_quota", str(exc)[:400])
        return 2
    except Exception as exc:
        _log_progress({"event": "ping_fail", "error": str(exc)[:400]})
        _write_status({"state": "failed", "phase": "ping", "error": str(exc)[:400], "ts": _now()})
        _write_results_md("failed_ping", str(exc)[:400])
        return 1

    completed: list[str] = []
    try:
        for exp, conditions in JOBS:
            _write_status({"state": "running", "experiment": exp, "ts": _now(), "completed": completed})
            _log_progress({"event": "experiment_start", "experiment": exp, "conditions": conditions})
            rows = run_batch(
                exp,
                seeds=1,
                condition_ids=conditions,
                parallel=1,
                max_rounds=60,
                skip_existing=True,
            )
            completed.append(exp)
            _log_progress({"event": "experiment_done", "experiment": exp, "n": len(rows)})
            try:
                path = write_aggregate_report(
                    exp,
                    output_dir=PROJECT_ROOT / "output" / "reports",
                    batch_path=PROJECT_ROOT / "output" / "runs" / f"batch_{exp}" / f"batch_{exp}_summary.json",
                )
                _log_progress({"event": "aggregate_ok", "experiment": exp, "path": str(path)})
            except Exception as exc:
                _log_progress({"event": "aggregate_skip", "experiment": exp, "error": str(exc)[:240]})
    except QuotaExhaustedError as exc:
        _log_progress({"event": "quota_exhausted", "error": str(exc)[:400], "completed": completed})
        _write_status({"state": "stopped_quota", "completed": completed, "ts": _now(), "error": str(exc)[:400]})
        _write_results_md("stopped_quota", str(exc)[:400])
        print("FULL_DEEPSEEK_STOPPED_QUOTA", flush=True)
        return 2
    except Exception as exc:
        _log_progress({"event": "failed", "error": str(exc)[:400], "trace": traceback.format_exc()[-800:]})
        _write_status({"state": "failed", "completed": completed, "error": str(exc)[:400], "ts": _now()})
        _write_results_md("failed", str(exc)[:400])
        print("FULL_DEEPSEEK_FAILED", flush=True)
        return 1

    _write_status({"state": "complete", "completed": completed, "ts": _now()})
    _write_results_md("complete")
    _log_progress({"event": "complete", "completed": completed})
    print("FULL_DEEPSEEK_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
