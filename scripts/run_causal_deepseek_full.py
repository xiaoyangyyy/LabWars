"""Paper-relevant DeepSeek Causal MRI: 60-round full-cast dual-engine decompile.

Pings first. Stops immediately on quota / payment / invalid key.
Saves a partial report after every MRI phase. Never prints the API key.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engine.causal.decompiler import CausalMRIReport  # noqa: E402
from src.engine.causal.estimands import (  # noqa: E402
    contrastive_event_effects,
    memory_irf,
    point_of_commitment,
    split_y,
)
from src.engine.causal.toy import coalition_value, contrastive_leave_one_out, exact_shapley, planted_factors  # noqa: E402
from src.engine.causal.twin import identity_holds, run_factual, run_twin  # noqa: E402
from src.engine.llm_adapter import LLMAdapter, OpenAIAdapter, QuotaExhaustedError  # noqa: E402
from src.engine.probe import ProbeAgent  # noqa: E402
from src.engine.run_log import extract_outcome  # noqa: E402
from src.engine.simulation import SimConfig  # noqa: E402
from src.experiments.causal_mri import summarize_report  # noqa: E402
from src.experiments.report import generate_report  # noqa: E402
from src.world.loader import PROJECT_ROOT  # noqa: E402

MODELS = ("deepseek-v4-flash", "deepseek-chat")
OUT_DIR = PROJECT_ROOT / "output" / "reports"
STATUS_PATH = PROJECT_ROOT / "output" / "runs" / "causal_mri_deepseek_status.json"
PROGRESS_PATH = PROJECT_ROOT / "output" / "runs" / "causal_mri_deepseek_progress.jsonl"
SUMMARY_PATH = OUT_DIR / "causal_mri_deepseek_summary.md"

os.environ.setdefault("LABWARS_PROGRESS", "1")
os.environ.setdefault("LABWARS_LLM_CONFIG", "config/llm.deepseek.yaml")
os.environ.setdefault("LABWARS_REQUEST_DELAY_SEC", "0.6")
os.environ.setdefault("LABWARS_LLM_MAX_RETRIES", "3")
os.environ.setdefault("LABWARS_POLICY_RETRIES", "1")


class CountingAdapter(LLMAdapter):
    """Logs live API misses without wrapping the inner client's secrets."""

    def __init__(self, inner: OpenAIAdapter) -> None:
        self.inner = inner
        self.calls = 0
        self.model = inner.model

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        self.calls += 1
        kind = "memory" if "memory interpretation" in system.lower() else "policy"
        print(f"LLM_LIVE call={self.calls} kind={kind} model={self.inner.model}", flush=True)
        return self.inner.complete_json(system, user)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _log(event: dict[str, Any]) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {"ts": _now(), **event}
    with PROGRESS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    print(json.dumps(row, ensure_ascii=False, default=str), flush=True)


def _write_status(**payload: Any) -> None:
    _write_json(STATUS_PATH, {"ts": _now(), **payload})


def _dump_effects(effects: list) -> list[dict[str, Any]]:
    from dataclasses import asdict

    return [asdict(e) for e in effects]


def _pick_adapter() -> CountingAdapter:
    last_error: Exception | None = None
    for model in MODELS:
        adapter = OpenAIAdapter(
            model=model,
            temperature=0.65,
            max_tokens=1024,
            api_key_env="DEEPSEEK_API_KEY",
            base_url="https://api.deepseek.com",
            request_delay_sec=0.6,
            top_p=0.88,
            max_retries=3,
        )
        try:
            ping = adapter.complete_json(
                "Output JSON only.",
                '{"task":"ping","reply_schema":{"ok":true}}',
            )
            _log({"event": "ping_ok", "model": model, "payload": ping})
            return CountingAdapter(adapter)
        except QuotaExhaustedError:
            raise
        except Exception as exc:
            last_error = exc
            _log({"event": "ping_fail", "model": model, "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
    raise SystemExit(f"DeepSeek ping failed: {last_error}")


def _write_summary_md(status: str, report: CausalMRIReport | None, reason: str | None, extra: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DeepSeek 因果引擎全量 MRI 结果",
        "",
        f"- 状态: **{status}**",
        f"- 时间: {_now()}",
    ]
    if reason:
        lines.append(f"- 停止原因: {reason[:400]}")
    if extra.get("model"):
        lines.append(f"- 模型: {extra['model']}")
    if extra.get("llm_live_calls") is not None:
        lines.append(f"- 现场 LLM 调用: {extra['llm_live_calls']}")
    if extra.get("phase"):
        lines.append(f"- 停在阶段: {extra['phase']}")
    if extra.get("json_path"):
        lines.append(f"- JSON: `{extra['json_path']}`")
    if extra.get("report_path"):
        lines.append(f"- Markdown: `{extra['report_path']}`")
    lines.append("")
    if report is None:
        lines.append("_没有完整 MRI 报告。_")
    else:
        replay = report.llm_replay or {}
        cache = replay.get("cache") or {}
        lines += [
            "## 关键数字",
            "",
            f"- identity_twin_ok: **{report.identity_twin_ok}**",
            f"- Y / protest_authorship: **{report.factual_y:.4f}**",
            f"- split-Y: {json.dumps(report.split_y, ensure_ascii=False, default=str)}",
            f"- public_private_divergence_mean: {float(report.split_y.get('public_private_divergence_mean') or 0):.4f}",
            f"- identity replay hits={replay.get('identity_run_hits')} misses={replay.get('identity_run_misses')}",
            f"- factual cache unique={cache.get('unique_prompts')} hits={cache.get('hits')} misses={cache.get('misses')} cached_errors={cache.get('cached_errors')}",
            "",
            "## Memory IRF",
            "",
        ]
        if not report.memory_irf:
            lines.append("_未完成。_")
        else:
            for row in report.memory_irf:
                lines.append(
                    f"- {row.get('factor_id')}: ATE={float(row.get('ate', 0.0)):+.4f} "
                    f"(factual={float(row.get('factual_y', 0.0)):.4f}, twin={float(row.get('twin_y', 0.0)):.4f})"
                )
        lines += ["", "## Contrastive skip", ""]
        if not report.contrastive:
            lines.append("_未完成。_")
        else:
            for row in report.contrastive:
                lines.append(
                    f"- {row.get('factor_id')}: ATE={float(row.get('ate', 0.0)):+.4f}"
                )
        if report.point_of_commitment:
            lines.append(f"- point_of_commitment: {report.point_of_commitment.get('factor_id')}")
        lines += ["", "## Probes", ""]
        if not report.probes:
            lines.append("_无。_")
        else:
            for probe in report.probes[:5]:
                lines.append(f"- {probe.get('variant')}: {probe.get('reason')}")
        lines += ["", "## Notes", ""]
        for note in report.notes:
            lines.append(f"- {note}")
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _persist_report(report: CausalMRIReport, log, extra: dict[str, Any]) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / f"causal_mri_{log.run_id}.json"
    json_path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    md_path = generate_report(log=log, output_dir=OUT_DIR)
    extra["json_path"] = str(json_path)
    extra["report_path"] = str(md_path)
    extra["run_id"] = log.run_id
    return extra


def _make_config(adapter: LLMAdapter, *, rounds: int, mvp: bool) -> SimConfig:
    return SimConfig(
        max_rounds=rounds,
        seed=0,
        mvp=mvp,
        interventions=[],
        llm_adapter=adapter,
        llm_provider="deepseek",
        policy_mode="dual_engine",
        cognitive_sampling_top_k=1,
        output_dir=PROJECT_ROOT / "output" / "runs",
    )


def run_mri(adapter: CountingAdapter, *, rounds: int, mvp: bool) -> tuple[str, CausalMRIReport, dict[str, Any]]:
    from dataclasses import asdict

    config = _make_config(adapter, rounds=rounds, mvp=mvp)
    extra: dict[str, Any] = {
        "model": adapter.model,
        "rounds": rounds,
        "mvp": mvp,
        "phase": "factual",
        "llm_live_calls": adapter.calls,
    }
    outcome = "protest_authorship"
    memory_rounds = [t for t in (10, 20, 30, 45) if t < rounds]
    report = CausalMRIReport(
        outcome=outcome,
        factual_y=0.0,
        split_y={},
        identity_twin_ok=False,
        notes=[
            "Abduction is event-keyed NoiseLog, not a global PRNG queue.",
            "LLM outputs are record-replayed from the factual prompt cache.",
            "Memory IRF is an interventional analogue, not a natural indirect effect.",
            "Contrastive skip lies on AND causes; Shapley on the planted SCM is the oracle.",
        ],
    )
    _write_status(state="running", phase="factual", rounds=rounds, mvp=mvp, model=adapter.model)
    _log({"event": "phase_start", "phase": "factual", "rounds": rounds, "mvp": mvp})
    factual = None
    try:
        factual = run_factual(config)
        extra["llm_live_calls"] = adapter.calls
        report.factual_run_id = factual.run_id
        report.factual_y = extract_outcome(factual, outcome)
        report.split_y = split_y(factual)
        report.llm_replay = {"cache": factual.outcomes.get("llm_trace_stats") or {}}
        _log({
            "event": "factual_done",
            "run_id": factual.run_id,
            "Y": report.factual_y,
            "split_y": report.split_y,
            "llm_live_calls": adapter.calls,
            "trace": factual.outcomes.get("llm_trace_stats"),
        })

        extra["phase"] = "identity"
        _write_status(state="running", phase="identity", run_id=factual.run_id, llm_live_calls=adapter.calls)
        twin0 = run_twin(config, [], llm_trace=factual.llm_cache)
        extra["llm_live_calls"] = adapter.calls
        report.identity_twin_ok = identity_holds(factual, twin0)
        twin_stats = twin0.outcomes.get("llm_trace_stats") or {}
        report.llm_replay = {
            "identity_run_hits": twin_stats.get("run_hits", 0),
            "identity_run_misses": twin_stats.get("run_misses", 0),
            "cache": factual.outcomes.get("llm_trace_stats") or {},
        }
        if report.identity_twin_ok and int(twin_stats.get("run_misses", 0) or 0) == 0:
            report.notes.append("Identity twin replayed every LLM prompt (zero cache misses).")
        elif not report.identity_twin_ok:
            report.notes.append("FAIL: no-op twin diverged from factual run.")
        _log({
            "event": "identity_done",
            "identity_twin_ok": report.identity_twin_ok,
            "hits": twin_stats.get("run_hits"),
            "misses": twin_stats.get("run_misses"),
            "llm_live_calls": adapter.calls,
        })
        if twin_stats.get("run_misses", 0) != 0:
            raise RuntimeError(
                f"identity twin made {twin_stats.get('run_misses')} LLM cache misses (must be 0)"
            )
        factual.outcomes["causal_mri"] = report.to_dict()
        extra = _persist_report(report, factual, extra)

        extra["phase"] = "memory_irf"
        _write_status(state="running", phase="memory_irf", run_id=factual.run_id, llm_live_calls=adapter.calls)
        report.memory_irf = []
        for t in memory_rounds:
            extra["phase"] = f"memory_irf_t={t}"
            _log({"event": "phase_start", "phase": extra["phase"], "llm_live_calls": adapter.calls})
            rows = memory_irf(config, factual, outcome, [t])
            report.memory_irf.extend(_dump_effects(rows))
            extra["llm_live_calls"] = adapter.calls
            factual.outcomes["causal_mri"] = report.to_dict()
            extra = _persist_report(report, factual, extra)
            _log({"event": "memory_irf_step", "t": t, "rows": report.memory_irf[-1:], "llm_live_calls": adapter.calls})

        extra["phase"] = "contrastive"
        _write_status(state="running", phase="contrastive", run_id=factual.run_id, llm_live_calls=adapter.calls)
        event_ids = [e["event_id"] for e in factual.events[:1]]
        if event_ids:
            contrastive = contrastive_event_effects(config, factual, outcome, event_ids)
            report.contrastive = _dump_effects(contrastive)
            locus = point_of_commitment(contrastive)
            report.point_of_commitment = asdict(locus) if locus else None
            extra["llm_live_calls"] = adapter.calls
            _log({"event": "contrastive_done", "rows": report.contrastive, "llm_live_calls": adapter.calls})

        extra["phase"] = "shapley_probes"
        factors = planted_factors()
        report.shapley_toy = exact_shapley(lambda s: coalition_value(s), factors)
        report.contrastive_toy_lie = contrastive_leave_one_out(factors, factors)
        report.notes.append(
            "Planted AND: factual knockout credits 1+1 (overcount); Shapley splits 0.5/0.5/0."
        )
        report.probes = ProbeAgent().suggest_from_mri(report)
        factual.outcomes["causal_mri"] = report.to_dict()
        heuristic = list(factual.outcomes.get("probe_suggestions") or [])
        factual.outcomes["probe_suggestions"] = heuristic + report.probes
        extra = _persist_report(report, factual, extra)
        extra["llm_live_calls"] = adapter.calls
        extra["phase"] = "complete"
        extra["summary"] = summarize_report(report)
        return "complete", report, extra
    except QuotaExhaustedError as exc:
        extra["llm_live_calls"] = adapter.calls
        if factual is not None:
            factual.outcomes["causal_mri"] = report.to_dict()
            try:
                extra = _persist_report(report, factual, extra)
            except Exception as persist_exc:
                extra["persist_error"] = str(persist_exc)[:240]
        exc.partial_log = factual
        exc.partial_report = report
        exc.partial_extra = extra
        raise


def _finish(status: str, report: CausalMRIReport | None, reason: str | None, extra: dict[str, Any], code: int) -> int:
    extra["llm_live_calls"] = extra.get("llm_live_calls")
    _write_summary_md(status, report, reason, extra)
    _write_status(state=status, reason=reason, **{k: v for k, v in extra.items() if k != "summary"})
    _log({"event": status, "reason": reason, **{k: extra.get(k) for k in ("phase", "json_path", "report_path", "llm_live_calls", "run_id")}})
    if extra.get("summary"):
        print(extra["summary"], flush=True)
    print(f"CAUSAL_MRI_{status.upper()}", flush=True)
    print(f"SUMMARY_PATH={SUMMARY_PATH}", flush=True)
    return code


def main() -> int:
    _write_status(state="starting")
    _log({"event": "start", "goal": "full-cast 60-round dual-engine MRI", "sampled_top_k": 1})
    adapter: CountingAdapter | None = None
    report: CausalMRIReport | None = None
    extra: dict[str, Any] = {"phase": "ping"}
    try:
        adapter = _pick_adapter()
        extra["model"] = adapter.model
    except QuotaExhaustedError as exc:
        return _finish("stopped_quota", None, str(exc)[:400], extra, 2)
    except SystemExit as exc:
        return _finish("failed_ping", None, str(exc)[:400], extra, 1)

    assert adapter is not None
    try:
        status, report, extra = run_mri(adapter, rounds=60, mvp=False)
        return _finish(status, report, None, extra, 0)
    except QuotaExhaustedError as exc:
        extra = getattr(exc, "partial_extra", extra)
        extra["llm_live_calls"] = adapter.calls
        extra["phase"] = extra.get("phase") or "unknown"
        report = getattr(exc, "partial_report", report)
        partial = getattr(exc, "partial_log", None)
        if partial is not None and report is not None:
            try:
                extra = _persist_report(report, partial, extra)
            except Exception as persist_exc:
                extra["persist_error"] = str(persist_exc)[:240]
        return _finish("stopped_quota", report, str(exc)[:400], extra, 2)
    except MemoryError as exc:
        _log({"event": "oom_degrade", "error": str(exc)[:300]})
        extra["degraded"] = True
        try:
            status, report, extra = run_mri(adapter, rounds=20, mvp=True)
            extra["degraded"] = True
            extra["degrade_reason"] = "OOM on 60-round full-cast; reran mvp 20"
            return _finish(status, report, extra.get("degrade_reason"), extra, 0)
        except QuotaExhaustedError as qexc:
            extra["llm_live_calls"] = adapter.calls
            return _finish("stopped_quota", report, str(qexc)[:400], extra, 2)
    except Exception as exc:
        extra["llm_live_calls"] = adapter.calls
        extra["trace"] = traceback.format_exc()[-1200:]
        _log({"event": "failed", "error": f"{type(exc).__name__}: {str(exc)[:400]}", "trace": extra["trace"]})
        return _finish("failed", report, f"{type(exc).__name__}: {str(exc)[:400]}", extra, 1)


if __name__ == "__main__":
    raise SystemExit(main())
