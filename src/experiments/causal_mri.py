"""Run the Causal Decompiler as an experiment product, not a sidecar."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.engine.causal import CausalDecompiler, CausalMRIReport, CausalOp
from src.engine.simulation import SimConfig
from src.experiments.report import generate_report
from src.world.loader import PROJECT_ROOT

DEFAULT_MRI_DIR = PROJECT_ROOT / "output" / "reports"


def run_causal_mri(
    config: SimConfig,
    *,
    outcome: str = "protest_authorship",
    extra_ops: list[CausalOp] | None = None,
    memory_rounds: list[int] | None = None,
    blame_event_ids: list[str] | None = None,
    blame_limit: int | None = None,
    include_toy_shapley: bool = True,
    auto_battery: bool = False,
    factual=None,
    write_output: bool = False,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    decompiler = CausalDecompiler()
    report = decompiler.decompile(
        config,
        outcome=outcome,
        extra_ops=extra_ops,
        memory_rounds=memory_rounds,
        blame_event_ids=blame_event_ids,
        blame_limit=blame_limit,
        include_toy_shapley=include_toy_shapley,
        auto_battery=auto_battery,
        factual=factual,
    )
    log = decompiler.last_log
    payload = {
        "report": report,
        "log": log,
        "summary": summarize_report(report),
    }
    if write_output and log is not None:
        out_dir = Path(output_dir) if output_dir else DEFAULT_MRI_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"causal_mri_{log.run_id}.json"
        json_path.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        generate_report(log=log, output_dir=out_dir)
        payload["json_path"] = json_path
    return payload


def summarize_report(report: CausalMRIReport) -> str:
    irf = report.memory_irf[0] if report.memory_irf else None
    skip = report.contrastive[0] if report.contrastive else None
    replay = report.llm_replay or {}
    lines = [
        f"run={report.factual_run_id} identity_twin_ok={report.identity_twin_ok}",
        f"Y={report.outcome} factual={report.factual_y:.4f}",
        (
            "split-Y: "
            + ", ".join(f"{k}={float(v):.3f}" for k, v in report.split_y.items())
        ),
        (
            f"LLM replay hits={replay.get('identity_run_hits', 0)} "
            f"misses={replay.get('identity_run_misses', 0)}"
        ),
    ]
    if report.findings:
        lines.extend(f"finding: {item}" for item in report.findings[:4])
    if irf:
        lines.append(f"memory IRF {irf.get('factor_id')} ATE={float(irf.get('ate', 0.0)):+.4f}")
    if skip:
        lines.append(f"contrastive {skip.get('factor_id')} ATE={float(skip.get('ate', 0.0)):+.4f}")
    if report.probes:
        top = report.probes[0]
        lines.append(f"next probe: {top.get('variant')} ({top.get('reason')})")
    return "\n".join(lines)
