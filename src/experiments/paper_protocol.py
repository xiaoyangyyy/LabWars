"""Top-venue paper protocol: Causal Decompiler battery + optional CRN contrasts.

Phases (all CRN-paired, all replay LLM traces):

1. Identity twin
2. Split-Y snapshot (public vs private)
3. Memory IRF over story beats
4. Contrastive skip vs budgeted story Shapley (AND-cause lie)
5. Three-worlds spillover / hypocrisy
6. Optional λ lesion (field vs LLM) — cache misses expected
7. Optional A/B/C/D CRN contrasts

This is the experiment code a reviewer can rerun. It does not call a paid API
unless the SimConfig says so.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.engine.causal import CausalDecompiler, CausalMRIReport
from src.engine.causal.twin import load_factual, sim_config_from_log
from src.engine.simulation import SimConfig
from src.experiments.paper_contrasts import run_paper_contrasts
from src.experiments.paper_tables import latex_split_y, render_paper_markdown
from src.experiments.report import generate_report
from src.world.loader import PROJECT_ROOT

DEFAULT_PAPER_DIR = PROJECT_ROOT / "output" / "reports"


@dataclass
class PaperProtocolResult:
    report: CausalMRIReport
    summary: str
    tables_markdown: str
    latex_split_y: str
    contrasts: dict[str, Any] = field(default_factory=dict)
    json_path: Path | None = None
    markdown_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "report": self.report.to_dict(),
            "tables_markdown": self.tables_markdown,
            "contrasts": self.contrasts,
            "json_path": str(self.json_path) if self.json_path else None,
            "markdown_path": str(self.markdown_path) if self.markdown_path else None,
        }


def _summarize(report: CausalMRIReport) -> str:
    findings = report.findings or []
    replay = report.llm_replay or {}
    lines = [
        f"run={report.factual_run_id} identity={report.identity_twin_ok} Y={report.factual_y:.4f}",
        f"replay hits={replay.get('identity_run_hits', 0)} misses={replay.get('identity_run_misses', 0)}",
    ]
    lines.extend(f"finding: {item}" for item in findings[:6])
    return "\n".join(lines)


def run_paper_protocol(
    config: SimConfig | None = None,
    *,
    from_jsonl: Path | str | None = None,
    outcome: str = "protest_authorship",
    auto_battery: bool = True,
    include_lambda: bool = False,
    include_toy_shapley: bool = True,
    contrasts: list[str] | None = None,
    contrast_seeds: list[int] | None = None,
    write_output: bool = False,
    output_dir: Path | str | None = None,
) -> PaperProtocolResult:
    decompiler = CausalDecompiler()
    factual = load_factual(from_jsonl) if from_jsonl else None
    cfg = config or (sim_config_from_log(factual) if factual is not None else SimConfig(mvp=True, max_rounds=8, llm_provider="scripted"))
    report = decompiler.decompile(
        cfg,
        outcome=outcome,
        auto_battery=auto_battery,
        include_lambda=include_lambda,
        include_toy_shapley=include_toy_shapley,
        blame_limit=None if auto_battery else 0,
        factual=factual,
    )
    contrast_payload: dict[str, Any] = {}
    table_rows: list[dict[str, Any]] = []
    if contrasts:
        contrast_payload = run_paper_contrasts(
            contrasts,
            seeds=contrast_seeds or [cfg.seed],
            max_rounds=cfg.max_rounds,
        )
        table_rows = contrast_payload.get("table_rows") or []
    markdown = render_paper_markdown(report.to_dict(), contrasts=table_rows)
    result = PaperProtocolResult(
        report=report,
        summary=_summarize(report),
        tables_markdown=markdown,
        latex_split_y=latex_split_y(report.to_dict()),
        contrasts=contrast_payload,
    )
    if write_output and decompiler.last_log is not None:
        out_dir = Path(output_dir) if output_dir else DEFAULT_PAPER_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        run_id = decompiler.last_log.run_id
        json_path = out_dir / f"paper_protocol_{run_id}.json"
        md_path = out_dir / f"paper_protocol_{run_id}.md"
        json_path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        md_path.write_text(markdown, encoding="utf-8")
        generate_report(log=decompiler.last_log, output_dir=out_dir)
        result.json_path = json_path
        result.markdown_path = md_path
    return result
