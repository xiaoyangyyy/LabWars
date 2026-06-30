"""Agent MRI decompilation report generator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.engine.run_log import RunLog
from src.experiments.metrics import compute_run_metrics, mediation_fraction
from src.experiments.runner import run_single
from src.world.loader import PROJECT_ROOT

TEMPLATE_PATH = PROJECT_ROOT / "config" / "report_template.md"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "output" / "reports"


def _format_timeline(timeline: list[dict[str, Any]]) -> str:
    if not timeline:
        return "_No salient memory nodes recorded._"
    lines = []
    for node in timeline:
        lines.append(
            f"- R{node['round']} | {node['agent']} | {node['event_ref']} | "
            f"strength={node.get('strength', 0):.2f} | valence={node.get('valence', 0):+.2f} | "
            f"{node.get('interpretation', '')}"
        )
    return "\n".join(lines)


def _format_trust_snapshots(snaps: dict[int, dict[str, float]]) -> str:
    if not snaps:
        return "_Trust snapshots unavailable._"
    lines = []
    for rnd in sorted(snaps):
        edges = ", ".join(f"{k.split('_', 1)[1]}={v:.3f}" for k, v in snaps[rnd].items())
        lines.append(f"- R{rnd}: {edges}")
    return "\n".join(lines)


def _format_curve(curve: list[dict[str, float]], key: str) -> str:
    if not curve:
        return "_No data._"
    sample = curve[:: max(1, len(curve) // 10)]
    return "\n".join(f"- R{p['round']}: {key}={p.get(key, 0):.3f}" for p in sample)



def _format_causal_path(path: dict[str, Any]) -> str:
    if not path or not path.get("nodes"):
        return "_No path-level causal chain extracted._"
    lines = [f"Finding: {path.get('finding', '')}".strip()]
    for node in path.get("nodes", []):
        metrics = []
        if "strength" in node:
            metrics.append(f"strength={float(node.get('strength', 0)):.2f}")
        if "valence" in node:
            metrics.append(f"valence={float(node.get('valence', 0)):+.2f}")
        if "intensity" in node:
            metrics.append(f"intensity={float(node.get('intensity', 0)):.2f}")
        suffix = f" ({', '.join(metrics)})" if metrics else ""
        lines.append(
            f"- R{node.get('round')} -> {node.get('kind')}:{node.get('label')} "
            f"[{node.get('event_id') or 'action'}]{suffix} - {node.get('detail', '')}"
        )
    outcome = path.get("outcome_summary", {})
    lines.append(
        "Outcome: "
        f"protest={float(outcome.get('protest_authorship', 0)):.3f}, "
        f"escalation={float(outcome.get('authorship_escalation_score', 0)):.3f}, "
        f"memory_cluster={float(outcome.get('memory_authorship_cluster_strength', 0)):.3f}, "
        f"promise_broken_R52={float(outcome.get('promise_broken_strength_r52', 0)):.3f}."
    )
    lines.append(f"Counterfactual: {path.get('counterfactual_hint', '')}")
    return "\n".join(lines)

def generate_report_from_log(log: RunLog, metrics: dict[str, Any] | None = None) -> str:
    metrics = metrics or compute_run_metrics(log)
    outcomes = metrics["outcomes"]
    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    latent = (
        f"- Primary outcomes: protest={outcomes.get('protest_authorship', 0):.0f}, "
        f"trust_pi_final={outcomes.get('trust_pi_final', 0):.3f}\n"
        f"- Memory cluster strength: {outcomes.get('memory_authorship_cluster_strength', 0):.3f}\n"
        f"- Authority compliance: {outcomes.get('authority_compliance', 0):.3f}"
    )

    memory_causal = (
        f"- Authorship memory cluster (R3鈥揜40): {outcomes.get('memory_authorship_cluster_strength', 0):.3f}\n"
        f"- Promise broken strength @R52: {outcomes.get('promise_broken_strength_r52', 0):.3f}\n"
        f"- Confound ladder proxy: memory contribution is reported as a continuous mediation fraction"
    )

    interventions = log.interventions_applied
    inter_lines = "\n".join(
        f"- R{i.get('round')}: {i.get('intervention_id')} ({i.get('variant')})"
        for i in interventions
    ) or "_No interventions applied._"

    div_peaks = metrics.get("divergence_peaks", [])
    div_text = "\n".join(
        f"- R{p['round']} divergence={p['divergence']:.3f} ({p.get('event_id')})" for p in div_peaks[:8]
    ) or "_No divergence ranking available._"

    failure = f"- Critic violations: {metrics.get('critic_count', 0)}\n"
    if log.critic_violations:
        failure += "\n".join(
            f"  - R{v.get('round')} {v.get('agent')}: {v.get('rule_id', v.get('message', 'violation'))}"
            for v in log.critic_violations[:5]
        )

    probes = log.outcomes.get("probe_suggestions") or []
    probe_text = "\n".join(
        f"- R{p.get('round')}: {p.get('variant')} - {p.get('reason')}" for p in probes
    ) or "_No probe suggestions._"

    replacements = {
        "{{run_id}}": log.run_id,
        "{{experiment_id}}": str(metrics.get("experiment_id") or log.config.get("experiment_id", "NA")),
        "{{condition_id}}": str(metrics.get("condition_id") or log.config.get("condition_id", "NA")),
        "{{seed}}": str(metrics.get("seed") or log.config.get("seed", "NA")),
        "{{timeline_section}}": _format_timeline(metrics.get("timeline", [])),
        "{{latent_section}}": latent,
        "{{trust_section}}": _format_trust_snapshots(metrics.get("trust_snapshots", {})),
        "{{authorship_section}}": _format_curve(metrics.get("authorship_dispute_curve", []), "authorship_dispute_index"),
        "{{memory_causal_section}}": memory_causal,
        "{{intervention_section}}": inter_lines,
        "{{divergence_section}}": div_text,
        "{{failure_section}}": failure,
        "{{probe_section}}": probe_text,
        "{{causal_path_section}}": _format_causal_path(metrics.get("path_level_causal_chain", {})),
    }
    text = template
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def generate_report(
    run_id: str | None = None,
    *,
    experiment_id: str = "A",
    condition_id: str = "A1",
    seed: int = 42,
    output_dir: Path | str | None = None,
    log: RunLog | None = None,
) -> Path:
    out_dir = Path(output_dir) if output_dir else DEFAULT_REPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if log is None:
        result = run_single(experiment_id, seed, condition_id)
        log = result["log"]
        metrics = result["metrics"]
    else:
        metrics = compute_run_metrics(log)

    rid = run_id or log.run_id
    report_text = generate_report_from_log(log, metrics)
    path = out_dir / f"report_{rid}.md"
    path.write_text(report_text, encoding="utf-8")

    meta = {"run_id": rid, "metrics": metrics}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def generate_finding_summary(
    control_logs: list[RunLog],
    treatment_logs: list[RunLog],
    intervention_label: str,
) -> str:
    med = mediation_fraction(control_logs, treatment_logs)
    y_c = sum(l.outcomes.get("protest_authorship", 0) for l in control_logs) / max(len(control_logs), 1)
    y_t = sum(l.outcomes.get("protest_authorship", 0) for l in treatment_logs) / max(len(treatment_logs), 1)
    delta_pct = (y_c - y_t) * 100
    return (
        f"`do({intervention_label})` shifted protest probability by {delta_pct:+.0f}pp. "
        f"Mediation fraction via authorship memory cluster: {med['mediation_fraction']:.0%}."
    )
