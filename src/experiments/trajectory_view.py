"""Trajectory export for visual inspection and lightweight viewers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.engine.run_log import RunLog


def build_trajectory_view(log: RunLog) -> dict[str, Any]:
    """Convert a run log into chart-ready state/action/event traces."""
    rounds: list[dict[str, Any]] = []
    actions_by_round: dict[int, list[dict[str, Any]]] = {}
    for action in log.actions:
        actions_by_round.setdefault(int(action.get("round", 0)), []).append(action)

    event_by_round = {int(e.get("round", 0)): e for e in log.events}
    for rec in log.round_records:
        rnd = int(rec.get("round", 0))
        event = event_by_round.get(rnd, {})
        metrics = rec.get("metrics", {})
        agent_deltas = rec.get("agent_deltas", {})
        round_actions = actions_by_round.get(rnd, [])
        rounds.append({
            "round": rnd,
            "event": {
                "event_id": event.get("event_id"),
                "type": event.get("type"),
                "generator": event.get("payload", {}).get("generator"),
                "candidates": event.get("payload", {}).get("event_candidates", []),
            },
            "metrics": {
                "authorship_dispute_index": metrics.get("authorship_dispute_index", 0.0),
                "public_private_divergence": metrics.get("public_private_divergence", 0.0),
                "trust_fragmentation": metrics.get("trust_fragmentation", 0.0),
                "credit_threat_density": metrics.get("credit_threat_density", 0.0),
                "career_hostage_index": metrics.get("career_hostage_index", 0.0),
                "pi_control_pressure_phd_a": metrics.get("pi_control_pressure_phd_a", 0.0),
            },
            "actions": [
                {
                    "agent": a.get("agent"),
                    "type": a.get("type"),
                    "target": a.get("target"),
                    "intensity": a.get("intensity", 0.0),
                    "selected_action": a.get("selected_action"),
                    "candidates": a.get("action_candidates", []),
                    "motives": a.get("private_motives", {}),
                }
                for a in round_actions
            ],
            "memory_reconsolidation": {
                aid: delta.get("memory_reconsolidation", {})
                for aid, delta in agent_deltas.items()
                if delta.get("memory_reconsolidation", {}).get("updated")
            },
        })

    return {
        "run_id": log.run_id,
        "config": log.config,
        "outcomes": log.outcomes,
        "rounds": rounds,
    }


def write_trajectory_view(log: RunLog, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_trajectory_view(log)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_trajectory_html(log: RunLog, path: Path | str) -> Path:
    """Write a self-contained HTML viewer for the trajectory JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(build_trajectory_view(log), ensure_ascii=False)
    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>LabWars Trajectory {log.run_id}</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f7f7f8; color: #1f2933; }}
    header {{ padding: 16px 20px; border-bottom: 1px solid #d8dee4; background: #fff; position: sticky; top: 0; }}
    main {{ display: grid; grid-template-columns: 300px 1fr; gap: 16px; padding: 16px; }}
    aside, section {{ background: #fff; border: 1px solid #d8dee4; border-radius: 8px; padding: 12px; }}
    .round {{ border-bottom: 1px solid #edf0f2; padding: 10px 0; }}
    .metric {{ display: grid; grid-template-columns: 170px 1fr 54px; align-items: center; gap: 8px; margin: 6px 0; }}
    .bar {{ height: 8px; background: #e5e7eb; border-radius: 999px; overflow: hidden; }}
    .bar span {{ display: block; height: 100%; background: #2563eb; }}
    button {{ width: 100%; text-align: left; padding: 8px; border: 0; background: transparent; border-radius: 6px; cursor: pointer; }}
    button:hover, button.active {{ background: #eef2ff; }}
    pre {{ white-space: pre-wrap; background: #111827; color: #f9fafb; padding: 12px; border-radius: 6px; overflow: auto; }}
    .pill {{ display: inline-block; padding: 2px 7px; border-radius: 999px; background: #eef2ff; margin-left: 6px; font-size: 12px; }}
  </style>
</head>
<body>
  <header><strong>LabWars Trajectory</strong><span class=\"pill\">run {log.run_id}</span></header>
  <main>
    <aside id=\"roundList\"></aside>
    <section id=\"detail\"></section>
  </main>
  <script>
    const data = {payload};
    const list = document.getElementById('roundList');
    const detail = document.getElementById('detail');
    function metricRow(k, v) {{
      const pct = Math.max(0, Math.min(1, Number(v || 0))) * 100;
      return `<div class=\"metric\"><span>${{k}}</span><div class=\"bar\"><span style=\"width:${{pct}}%\"></span></div><span>${{Number(v || 0).toFixed(3)}}</span></div>`;
    }}
    function renderRound(r) {{
      detail.innerHTML = `<h2>Round ${{r.round}} <span class=\"pill\">${{r.event.type}}</span></h2>` +
        `<p><strong>Event:</strong> ${{r.event.event_id}} | generator=${{r.event.generator || ''}}</p>` +
        `<h3>Metrics</h3>` + Object.entries(r.metrics).map(([k,v]) => metricRow(k,v)).join('') +
        `<h3>Actions</h3><pre>${{JSON.stringify(r.actions, null, 2)}}</pre>` +
        `<h3>Event Candidates</h3><pre>${{JSON.stringify(r.event.candidates, null, 2)}}</pre>` +
        `<h3>Memory Reconsolidation</h3><pre>${{JSON.stringify(r.memory_reconsolidation, null, 2)}}</pre>`;
    }}
    data.rounds.forEach((r, i) => {{
      const b = document.createElement('button');
      b.textContent = `R${{r.round}} · ${{r.event.type}}`;
      b.onclick = () => {{ document.querySelectorAll('button').forEach(x => x.classList.remove('active')); b.classList.add('active'); renderRound(r); }};
      list.appendChild(b);
      if (i === 0) {{ b.classList.add('active'); renderRound(r); }}
    }});
  </script>
</body>
</html>"""
    path.write_text(html, encoding="utf-8")
    return path
