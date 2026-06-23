"""Probe Agent — suggest next interventions from metric anomalies."""

from __future__ import annotations

from typing import Any


class ProbeAgent:
    def suggest(self, round_records: list[dict[str, Any]], config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if not round_records:
            return []

        latest = round_records[-1]
        metrics = latest.get("metrics", {})
        rnd = latest.get("round", 0)
        suggestions: list[dict[str, Any]] = []

        tf = metrics.get("trust_fragmentation", 0)
        if tf > 0.55:
            suggestions.append({
                "round": rnd + 1,
                "intervention_type": "memory_intervention",
                "variant": "memory_correct_false_rumor",
                "reason": f"trust_fragmentation={tf:.3f} elevated; test trust recovery",
            })

        adi = metrics.get("authorship_dispute_index", 0)
        if adi > 0.45 and rnd >= 30:
            suggestions.append({
                "round": rnd + 1,
                "intervention_type": "authorship_framing",
                "variant": "merit_frame",
                "reason": f"authorship_dispute_index={adi:.3f}; test merit framing",
            })

        ppd = metrics.get("public_private_divergence", 0)
        if ppd > 0.35:
            suggestions.append({
                "round": rnd + 1,
                "intervention_type": "memory_intervention",
                "variant": "memory_strengthen_betrayal",
                "target_event": "E031",
                "reason": f"public_private_divergence={ppd:.3f}; probe betrayal memory effect",
            })

        return suggestions
