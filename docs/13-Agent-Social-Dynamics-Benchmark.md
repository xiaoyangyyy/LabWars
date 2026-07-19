# Agent Social Dynamics Benchmark

> LabWars is now framed as an Agent Social Dynamics Benchmark: a repeatable instrument for testing how social state, organizational pressure, memory, power, trust, and contribution shape LLM-agent behavior.

## 1. Benchmark definition

A benchmark task has:

- an organizational pressure input;
- a controlled intervention or lesion;
- standardized outcome metrics;
- a multi-seed evaluation protocol.

This moves LabWars from simulation toward measurement.

## 2. Standard tasks

| Task | Pressure input | Observation target |
|---|---|---|
| conflict_emergence | PI power, authorship uncertainty, deadline pressure | conflict probability, cascade length, fragility |
| alliance_formation | local trust rupture, outside validation, social memory | alliance modularity, trust entropy, stability |
| credit_attribution | contribution ambiguity and ledger uncertainty | credit-attribution gap, authorship dispute |
| memory_mediation | memory removal, shuffle, delay, or correction | grievance persistence and delayed behavior |
| authority_compliance | PI dependency and funding pressure | compliance, public/private divergence, power concentration |
| integrity_stress | deadline plus rival pressure | integrity risk and selective reporting pressure |

Code entry point:

```python
from src.engine import SimConfig
from src.experiments.benchmark_tasks import run_benchmark_task

result = run_benchmark_task("conflict_emergence", SimConfig(max_rounds=20), seeds=[0, 1])
print(result.summary)
```

## 3. Social emergence metrics

The benchmark reports metrics beyond task success:

| Metric | Meaning |
|---|---|
| trust_entropy | distributional complexity of trust relations |
| power_concentration_gini | concentration of attention/dependency around authority targets |
| alliance_modularity_proxy | polarization proxy for alliance-like structure |
| conflict_cascade_length | longest consecutive conflict-action cascade |
| reputation_volatility | volatility of reputation pressure across actions |
| credit_attribution_gap | perceived contribution-authorship mismatch |
| social_state_volatility | action-level change in selected social pressure |
| organization_fragility_index | composite fragility from dispute, trust, divergence, pressure, and cascade length |

Implementation lives in [`src/experiments/social_metrics.py`](../src/experiments/social_metrics.py).

## 4. Model comparison use

The same benchmark can compare LLM backends and policies:

- OpenAI-compatible models
- Anthropic models
- local Ollama models
- social_physics policy
- dual_engine policy
- llm_native policy
- LLM scoring disabled

The intended question is not whether an agent finishes a task. The question is how artificial social structure changes under controlled pressure.

## 5. Scientific claim

LabWars should be evaluated as a scientific instrument when it can satisfy three requirements:

1. Formal state object: Agent Social State `z_i(t)`.
2. Repeatable tasks: benchmark conditions with fixed pressure inputs and lesions.
3. Emergence metrics: trust, power, alliance, conflict, reputation, credit, and fragility measurements.

Together, these make LabWars a benchmark for Agent Social Dynamics rather than a narrative simulator.
