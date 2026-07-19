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

## 6. Scale protocol

LabWars now separates the benchmark into two layers:

- `canonical_mri`: 14 agents, 60 rounds, best for interpretable causal traces.
- `scale_baseline`: 50/100/200 agents, 100-1000 rounds, best for testing whether social-emergence metrics remain stable beyond one authored cast.

Command-line entry point:

```powershell
python -m src.experiments scale --population-sizes 14,50,100,200 --rounds 100 --seeds 5 --llm-provider scripted --policy-mode social_physics
```

The scale layer uses deterministic archetype jitter and hierarchical lab assignment; it does not add new hand-authored conflict scripts. See [`docs/14-Scale-and-Theory-Protocol.md`](14-Scale-and-Theory-Protocol.md).

## 7. Repeatable protocol matrix

The benchmark also exposes a fixed experimental-science matrix:

```text
population_size = 10, 50, 100, 500
rounds = 500
seeds = 100
conditions = baseline, no_memory, no_status, no_trust, no_hierarchy
```

Command:

```powershell
python -m src.experiments protocol --population-sizes 10,50,100,500 --rounds 500 --seeds 100 --conditions baseline,no_memory,no_status,no_trust,no_hierarchy --llm-provider scripted --policy-mode social_physics
```

The point is to compare each lesion against its same-size baseline, not to interpret one dramatic run.
## 8. Emergence-pattern validation

Outcome means are not enough. LabWars reports distributional emergence metrics:

- `power_law_alpha` and `power_law_fit_r2` for heavy-tailed power/attention concentration;
- `network_modularity_q` for alliance-like trust clustering;
- `cascade_tail_alpha` and `cascade_tail_r2` for conflict cascade tails;
- `emergent_pattern_score` as a compact joint signature.

## 9. Policy-regime comparison

Use `policy-compare` to compare:

- `rule_baseline`: social physics only;
- `llm_native`: LLM proposes candidate actions;
- `hybrid`: social field plus LLM candidate scoring;
- `hybrid_sampled`: hybrid with top-k cognitive sampling.

```powershell
python -m src.experiments policy-compare --population-size 50 --rounds 60 --seeds 3 --regimes rule_baseline,llm_native,hybrid,hybrid_sampled --llm-provider scripted --sampled-top-k 20
```

See [`docs/16-Emergence-and-Cognitive-Sampling.md`](16-Emergence-and-Cognitive-Sampling.md).

## 10. Reviewer challenge protocols

LabWars includes explicit challenge protocols:

- `egalitarian-challenge`: tests emergence from equal initial capability/resource/status/network.
- `policy-compare`: tests H1/H2/H3 about hybrid cognition with action entropy, coalition persistence, and cascade probability.
- `sampling-frontier`: sweeps k=0/5/10/20/50/100/full to measure the LLM-call vs emergence frontier.

See [`docs/17-Reviewer-Challenge-Protocols.md`](17-Reviewer-Challenge-Protocols.md).
