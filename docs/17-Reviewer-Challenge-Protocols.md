# Reviewer Challenge Protocols

This document turns three likely reviewer objections into explicit experiments.

## Challenge 1: Emergence is not manually manufactured

Reviewer question:

> Did the system produce hierarchy, inequality, and coalitions only because the initial profiles already encoded hierarchy and inequality?

Protocol:

```powershell
python -m src.experiments egalitarian-challenge --population-size 500 --rounds 500 --seeds 10 --llm-provider scripted --policy-mode social_physics
```

Conditions:

- `canonical_initialization`: normal scalable population;
- `egalitarian_initialization`: same capability, resource, status, contribution, trust, dependency, alliance, and network conditions.

Key measurements:

- `power_concentration_gini`
- `credit_attribution_gap`
- `network_modularity_q`
- `coalition_strength`
- `coalition_persistence`
- `cascade_probability`
- `emergent_pattern_score`

Strong result:

> Even from equal initial conditions, the system develops non-zero power concentration, coalition structure, and conflict cascades over long horizons.

This does not prove human realism, but it directly argues against the simplest "you initialized the outcome" critique.

## Challenge 2: Why hybrid rather than rules?

Hypotheses:

| Hypothesis | Claim | Metric |
|---|---|---|
| H1 | LLM cognition increases strategic diversity | `action_entropy` |
| H2 | LLM cognition stabilizes long-term alliances | `coalition_persistence` |
| H3 | LLM cognition can increase failure risk | `cascade_probability` |

Protocol:

```powershell
python -m src.experiments policy-compare --population-size 50 --rounds 60 --seeds 10 --regimes rule_baseline,llm_native,hybrid,hybrid_sampled --llm-provider scripted --sampled-top-k 20
```

For a real LLM backend, begin smaller:

```powershell
python -m src.experiments policy-compare --population-size 20 --rounds 30 --seeds 3 --regimes rule_baseline,hybrid_sampled --llm-provider openai --sampled-top-k 5
```

The paper should not say "hybrid gets a higher score". It should say which mechanism changes: diversity, alliance persistence, or cascade risk.

## Challenge 3: Cognitive sampling has a compute frontier

Reviewer question:

> Why top-k=20? Is it arbitrary?

Protocol:

```powershell
python -m src.experiments sampling-frontier --population-size 100 --rounds 100 --seeds 5 --k-values 0,5,10,20,50,100,full --llm-provider scripted
```

Metrics:

- x-axis: `llm_sampled_action_fraction`
- y-axis candidates:
  - `emergent_pattern_score`
  - `action_entropy`
  - `coalition_persistence`
  - `cascade_probability`
  - `organization_fragility_index`

Strong result:

> A small k, such as 20, approaches the full-hybrid emergence signature at a fraction of the LLM calls.

This frames cognitive sampling as an experimental contribution, not just a cost-saving trick.

## Combined reviewer-facing run set

Fast smoke:

```powershell
python -m src.experiments egalitarian-challenge --population-size 20 --rounds 3 --seeds 1
python -m src.experiments policy-compare --population-size 20 --rounds 3 --seeds 1 --regimes rule_baseline,hybrid_sampled --sampled-top-k 4
python -m src.experiments sampling-frontier --population-size 20 --rounds 3 --seeds 1 --k-values 0,4,full
```

Paper-scale scripted runs:

```powershell
python -m src.experiments egalitarian-challenge --population-size 500 --rounds 500 --seeds 10 --llm-provider scripted --policy-mode social_physics
python -m src.experiments policy-compare --population-size 50 --rounds 60 --seeds 10 --regimes rule_baseline,llm_native,hybrid,hybrid_sampled --llm-provider scripted --sampled-top-k 20
python -m src.experiments sampling-frontier --population-size 100 --rounds 100 --seeds 5 --k-values 0,5,10,20,50,100,full --llm-provider scripted
```