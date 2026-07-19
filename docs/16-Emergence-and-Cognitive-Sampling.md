# Emergence Validation and Cognitive Sampling

This document addresses three reviewer-level concerns:

1. LabWars should validate emergence, not only outcome deltas.
2. The relation between rule baselines, LLM-native agents, and hybrid cognition must be explicit.
3. Large populations require selective LLM cognition rather than all-agent all-round LLM calls.

## 1. Emergence is distributional

A social simulator is weak if it only reports mean outcomes. LabWars now reports distribution-level pattern metrics:

| Pattern | Metric | Question |
|---|---|---|
| power-center formation | `power_law_alpha`, `power_law_fit_r2` | do action/dependency targets become heavy-tailed rather than uniform? |
| alliance clustering | `network_modularity_q` | do trust relations cluster into community-like blocs? |
| conflict propagation | `cascade_tail_alpha`, `cascade_tail_r2` | do conflict episodes form cascade-size tails? |
| joint emergence | `emergent_pattern_score` | do power, modularity, and cascades jointly become non-degenerate? |

The goal is not to force a perfect power law. The goal is to test whether simulated organizations produce heavy-tailed, modular, and cascade-like signatures that are common in real social systems.

## 2. Policy-regime comparison

LabWars should always distinguish three regimes:

| Regime | Meaning | Purpose |
|---|---|---|
| `rule_baseline` | `policy_mode=social_physics` | tests whether social dynamics alone generate structure |
| `llm_native` | LLM proposes candidate actions | tests pure language-policy behavior |
| `hybrid` | social field candidates + LLM scoring | tests the core LabWars architecture |
| `hybrid_sampled` | hybrid, but only top-k uncertain agents call LLM | scalable approximation of selective cognition |

Command:

```powershell
python -m src.experiments policy-compare --population-size 50 --rounds 60 --seeds 3 --regimes rule_baseline,llm_native,hybrid,hybrid_sampled --llm-provider scripted --sampled-top-k 20
```

With a real OpenAI-compatible LLM, start smaller:

```powershell
$env:DEEPSEEK_API_KEY = Read-Host "Paste DEEPSEEK_API_KEY"
$env:LABWARS_LLM_CONFIG = "config/llm.deepseek.yaml"
python -m src.experiments policy-compare --population-size 20 --rounds 30 --seeds 1 --regimes rule_baseline,hybrid_sampled --llm-provider openai --sampled-top-k 5
```

## 3. Cognitive sampling

Large organizations cannot call an LLM for every agent every round. LabWars therefore supports top-k cognitive sampling:

```text
all agents receive social-physics state updates
all agents generate field candidates
only top-k agents by uncertainty/social pressure receive LLM cognitive scoring/rendering
unsampled agents use deterministic field-constrained rendering
```

The sampling score is a weighted detector:

```text
score_i = 0.34 uncertainty
        + 0.24 memory_pressure
        + 0.18 trust_deficit
        + 0.14 power_constraint
        + 0.10 total_social_pressure
```

Every action logs:

- whether cognitive sampling was enabled;
- whether that agent was sampled;
- sampling score and rank;
- top-k and threshold.

This turns compute reduction into an auditable experimental variable.

## 4. Recommended large-scale design

For 500 agents, avoid all-agent LLM calls:

```powershell
python -m src.experiments policy-compare --population-size 500 --rounds 500 --seeds 10 --regimes rule_baseline,hybrid_sampled --llm-provider scripted --sampled-top-k 20
```

For a paid LLM backend, use:

```powershell
python -m src.experiments policy-compare --population-size 100 --rounds 100 --seeds 3 --regimes hybrid_sampled --llm-provider openai --sampled-top-k 20
```

The paper claim should be framed as:

> LabWars separates full-population social dynamics from selectively allocated LLM cognition, allowing organization-level scale while preserving interpretable cognitive intervention points.