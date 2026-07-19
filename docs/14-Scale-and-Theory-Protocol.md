# Scale and Theory Protocol for Agent Social Dynamics

LabWars should be read as an Agent Social Dynamics instrument, not only as a 14-agent narrative sandbox. The canonical 14-agent lab is the hand-authored mechanism probe; the scalable population layer tests whether the same social-state metrics remain non-degenerate under larger organizations.

## 1. Why scale matters

A reviewer can reasonably object that 14 agents across 60 rounds looks like a story. The scale protocol directly addresses this by adding:

- 50-200 synthetic agents for benchmark-scale studies;
- 100-1000 round long-horizon event generation;
- multiple random seeds;
- fixed emergence metrics rather than anecdotal scenes.

The purpose is not to claim that larger is automatically more realistic. The purpose is to separate mechanism validity from a single authored cast.

## 2. Canonical vs scalable modes

| Mode | Population | Rounds | Main use |
|---|---:|---:|---|
| canonical_mri | 14 agents | 60 | interpretable mechanism tracing and report examples |
| scale_baseline | 50/100/200 agents | 100-1000 | social physics robustness and metric stability |
| model_comparison | 14-50 agents | 60-200 | LLM backend and policy-mode comparison |

Large scale should normally use `policy_mode=social_physics` with `llm_provider=scripted`. LLM-backed large runs are possible, but they are expensive because every active agent produces constrained public/private rendering.

## 3. Population synthesis

The scalable population generator keeps the original LabWars archetypes but expands them into a hierarchical organization:

- multiple PI-led labs;
- PhD, master, postdoc, engineer, visiting-student clones with deterministic trait jitter;
- external reviewers/collaborators/program officers/alumni;
- denser within-lab relationships and weaker cross-lab ties.

This is deliberately not new hand-written drama. The same action field, memory, belief, trust, contribution, and relationship updates drive all agents.

## 4. Theory anchors

The implemented variables map to known social-science and cognitive concepts:

| LabWars object | Theory anchor | Interpretation |
|---|---|---|
| trust network | structural balance / network formation | relations become clustered or polarized under repeated interaction |
| contribution ledger | status and credit attribution theory | ambiguous effort creates entitlement and status competition |
| PI dependency | authority and compliance theory | resource asymmetry produces public compliance and private divergence |
| memory salience and reconsolidation | cognitive memory theory | grievances persist when emotionally rehearsed |
| action field | bounded rationality / BDI-style candidate generation | agents select from pressure-shaped feasible actions rather than unconstrained roleplay |
| homophily-like clustering | computational sociology | similar role/lab positions tend to form denser cooperative ties |

The formal state object remains:

```text
z_i(t) = [r_i(t), tau_i(t), p_i(t), c_i(t), u_i(t), m_i(t)]
z_i(t+1) = F(z_i(t), z_j(t), E_t, a_t, mu_i(t))
```

where reputation, trust, power, contribution entitlement, uncertainty, and memory pressure jointly determine the next action distribution.

## 5. Hard metrics

Scale runs should report at minimum:

- Trust Collapse / trust entropy;
- Power Concentration Gini;
- Alliance Modularity Proxy;
- Credit Attribution Gap;
- Conflict Cascade Length;
- Reputation Volatility;
- Social State Volatility;
- Organization Fragility Index.

These metrics are implemented in `src/experiments/social_metrics.py` and summarized by `src/experiments/scale.py`.

## 6. Recommended reviewer-facing commands

Fast smoke run:

```powershell
python -m src.experiments scale --population-sizes 14,50 --rounds 20 --seeds 1 --llm-provider scripted --policy-mode social_physics
```

Scale baseline:

```powershell
python -m src.experiments scale --population-sizes 14,50,100,200 --rounds 100 --seeds 5 --llm-provider scripted --policy-mode social_physics
```

Long-horizon baseline:

```powershell
python -m src.experiments scale --population-sizes 50,100,200 --rounds 1000 --seeds 10 --llm-provider scripted --policy-mode social_physics
```

LLM-backed model comparison should start smaller:

```powershell
$env:DEEPSEEK_API_KEY = Read-Host "Paste DEEPSEEK_API_KEY"
$env:LABWARS_LLM_CONFIG = "config/llm.deepseek.yaml"
python -m src.experiments scale --population-sizes 14,50 --rounds 60 --seeds 2 --llm-provider openai --policy-mode dual_engine
```

## 7. Claim boundary

The current scalable layer supports the claim that LabWars is moving from a single simulated story toward a benchmarkable artificial organization. It does not yet prove external validity to human labs. The right next validation step is to compare metric signatures against historical organizational cases, lab ethnographies, or controlled human multi-player studies.

## 8. Standard experimental protocol CLI

The scale runner varies only population size. The protocol runner varies population size and mechanism lesions together:

```powershell
python -m src.experiments protocol --population-sizes 10,50,100,500 --rounds 500 --seeds 100 --conditions baseline,no_memory,no_status,no_trust,no_hierarchy --llm-provider scripted --policy-mode social_physics
```

Outputs:

- `agent_social_dynamics_protocol_v1_social_physics_500r.json`
- `agent_social_dynamics_protocol_v1_social_physics_500r.md`

Use the JSON for statistics and the Markdown file for reviewer-readable summaries.