# 行为生成与 LLM 分工审计

本文专门澄清 LabWars 当前的行为生成机制，避免把系统误读成“LLM 自由写剧情”或“纯手写规则触发”。

结论先说清楚：

> LabWars 当前是 **continuous latent action field proposes candidates -> LLM scores candidate plausibility -> fused sampler selects primary action -> LLM renders public/private stance and memory-compatible narration**。

这不是 LLM 自由行动系统，也不是旧式阈值规则系统，而是一种可复现、可消融、可审计的 hybrid social simulation。

---

## 1. 当前真实行为链路

实现位置：

| 环节 | 文件 |
|---|---|
| 候选动作生成 | `src/engine/action_selection.py` |
| LLM 候选评分与融合采样 | `src/engine/role_policy.py` |
| scoring/rendering prompt | `src/engine/prompts.py` |
| LLM drift 审计 | `src/engine/critic.py` |
| action-field 参数 | `config/action_field.yaml` |
| action-field 消融 | `src/experiments/action_field_ablation.py` |

真实流程是：

```text
agent latent state + event + recall + relationship + project state
  -> generate_action_candidates()                    # structural field prior
  -> LLM scores each candidate plausibility           # subjective model layer
  -> fused_tendency = field_tendency + mix*(llm_score - 0.5)
  -> softmax(fused_tendency) + seeded sampling
  -> selected primary_action
  -> LLM renders public_position / private_intent / wording
  -> RolePolicyAgent overwrites raw.primary_action with selected action
  -> Critic checks wording/action drift
```

代码上最关键的是：

```python
raw["primary_action"] = {
    "type": selected_payload["type"],
    "target": selected_payload["target"],
    "intensity": selected_payload["intensity"],
}
```

也就是说，LLM 参与评分，但不能在 rendering 阶段把 `confront` 偷换成 `comply`，也不能把 `document_contribution` 偷换成 `ask_for_authorship`。

---

## 2. LLM 做什么，不做什么

| 层 | 是否由 LLM 决定 | 说明 |
|---|---:|---|
| action candidate set | 否 | 由 continuous action field 从 allowed action space 生成 |
| candidate plausibility score | 是 | LLM 根据状态、记忆、关系、事件给每个候选动作打分 |
| primary action type | 共同决定 | field prior + LLM score 融合后采样 |
| action target | 主要由 field 决定 | selected candidate 固定，normalizer 只做合法性修正 |
| action intensity | 主要由 field 决定 | tendency -> intensity curve 生成 |
| public_position | 是 | LLM 生成公开立场 |
| private_intent | 是 | LLM 生成私下目标/策略描述 |
| memory interpretation | 是 | LLM 生成第一人称主观记忆文本 |
| 数值状态更新 | 否 | 情绪、信念、关系、ledger 由动力学函数更新 |
| 事件概率 | 否 | 由 state-driven event field 计算 |

准确表述是：

```text
LLM is a candidate plausibility scorer plus interpretation/stance renderer, not an unconstrained behavioral policy.
```

---

## 3. 为什么不让 LLM 完全自由选 action？

LLM 直接选 action 的优点是 open-ended，但风险很大：

| 风险 | 影响 |
|---|---|
| 不可复现 | 同一状态不同调用可能给出完全不同动作 |
| 容易剧情化 | LLM 倾向写戏剧性冲突，而不是连续社会动力学 |
| 难做反事实 | 无法判断结果来自状态变量还是 prompt 风格 |
| 难做消融 | 无法单独移除某个 motive weight 或 field prior |
| 难审计 | 行为来源不容易反编译 |

所以当前方案选择中间路线：

```text
可复现 continuous policy prior + LLM subjective plausibility scoring + LLM narration
```

这让系统可以回答：

```text
如果删除 PI 承诺记忆，LLM score 和 fused probability 怎么变？
如果把 confront 对 resentment_drive 的权重置零，署名抗议还会出现吗？
如果关闭 LLM scoring，只用 field prior，行为轨迹差多少？
```

---

## 4. Action Field 不是阈值法，但仍是参数化 baseline

`action_selection.py` 已经不是：

```python
if resentment > 0.7:
    action = "confront"
```

而是：

```text
field_tendency = Σ motive_i * weight_i + event_affinity + noise + repetition_effect
llm_score = LLM(candidate, state, memory, relationship, event)
fused_tendency = field_tendency + mix*(llm_score - 0.5)
probability = softmax(fused_tendency / temperature)
selected_action = sample(probability)
```

但 motive weights 仍然是结构先验，不应该被宣传成“完全无启发式”。更准确地说：

```text
calibratable social-psychological field prior
+ LLM candidate plausibility scoring
+ continuous sampling
+ LLM interpretation layer
```

---

## 5. Motive weights 是 baseline/prior，不是最终真理

当前 motive weights 外置到：

```text
config/action_field.yaml
```

其中包括：

```yaml
action_field:
  baseline_tendency: 0.10
  repetition_penalty: 0.045
  temperature_base: 0.20
  temperature_arousal_scale: 0.22

motive_weights:
  confront:
    resentment_drive: 0.35
    authorship_anxiety: 0.25
    authority_pressure: 0.10
    caution: -0.35
```

每个 action log 会保留：

```json
{
  "action_candidates": [...],
  "selected_action": {
    "type": "confront",
    "field_probability": 0.184,
    "llm_score": 0.720,
    "fused_tendency": 0.558,
    "scoring_source": "field_llm_fused"
  },
  "llm_action_scoring": {
    "enabled": true,
    "source": "field_llm_fused",
    "mix": 0.35
  }
}
```

这样可以区分：

| 来源 | 含义 |
|---|---|
| `field_probability` | 手写结构场给出的 baseline 概率 |
| `llm_score` | LLM 认为该候选动作对当前 agent 是否心理合理 |
| `fused_tendency` | 融合后的采样倾向 |
| `scoring_source` | 当前是 field-only 还是 field+LLM |

---

## 6. 如何做消融？

可以分别消融两层：

| 消融 | 做法 | 看什么 |
|---|---|---|
| field weight ablation | override `config/action_field.yaml` 中某个 motive weight | 冲突是否由手写 prior 主导 |
| LLM scoring ablation | `SimConfig(enable_llm_action_scoring=False)` | LLM 主观评分是否改变轨迹 |
| mix sensitivity | 调 `llm_action_score_mix` | 系统对 LLM 评分权重是否敏感 |
| memory delete | 删除 promise/authorship memory | 长程记忆是否中介 R52 escalation |

---

## 7. 如何描述当前系统才准确？

不准确：

```text
LLM agents freely decide actions based on long-term memory.
```

更准确：

```text
LabWars uses continuous latent action fields as a structural behavioral prior, asks LLMs to score candidate action plausibility under subjective memory and relationship context, then samples actions from the fused field. LLMs also generate memory interpretations and public/private stance under the selected-action constraint.
```

中文：

```text
LabWars 使用连续 latent action field 作为结构性行为先验，让 LLM 在主观记忆和关系语境下对候选动作评分，再从融合后的行为场中采样真实行动。LLM 同时在 selected action 约束下生成记忆解释、公开立场和私下意图。
```

---

## 8. 现在比旧版强在哪里？

旧版：

```text
field samples action -> LLM explains
```

新版：

```text
field proposes candidates -> LLM scores -> fused sampler selects -> LLM explains
```

关键提升是：LLM 不再只是“包装动作”，它会影响候选动作概率；但它仍不能绕过 action space、状态审计和因果可复现性。

---

## 9. 最终边界

LabWars 当前最强的地方不是“LLM 自己演出了内斗”，而是：

```text
它把科研内斗拆成可观测、可评分、可干预、可消融、可反事实比较的社会动力学变量。
```

这比单纯让 LLM 写剧情更适合做研究平台，也比纯手写权重更接近主观社会行为仿真。
