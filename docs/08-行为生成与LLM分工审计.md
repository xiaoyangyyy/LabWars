# 行为生成与 LLM 分工审计

本文专门澄清 LabWars 当前的行为生成机制，避免把系统误读成“LLM 根据记忆自主选择 action”。

结论先说清楚：

> LabWars 当前不是 LLM 自主行动系统，而是 **continuous latent action field 采样真实 primary action，LLM 在 sampled action 约束下生成 public/private stance、memory interpretation 和 wording**。

这是一种可复现、可消融、可审计的社会动力学仿真架构。

---

## 1. 当前真实行为链路

实现位置：

| 环节 | 文件 |
|---|---|
| 候选动作生成 | `src/engine/action_selection.py` |
| sampled action 固定 | `src/engine/role_policy.py` |
| LLM prompt 约束 | `src/engine/prompts.py` |
| LLM drift 审计 | `src/engine/critic.py` |
| action-field 参数 | `config/action_field.yaml` |
| action-field 消融 | `src/experiments/action_field_ablation.py` |

真实流程是：

```text
agent latent state + event + recall + relationship + project state
  -> generate_action_candidates()
  -> sample_action_candidate()
  -> selected primary_action
  -> LLM receives sampled_action and action_candidates
  -> LLM generates public_position / private_intent / wording
  -> RolePolicyAgent overwrites raw.primary_action with selected action
  -> Critic checks wording/action drift
```

代码上最关键的是：

```python
raw["primary_action"] = {
    "type": selected.type,
    "target": selected.target,
    "intensity": selected.intensity,
}
```

也就是说，LLM 不能把 `confront` 改成 `comply`，也不能把 `document_contribution` 改成 `ask_for_authorship`。

---

## 2. LLM 做什么，不做什么

| 层 | 是否由 LLM 决定 | 说明 |
|---|---:|---|
| primary action type | 否 | 由 continuous action field 采样 |
| action target | 否 | 由 sampled candidate 固定 |
| action intensity | 否 | 由 tendency -> intensity curve 生成 |
| public_position | 是 | LLM 生成公开立场 |
| private_intent | 是 | LLM 生成私下目标/策略描述 |
| memory interpretation | 是 | LLM 生成第一人称主观记忆文本 |
| 数值状态更新 | 否 | 情绪、信念、关系、ledger 由动力学函数更新 |
| 事件概率 | 否 | 由 state-driven event field 计算 |

所以准确表述是：

```text
LLM is an interpretation and stance renderer, not the primary behavioral policy.
```

---

## 3. 为什么不让 LLM 直接选 action？

LLM 直接选 action 的优点是 open-ended，但风险很大：

| 风险 | 影响 |
|---|---|
| 不可复现 | 同一状态不同调用可能给出完全不同动作 |
| 容易剧情化 | LLM 倾向写戏剧性冲突，而不是连续社会动力学 |
| 难做反事实 | 无法判断结果来自状态变量还是 prompt 风格 |
| 难做消融 | 无法单独移除某个 motive weight |
| 难审计 | 行为来源不容易反编译 |

LabWars 的目标不是生成好看的故事，而是反编译机制。因此当前设计优先选择：

```text
可复现 continuous policy + LLM subjective narration
```

这让系统可以回答：

```text
如果删除 PI 承诺记忆，action probability 怎么变？
如果把 confront 对 resentment_drive 的权重置零，署名抗议还会出现吗？
如果关闭 state events，只 replay anchors，结果差多少？
```

---

## 4. Action Field 不是阈值法，但仍是参数化模型

`action_selection.py` 已经不是：

```python
if resentment > 0.7:
    action = "confront"
```

而是：

```text
action_tendency = Σ motive_i * weight_i
probability = softmax(tendency / temperature)
selected_action = sample(probability)
```

但这仍然不是“LLM 自发社会行为”。更准确地说：

```text
hand-designed social-psychological dynamical policy
+ continuous sampling
+ calibratable parameters
+ LLM interpretation layer
```

这句话应该成为项目对外介绍的边界。

---

## 5. Motive weights 已经外置为可校准参数

当前不再把 motive weights 只藏在代码里，而是外置到：

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

event_affinity:
  authorship_ambiguity:
    privately_lobby_pi: 0.16
    ask_for_authorship: 0.13
    document_contribution: 0.10
    confront: 0.06
```

每个 `ActionCandidate` 会记录参数来源：

```json
{
  "type": "confront",
  "tendency": 0.481,
  "probability": 0.184,
  "parameter_source": "D:/Labwar/config/action_field.yaml"
}
```

如果使用 runtime override，会记录：

```json
"parameter_source": "runtime_override"
```

---

## 6. 如何做 action-field 消融？

实现位置：`src/experiments/action_field_ablation.py`

例子：测试“confront 是否过度依赖 resentment_drive”：

```python
from src.engine.simulation import SimConfig
from src.experiments.action_field_ablation import run_action_field_ablation

result = run_action_field_ablation(
    SimConfig(max_rounds=60, interventions=[]),
    {
        "motive_weights": {
            "confront": {"resentment_drive": 0.0}
        }
    },
    seeds=list(range(10)),
    outcome="authorship_escalation_score",
)
```

输出会比较：

| 字段 | 含义 |
|---|---|
| `Y_control` | baseline action-field 下的结果 |
| `Y_ablation` | override 权重后的结果 |
| `delta` | ablation - control |
| `control_mean` | 多 seed baseline 均值 |
| `ablation_mean` | 多 seed ablation 均值 |

这可以避免把“手写权重导致的冲突”误解成“agent 自然表现出的社会行为”。

---

## 7. 如何描述当前系统才准确？

不准确：

```text
LLM agents decide actions based on long-term memory.
```

更准确：

```text
LabWars uses continuous latent action fields to sample behavior from agent state, memory, relationship, project pressure, and institutional power. LLMs generate subjective memory interpretations and public/private stance under the sampled-action constraint.
```

中文：

```text
LabWars 用连续 latent action field 从人格、信念、情绪、记忆、关系、项目压力和制度性权力中采样真实行动；LLM 在 sampled action 约束下生成主观记忆解释、公开立场和私下意图。
```

---

## 8. 下一步如果要更像“大模型行为”怎么办？

可以做双轨 policy，对比两种机制：

| policy | 描述 | 用途 |
|---|---|---|
| `latent_field_policy` | 当前默认，可复现、可消融 | 主仿真和因果实验 |
| `llm_free_policy` | LLM 直接从状态/记忆中选择 action | 对照实验，观察 LLM 自发行为 |
| `hybrid_policy` | LLM 给 action tendency modifier，不直接决定 action | 在可控性和开放性之间折中 |

推荐下一步不是直接替换当前机制，而是新增对照实验：

```text
same seed / same world / same memory
  latent_field_policy vs llm_free_policy vs hybrid_policy
```

这样才能判断：

```text
哪些冲突来自连续社会动力学？
哪些冲突来自 LLM 的叙事偏好？
哪些冲突对 motive weights 敏感？
```

---

## 9. 最终边界

LabWars 当前最强的地方不是“LLM 自己演出了内斗”，而是：

```text
它把科研内斗拆成可观测、可干预、可消融的社会动力学变量。
```

这比单纯让 LLM 写剧情更适合做研究平台。
