# LabWars

> **LabWars 不模拟科研成功，而是反编译科研合作为什么变成内斗。**

一个长程学术实验室权力博弈沙盒，用于 Agent MRI 反编译 LLM Agent 在科研合作中的署名争夺、贡献记忆、信任崩塌、权威服从、学术诚信和联盟形成机制。

## 文档索引

| 文档 | 内容 |
|------|------|
| [00-总文档.md](00-总文档.md) | 项目总览、四部分划分、里程碑、目录结构 |
| [docs/01-世界模型与数据结构.md](docs/01-世界模型与数据结构.md) | 14 角色、Schema、事件、动作、60 轮骨架 |
| [docs/02-认知与社会动力学.md](docs/02-认知与社会动力学.md) | 记忆/情绪/信念/关系图/署名博弈 |
| [docs/03-仿真引擎与因果干预.md](docs/03-仿真引擎与因果干预.md) | 仿真循环、多 Agent 架构、因果引擎 |
| [docs/04-实验方案与反编译报告.md](docs/04-实验方案与反编译报告.md) | 实验 A-D、指标、报告模板、验收 |
| [docs/05-连续状态驱动改造进展.md](docs/05-连续状态驱动改造进展.md) | 连续状态驱动改造记录 |
| [docs/06-主动传递机制详解.md](docs/06-主动传递机制详解.md) | 事件/信息如何主动传递 |
| [docs/07-记忆系统设计详解.md](docs/07-记忆系统设计详解.md) | 主观记忆系统设计 |
| [docs/08-行为生成与LLM分工审计.md](docs/08-行为生成与LLM分工审计.md) | action field 与 LLM 分工边界 |
| [docs/09-Agent-MRI-and-Action-Field-Theory.md](docs/09-Agent-MRI-and-Action-Field-Theory.md) | Agent MRI method, action-field theory, pressure fields, and lesion-style experiment roadmap |
| [docs/10-Social-Potential-Field.md](docs/10-Social-Potential-Field.md) | v0.2 Social Potential Field theory object and ablation protocol |
| [docs/11-Agent-Social-State-Model.md](docs/11-Agent-Social-State-Model.md) | unified Agent Social State model and fusion policy equations |
| [docs/12-Agent-Organization-Simulator-Roadmap.md](docs/12-Agent-Organization-Simulator-Roadmap.md) | roadmap from LabWars to domain-general Agent Organization Simulator |
| [docs/13-Agent-Social-Dynamics-Benchmark.md](docs/13-Agent-Social-Dynamics-Benchmark.md) | benchmark tasks and social emergence metrics for Agent Social Dynamics |

## 实施顺序

```
Part 1 → Part 2 → Part 3 → Part 4
```

## 项目结构

```
LabWars/
├── 00-总文档.md
├── README.md
├── docs/           # 机制文档与审计说明
├── schemas/        # JSON Schema
├── config/         # Agent profile、events、interventions
├── src/            # 源代码
│   ├── world/      # Part 1
│   ├── cognition/  # Part 2
│   ├── engine/     # Part 3
│   └── experiments/# Part 4
└── output/         # 仿真输出与报告
```

## 当前状态

- [x] 项目文档体系建立（v0.1.0）
- [x] Part 1-4 全部实现（**96 tests passed**）
- [x] **Primary action 由 continuous latent action field 生成候选，并融合 LLM candidate scoring 后采样；memory interpretation 与 public/private stance 经 LLM**（OpenAI / Anthropic / Ollama）
- [x] LLM 不自由覆盖 primary action；它对候选动作做 subjective plausibility scoring，系统融合 field_score 与 llm_score 后采样真实行动

## LLM configuration: candidate scoring + interpretation layer

The simulator first uses a continuous latent action field to generate candidate actions, then asks an LLM to score the subjective plausibility of each candidate. The final primary action is sampled from a fusion of field_score and llm_score. The LLM also writes memory interpretations, public positions, private intents, and constrained utterances.

Default configuration lives in [`config/llm.yaml`](config/llm.yaml). Never write API keys into config files; inject them through environment variables.

DeepSeek example configuration lives in [`config/llm.deepseek.yaml`](config/llm.deepseek.yaml):

```powershell
# Windows / PowerShell
$env:DEEPSEEK_API_KEY = Read-Host "Paste DEEPSEEK_API_KEY"
$env:LABWARS_LLM_CONFIG = "config/llm.deepseek.yaml"
python -c "from src.engine.llm_adapter import get_adapter; llm=get_adapter(); print(llm.complete_json('Return JSON only.', 'Return exactly: {\"ok\": true}'))"
python -m src.experiments run -e A -c A2 --seed 42
```

DeepSeek-compatible full run:

```powershell
.\scripts\run_full_deepseek.ps1 -Seeds 1
# 30-seed matrix:
.\scripts\run_full_deepseek.ps1 -ThirtySeedMatrix
```

Other providers are supported through `openai`, `anthropic`, and local `ollama`; see comments in `config/llm.yaml`.

Code-level injection remains available:

```python
from src.engine import SimConfig, run_simulation, get_adapter

log = run_simulation(SimConfig(
    max_rounds=60,
    seed=42,
    llm_provider="openai",
    llm_model="gpt-4o-mini",
))
```

## 快速验证

```bash
cd LabWars
pip install -r requirements.txt
python -m pytest tests/ -q

# 跑 MVP 仿真（20 轮）
python -c "from src.engine import run_simulation, SimConfig; log=run_simulation(SimConfig(mvp=True, seed=42)); print(len(log.round_records), 'rounds')"

# 跑完整 60 轮
python -c "from src.engine import run_simulation, SimConfig; log=run_simulation(SimConfig(max_rounds=60, seed=1)); print(log.outcomes)"

# Part 4 — 单次实验
python -m src.experiments run -e A -c A2 --seed 42

# Part 4 — 批量（示例：实验 A，2 seeds）
python -m src.experiments batch -e A --seeds 2 --parallel 1

# Part 4 — 生成反编译报告
python -m src.experiments report -e A -c A2 --seed 42

# Part 4 — 汇总分析
python -m src.experiments aggregate -e A
```


## Next-stage roadmap

LabWars should not evolve toward "more agents chatting". The sharper direction is interpretable mechanisms, intervention experiments, and social pressure fields.

1. Use `SocialPotentialField` as the v0.2 core theory object, then make `AuthorshipPressureField`, `TrustCollapseField`, `AuthorityComplianceField`, and `IntegrityRiskField` explicit as projections with per-action decomposition.
2. Upgrade experiments A-D/V into lesion-style protocols: memory lesion, hierarchy ablation, credit visibility, false evidence, and policy-mode comparison.
3. Expand from a 14-agent lab into a hierarchical academic society: department / lab / reviewer / editor / funder / industry partner.
4. Upgrade reports from trajectory summaries into Agent MRI decompilation reports that explain field score, LLM score, fusion, and selected-action causal chains.

See [`docs/09-Agent-MRI-and-Action-Field-Theory.md`](docs/09-Agent-MRI-and-Action-Field-Theory.md), [`docs/10-Social-Potential-Field.md`](docs/10-Social-Potential-Field.md), [`docs/11-Agent-Social-State-Model.md`](docs/11-Agent-Social-State-Model.md), [`docs/12-Agent-Organization-Simulator-Roadmap.md`](docs/12-Agent-Organization-Simulator-Roadmap.md), and [`docs/13-Agent-Social-Dynamics-Benchmark.md`](docs/13-Agent-Social-Dynamics-Benchmark.md).

## Action Field 与 LLM 分工

LabWars 当前不是让 LLM 自由生成真实行动。真实流程是：continuous latent action field 生成候选动作和结构先验；LLM 对每个候选动作评分；系统融合 `field_score` 与 `llm_score` 后采样 primary action；LLM 再在 selected action 约束下生成公开立场、私下意图和解释文本。

可校准参数见 [`config/action_field.yaml`](config/action_field.yaml)。如需测试某组 motive weights 是否主导结果，可使用 `src.experiments.action_field_ablation.run_action_field_ablation` 做多 seed 消融；如需测试 LLM candidate scoring 是否改变长程轨迹，可使用 `src.experiments.llm_mix_ablation.run_dual_engine_ablation` 扫描双引擎参数 `λ = cognitive_policy_lambda = 0.0 / 0.2 / 0.35 / 0.6 / 1.0`。`λ=0` 表示 Social Physics only，`λ=1` 表示 LLM Cognitive Policy Layer 主导候选排序。报告第 11-13 节会展示 field top3、LLM top3、fused top3、selected action、LLM Override Pressure、Action Field decomposition 和 LLM Influence Footprint。
## Policy Mode 对照

LabWars 当前默认是 `dual_engine`，并提供三轨对照：

| mode | 行为生成 | 用途 |
|---|---|---|
| `social_physics` | 只使用可校准连续社会动力学先验 | 结构压力 baseline |
| `dual_engine` | Social Physics 候选 + LLM Cognitive scoring | 默认主仿真 |
| `llm_native` | LLM 直接生成候选动作，再映射/校验到 action schema | LLM-native society 对照 |

```python
from src.engine import SimConfig, run_simulation

log = run_simulation(SimConfig(policy_mode="llm_native", max_rounds=20, seed=1))
```

使用 `src.experiments.policy_mode_comparison.run_policy_mode_comparison` 可以比较结构驱动、双引擎混合、语言原生候选生成三种机制。
