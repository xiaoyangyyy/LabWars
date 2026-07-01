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

## LLM 配置（候选评分 + 解释层）

仿真先由 continuous latent action field 生成候选动作，再由 LLM 对候选动作进行 subjective plausibility scoring；系统融合 field_score 与 llm_score 后采样 primary action。LLM 同时负责记忆解释、公开立场、私下意图和话术生成。配置见 [`config/llm.yaml`](config/llm.yaml)：

```yaml
# 智算 OpenAI 兼容（已验证 ai.azya.top）
provider: openai
base_url: https://ai.azya.top/v1
model: qwen3.5
temperature: 0.7
request_delay_sec: 1.1   # 避免 429 限速（约 1 次/秒）
api_key_env: OPENAI_API_KEY
```

```powershell
# Windows
$env:OPENAI_API_KEY="sk-xxx"
python scripts/test_zhisuan_api.py          # 连通性自检
python -m src.experiments run -e A -c A2 --seed 42
```

其他 provider：`anthropic` / 本地 `ollama` — 见 `config/llm.yaml` 注释。

代码注入（自定义 backend）：

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


## Action Field 与 LLM 分工

LabWars 当前不是让 LLM 自由生成真实行动。真实流程是：continuous latent action field 生成候选动作和结构先验；LLM 对每个候选动作评分；系统融合 `field_score` 与 `llm_score` 后采样 primary action；LLM 再在 selected action 约束下生成公开立场、私下意图和解释文本。

可校准参数见 [`config/action_field.yaml`](config/action_field.yaml)。如需测试某组 motive weights 是否主导结果，可使用 `src.experiments.action_field_ablation.run_action_field_ablation` 做多 seed 消融；如需测试 LLM candidate scoring 是否改变长程轨迹，可使用 `src.experiments.llm_mix_ablation.run_llm_mix_ablation` 扫描 `llm_action_score_mix = 0.0 / 0.2 / 0.35 / 0.6 / 1.0`。报告第 11 节会展示 field top3、LLM top3、fused top3、selected action 与 LLM Override Pressure。
