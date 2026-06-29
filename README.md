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
| [docs/04-实验方案与反编译报告.md](docs/04-实验方案与反编译报告.md) | 实验 A–D、指标、报告模板、验收 |

## 实施顺序

```
Part 1 → Part 2 → Part 3 → Part 4
```

## 项目结构

```
LabWars/
├── 00-总文档.md
├── README.md
├── docs/           # 4 个子文档
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
- [x] Part 1–4 全部实现（**54 tests passed**）
- [x] **Agent 决策与 memory interpretation 经 LLM**（OpenAI / Anthropic / Ollama）
- [x] LLM 不直接选择 primary action；连续 latent action field 采样动作，LLM 负责 public/private 解释

## LLM 配置（必配）

仿真通过连续 latent action field 采样 primary action；大模型负责记忆解释与 public/private 话术生成。配置见 [`config/llm.yaml`](config/llm.yaml)：

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

LabWars 当前不是让 LLM 直接选择真实行动。真实 primary action 由连续 latent action field 生成候选并采样；LLM 负责在 sampled action 约束下生成公开立场、私下意图和解释文本。

可校准参数见 [`config/action_field.yaml`](config/action_field.yaml)。如需测试某组 motive weights 是否主导结果，可使用 `src.experiments.action_field_ablation.run_action_field_ablation` 做多 seed 消融。
