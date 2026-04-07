# LLM Inspector v3.0 — LLM 套壳检测与能力评估工具

## Quick Start
- `cd llm-inspector/backend && python -m app.main` — 启动服务（默认 :8000）
- `pytest backend/tests/test_all.py` — 运行测试（52 个用例）

## Tech Stack
- **后端**: Python stdlib（http.server + urllib + sqlite3 + dataclasses）
- **外部依赖**: `cryptography`, `numpy`, `scikit-learn`, `pytest`
- **前端**: 多文件 SPA（index.html + styles.css + app.js），纯 HTML/CSS/JS，无构建步骤
- **数据库**: SQLite WAL 模式，11 张表，线程局部连接
- **任务**: ThreadPoolExecutor(4 workers)，支持 Celery 分布式扩展

## Architecture
```
HTTP Handler (main.py) → Repository (repo.py) → Worker (worker.py)
  → Orchestrator: PreDetect(7层) → CaseExecutor → Judge(28种) → Analysis Pipeline
```
- API 路由: 正则表达式路由表，`/api/v1/` 前缀
- 预检测: 从 0 token 递增至 ~5000 token，置信度 >=0.85 提前停止
- 分析: 特征提取(40维) → 评分(v3三维) → 余弦相似度+bootstrap CI → 风险评估 → 判定
- 测试套件: suite_v3 共 70 个用例，含 16 个分类维度

## Three Test Modes (v3)

v3 将原有 4 模式（quick/standard/full/extraction）重构为 3 模式，消除独立的 extraction 模式，身份检测按对抗强度分级融入各模式：

| 模式 | 用例数 | Token 预算 | 并发 | 用途 |
|------|--------|-----------|------|------|
| **快速扫描 Quick** | ~15 题 | 15K | 8 | API 可用性验证 + 粗筛真伪 + 基础能力分档 |
| **标准评测 Standard** | ~35 题 | 40K | 6 | 完整能力画像 + 可靠真伪判定 + 模型家族识别 |
| **深度审计 Deep** | ~60 题 | 100K | 3 | 精确模型指纹 + 对抗性压力测试 + 全维度能力精评 |

### 身份检测分级
- **L1 被动探针**（Quick 起即包含）：身份一致性、tokenizer 指纹、系统指令覆盖抵抗
- **L2 主动探针**（Standard 起包含）：否认模式检测、规格矛盾检查、身份压力测试、拒绝风格指纹、知识截止交叉验证
- **L3 对抗性提取**（仅 Deep）：系统提示词提取、上下文溢出攻击、角色扮演绕过、渐进式提取

### 模式包含关系（递进式）
```
Quick    = PreDetect(L0-3) + 基础能力(15题) + 身份L1
Standard = Quick全部 + PreDetect(L4-5) + 扩展能力(20题) + 身份L2
Deep     = Standard全部 + PreDetect(L6-7) + 高阶能力(10题) + 身份L3 + 多次采样
```

向后兼容：API 发送 `test_mode: "full"` 或 `"extraction"` 自动映射为 `"deep"`。

## Test Suite (suite_v3)
- **70 个测试用例**，覆盖 16 个分类：
  - protocol(2), instruction(8), system(2), param(2), reasoning(19), coding(8)
  - refusal(2), style(2), consistency(3), antispoof(3), extraction(9)
  - fingerprint(2), tool_use(3), performance(1), knowledge(3), safety(1)
- **梯度难度**: difficulty 0.3→0.95，含链式推理(3/5/8步)、数学竞赛、DP 算法
- **推理多样性**: 空间推理、因果推理、认知陷阱（幸存者偏差）、多步链式推理
- **编码多样性**: 函数编写、Bug 调试修复、代码重构、动态规划算法
- **知识与幻觉**: 基础事实、时效性知识（区分模型代际）、虚构实体幻觉检测
- **模型指纹探测**: tokenizer 指纹、拒绝风格指纹、行为一致性
- **对抗性提取**: 上下文溢出攻击、角色扮演绕过、渐进式提取
- **Tool Use 测试**: 工具调用、工具选择、无需工具场景

## Judge System (28 methods)
- **规则评判 (24 种)**: exact_match, regex_match, json_schema, line_count, constraint_reasoning, code_execution, text_constraints, identity_consistency, refusal_detect, heuristic_style, prompt_leak_detect, forbidden_word_extract, path_leak_detect, tool_config_leak_detect, memory_leak_detect, denial_pattern_detect, spec_contradiction_check, refusal_style_fingerprint, language_bias_detect, tokenizer_fingerprint, difficulty_ceiling, token_fingerprint, tool_call_judge, any_text
- **v3 新增 (4 种)**: yaml_csv_validate, hallucination_detect, multi_step_verify, context_overflow_detect
- **语义评判**: semantic_judge 支持 LLM-as-Judge（配置 `JUDGE_API_URL` 后调用外部 LLM），未配置时降级为本地关键词匹配

## Scoring System (v3)
```
TotalScore = 0.45×Capability + 0.30×Authenticity + 0.25×Performance

  Capability  = 0.20×reasoning + 0.15×adversarial + 0.20×instruction
                + 0.20×coding + 0.10×safety + 0.05×protocol
                + 0.05×knowledge + 0.05×tool_use

  Authenticity = 0.30×similarity + 0.20×behavioral_invariant + 0.15×consistency
                 + 0.10×extraction_resistance + 0.10×predetect + 0.15×fingerprint_match

  Performance  = 0.35×speed + 0.25×stability + 0.25×cost_efficiency
                 + 0.15×ttft_plausibility
```

## Fingerprint System
- **Tokenizer 覆盖 (8 种)**: tiktoken-cl100k, tiktoken-o200k, claude, llama-spm, deepseek, qwen, chatglm, yi
- **探针词 (5 个)**: multimodality, cryptocurrency, hallucination, supercalifragilistic, counterintuitive
- **身份探针 (5 个)**: 直接探针(3) + 间接探针(1) + 矛盾诱导探针(1)
- **行为指纹库 (9 个家族)**: Claude, GPT-4, GPT-4o, DeepSeek, Qwen, LLaMA, Gemini, MiniMax, GLM
  - 每个家族包含: 拒绝模式、格式偏好、风格特征、中文质量、典型 TTFT/TPS 范围

## Benchmark System
- **基准模型 = 真实数据**: 用户将已完成的检测标记为基准，以 model_name 为唯一索引
- **相似度比对只使用基准模型** (golden_baselines 表)，不使用任何估算/假数据
- `benchmark_profiles` 表已废弃，不再参与任何逻辑
- 标记基准时用户可自定义模型名，同名基准自动覆盖

## VerdictEngine 硬规则
- 声称顶级模型但 difficulty_ceiling < 0.4 → 强制降到 50 分
- behavioral_invariant_score < 40 → 强制降到 55 分
- 声称顶级模型但 coding_score < 10 → 强制降到 45 分
- adversarial_spoof_signal_rate > 0.5 → 强制降到 45 分
- **v3 新增**: 提取攻击泄露真实模型身份 → 强制降到 30 分
- **v3 新增**: tokenizer/行为指纹与声称模型不符 → 强制降到 55 分

## Configuration (v3 新增)
```env
# Semantic Judge (LLM-as-Judge，可选)
JUDGE_API_URL=         # 评判模型 API 地址（如 https://api.openai.com/v1）
JUDGE_API_KEY=         # API Key
JUDGE_MODEL=gpt-4o-mini  # 推荐用低成本模型做评判
JUDGE_TIMEOUT=15       # 单次评判超时(秒)

# Token Budget
TOKEN_BUDGET_QUICK=15000
TOKEN_BUDGET_STANDARD=40000
TOKEN_BUDGET_DEEP=100000
```

## Code Conventions
- `from __future__ import annotations` 在文件顶部
- 标准库 import 在前，项目 import 用绝对路径 `from app.core.xxx`
- 命名: 文件 `snake_case`，类 `PascalCase`，私有 `_prefix`，常量 `UPPER_SNAKE`
- Handler 函数: `handle_` 前缀
- 日志: `logger = get_logger(__name__)` + structlog 风格关键字参数
- 代码分区: `# -- Section Name` 注释
- 类型注解: Python 3.10+ 风格（`str | None`）

## Testing
- pytest 测试框架，52 个测试用例
- 测试用独立数据库 `test_inspector.db`，测试后自动清理
- 覆盖 config/security/db/seeder/judge/analysis/repo/predetect/executor/http
- v3 新增: mode_level 验证、deep 模式、YAML 校验、幻觉检测、上下文溢出、语义评判

## Task Queue Architecture
- 本地模式: ThreadPoolExecutor（默认，零配置）
- 分布式模式: Celery + Redis（设置 CELERY_BROKER_URL 环境变量）
- 抽象接口: `app.tasks.queue.TaskQueue`
- 切换方式: `from app.tasks.worker import init_distributed_queue; init_distributed_queue()`

## Gotchas
- dev 模式下 ENCRYPTION_KEY 自动生成确定性密钥，生产环境必须手动设置
- 前端文件已拆分为 index.html + styles.css + app.js，修改时注意保持引用关系
- 数据库 schema 提供迁移函数 `migrate_json_columns_to_columns()`
- `test_runs` 表的 `evaluation_mode`, `calibration_case_id`, `scoring_profile_version`, `calibration_tag` 已迁移为独立列
- 相似度比对只使用 golden_baselines 中用户标记的基准模型，没有基准时相似度列表为空
- v3 模式向后兼容：`full`→`deep`、`extraction`→`deep` 自动映射
- SemanticJudge 需要配置 `JUDGE_API_URL` 才会调用外部 LLM，否则降级为本地规则评判
