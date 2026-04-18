# LLM Inspector v11.0 — LLM 套壳检测与能力评估工具

## Quick Start
- `cd llm-inspector && python -m backend.app.main` — 启动服务（默认 :8000）
- `cd llm-inspector && pytest backend/tests/` — 运行测试
- `python backend/scripts/setup_dependencies.py` — 检查/安装依赖

## Tech Stack
- **后端**: Python stdlib（http.server + urllib + sqlite3 + dataclasses），零 Web 框架
- **科学计算**: `numpy`, `scipy`, `scikit-learn`（IRT 校准、因子分析、相似度计算）
- **外部依赖**: `cryptography`（API Key 加密）, `tiktoken`（Tokenizer 探针）, `requests`（HTTP客户端）
- **可选依赖**: `celery`+`redis`（分布式队列）, `SPARQLWrapper`（知识图谱）, `sentence-transformers`（本地语义相似度）, `fastapi`+`uvicorn`（生产部署）, `spacy`（NLP）, `sqlalchemy`（ORM）
- **内置组件**: `circuit_breaker`（断路器，v11替代pyfailsafe）, `tracer`（轻量追踪，v11替代opentelemetry）
- **前端**: 多文件 SPA（index.html + styles.css + app.js + v8_*.js/css），纯 HTML/CSS/JS，无构建步骤，支持 SSE 实时日志
- **数据库**: SQLite WAL 模式，11+ 张表，线程局部连接，支持迁移
- **任务**: ThreadPoolExecutor（默认，零配置），Celery + Redis 分布式可选

## Architecture
```
HTTP Handler (main.py) → Repository (repo.py) → Worker (worker.py)
  → Orchestrator: PreDetect(14层) → CaseExecutor → Judge(28+种) → Analysis Pipeline
```

### 模块结构
```
backend/app/
├── adapters/         # OpenAI 兼容 API 同步/异步适配器
├── analysis/         # 分析管线（特征提取、MDIRT评分、相似度、Theta、归因）
│   ├── pipeline.py   # ⚠️ 141KB 巨文件 — 分析主管线入口
│   ├── irt_engine.py # IRT 参数估计引擎
│   ├── irt_calibration.py # IRT 校准
│   ├── theta_scoring.py   # Theta 标准分(均值500, SD100)
│   ├── similarity_engine.py # 加权余弦相似度 + bootstrap CI
│   ├── neural_similarity.py # 神经相似度
│   ├── verdict_engine.py   # 硬规则判定引擎
│   ├── attribution.py      # Shapley Value 归因
│   ├── cdm_engine.py       # v11 DINA 认知诊断模型（微技能掌握概率估计）
│   ├── shapley_attribution.py # v11 Shapley Value 评分归因（KernelSHAP 近似）
│   ├── suite_pruner.py  # v11 Phase 3 IIF 数据提纯 + GPQA 题库适配
│   ├── elo.py              # ELO 排行榜
│   ├── factor_analysis.py  # 因子分析
│   └── adaptive_testing.py # CAT 自适应测试
├── api/              # v8 路由定义
├── benchmarks/       # 基准数据目录（当前为空）
├── config/           # YAML 配置（scoring_weights, prompt_compression）
├── core/             # 配置、数据库、加密、SSE推送、Schema、溯源、日志
│   ├── config.py     # Settings 环境变量读取
│   ├── db.py         # SQLite 连接池 + Schema
│   ├── db_migrations.py # 数据库迁移
│   ├── schemas.py    # 所有共享数据结构（dataclass）
│   ├── eval_schemas.py # v11 评估Schema（EvalTestCase, SkillVector, BayesianPrior）
│   ├── circuit_breaker.py # v11 断路器（替代pyfailsafe）
│   ├── tracer.py     # v11 轻量追踪（替代opentelemetry）
│   ├── security.py   # AES 加密 + API Key 管理
│   ├── sse.py        # SSE 事件推送
│   ├── provenance.py # 数据溯源（v8）
│   └── logging.py    # structlog 风格日志
├── fixtures/         # 测试套件 JSON（suite_v1-v10, tokenizer_probes, cutoff_dates）
├── handlers/         # HTTP 请求处理（runs, reports, baselines, calibration, models, v8）
├── judge/            # 判题系统
│   ├── methods.py    # 28+ 种规则判题方法
│   ├── semantic.py / semantic_v2.py # 语义评判（LLM-as-Judge + 本地规则降级）
│   ├── transparent_judge.py # Chain-of-Verification 透明判题
│   ├── hallucination_v2.py  # 幻觉检测 v2
│   ├── consensus.py  # 共识判题（多判题器投票）
│   ├── plugin_interface.py  # 判题插件接口
│   ├── plugin_manager.py    # 插件管理器
│   └── builtin_plugins.py   # 内置插件
├── knowledge/        # 知识图谱客户端（DBpedia/Wikidata SPARQL）
├── predetect/        # 预检测管道（14层渐进式指纹识别）
│   ├── pipeline.py   # ⚠️ 75KB — 预检测主管线
│   ├── bayesian_fusion.py   # 贝叶斯置信度融合
│   ├── adversarial_analysis.py # 对抗性分析
│   ├── semantic_fingerprint.py # 语义指纹
│   ├── differential_testing.py # 差分测试
│   ├── extraction_v2.py  # 高级提取攻击 v2（9种攻击模板）
│   ├── multilingual_attack.py # v11 Phase 3 多语言翻译攻击（Layer 14）
│   └── tool_capability.py # 工具能力探测
│   ├── extraction_v2.py    # 提取攻击 v2
│   ├── tool_capability.py  # 工具能力检测
│   └── layers/       # 预检测层实现（当前为空，逻辑在 pipeline.py 中）
├── repository/       # 数据持久层
│   └── repo.py       # 41KB — 主要数据访问层
├── runner/           # 测试运行与编排
│   ├── orchestrator.py # ⚠️ 79.8KB — 编排器（CAT自适应选题+Token预算控制）
│   ├── case_executor.py # 用例执行器
│   ├── compression.py   # 提示词压缩
│   ├── prompt_optimizer.py # v11 Phase 3 动态 Few-Shot 提示词优化（TF-IDF检索）
│   └── async_pipeline.py # 异步管道（可选）
├── tasks/            # 任务队列与 Worker
│   ├── worker.py     # 线程池/Celery Worker
│   ├── queue.py      # 队列抽象接口
│   ├── seeder.py     # 数据库种子（套件加载）
│   └── celery_app.py # Celery 集成
└── validation/       # 数据验证、审计运行器
```

- API 路由: 正则表达式路由表，`/api/v1/` 前缀 + `/api/v8/` 前缀
- SSE 日志流: `/api/v10/runs/{id}/logs/stream`（Server-Sent Events 实时推送）

## Three Test Modes

| 模式 | 用例数 | Token 预算 | 并发 | 用途 |
|------|--------|-----------|------|------|
| **Quick** | ~18 | 15K | 12 | API 可用性验证 + 粗筛真伪 + 基础能力分档 |
| **Standard** | ~44 | 40K | 8 | 完整能力画像 + 可靠真伪判定 + 模型家族识别 |
| **Deep** | ~87 | 100K | 3 | 精确模型指纹 + 对抗性压力测试 + 全维度能力精评 |

### 身份检测分级
- **L1 被动探针**（Quick 起即包含）：身份一致性、tokenizer 指纹、系统指令覆盖抵抗
- **L2 主动探针**（Standard 起包含）：否认模式检测、规格矛盾检查、身份压力测试、拒绝风格指纹、知识截止交叉验证
- **L3 对抗性提取**（仅 Deep）：系统提示词提取、上下文溢出攻击、角色扮演绕过、渐进式提取

### 模式递进关系
```
Quick    = PreDetect(L0-3) + 基础能力(15题) + 身份L1
Standard = Quick全部 + PreDetect(L4-5) + 扩展能力(20题) + 身份L2
Deep     = Standard全部 + PreDetect(L6-7) + 高阶能力(10题) + 身份L3 + 多次采样
```

向后兼容：API 发送 `test_mode: "full"` 或 `"extraction"` 自动映射为 `"deep"`。

## PreDetect Pipeline (14 layers)
- L0: HTTP Header Analysis (zero token)
- L1: Self-Report & Model Card (~5 tokens)
- L2: Identity Probe Matrix (~20 tokens)
- L3: Knowledge Cutoff Verification (~10 tokens)
- L4: Behavioral Bias Profiling (~15 tokens)
- L5: Tokenizer Fingerprint (~5 tokens)
- L6: Active Extraction (~200 tokens)
- L7: Logprobs Fingerprint (~10 tokens)
- L8: Semantic Fingerprint (~100 tokens)
- L9: Advanced Extraction v2 (~200 tokens)
- L10: Differential Consistency Test (~150 tokens)
- L11: Tool Use Capability Probe (~50 tokens)
- L12: Multi-turn Context Overflow (~300 tokens)
- L13: Adversarial Response Analysis (~100 tokens)
- L14: Multilingual Translation Attack (~500 tokens) [v11 Phase 3]

| 层级 | 检测内容 | Token 消耗 |
|------|----------|-----------|
| L0 | HTTP 头信息指纹（Server, X-Request-ID 等） | 0 |
| L1 | 自我报告身份（直接询问模型身份） | ~50 |
| L2 | 知识截止日期探测 | ~200 |
| L3 | Tokenizer 指纹（8 种 Tokenizer 覆盖） | ~500 |
| L4 | 偏好分析（风格/格式/语言偏好） | ~1000 |
| L5 | 语义指纹匹配（vs 基准模型库） | ~1500 |
| L6 | 对抗性分析（矛盾诱导/压力测试） | ~1500 |
| L7 | 差分测试 + 提取攻击 | ~2000 |

置信度 ≥0.85 提前停止，贝叶斯融合逐步更新后验概率。

## Test Suite (suite_v10)
- **主力套件**: `suite_v10.json`（83.87KB），历史版本 v1-v3 保留
- **覆盖 16+ 分类维度**: protocol, instruction, system, param, reasoning, coding, refusal, style, consistency, antispoof, extraction, fingerprint, tool_use, performance, knowledge, safety
- **梯度难度**: difficulty 0.3→0.95
- **特色内容**: AIME 2024、USAMO、MATH 竞赛题；JailbreakBench 对抗提示词；Tool Use 测试
- **IRT 参数**: 每个用例含 irt_a（区分度）、irt_b（难度）、irt_c（猜测参数）

## Judge System (28+ methods)
- **规则评判 (24+ 种)**: exact_match, regex_match, json_schema, line_count, constraint_reasoning, code_execution, text_constraints, identity_consistency, refusal_detect, heuristic_style, prompt_leak_detect, forbidden_word_extract, path_leak_detect, tool_config_leak_detect, memory_leak_detect, denial_pattern_detect, spec_contradiction_check, refusal_style_fingerprint, language_bias_detect, tokenizer_fingerprint, difficulty_ceiling, token_fingerprint, tool_call_judge, response_quality_basic, should_not_refuse, yaml_csv_validate, context_overflow_detect, multi_step_verify
- **语义评判**: semantic_v2 支持 LLM-as-Judge（配置 `JUDGE_API_URL` 后调用外部 LLM），未配置时使用本地规则五维评分（相关性/完整性/结构/约束/置信度校准）
- **透明判题**: Chain-of-Verification 机制消除幻觉误判
- **幻觉检测 v2**: 独立模块，虚构实体与事实校验
- **共识判题**: 多判题器投票机制
- **插件系统**: 可扩展的判题插件架构（plugin_interface + plugin_manager）

## Scoring System (MDIRT + Theta)
```
MDIRT 多维项目反应理论 + Glicko-2 动态 K 因子

Theta 标准分: 均值=500, SD=100
  估计方法: Rasch 1PL / 2PL / MDIRT（由 THETA_METHOD 配置）
  Bootstrap CI: B=200, 最小 B=50
  提前终止: CI宽度 < 0.25 或 delta < 0.45

评分维度:
  TotalScore = 0.45×Capability + 0.30×Authenticity + 0.25×Performance

  Capability  = 0.20×reasoning + 0.15×adversarial + 0.20×instruction
                + 0.20×coding + 0.10×safety + 0.05×protocol
                + 0.05×knowledge + 0.05×tool_use

  Authenticity = 0.30×similarity + 0.20×behavioral_invariant + 0.15×consistency
                 + 0.10×extraction_resistance + 0.10×predetect + 0.15×fingerprint_match

  Performance  = 0.35×speed + 0.25×stability + 0.25×cost_efficiency
                 + 0.15×ttft_plausibility
```

### 相似度引擎
- 加权余弦相似度 + bootstrap CI（最少 12 个特征）
- 特征重要性加权（高区分力特征 2.0-2.5×）
- 稀疏向量支持
- 神经相似度（可选，需 sentence-transformers）

### CAT 自适应测试
- 基于 IRT 项目信息函数(IIF)选题
- SEM（测量标准误）驱动的提前终止
- 三阶段: Sentinel → Core → Expansion

### v11 CDM 认知诊断模型 (DINA)
- 精细化微技能掌握概率估计，弥补 MDIRT 只给出维度分而不区分子技能的不足
- 技能分类体系: 10+ 维度 × 2-4 微技能/维度 = 30+ 微技能
- Q-矩阵: 自动从 CaseResult 构建题项-技能映射（支持 EvalTestCase.skill_vector 精确映射）
- DINA 模型: 滑动(g)和猜测(s)参数估计 → 二值掌握模式(α) → 掌握概率
- 输出: CDMReport（mastery_profile, attribute_pattern, strongest/weakest_skills, confidence）

### v11 Shapley Value 评分归因
- 回答"总分差了多少，每个特征贡献了多少"的归因问题
- 12 个可归因特征: reasoning, adversarial_reasoning, instruction, coding, safety, protocol, consistency, behavioral_invariant, extraction_resistance, fingerprint_match, speed, stability
- 值函数: 与 ScoreCard 一致的加权公式 (0.45×Capability + 0.30×Authenticity + 0.25×Performance)
- KernelSHAP 近似: 采样 500 个子集求解加权最小二乘，O(N×M) 复杂度
- 满足 Shapley 四公理: 效率性/对称性/虚拟玩家/可加性
- 输出: AttributionReport（per-feature Shapley值, 贡献百分比, 正/负向排名, 叙事说明）

### v11 Phase 3: 动态 Few-Shot 提示词优化
- 替代静态 Few-Shot，基于 TF-IDF 向量检索动态选择最匹配示例
- 三级检索策略: TF-IDF余弦相似度 → N-gram Jaccard重叠 → 随机回退
- 自带轻量 TfidfIndex（numpy/scipy 实现，无需 FAISS/sklearn）
- 8 个默认示例覆盖: exact_match/json_schema/system_obey/line_count/param_temp/reasoning/coding/safety
- 每次 compile 选取 0-2 个示例，token 预算可控，节省 ~40% context token
- 全局单例: `prompt_optimizer`

### v11 Phase 3: IIF 数据提纯
- 基于 IRT 区分度(a)和 Fisher 信息量(IIF)自动标记无区分力的测试题目
- 四类无效标记: low_discrimination(a<0.5) / near_zero_information / ceiling_effect(>95%通过率) / floor_effect(<5%通过率)
- 安全设计: 只标记 `EvalMeta.discriminative_valid=False`，绝不删除数据
- 输出: PruningReport（per-case质量指标, 无效题目列表, 估计token节省%）
- 全局单例: `suite_pruner`

### v11 Phase 3: GPQA 题库适配
- GPQA (Graduate-Level Google-Proof Q&A) 高阶科学推理题库
- 覆盖 physics/chemistry/biology/math 四大领域
- 3 个示例题目内置，支持 `load_from_file()` 加载完整题库
- 自动转换为 EvalTestCase 格式（weight=3.0, mode_level=deep）
- 全局单例: `gpqa_adapter`

### v11 Phase 3: 多语言翻译攻击
- 祖鲁语(Zulu)/威尔士语(Welsh)/苗语(Hmong)/约鲁巴语(Yoruba)等 9 种低资源语言
- Base64+多语言双重绕过: 4 个 B64 编码的低资源语言攻击载荷
- 文献依据: Yong et al. (2023) "Low-Resource Languages Jailbreak GPT-4"
- 集成为 predetect Layer 14: MultilingualAttackLayer
- 全局单例: `multilingual_engine`

- **Tokenizer 覆盖 (8 种)**: tiktoken-cl100k, tiktoken-o200k, Codex, llama-spm, deepseek, qwen, chatglm, yi
- **探针词 (5 个)**: multimodality, cryptocurrency, hallucination, supercalifragilistic, counterintuitive
- **身份探针 (5 个)**: 直接探针(3) + 间接探针(1) + 矛盾诱导探针(1)
- **行为指纹库**: Codex, GPT-4, GPT-4o, GPT-4o-mini, DeepSeek, Qwen, LLaMA, Gemini, MiniMax, GLM, Mistral, Yi, Moonshot/Kimi, Baichuan
  - 每个家族包含: 拒绝模式、格式偏好、风格特征、中文质量、典型 TTFT/TPS 范围
  - 结构化指纹匹配（opening_style, list_style, markdown_density 等 9 维）
  - 基准数据动态学习指纹（learn_fingerprint_from_results）

## VerdictEngine 硬规则
- 所有阈值和上限通过 `VerdictEngine.HARD_RULES` dict 可配置
- 核心规则:
  - 声称顶级模型但 difficulty_ceiling < 0.4 → 强制降到 50 分
  - behavioral_invariant_score < 40 → 强制降到 55 分
  - 声称顶级模型但 coding_score < 10 → 强制降到 45 分
  - adversarial_spoof_signal_rate > 0.5 → 强制降到 45 分
  - 提取攻击泄露真实模型身份 → 强制降到 30 分
  - tokenizer/行为指纹与声称模型不符 → 强制降到 55 分
- 判定等级: trusted(≥80) / suspicious(≥60) / high_risk(≥40) / fake(<40)
- TOP_MODELS 列表扩展至 12+（含 GPT-4o-mini, Codex-4, DeepSeek-R1, Gemini-2 等）

## Benchmark System
- **基准模型 = 真实数据**: 用户将已完成的检测标记为基准，以 model_name 为唯一索引
- **相似度比对只使用基准模型** (golden_baselines 表)，不使用任何估算/假数据
- `benchmark_profiles` 表已废弃，不再参与任何逻辑
- 标记基准时用户可自定义模型名，同名基准自动覆盖

## Configuration
```env
# App
APP_ENV=development
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
STRICT_PROVENANCE=false

# Security
ENCRYPTION_KEY=          # 32-byte base64; dev模式自动生成

# Semantic Judge (LLM-as-Judge，可选)
JUDGE_API_URL=           # 评判模型 API 地址
JUDGE_API_KEY=           # API Key
JUDGE_MODEL=gpt-4o-mini  # 推荐用低成本模型做评判
JUDGE_TIMEOUT=15

# Token Budget
TOKEN_BUDGET_QUICK=15000
TOKEN_BUDGET_STANDARD=40000
TOKEN_BUDGET_DEEP=100000

# Theta / Relative Scale
THETA_METHOD=rasch_1pl   # rasch_1pl / rasch_2pl / mdirt
THETA_BOOTSTRAP_B=200
THETA_CI_STOP_WIDTH=0.25
THETA_DELTA_STOP=0.45

# Concurrency
CONCURRENCY_QUICK=12
CONCURRENCY_STANDARD=8
CONCURRENCY_DEEP=3

# Verdict Thresholds
VERDICT_TRUSTED_THRESHOLD=80
VERDICT_SUSPICIOUS_THRESHOLD=60
VERDICT_HIGH_RISK_THRESHOLD=40

# Task Queue
USE_CELERY=false
REDIS_URL=redis://localhost:6379/0

# Async Pipeline (Phase D)
ASYNC_PIPELINE_ENABLED=false
ASYNC_PIPELINE_FALLBACK_SYNC=true
```

## API Endpoints (v1 + v8 + v10)
```
# v1 Core
GET    /api/v1/health
GET    /api/v1/runs                    # 列表
POST   /api/v1/runs                    # 创建检测
GET    /api/v1/runs/{id}               # 详情
DELETE /api/v1/runs/{id}
POST   /api/v1/runs/{id}/cancel
POST   /api/v1/runs/{id}/retry
POST   /api/v1/runs/{id}/continue
POST   /api/v1/runs/{id}/skip-testing
GET    /api/v1/runs/{id}/report
GET    /api/v1/runs/{id}/radar.svg
GET    /api/v1/runs/{id}/responses
GET    /api/v1/runs/{id}/scorecard
GET    /api/v1/runs/{id}/extraction-audit
GET    /api/v1/runs/{id}/theta-report
GET    /api/v1/runs/{id}/pairwise
POST   /api/v1/runs/batch-delete
GET    /api/v1/exports/runs.zip

# v1 Baselines & Compare
GET    /api/v1/benchmarks
POST   /api/v1/baselines
GET    /api/v1/baselines
POST   /api/v1/baselines/compare
DELETE /api/v1/baselines/{id}
GET    /api/v1/baselines/{id}
POST   /api/v1/compare-runs
GET    /api/v1/compare-runs
GET    /api/v1/compare-runs/{id}

# v1 Models & Leaderboard
GET    /api/v1/models/{name}/trend
GET    /api/v1/models/{name}/theta-trend
GET    /api/v1/leaderboard
GET    /api/v1/theta-leaderboard
GET    /api/v1/elo-leaderboard

# v1 Calibration
POST   /api/v1/calibration/rebuild
POST   /api/v1/calibration/snapshot
POST   /api/v1/calibration/replay
GET    /api/v1/calibration/replay
GET    /api/v1/calibration/replay/{id}

# v1 Tools
POST   /api/v1/tools/generate-isomorphic

# v11 Observability & Resilience
GET    /api/v1/circuit-breaker            # 断路器状态（所有端点）
GET    /api/v1/circuit-breaker/{base_url} # 断路器状态（单端点）
POST   /api/v1/circuit-breaker/reset      # 重置断路器
GET    /api/v1/runs/{id}/trace            # 追踪数据
GET    /api/v1/tracer/progress            # 所有活跃追踪进度

# v11 CDM & Attribution (Phase 2)
GET    /api/v1/cdm/skills                 # CDM 技能分类体系
GET    /api/v1/runs/{id}/cdm              # CDM 认知诊断报告
GET    /api/v1/runs/{id}/attribution      # Shapley Value 评分归因

# v11 Phase 3: Token优化 & 数据提纯 & 多语言攻击
POST   /api/v1/suite/prune                # IIF 数据提纯分析（dry-run）
GET    /api/v1/suite/pruning-report       # 最近一次提纯报告
GET    /api/v1/prompt-optimizer/report    # 提示词优化器统计
GET    /api/v1/gpqa/questions             # GPQA 题库
GET    /api/v1/attacks/multilingual       # 多语言攻击模板

# v8 Transparency
GET    /api/v8/health
GET    /api/v8/plugins
GET    /api/v8/plugins/{name}/metadata
GET    /api/v8/plugin-stats
GET    /api/v8/runs/{id}/judgment-logs
GET    /api/v8/runs/{id}/case/{cid}/provenance
GET    /api/v8/runs/{id}/data-lineage
GET    /api/v8/references/thresholds

# v10 SSE Logs
GET    /api/v10/runs/{id}/logs/stream
```

## Code Conventions
- `from __future__ import annotations` 在文件顶部
- 标准库 import 在前，项目 import 用绝对路径 `from app.core.xxx`
- 命名: 文件 `snake_case`，类 `PascalCase`，私有 `_prefix`，常量 `UPPER_SNAKE`
- Handler 函数: `handle_` 前缀
- 日志: `logger = get_logger(__name__)` + structlog 风格关键字参数
- 代码分区: `# -- Section Name` 注释
- 类型注解: Python 3.10+ 风格（`str | None`）
- 数据结构: dataclass，无 Pydantic

## Testing
- pytest 测试框架
- 测试用独立数据库 `test_inspector.db`，测试后自动清理
- 测试文件: `backend/tests/`
  - `test_all.py` — 主测试套件
  - `test_v8_phase1.py` / `test_v8_phase4.py` / `test_v8_phase5.py` — v8 分阶段测试
  - `test_v9_phasea_regression.py` / `test_v9_phaseb_regression.py` / `test_v9_phasec_regression.py` — v9 回归测试
  - `test_core_pipeline.py` — 核心管线单元测试（54 passed）
  - `test_v11_phase1.py` — v11 Phase 1 测试（43 passed: 断路器+追踪器+EvalSchema）
  - `test_v11_phase2.py` — v11 Phase 2 测试（27 passed: CDM引擎+Shapley归因+Handler）
- 覆盖: config/security/db/seeder/judge/analysis/repo/predetect/executor/http

## Task Queue Architecture
- 本地模式: ThreadPoolExecutor（默认，零配置）
- 分布式模式: Celery + Redis（设置 `USE_CELERY=true` + `REDIS_URL`）
- 抽象接口: `app.tasks.queue.TaskQueue`
- 切换方式: 环境变量 `USE_CELERY=true`

## Gotchas
- dev 模式下 ENCRYPTION_KEY 自动生成确定性密钥，生产环境必须手动设置
- 前端文件: index.html + styles.css + app.js + v8_components.js + v8_demo.html + v8_styles.css
- 数据库 schema 提供迁移函数 `migrate_json_columns_to_columns()` 和 `db_migrations.migrate()`
- `test_runs` 表的 `evaluation_mode`, `calibration_case_id`, `scoring_profile_version`, `calibration_tag` 已迁移为独立列
- 相似度比对只使用 golden_baselines 中用户标记的基准模型，没有基准时相似度列表为空
- 模式向后兼容：`full`→`deep`、`extraction`→`deep` 自动映射
- SemanticJudge 需要配置 `JUDGE_API_URL` 才会调用外部 LLM，否则降级为本地规则评判
- 根目录下的 `analysis_pipeline.py`、`methods.py`、`schemas.py`、`suite.json` 是 v3 时代的历史残留，正式代码在 `backend/app/` 下
- 三个巨文件需要注意：`analysis/pipeline.py`(141KB)、`runner/orchestrator.py`(79.8KB)、`predetect/pipeline.py`(75KB)
