# LLM Inspector v17.0

> 套壳检测 · 能力评估 · 协议层硬证据 · 字段层硬证据 · 价格硬规则 · model_registry 实时新陈代谢 · 题库实时同步 · 后台守护任务（v17 全部 13 阶段完成）

**v17.0.0 全部 13 阶段（Phase 0–12）已完成**，1219 passed / 4 skipped / 0 failed。在 v16 基础上落地了 11 个真空（G-01..G-11）：协议字段层硬证据 + 模型情报与题库的实时新陈代谢 + 后台守护任务。

> 📖 [v17 升级方案](docs/UPGRADE_PLAN_V17.md) · [v17 变更日志](docs/CHANGELOG_v17.md) · [v16 → v17 迁移](docs/MIGRATION_v16_to_v17.md)
>
> 历史：v15 升级方案 [`docs/UPGRADE_PLAN_V15.md`](docs/UPGRADE_PLAN_V15.md) · v14 升级方案 [`docs/UPGRADE_PLAN_V14.md`](docs/UPGRADE_PLAN_V14.md)

LLM Inspector 是一款面向 OpenAI 兼容 API 的**模型真伪鉴别与能力评估工具**。它通过渐进式指纹识别、多维度测试套件和统计学评分系统，帮助用户判断所接入的 LLM API 是否是其声称的模型。

## Quick Start

```bash
# Windows
start.bat

# macOS / Linux
./start.sh

# 手动（任意平台）
cd llm-inspector
pip install -r backend/requirements.txt
python backend/start.py --port 9999
```

访问 http://localhost:9999

## 三种检测模式

| 模式 | 用例数 | Token 预算 | 并发 | 适用场景 |
|------|--------|-----------|------|----------|
| **Quick** | ~18 | 15K | 12 | API 可用性验证 + 粗筛真伪 |
| **Standard** | ~44 | 40K | 8 | 完整能力画像 + 可靠真伪判定 |
| **Deep** | ~87 | 100K | 3 | 精确模型指纹 + 对抗性压力测试 |

## 预检测管道（v15：L0-L23，共 24 层）

| 层 | 名称 | Token | 说明 |
|----|------|-------|------|
| L0 | HTTP Header | 0 | Server/X-Request-ID 指纹 |
| L1 | Self-Report | ~50 | 直接问模型身份 |
| L2 | Identity Matrix | ~200 | 5 维身份探针 |
| L3 | Knowledge Cutoff | ~500 | 8 种 Tokenizer 覆盖 |
| L4 | Behavioral Bias | ~1000 | 风格/格式/语言偏好 |
| L5 | Semantic Fingerprint | ~1500 | 语义嵌入比对 |
| L6 | Active Extraction | ~1500 | 主动身份提取 |
| L7 | Logprobs | ~2000 | 差分测试+提取攻击 |
| L8 | Semantic FP v2 | ~100 | 语义指纹 |
| L9 | Advanced Extraction | ~200 | 9 种攻击模板 |
| L10 | Differential Testing | ~150 | 差分一致性 |
| L11 | Tool Capability | ~50 | 工具能力探测 |
| L12 | Multi-turn Overflow | ~300 | 上下文溢出 |
| L13 | Adversarial Analysis | ~100 | 对抗性响应分析 |
| L14 | Multilingual Attack | ~500 | 13 种低资源语言攻击（Yong et al. 2023）|
| L15 | ASCII Art Attack | ~150 | 视觉注入绕过检测（Jiang et al. 2024）|
| L16 | Indirect Injection | ~150 | RAG 式间接注入检测（Greshake et al. 2023）|
| L17 | Identity Exposure | 0 | 贝叶斯后验推断实际模型家族（16 家族，v14 Phase 3）|
| L18 | Timing Side-Channel | 0 | TTFT/TPS KL 散度对比 6 家族参考分布（Yu et al. 2024，v14 Phase 5）|
| L19 | Token Distribution | 0 | 响应长度 Wasserstein 距离 + 4-gram 重复率（Carlini et al. 2023，v14 Phase 5）|
| L20 | Self-Paradox Probe | ~200 | 自我矛盾诱导——探测身份声明一致性（v15 Phase 6）|
| L21 | Multi-Step Drift | ~300 | 多轮漂移检测——上下文积压后的答案偏移（v15 Phase 6）|
| L22 | Prompt Reconstruction | ~200 | 提示词重构——反推系统提示词结构（v15 Phase 6）|
| L23 | Adversarial Tools | ~150 | 对抗性 Tool-Call 探测——工具调用绕过（v15 Phase 6）|

置信度 ≥ 0.85 提前停止；贝叶斯融合逐步更新后验概率。

## 评分体系

```
总分 = 0.45×能力 + 0.30×真实性 + 0.25×性能

能力    = f(推理, 对抗, 指令, 编码, 安全, 协议, 知识, 工具)
真实性  = f(相似度, 行为不变性, 一致性, 提取抵抗, 预检测, 指纹)
性能    = f(速度, 稳定性, 成本效率, TTFT可信度)
```

**双刻度展示**（v13 新增）：
- **Stanine-9**（1-9 阶）：心理测量学标准，来源 Canfield (1951)
- **百分位**（0-100%）：基于 HELM v1.10 参考分布
- **θ 逻辑分**：IRT 原生刻度，均值 0，SD 1

权重来源：HELM v1.10 + LMSYS Chatbot Arena，非负最小二乘回归（R² ≈ 0.96）。

## 数据溯源

所有常量、阈值、权重均在 `backend/app/_data/SOURCES.yaml` 注册：

```bash
# 验证数据溯源完整性
python backend/start.py --verify-sources
```

每条记录包含：`source_url`、`retrieved_at`、`license`、`source_type`。

## 安装与配置

### 依赖安装

```bash
pip install -r backend/requirements.txt
# 或最小安装（无可选功能）
pip install pyyaml numpy scipy cryptography requests tiktoken
```

### 环境变量（可选）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PORT` | 8000 | 服务端口 |
| `HOST` | 127.0.0.1 | 绑定地址 |
| `JUDGE_API_URL` | — | LLM-as-Judge 外部 API |
| `JUDGE_API_KEY` | — | Judge API 密钥 |
| `STRICT_PROVENANCE` | false | 溯源严格模式 |
| `USE_CELERY` | false | 启用分布式任务队列 |

## 题库

### suite_v13.json（90 道真实基准题）

全部含 `source_ref` 和 `license`：

| 数据集 | 题数 | 模式 | 许可证 |
|--------|------|------|--------|
| GPQA Diamond | 15 | Deep | MIT |
| AIME 2024 | 13 | Deep | Public domain |
| MATH-500 | 14 | Standard | MIT |
| SWE-bench Lite | 10 | Deep | MIT |
| MMLU-Pro | 20 | Standard | MIT |
| CMMLU | 10 | Standard | CC BY-NC-SA 4.0 |
| JailbreakBench | 8 | Deep | MIT |

### suite_v15.json（v15 数据集导入管线）

通过 `POST /api/v15/dataset/import` 持续追加；格式要求：`id`/`category`/`name`/`user_prompt`/`judge_method`（必填），`max_tokens`/`n_samples`/`weight` 缺省自动补全。

## API 端点

```
# v1 核心
GET  /api/v1/health
POST /api/v1/runs                        # 创建检测
GET  /api/v1/runs/{id}                   # 详情
GET  /api/v1/runs/{id}/report            # 报告
GET  /api/v1/runs/{id}/trace             # 追踪数据
GET  /api/v10/runs/{id}/logs/stream      # SSE 实时日志
GET  /api/v1/leaderboard                 # ELO 排行榜
GET  /api/v1/runs/{id}/timeline.svg      # 执行时间轴

# v14 扩展
GET  /api/v14/bt-leaderboard             # Bradley-Terry 强度排行榜
GET  /api/v14/model-taxonomy             # 模型家族分类表
GET  /api/v14/runs/{id}/identity-exposure  # 真实模型暴露报告
GET  /api/v14/runs/{id}/system-prompt    # 提取的系统提示词
GET  /api/v14/runs/{id}/judge-chain      # 判题路径日志
GET  /api/v14/runs/{id}/predetect-trace  # 24 层预检测 JSONL 日志（分页）

# v15 新增
GET  /api/v15/health                     # v15 命名空间健康检查
GET  /api/v15/runs/{id}/preflight        # 预检连通验证结果（A1-A5 步骤报告）
GET  /api/v15/runs/{id}/evidence-ledger  # 贝叶斯证据台账（包装器检测）
GET  /api/v15/runs/{id}/model-card-diff  # 模型卡差异对比报告
GET  /api/v15/runs/{id}/token-audit      # Token 效率审计
GET  /api/v15/cache-stats                # 全局响应缓存指标
POST /api/v15/cache/evict                # 驱逐过期缓存条目
POST /api/v15/dataset/import             # 导入测试用例到 suite_v15
POST /api/v15/dataset/validate           # 验证单条测试用例格式
GET  /api/v15/judge-registry             # 列出所有注册的评判方法
GET  /api/v15/judge-registry/{method}    # 单个评判方法详情

# v17 新增
GET  /api/v17/maintenance/status         # 守护线程 + 4 个 job 的实时健康状态
```

## v15 新特性（全部完成）

| 主题 | 阶段 | 说明 |
|------|------|------|
| Bug 修复 & 版本对齐 | ✅ Phase 0 | suite_v15 基础、version.json、Migration007 缓存表 |
| 证据台账 | ✅ Phase 1 | `authenticity/evidence_ledger.py`：贝叶斯奇数融合包装器检测概率 |
| 模型卡差异 | ✅ Phase 2 | `authenticity/model_card_diff.py`：声称模型 vs 疑似模型差异报告 |
| 叙事生成器 | ✅ Phase 3 | `analysis/narrative_builder.py`：纯规则、零 Token 报告叙事文本 |
| Bug 修复补丁 | ✅ Phase 4 | 多处线上 bug 修复（LLMRequest 字段、ScoreCard null 化等） |
| 预检连通验证 | ✅ Phase 5 | `preflight/`：A1 输入校验 → A2 TCP → A3 Auth → A4 Schema → A5 能力探测 |
| 对抗预检测 L20-L23 | ✅ Phase 6 | 4 层 Deep 专属层：自我矛盾/多轮漂移/提示词重构/对抗工具调用 |
| 校准指标 | ✅ Phase 7 | `analysis/calibration_metrics.py`：Brier 分/对数损失/ECE/可靠性曲线 |
| 不确定性量化 | ✅ Phase 8 | `analysis/uncertainty.py`：bootstrap CI/SEM/HDI/加权 CI |
| 评判校准 | ✅ Phase 9 | `analysis/judge_calibration.py`：Fleiss κ/Cohen κ/偏差检测 |
| 响应缓存 | ✅ Phase 10 | `runner/cache_strategy.py`：temperature=0 的 SHA-256 键 TTL 缓存 |
| 数据集导入管线 | ✅ Phase 11 | `runner/import_dataset.py`：DatasetImporter + ImportReport |
| 评判注册中心 | ✅ Phase 12 | `analysis/judge_registry.py` + `_data/judge_registry.yaml` |
| 预检错误分类 | ✅ Phase 13 | `preflight/error_taxonomy.py`：15 个 ErrorCode + `preflight/connection_check.py` |

## v17 新特性（全部完成，详见 [`docs/CHANGELOG_v17.md`](docs/CHANGELOG_v17.md)）

| 主题 | 阶段 | 模块 | 说明 |
|---|---|---|---|
| 死代码 + 套件归档 | ✅ Phase 0 | — | 删除 `adaptive_testing` / `factor_analysis` / `async_pipeline` / `celery_*` / `benchmarks/` (~1.7k LOC)；`suite_v10` → `archive/` |
| **协议级硬证据 L0.5** | ✅ Phase 1 | `predetect/protocol_validator.py` + `adapters/contamination_probe.py` | SSE 帧 / id 前缀 (`chatcmpl-`/`msg_`) / error schema / 跨家族鉴权污染 → 3 条硬规则 cap |
| **字段级硬证据 L0.6** | ✅ Phase 2 | `predetect/field_evidence.py` + `judge/methods.py` | `system_fingerprint` 正则 / `reasoning_tokens` / `cache_read_input_tokens` / `thinking.signature` + 6 条 wrapper 反模式正则 |
| **价格层硬规则** | ✅ Phase 3 | `_data/official_prices.yaml` (30 模型) + `authenticity/price_evidence.py` | claimed vs 官方价：`<30%` cap=30, `<60%` cap=60 |
| **timing 基线真值化** | ✅ Phase 4 | `scripts/sample_timing_references.py` 增强 + L18/L19 placeholder gate | p10/p25/p50/p75/p90 + raw-data SHA256 + 全 placeholder 时强制 confidence=0 |
| **model_registry Schema** | ✅ Phase 5 | Migration 008 + `repository/registry_repo.py` | 28 列单一真相源 + audit log + 优先级合并 (`official_api > openrouter > changelog > self_probed > manual`) |
| **Tier 1+2 实时同步** | ✅ Phase 6 | `runner/model_registry_sync.py` | OpenAI/Anthropic/Google/xAI/DeepSeek/Mistral `/v1/models` + OpenRouter（200+ 模型，免 key） |
| **Tier 3 文档解析** | ✅ Phase 7 | `runner/changelog_harvester.py` | RSS+LLM 抽取 + **anti-hallucination gate**（evidence_quote verbatim 校验） |
| **Tier 4 自注册探针** | ✅ Phase 8 | `runner/self_probe_register.py` | 4 阶段 ≤1k token 探针：cutoff 二分定位 / tokenizer 指纹 / timing / self-report |
| **题库实时同步** | ✅ Phase 9 | `runner/dataset_sync.py` | LiveBench / SWE-bench Verified / HLE 增量 ingest，仅插入新 case_id |
| **suite_pruner + 枯竭告警** | ✅ Phase 10 | Migration 009 + `tasks/pruner_job.py` | `case_quality_flags` 表 + ceiling/floor 标记 + `suite_exhaustion_warning` SSE |
| **基线池清洗** | ✅ Phase 11 | `analysis/baseline_pool.py` | similarity 比对仅用 registry-eligible（active ∧ aged≥30d ∧ fresh≤14d ∧ 来源 ∈ {官方API, openrouter, changelog}），冷启动 fallback |
| **守护任务 + 状态端点** | ✅ Phase 12 | `tasks/maintenance_jobs.py` + `/api/v17/maintenance/status` | 单守护线程跑 4 个周期任务（6h/24h/7d/1h），错峰启动 + 失败隔离 |

### v17 新硬规则矩阵

VerdictEngine 新增 **6 条 v17 硬规则**（按 cap 严重度排列）：

| 规则 | Cap | 触发条件 |
|---|---|---|
| `protocol_auth_pollution_cap` | 35 | 跨家族鉴权污染（同端点接受 Bearer + x-api-key） |
| `price_below_30pct_cap` | 30 | 声称价 < 官方价 30% |
| `protocol_error_schema_cap` | 50 | 错误体 schema 不符合声称厂商契约 |
| `field_malformed_fingerprint_cap` | 50 | `system_fingerprint` 字段格式畸形（主动伪造） |
| `protocol_id_prefix_cap` | 55 | `chatcmpl-` / `msg_` 前缀不匹配 |
| `price_below_60pct_cap` | 60 | 声称价 < 官方价 60% |

## 后台守护任务（v17 默认开启）

`start.bat` / `start.sh` 默认 `MAINTENANCE_JOBS_ENABLED=1`，启动后单守护线程自动跑：

| Job | 频率 | 行为 | 模块 |
|---|---|---|---|
| `model_registry_sync` | **6h** | OpenRouter + 各家 `/v1/models` 拉新 + 价格 + sweep_deprecated | `runner/model_registry_sync.py` |
| `changelog_harvester` | **24h** | RSS / 公告页 + LLM 抽取（anti-hallucination gate） | `runner/changelog_harvester.py` |
| `dataset_sync` | **7d** | LiveBench / SWE-bench Verified / HLE 新题 ingest | `runner/dataset_sync.py` |
| `pruner_job` | **1h** | ceiling/floor 标记 + suite_exhaustion 告警 | `tasks/pruner_job.py` |

**错峰**：初始延迟 60s / 300s / 600s / 120s，避免冷启动同时打外部 API。
**失败隔离**：任意 job 抛异常仅自己计 `failures += 1`，不影响其他 job 重新调度。
**禁用**：把 env 设为 `0` 或删掉那行即可，老部署默认仍然 opt-in。

### 实时观察守护线程健康

```bash
curl http://localhost:9999/api/v17/maintenance/status
```

返回示例：

```json
{
  "status": "ok", "api_version": "v17",
  "enabled": true, "running": true,
  "jobs": [
    {"name": "model_registry_sync", "interval_sec": 21600, "successes": 12,
     "failures": 0, "last_error": null,
     "next_run_at": 1777310407.521, "next_run_in_sec": 4823},
    {"name": "changelog_harvester", "...": "..."},
    {"name": "dataset_sync",        "...": "..."},
    {"name": "pruner_job",          "...": "..."}
  ]
}
```

## 一次性命令（手动触发，**不**进守护线程）

### `sample_timing_references.py` — 时序基线实测

**作用**：给 L18 (Timing KL 散度) / L19 (Token 分布 Wasserstein) 提供可信的官方时序基线。不跑这个脚本时基线是 placeholder，L18/L19 强制 `confidence=0`，时序证据完全停用。

**何时跑**：

- 第一次部署完成
- 每次发新版前（建议 CI release 步骤）
- 怀疑某家厂商网络/边缘节点变更后

**怎么跑**：

```bash
# 准备 key（配几家就跑几家，缺失的家族会跳过）
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=...
export XAI_API_KEY=...
export DEEPSEEK_API_KEY=...
export MISTRAL_API_KEY=...

# 跑 6 家族，每家 100 次实测
python -m backend.scripts.sample_timing_references --all --samples 100

# 验证 provenance 已升级
python -c "import json; \
  d=json.load(open('llm-inspector/backend/app/_data/timing_references.json')); \
  print(d['_provenance']['note'])"
# 期望输出：v17.0-self-measurement
```

**成本**：约 5–10 分钟 + 每家 ~$0.05–0.10 token 费用（短 prompt）。

**为什么没进守护线程**：基线不应频繁刷新，否则会把短期网络抖动固化进基线；且要 API key（守护线程跑时 key 可能没配会刷错误日志）。

### 其他手动命令

```bash
# 立即跑一次 model_registry_sync（不影响守护轮次）
python -m app.runner.model_registry_sync --once --sweep-deprecated

# 立即拉一次 LiveBench / SWE-bench / HLE
python -m app.runner.dataset_sync --pull --max-rows 1000

# 立即跑 pruner（按 case 计算 pass-rate + ceiling/floor）
python -m app.tasks.pruner_job --once

# 解析 RSS / 公告页（需 LLM 抽取器，默认 noop）
python -m app.runner.changelog_harvester --once
```

## 开发

```bash
# 运行测试（v17.0：1219 passed, 4 skipped）
pytest backend/tests/ -q

# 验证 SOURCES.yaml 完整性
python backend/start.py --verify-sources

# 验证测试套件 JSON 格式
python backend/scripts/validate_suite.py

# 构建 suite_v13
python backend/scripts/build_suite_v13.py

# 拟合权重（需要 golden_baselines 数据）
python backend/scripts/fit_weights.py --from-helm

# 构建参考嵌入
python backend/scripts/build_reference_embeddings.py
```

## 架构

```
HTTP Handler (main.py)
  -> Repository (repo.py)
  -> Worker (worker.py) + Watchdog
    -> Orchestrator (run_lifecycle.py)
      -> PreflightCheck (preflight/) — 5 步连通验证（A1-A5）
      -> PreDetectionPipeline (24 layers, L0-L23)
      -> CaseExecutor (CacheStrategy TTL 缓存, IRT 自适应采样)
        -> DualJudge (rule + semantic + Cohen's κ)
        -> JudgeChainRunner (4 级降级链)
      -> AnalysisPipeline (IRT + Theta + Stanine + Shapley + CDM
                           + CalibrationMetrics + Uncertainty + NarrativeBuilder)
      -> EvidenceLedger (贝叶斯奇数融合) -> ModelCardDiff
      -> KnowledgeGraphClient (Wikidata + DBpedia fan-out)
```

## 引用

| 来源 | 用途 |
|------|------|
| Rasch (1960) | IRT 1PL θ 估计 |
| Birnbaum (1968) | IRT 2PL 区分度 |
| Cohen (1960) | Cohen's κ 一致性 |
| Fleiss (1971) | Fleiss's κ（≥3 评判器） |
| Canfield (1951) | Stanine-9 边界 |
| Bradley & Terry (1952) | Bradley-Terry 强度排行榜 |
| Yong et al. (2023) arXiv:2310.02446 | 多语言攻击 |
| Jiang et al. (2024) arXiv:2402.11753 | ASCII Art 注入 |
| Greshake et al. (2023) arXiv:2302.12173 | 间接提示注入 |
| Yu et al. (2024) | 时序侧信道指纹 |
| Carlini et al. (2023) arXiv:2403.06634 | Token 分布侧信道 |
| HELM v1.10 (Stanford CRFM) | 权重回归锚点 |
| LMSYS Chatbot Arena | 权重回归锚点 |
| NIST SP 330-2019 | 数值容差判题（相对误差 ≤5%） |
| Hendrycks et al. (2021) | MMLU strict letter match |
| Reimers & Gurevych (2019) | NLI 语义蕴含判题 |
