# LLM Inspector v15.0

> 套壳检测 · 能力评估 · 数据溯源 · 24 层对抗探针 · 预检连通验证 · 评判注册中心（v15.0 全部 14 阶段完成）

**v15.0.0 全部 14 阶段（Phase 0–13）已完成**。v14.0.0 全部 9 个阶段亦已完成。详见 [CHANGELOG.md](CHANGELOG.md) 和 [迁移指南](docs/MIGRATION_v14_to_v15.md)。

> 📖 [v15 升级方案](docs/UPGRADE_PLAN_V15.md) · 📖 [v14 升级方案](docs/UPGRADE_PLAN_V14.md)

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

## 开发

```bash
# 运行测试（825 passed, 4 skipped）
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
