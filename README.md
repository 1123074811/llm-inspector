# LLM Inspector v15.0

> 套壳检测 · 能力评估 · 数据溯源 · 20层对抗探针 · 真实模型暴露引擎（v15.0 Bug修复与版本对齐）

**v15.0.0 Phase 0 + Phase 4 已完成**。v14.0.0 全部 9 个阶段亦已完成。详见 [CHANGELOG.md](CHANGELOG.md) 和 [迁移指南](docs/MIGRATION_v13_to_v14.md)。

> 📖 [v15 升级方案](docs/UPGRADE_PLAN_V15.md)

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

## 预检测管道（v13：L0-L16；v14 规划：L17-L20）

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
| L17 | Identity Exposure *(v14 Phase 3)* | 0 | 重分析前层证据，贝叶斯后验推断实际模型家族（16 家族分类） |
| L18 | Timing Side-Channel *(v14 Phase 5)* | 0 | TTFT/TPS KL 散度对比 6 家族参考分布（Yu et al. 2024）|
| L19 | Token Distribution *(v14 Phase 5)* | 0 | 响应长度 Wasserstein 距离 + 4-gram 重复率（Carlini et al. 2023）|

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

## 题库 v13（suite_v13.json）

90 道真实基准题，全部含 `source_ref` 和 `license`：

| 数据集 | 题数 | 模式 | 许可证 |
|--------|------|------|--------|
| GPQA Diamond | 15 | Deep | MIT |
| AIME 2024 | 13 | Deep | Public domain |
| MATH-500 | 14 | Standard | MIT |
| SWE-bench Lite | 10 | Deep | MIT |
| MMLU-Pro | 20 | Standard | MIT |
| CMMLU | 10 | Standard | CC BY-NC-SA 4.0 |
| JailbreakBench | 8 | Deep | MIT |

## API 端点

```
GET  /api/v1/health
POST /api/v1/runs                        # 创建检测
GET  /api/v1/runs/{id}                   # 详情
GET  /api/v1/runs/{id}/report            # 报告
GET  /api/v1/runs/{id}/timeline.svg      # 执行时间轴 (v13新增)
GET  /api/v1/runs/{id}/trace             # 追踪数据
GET  /api/v10/runs/{id}/logs/stream      # SSE 实时日志
GET  /api/v1/leaderboard                 # ELO 排行榜
GET  /api/v14/bt-leaderboard             # Bradley-Terry 强度排行榜（Phase 2）
GET  /api/v14/model-taxonomy             # 模型家族分类表（Phase 3）
GET  /api/v14/runs/{id}/identity-exposure # 真实模型暴露报告（Phase 3）
GET  /api/v14/runs/{id}/system-prompt    # 提取的系统提示词（Phase 3）
GET  /api/v14/runs/{id}/judge-chain     # 判题路径日志（Phase 4）
GET  /api/v14/runs/{id}/predetect-trace # 20 层预检测 JSONL 日志，分页（Phase 5）
```

## v14 新特性（进行中）

| 主题 | 状态 | 说明 |
|------|------|------|
| 代码清理 & 数据链 | ✅ Phase 1 完成 | 删除死代码，suite_v13 补齐 source_url，v8 路由标 deprecated |
| 评分重构 | ✅ Phase 2 完成 | 消除 19 处 return 50.0 假数据，NNLS 权重落盘，ScoreCard.completeness，Bradley-Terry 排行榜 |
| 真实模型暴露引擎 | ✅ Phase 3 完成 | 16 家族 model_taxonomy.yaml + 系统提示词抽取 + Layer17 + "疑似实际模型"卡片 |
| 判题加固 | ✅ Phase 4 完成 | numeric_tolerance / multi_choice / NLI 本地判题 / DBpedia 幻觉验证 / Fleiss κ / 4 级降级链 |
| 预检测扩展 | ✅ Phase 5 完成 | L18 时序侧信道 + L19 Token 分布侧信道 + 20 层 JSONL 日志落盘 + predetect-trace API |
| Token 效率 | ✅ Phase 6 完成 | PromptOptimizer 入链（B5 修复）+ IRT 自适应采样 + tiktoken 精确计数 |
| 100% 进度保障 | ✅ Phase 7 完成 | B7 进度公式修正 + B8 Watchdog 分页 + 分级重试策略 + 断路器历史 API |
| 前端 & 报告 | ✅ Phase 8 完成 | safeFetch 错误边界 + 排行榜分页 + 动态雷达图 + v14 报告卡片 |

详见 [docs/UPGRADE_PLAN_V14.md](docs/UPGRADE_PLAN_V14.md)。

## 开发

```bash
# 运行测试（471 passed, 4 skipped）
pytest backend/tests/ -q

# 验证 SOURCES.yaml 完整性
python backend/start.py --verify-sources

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
      -> PreDetectionPipeline (16 layers)
      -> CaseExecutor -> DualJudge (rule + semantic + Cohen's kappa)
      -> AnalysisPipeline (IRT + Theta + Stanine + Shapley + CDM)
      -> KnowledgeGraphClient (Wikidata + DBpedia fan-out)
```

## 引用

| 来源 | 用途 |
|------|------|
| Rasch (1960) | IRT 1PL θ 估计 |
| Birnbaum (1968) | IRT 2PL 区分度 |
| Cohen (1960) | Cohen's κ 一致性 |
| Canfield (1951) | Stanine-9 边界 |
| Yong et al. (2023) arXiv:2310.02446 | 多语言攻击 |
| Jiang et al. (2024) arXiv:2402.11753 | ASCII Art 注入 |
| Greshake et al. (2023) arXiv:2302.12173 | 间接提示注入 |
| HELM v1.10 (Stanford CRFM) | 权重回归锚点 |
| LMSYS Chatbot Arena | 权重回归锚点 |
