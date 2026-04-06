# LLM Inspector v1.0

LLM 套壳检测工具 —— 检测 LLM API 是否真正提供其声称的模型服务，还是在背后代理/包装了其他模型。

## 工作原理

LLM Inspector 通过多层行为指纹分析，识别 API 背后的真实模型：

1. **预检测（Pre-detection）**：7 层渐进式指纹识别管道（HTTP 头信息、自我报告、身份探测、知识截止日期、偏好/格式分析、**主动提示词提取、多轮上下文过载**），以最小 token 消耗快速锁定候选模型
2. **全量测试**：v1 套件 19 个用例（6 类别）+ v2 套件 33 个用例（11 类别）+ **v3 精简套件 37 个用例（降本增效优化）** + **提取审计套件 20 个用例（3 类别）**，覆盖协议合规、指令遵循、推理、编码、反欺骗、**主动身份提取**等维度
3. **特征提取与比对**：从响应中提取行为特征向量，与内置 108 个基准模型画像（覆盖 20+ 厂商）进行余弦相似度比对 + Bootstrap 置信区间估算
4. **v2 三维评分**：Capability（能力）+ Authenticity（真实性）+ Performance（性能）三维度评分卡
5. **信任判定**：输出 `trusted` / `suspicious` / `high_risk` / `fake` 四级信任判定
6. **A/B 对比**：支持两次检测结果的显著性差异分析

## 技术栈

- **后端**：Python 3.12，零外部框架依赖（纯标准库实现 HTTP 服务器、数据库、任务队列）
- **前端**：单文件 SPA（纯 HTML/CSS/JS，无构建步骤）
- **数据库**：SQLite（WAL 模式），12 张表
- **加密**：AES-256-GCM（API 密钥静态加密），外部依赖 `cryptography`
- **科学计算**：`numpy` + `scikit-learn`（特征比对与统计分析）

## 项目结构

```
llm-inspector/
├── start.bat                      # Windows 一键启动
├── stop.bat                       # Windows 停止服务
├── frontend/
│   └── index.html                 # 前端单文件应用（~1080 行）
├── backend/
│   ├── app/
│   │   ├── main.py                # HTTP 服务器入口（37 个 API 端点）
│   │   ├── core/
│   │   │   ├── config.py          # 环境配置
│   │   │   ├── db.py              # SQLite 数据库层（12 张表）
│   │   │   ├── schemas.py         # 数据模型（dataclasses）
│   │   │   ├── security.py        # 加密 + SSRF 防护
│   │   │   └── logging.py         # 结构化 JSON 日志
│   │   ├── adapters/
│   │   │   └── openai_compat.py   # OpenAI 兼容 API 适配器
│   │   ├── predetect/
│   │   │   └── pipeline.py        # 7 层预检测管道（含 Layer6 主动提取）
│   │   ├── runner/
│   │   │   ├── orchestrator.py    # 测试编排器（含 extraction 模式）
│   │   │   └── case_executor.py   # 单用例执行器
│   │   ├── judge/
│   │   │   └── methods.py         # 21 种判定方法（含 11 种提取审计专用）
│   │   ├── analysis/
│   │   │   └── pipeline.py        # 特征提取 + v2 评分 + 提取审计 + 报告生成
│   │   ├── repository/
│   │   │   └── repo.py            # 数据库 CRUD
│   │   ├── tasks/
│   │   │   ├── worker.py          # 线程池后台 Worker（4 线程）
│   │   │   └── seeder.py          # 数据库初始化种子
│   │   └── fixtures/
│   │       ├── suite_v1.json      # 19 个测试用例（v1）
│   │       ├── suite_v2.json      # 33 个测试用例（v2，11 类别）
│   │       ├── suite_v3.json      # 37 个测试用例（v3，精简优化版，默认）
│   │       ├── suite_extraction.json  # 20 个提取审计用例（新增）
│   │       └── benchmarks/
│   │           └── default_profiles.json  # 108 个基准模型画像
│   └── tests/
│       └── test_all.py            # 测试套件（~60+ 用例）
```

## 快速开始

### 环境要求

- Python 3.12+
- `cryptography>=41`
- `numpy>=1.24`
- `scikit-learn>=1.3`

### 安装与运行

```bash
cd llm-inspector

# 创建并激活虚拟环境
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# 安装依赖
pip install cryptography numpy scikit-learn

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入必要配置

# 启动服务
cd backend
python -m app.main
```

服务默认运行在 `http://localhost:8000`。

**Windows 快捷启动**：直接运行 `start.bat`。

**PowerShell 一键启动**（自动创建 .venv + 安装依赖 + 健康检查）：

```powershell
cd llm-inspector
.\start.bat              # 或手动：python backend/app/main.py
```

## 配置说明

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `APP_ENV` | `development` | 运行环境（development / production） |
| `PORT` | `8000` | 服务端口 |
| `DATABASE_URL` | `sqlite:///./llm_inspector.db` | 数据库连接 |
| `ENCRYPTION_KEY` | 自动生成（开发模式） | AES-256 加密密钥，**生产环境必须手动设置** |
| `API_KEY_TTL_HOURS` | `72` | API 密钥保留时长（小时） |
| `INTER_REQUEST_DELAY_MS` | `150` | 请求间隔（防限流） |
| `PREDETECT_CONFIDENCE_THRESHOLD` | `0.85` | 预检测置信度阈值 |
| `DEFAULT_REQUEST_TIMEOUT_SEC` | `60` | 请求超时时间 |
| `RAW_RESPONSE_TTL_DAYS` | `7` | 原始响应保留天数 |

生成加密密钥：

```bash
python3 -c "import secrets,base64; print(base64.b64encode(secrets.token_bytes(32)).decode())"
```

## API 接口

### 核心端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/health` | 健康检查 + Worker 状态 |
| POST | `/api/v1/runs` | 创建检测任务 |
| GET | `/api/v1/runs` | 列出所有检测任务 |
| GET | `/api/v1/runs/:id` | 获取任务状态与进度 |
| DELETE | `/api/v1/runs/:id` | 删除检测任务 |
| POST | `/api/v1/runs/:id/cancel` | 取消运行中的任务 |
| POST | `/api/v1/runs/:id/retry` | 重试失败的任务 |
| POST | `/api/v1/runs/:id/continue` | pre_detected 状态下继续执行全量测试 |
| POST | `/api/v1/runs/:id/skip-testing` | pre_detected 状态下跳过测试并直接生成报告 |
| GET | `/api/v1/runs/:id/report` | 获取分析报告 |
| GET | `/api/v1/runs/:id/responses` | 获取测试响应详情 |
| GET | `/api/v1/runs/:id/scorecard` | 获取 v2 评分卡（各维度分值，单位 0–10,000） |
| GET | `/api/v1/runs/:id/extraction-audit` | 获取提取审计报告（extraction 模式） |
| GET | `/api/v1/runs/:id/theta-report` | 获取 Rasch IRT theta 能力估计报告 |
| GET | `/api/v1/runs/:id/pairwise` | 获取 Bradley-Terry 两两对比结果 |
| GET | `/api/v1/baselines/:id` | 获取指定 Golden Baseline 详情 |
| GET | `/api/v1/baselines/:id/comparisons` | 获取该 Baseline 的历史对比记录 |
| GET | `/api/v1/runs/:id/report.csv` | 导出报告 CSV |
| GET | `/api/v1/runs/:id/radar.svg` | 导出雷达图 SVG |

### 比对与排行

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/compare-runs` | 创建 A/B 对比任务 |
| GET | `/api/v1/compare-runs` | 列出所有对比任务 |
| GET | `/api/v1/compare-runs/:id` | 获取对比结果 |
| POST | `/api/v1/calibration/replay` | 创建校准回放任务（批量评估历史 run） |
| GET | `/api/v1/calibration/replay` | 列出校准回放任务 |
| GET | `/api/v1/calibration/replay/:id` | 获取校准回放结果 |
| POST | `/api/v1/baselines` | 将已完成 run 标记为 Golden Baseline |
| GET | `/api/v1/baselines` | 列出 Golden Baseline（支持 model_name/active_only） |
| POST | `/api/v1/baselines/compare` | 将某次 run 与 Golden Baseline 对比 |
| DELETE | `/api/v1/baselines/:id` | 失活（软删除）Golden Baseline |
| GET | `/api/v1/benchmarks` | 列出基准模型（支持 suite_version 过滤） |
| GET | `/api/v1/models/:name/trend` | 模型历史评分趋势（0–10,000 单位） |
| GET | `/api/v1/models/:name/theta-trend` | 模型历史 theta 趋势 |
| GET | `/api/v1/leaderboard` | 模型排行榜（支持 sort_by/limit） |
| GET | `/api/v1/theta-leaderboard` | 基于 Rasch theta 的模型排行榜 |
| GET | `/api/v1/exports/runs.zip` | 批量导出多次 run（CSV + 雷达图 ZIP） |
| POST | `/api/v1/calibration/rebuild` | 重建校准参数（version 参数） |
| POST | `/api/v1/calibration/snapshot` | 快照当前校准状态 |
| POST | `/api/v1/tools/generate-isomorphic` | 生成同构题（apply=true 写入 suite_v2.json） |

### 创建检测任务示例

```bash
curl -X POST http://localhost:8000/api/v1/runs \
  -H "Content-Type: application/json" \
  -d '{
    "base_url": "https://api.example.com/v1",
    "api_key": "sk-xxx",
    "model": "gpt-4o"
  }'
```

## 检测流程

```
HTTP Handler → Repository → Worker(ThreadPool)
  → Orchestrator: PreDetect(7层) → CaseExecutor → Judge(21种) → Analysis Pipeline
```

### 预检测 7 层管道

| 层级 | 名称 | 探测内容 |
|------|------|----------|
| 0 | HTTP Header | 响应头信息指纹 |
| 1 | Self-report | 模型自我身份报告 |
| 2 | Identity Probe | 多探针交叉验证身份 |
| 3 | Knowledge Cutoff | 知识截止日期 |
| 4 | Bias/Format | 偏好与格式分析 |
| 5 | Tokenizer FP | Tokenizer 类型指纹 |
| 6 | Active Extraction | **主动提示词提取 + 探针混淆（仅 extraction 模式）** |
| 6b | Multi-turn Overload | **多轮上下文过载攻击（仅 extraction 模式）** |

从 0 token 递增至 ~500 token，置信度 ≥0.85 时提前停止。

### v2 三维评分体系

```
TotalScore = 0.45 × Capability + 0.35 × Authenticity + 0.20 × Performance

Capability  = 0.25×reasoning + 0.25×instruction + 0.20×coding
              + 0.15×safety + 0.15×protocol
Authenticity = 0.40×similarity + 0.25×predetect + 0.15×consistency
               + 0.10×temp + 0.10×usage
Performance  = 0.40×speed + 0.30×stability + 0.30×cost_efficiency
```

> **分值单位说明**：内部计算使用 0.0–1.0 浮点数；API 响应（`/scorecard`、`/leaderboard`、`/trend`、baseline 端点）统一乘以 100 后以整数输出，范围 **0–10,000**。信任判定阈值（`trusted`/`suspicious` 等）基于 0–100 百分制（即内部值 × 100）。

**各 API 端点分值输出格式汇总**：

| 端点 | 输出分值范围 | 格式说明 |
|------|-------------|----------|
| `GET /runs/:id/scorecard` | 0–10,000 | 整数，各维度独立缩放 |
| `GET /runs/:id/report` | 0–100 | 百分制，用于 trust 判定 |
| `GET /leaderboard` | 0–10,000 | 整数 |
| `GET /models/:name/trend` | 0–10,000 | 整数 |
| `POST /baselines/compare` (`score_delta`) | −10,000–+10,000 | 带符号整数 |

### Golden Baseline

Golden Baseline 用于将某次检测 run 标记为"标准答案"，后续可与新 run 对比以判断是否存在显著漂移。

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/baselines` | 将已完成 run 标记为 Golden Baseline |
| GET | `/api/v1/baselines` | 列出 Golden Baseline（支持 `model_name` / `active_only` 过滤） |
| GET | `/api/v1/baselines/:id` | 获取指定 Golden Baseline 详情 |
| GET | `/api/v1/baselines/:id/comparisons` | 获取该 Baseline 的历史对比记录 |
| POST | `/api/v1/baselines/compare` | 将某次 run 与 Golden Baseline 对比 |
| DELETE | `/api/v1/baselines/:id` | 失活（软删除）Golden Baseline |

**Baseline 对比返回字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `verdict` | string | 对比判定：`match` / `suspicious` / `mismatch` |
| `cosine_similarity` | float | 特征向量余弦相似度（0–1） |
| `score_delta` | float | 总分之差（被比较方 - Baseline），单位 0–10,000 |
| `feature_drift_top5` | array | 漂移最大的 5 个特征，格式 `[{dimension, baseline_val, run_val, delta}]` |

### 信任判定等级

| 评分条件 | 等级 | 含义 |
|----------|------|------|
| auth≥85 且 score≥75 | `trusted` | 可信 |
| auth≥70 或 score≥65 | `suspicious` | 可疑 |
| auth≥50 或 score≥45 | `high_risk` | 高风险 |
| 低于上述阈值 | `fake` | 疑似套壳 |

## 降本增效优化（v3 精简套件）

v3 精简套件通过以下优化实现 **~57% token 节省**：

| 优化项 | 预计节省 Token | 说明 |
|--------|--------------|------|
| 用例精简（64→37） | ~8,000-12,000 | 删除对主流模型区分度低的用例 |
| 自适应采样数 | ~3,000-5,000 | 确定性 judge 仅采样 1 次 |
| 调低 per-judge token ceiling | ~2,000-4,000 | 约束推理从 1200→700 |
| 更激进的 early-stop | ~5,000-8,000 | Standard 相似度阈值 0.88→0.85 |
| 预检测层级跳过 | ~100-300 | 高置信时跳过 Layer 3-5 |
| Bootstrap 降采样 | 计算时间减半 | 50-200 次自适应 |
| Smart 模式 | ~10,000-20,000 | 动态调整测试规模 |

**精度保障**：不可删除的安全底线用例包括 `identity_consistency`、`antispoof_override`、`instruction_exact_match`、`refusal_harmful`、`reasoning_constraint`、`system_obedience`、`consistency_temperature`、`extraction_dump_verbatim`。

## 检测模式

| 模式 | 说明 |
|------|------|
| `quick` | 快速检测，跳过风格测试，支持提前终止 |
| `standard` | 标准检测，运行完整测试套件 |
| `smart` | **智能检测模式**：根据预检测置信度动态调整测试规模，预检测高置信时仅需 6 个验证用例，预计节省 ~57% token |
| `full` | 全量检测，所有测试用例 + 多次采样 |
| `extraction` | **提取审计模式：主动提示词提取 + 身份压力测试 + 行为指纹对比** |

### Smart 模式详解

Smart 模式根据预检测结果动态决定测试规模：

| 预检测置信度 | Token 预算 | Phase1 用例数 | Phase2/3 用例数 | 场景 |
|-------------|-----------|-------------|----------------|------|
| ≥ 0.90 | 8,000 | 6 | 0 | 高置信验证模式 |
| 0.70 - 0.90 | 15,000 | 10 | 4 | 定向鉴别模式 |
| 0.50 - 0.70 | 25,000 | 12 | 8 + 4 | 标准检测模式 |
| < 0.50 | 35,000 | 14 | 10 + 6 | 全面检测模式 |

**加权平均节省约 57% token**（约 60% 的检测在预检测阶段就能达到 0.70+ 置信度）。

## 提取审计模式（新增）

`extraction` 模式是专门针对套壳检测的主动攻击模式，在标准行为测试的基础上新增：

### Layer 6：主动提取（Active Extraction）

向目标 API 发送 10 条梯度递增的探针，通过**探针混淆**（同一语义生成多个低特征变体）绕过网关过滤：

| 探针类型 | 攻击目标 |
|----------|----------|
| 直接输出 | 提取 system prompt 全文 |
| 调试模式 | 触发系统配置输出 |
| 禁词列表 | 暴露被屏蔽的真实模型名 |
| 文件路径 | 泄露代理服务目录结构 |
| 工具配置 | 暴露代理平台工具特征 |
| 身份压力 | 突破 persona 伪装层 |

### Layer 6b：多轮上下文过载（Multi-turn Overload）

填充 15-20 轮无关技术问答使模型"忘记"身份伪装指令，然后突然询问原始配置（指令飘移攻击）。

### 提取审计报告

检测完成后自动生成 `extraction_audit` 节，包含：
- **真实模型暴露**：是否检测到底层模型名称
- **禁词列表泄露**：禁词中是否出现竞争对手模型名
- **规格矛盾**：自报上下文窗口与声称模型是否匹配
- **文件路径泄露**：是否暴露代理服务路径
- **语言偏向检测**：英文 prompt 下的非预期中文输出
- **证据链**：所有信号的完整追溯记录

### TTFT 代理延迟分析

自动分析 First Token Time 的分布特征，通过双峰检测、IQR 离散度、基线偏差推断是否存在二次转发代理层。

```bash
# 使用 extraction 模式创建检测任务
curl -X POST http://localhost:8000/api/v1/runs \
  -H "Content-Type: application/json" \
  -d '{"base_url": "https://api.example.com/v1", "api_key": "sk-xxx", "model": "gpt-4o", "test_mode": "extraction"}'

# 获取提取审计报告
GET /api/v1/runs/{run_id}/extraction-audit
```

## 测试类别（v2 · 33 用例 · 11 类别）

| 类别 | 用例数 | 说明 |
|------|--------|------|
| Protocol | 2 | 基础聊天、usage 字段 |
| Instruction | 5 | 格式约束、语言切换、字数限制 |
| System | 4 | 系统提示遵从、角色扮演 |
| Reasoning | 13 | 数学推理、逻辑问题、约束优先推理（糖果/烧绳/试毒同构题） |
| Coding | 3 | 代码生成与执行 |
| Consistency | 2 | 多次采样一致性 |
| Antispoof | 3 | 身份探测、矛盾检测 |
| Parameter | 3 | temperature、max_tokens 行为 |
| Style | 3 | Markdown 偏好、格式习惯 |
| Refusal | 2 | 敏感内容处理、拒绝措辞 |
| Performance | 2 | 响应速度、吞吐量 |

### 11 种基础判定方法

`exact_match` · `regex_match` · `json_schema` · `line_count` · `refusal_detect` · `heuristic_style` · `code_execution` · `identity_consistency` · `any_text` · `constraint_reasoning` · `text_constraints`

### 10 种提取审计专用判定方法（新增）

`prompt_leak_detect` · `forbidden_word_extract` · `path_leak_detect` · `tool_config_leak_detect` · `memory_leak_detect` · `denial_pattern_detect` · `spec_contradiction_check` · `refusal_style_fingerprint` · `language_bias_detect` · `tokenizer_fingerprint`

新增说明：
- `constraint_reasoning`：约束优先判定，重点检查“关键约束命中 + 边界证明信号 + 反模板误触发”，不以思维链长度作为加分项。
- `text_constraints`：统一字符计数规则（去空白与常见标点后计数）+ 禁字校验，避免平台计数口径不一致导致误判。

## 基准模型库

内置 **108 个基准模型画像**，覆盖 20+ 厂商：

| 厂商 | 代表模型 |
|------|----------|
| OpenAI | GPT-4o, GPT-4.1, GPT-5/5.1/5.3/5.4, o1/o3/o4-mini, Codex-mini |
| Anthropic | Claude 3/3.5/4/4.5/4.6 全系列 |
| Google | Gemini 1.0/1.5/2.0/2.5/3.0, Gemma 2/3 |
| Meta | LLaMA 2/3/3.1/3.2/3.3/4 |
| DeepSeek | V2/V2.5/V3, R1, Coder-V2 |
| Alibaba | Qwen 2/2.5/3, Qwen-Turbo/Max |
| Mistral | Large/Medium/Small, Mixtral, Codestral, Pixtral |
| Zhipu | GLM-4/4-Plus/4-Flash |
| 01.AI | Yi-Large/Lightning/34B |
| 更多 | Baichuan, ERNIE, Spark, Hunyuan, Doubao, Moonshot, MiniMax, StepFun, Cohere, Phi, DBRX, Jamba, Falcon, InternLM |

## 安全特性

- API 密钥使用 AES-256-GCM 静态加密存储
- SSRF 防护：阻止对内网 / 私有 IP 地址的请求
- 可配置 CORS 策略
- API 密钥自动过期清理

## 工具脚本（新增）

### 1) 同构题批量生成

```bash
cd llm-inspector
PYTHONPATH=backend python backend/tools/generate_isomorphic_cases.py --preview
PYTHONPATH=backend python backend/tools/generate_isomorphic_cases.py --apply
```

- `--preview`：预览将新增的同构题，不写入文件
- `--apply`：写入 `backend/app/fixtures/suite_v2.json`

### 2) 导出单次检测 CSV

```bash
cd llm-inspector
PYTHONPATH=backend python backend/tools/export_run_report.py --run-id <RUN_ID> --out backend/output/<RUN_ID>.csv
```

导出内容包含：总分、三大主分、关键子分、判定等级。

### 3) 导出雷达图（SVG）

```bash
cd llm-inspector
PYTHONPATH=backend python backend/tools/export_radar_svg.py --run-id <RUN_ID> --out backend/output/<RUN_ID>-radar.svg
```

> 采用 SVG 输出（无第三方依赖），可直接用浏览器打开或后续转 PNG。

### 4) 运行校准回放（新增）

```bash
cd llm-inspector
PYTHONPATH=backend python backend/tools/run_calibration.py \
  --cases backend/app/fixtures/calibration/cases.json \
  --out-json backend/output/calibration-result.json \
  --out-csv backend/output/calibration-result.csv
```

- `--skip-submit`：仅评估 cases 文件中的既有 `run_id`，不新建检测任务
- 输出包含：分类准确率、macro precision/recall/F1、混淆矩阵、逐样本对照

## API 直连导出（新增）

### 1) 导出报告 CSV

```bash
GET /api/v1/runs/{run_id}/report.csv
```

### 2) 导出雷达图 SVG

```bash
GET /api/v1/runs/{run_id}/radar.svg
```

### 3) 触发同构题生成

```bash
POST /api/v1/tools/generate-isomorphic?apply=false   # 预览
POST /api/v1/tools/generate-isomorphic?apply=true    # 写入 suite_v2.json
```

## 前端一键入口（新增）

- 首页新增“题库维护工具”卡片，可一键：
  - 预览同构题增量
  - 写入同构题
- 历史记录页新增：
  - 状态筛选 + 模型关键字筛选
  - 分页浏览（每页 10 条）
  - 每条 run 快速导出（CSV / 雷达图）
  - 勾选后批量导出 ZIP（CSV / 雷达图 / CSV+雷达图）
- 任务详情页在完成后新增：
  - 导出 CSV
  - 导出雷达图（SVG）
  - 内嵌雷达图预览（iframe 直连 `/radar.svg`）

## 运行测试

```bash
cd llm-inspector
PYTHONPATH=backend python backend/tests/test_all.py
```

自制测试框架（非 pytest），覆盖 config / security / db / seeder / judge / analysis / repo / predetect / executor / http 共 ~60+ 用例。

## 许可证

本项目仅供学习和安全研究使用。
