# LLM Inspector v9.0

LLM 套壳检测与能力评估工具 —— 检测 LLM API 是否真正提供其声称的模型服务，还是在背后代理/包装了其他模型。

**v6.0 重大升级**：
- **彻底清除假数据**：移除所有硬编码假数据（GLOBAL_FEATURE_MEANS等），实现100%数据驱动
- **关键Bug修复**：修复8个影响准确性的核心bug（导入错误、逻辑反转、词边界匹配等）
- **评分体系重建**：基于IRT区分度的动态权重、Safety评分激励修正、性能基准线动态化
- **套壳检测增强**：新增5种检测方法（行为一致性差分、Token计费异常、响应指纹多样性等）
- **相似度引擎重构**：特征归一化一致性、Bootstrap置信区间优化、数据驱动特征重要性
- **预检测优化**：CUTOFF_MAP动态化、贝叶斯置信度合并、身份探针防诱导
- **测试用例优化**：删除低区分力用例、自适应采样、Token预算预估、快速模式早停
- **代码架构重构**：pipeline.py模块化拆分（4个新模块）、死代码清理、导入优化
- **安全加固升级**：输入长度验证、文件名安全处理、XSS防护完善
- **前端体验提升**：相似度排名可视化、置信度分级显示、移动端适配
- **测试覆盖完善**：新增20+测试用例、单元测试+集成测试全覆盖

## 工作原理

LLM Inspector 通过多层行为指纹分析，识别 API 背后的真实模型：

1. **预检测（Pre-detection）**：8 层渐进式指纹识别管道（HTTP 头信息、自我报告、身份探测、知识截止日期、偏好/格式分析、Tokenizer 指纹、主动提示词提取、多轮上下文过载），以最小 token 消耗快速锁定候选模型
2. **全量测试**：suite_v3 精简套件 70 个测试用例（16 分类维度），覆盖协议合规、指令遵循、推理、编码、反欺骗、身份检测等维度
3. **特征提取与比对**：从响应中提取 40 维行为特征向量，与用户标记的 Golden Baseline 进行余弦相似度比对 + Bootstrap 置信区间估算
4. **v3 三维评分**：Capability（能力）+ Authenticity（真实性）+ Performance（性能）三维度评分卡
5. **信任判定**：输出 `trusted` / `suspicious` / `high_risk` / `fake` 四级信任判定
6. **A/B 对比**：支持两次检测结果的显著性差异分析

## 技术栈

- **后端**：Python 3.12+，零外部框架依赖（纯标准库实现 HTTP 服务器、数据库、任务队列）
- **外部依赖**：`cryptography`, `numpy`, `scikit-learn`, `pytest`
- **前端**：多文件 SPA（index.html + styles.css + app.js），纯 HTML/CSS/JS，无构建步骤
- **数据库**：SQLite（WAL 模式），11 张表，线程局部连接
- **任务队列**：ThreadPoolExecutor（4 workers），支持 Celery 分布式扩展

## 项目结构

```
llm-inspector/
├── start.bat                      # Windows 一键启动
├── stop.bat                       # Windows 停止服务
├── backend/
│   ├── app/
│   │   ├── main.py                # HTTP 服务器入口（37 个 API 端点）
│   │   ├── core/
│   │   │   ├── config.py          # 环境配置
│   │   │   ├── db.py              # SQLite 数据库层（11 张表）
│   │   │   ├── schemas.py         # 数据模型（dataclasses）
│   │   │   ├── security.py        # 加密 + SSRF 防护
│   │   │   └── logging.py        # 结构化 JSON 日志
│   │   ├── adapters/
│   │   │   └── openai_compat.py  # OpenAI 兼容 API 适配器
│   │   ├── predetect/
│   │   │   └── pipeline.py       # 8 层预检测管道
│   │   ├── runner/
│   │   │   ├── orchestrator.py   # 测试编排器
│   │   │   └── case_executor.py  # 单用例执行器
│   │   ├── judge/
│   │   │   └── methods.py        # 28 种判定方法
│   │   ├── analysis/
│   │   │   ├── pipeline.py       # 原始管道（保持兼容）
│   │   │   ├── feature_extractor.py  # V6: 特征提取模块
│   │   │   ├── score_calculator.py    # V6: 评分计算模块
│   │   │   ├── similarity_engine.py   # V6: 相似度引擎模块
│   │   │   ├── verdict_engine.py      # V6: 判决引擎模块
│   │   │   └── pipeline_new.py        # V6: 重构后的主管道
│   │   ├── repository/
│   │   │   └── repo.py           # 数据库 CRUD
│   │   ├── tasks/
│   │   │   ├── worker.py         # 线程池后台 Worker
│   │   │   └── seeder.py         # 数据库初始化种子
│   │   └── fixtures/
│   │       ├── suite_v3.json     # 70 个测试用例（v3）
│   │       ├── cutoff_map.json   # V6: 动态CUTOFF_MAP配置
│   │       └── tokenizer_probes.json  # V6: Tokenizer探针数据
│   ├── scripts/
│   │   └── verify_tokenizer_probes.py  # V6: Tokenizer探针验证脚本
│   └── tests/
│       └── test_all.py           # 测试套件（V6扩展至70+用例）
└── frontend/
    ├── index.html                 # 前端入口
    ├── styles.css                 # 样式文件
    └── app.js                     # 前端逻辑
```

## 快速开始

### 环境要求

- Python 3.12+
- `cryptography>=41`
- `numpy>=1.24`
- `scikit-learn>=1.3`
- `pytest>=7`

### 安装与运行

```bash
cd llm-inspector

# 创建并激活虚拟环境
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# 安装依赖
pip install cryptography numpy scikit-learn pytest

# 启动服务
cd backend
python -m app.main
```

服务默认运行在 `http://localhost:8000`。

**Windows 快捷启动**：直接运行 `start.bat`。

**运行测试**：

```bash
pytest backend/tests/test_all.py
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
| `CELERY_BROKER_URL` | 空 | Celery 分布式模式（可选） |

### Token 预算配置（v3 新增）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TOKEN_BUDGET_QUICK` | `15000` | 快速扫描模式 token 预算 |
| `TOKEN_BUDGET_STANDARD` | `40000` | 标准评测模式 token 预算 |
| `TOKEN_BUDGET_DEEP` | `100000` | 深度审计模式 token 预算 |

### 语义评判配置（v3 新增，可选）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `JUDGE_API_URL` | 空 | 评判模型 API 地址（如 https://api.openai.com/v1） |
| `JUDGE_API_KEY` | 空 | API Key |
| `JUDGE_MODEL` | `gpt-4o-mini` | 推荐用低成本模型做评判 |
| `JUDGE_TIMEOUT` | `15` | 单次评判超时（秒） |

生成加密密钥：

```bash
python3 -c "import secrets,base64; print(base64.b64encode(secrets.token_bytes(32)).decode())"
```

## 检测模式（v3）

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

## 测试套件（suite_v3）

- **78 个测试用例**，覆盖 16 个分类：
  - protocol(2), instruction(8), system(2), param(2), reasoning(19), coding(8)
  - refusal(2), style(2), consistency(3), antispoof(3), extraction(9)
  - fingerprint(2), tool_use(3), performance(1), knowledge(7), safety(1)
- **v4.0 新增**：4个hallucination检测用例，覆盖虚假学术论文、虚假技术、虚假历史事件、双虚假实体等场景
- **梯度难度**：difficulty 0.3→0.95，含链式推理(3/5/8步)、数学竞赛、DP 算法
- **推理多样性**：空间推理、因果推理、认知陷阱（幸存者偏差）、多步链式推理
- **编码多样性**：函数编写、Bug 调试修复、代码重构、动态规划算法
- **知识与幻觉**：基础事实、时效性知识（区分模型代际）、虚构实体幻觉检测
- **模型指纹探测**：tokenizer 指纹、拒绝风格指纹、行为一致性
- **对抗性提取**：上下文溢出攻击、角色扮演绕过、渐进式提取
- **Tool Use 测试**：工具调用、工具选择、无需工具场景

## 预检测 8 层管道

| 层级 | 名称 | 探测内容 |
|------|------|----------|
| 0 | HTTP Header | 响应头信息指纹 |
| 1 | Self-report | 模型自我身份报告 |
| 2 | Identity Probe | 多探针交叉验证身份 |
| 3 | Knowledge Cutoff | 知识截止日期 |
| 4 | Bias/Format | 偏好与格式分析 |
| 5 | Tokenizer FP | Tokenizer 类型指纹 |
| 6 | Active Extraction | 主动提示词提取（仅 Deep 模式） |
| 7 | Multi-turn Overload | 多轮上下文过载攻击（仅 Deep 模式） |

从 0 token 递增至 ~5000 token，置信度 ≥0.85 时提前停止。

## Judge 系统（28 种方法）

### 规则评判（24 种）

`exact_match` · `regex_match` · `json_schema` · `line_count` · `constraint_reasoning` · `code_execution` · `text_constraints` · `identity_consistency` · `refusal_detect` · `heuristic_style` · `prompt_leak_detect` · `forbidden_word_extract` · `path_leak_detect` · `tool_config_leak_detect` · `memory_leak_detect` · `denial_pattern_detect` · `spec_contradiction_check` · `refusal_style_fingerprint` · `language_bias_detect` · `tokenizer_fingerprint` · `difficulty_ceiling` · `token_fingerprint` · `tool_call_judge` · `any_text`

### v3 新增（4 种）

`yaml_csv_validate` · `hallucination_detect` · `multi_step_verify` · `context_overflow_detect`

### 语义评判

`semantic_judge` 支持 LLM-as-Judge（配置 `JUDGE_API_URL` 后调用外部 LLM），未配置时降级为本地关键词匹配。

## v3 三维评分体系

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

**分值单位说明**：内部计算使用 0.0–1.0 浮点数；API 响应统一乘以 100 后以整数输出，范围 **0–10,000**。

## VerdictEngine 硬规则

- 声称顶级模型但 difficulty_ceiling < 0.4 → 强制降到 50 分
- behavioral_invariant_score < 40 → 强制降到 55 分
- 声称顶级模型但 coding_score < 10 → 强制降到 45 分
- adversarial_spoof_signal_rate > 0.5 → 强制降到 45 分
- **v3 新增**：提取攻击泄露真实模型身份 → 强制降到 30 分
- **v3 新增**：tokenizer/行为指纹与声称模型不符 → 强制降到 55 分

## 信任判定等级

| 评分条件 | 等级 | 含义 |
|----------|------|------|
| auth≥85 且 score≥75 | `trusted` | 可信 |
| auth≥70 或 score≥65 | `suspicious` | 可疑 |
| auth≥50 或 score≥45 | `high_risk` | 高风险 |
| 低于上述阈值 | `fake` | 疑似套壳 |

## Golden Baseline

Golden Baseline 用于将某次检测 run 标记为"标准答案"，后续可与新 run 对比以判断是否存在显著漂移。

- **基准模型 = 真实数据**：用户将已完成的检测标记为基准，以 model_name 为唯一索引
- **相似度比对只使用基准模型**（golden_baselines 表），不使用任何估算/假数据
- 标记基准时用户可自定义模型名，同名基准自动覆盖

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
| GET | `/api/v1/runs/:id/scorecard` | 获取 v3 评分卡 |
| GET | `/api/v1/runs/:id/radar.svg` | 导出9维度雷达图 SVG |

### Baseline 与比对

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/baselines` | 将已完成 run 标记为 Golden Baseline |
| GET | `/api/v1/baselines` | 列出 Golden Baseline |
| GET | `/api/v1/baselines/:id` | 获取指定 Golden Baseline 详情 |
| POST | `/api/v1/baselines/compare` | 将某次 run 与 Golden Baseline 对比 |
| DELETE | `/api/v1/baselines/:id` | 失活（软删除）Golden Baseline |

### 排行榜

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/leaderboard` | 模型排行榜 |
| GET | `/api/v1/models/:name/trend` | 模型历史评分趋势 |
| GET | `/api/v1/runs/:id/pairwise` | Bradley-Terry 两两对比结果 |

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

指定测试模式：

```bash
curl -X POST http://localhost:8000/api/v1/runs \
  -H "Content-Type: application/json" \
  -d '{
    "base_url": "https://api.example.com/v1",
    "api_key": "sk-xxx",
    "model": "gpt-4o",
    "test_mode": "deep"
  }'
```

## 任务队列架构

- **本地模式**：ThreadPoolExecutor（默认，零配置）
- **分布式模式**：Celery + Redis（设置 `CELERY_BROKER_URL` 环境变量）
- **抽象接口**：`app.tasks.queue.TaskQueue`
- **切换方式**：`from app.tasks.worker import init_distributed_queue; init_distributed_queue()`

## 安全特性

- API 密钥使用 AES-256-GCM 静态加密存储
- SSRF 防护：阻止对内网 / 私有 IP 地址的请求
- 可配置 CORS 策略
- API 密钥自动过期清理
- dev 模式下 ENCRYPTION_KEY 自动生成确定性密钥，生产环境必须手动设置

## Fingerprint 系统

- **Tokenizer 覆盖（8 种）**：tiktoken-cl100k, tiktoken-o200k, claude, llama-spm, deepseek, qwen, chatglm, yi
- **探针词（5 个）**：multimodality, cryptocurrency, hallucination, supercalifragilistic, counterintuitive
- **身份探针（5 个）**：直接探针(3) + 间接探针(1) + 矛盾诱导探针(1)
- **行为指纹库（9 个家族）**：Claude, GPT-4, GPT-4o, DeepSeek, Qwen, LLaMA, Gemini, MiniMax, GLM

## 代码规范

- `from __future__ import annotations` 在文件顶部
- 标准库 import 在前，项目 import 用绝对路径 `from app.core.xxx`
- 命名：文件 `snake_case`，类 `PascalCase`，私有 `_prefix`，常量 `UPPER_SNAKE`
- Handler 函数：`handle_` 前缀
- 日志：`logger = get_logger(__name__)` + structlog 风格关键字参数
- 代码分区：`# -- Section Name` 注释
- 类型注解：Python 3.10+ 风格（`str | None`）

## 数据库

- SQLite WAL 模式，11 张表，线程局部连接
- `test_runs` 表的 `evaluation_mode`, `calibration_case_id`, `scoring_profile_version`, `calibration_tag` 已迁移为独立列
- 提供迁移函数 `migrate_json_columns_to_columns()`

## Gotchas

- dev 模式下 ENCRYPTION_KEY 自动生成确定性密钥，生产环境必须手动设置
- 前端文件已拆分为 index.html + styles.css + app.js，修改时注意保持引用关系
- 相似度比对只使用 golden_baselines 中用户标记的基准模型，没有基准时相似度列表为空
- v3 模式向后兼容：`full`→`deep`、`extraction`→`deep` 自动映射
- SemanticJudge 需要配置 `JUDGE_API_URL` 才会调用外部 LLM，否则降级为本地规则评判

## 更新日志

### v5.0 (2026-04-10)

**核心架构升级** —— 从规则驱动到数据驱动，从静态阈值到自适应校准：

**判题系统智能化 (P0)**：
- 🧠 **语义判题引擎 2.0**：引入本地嵌入模型（BGE-large-zh）+ LLM-as-Judge 结构化评分，三层级联评判（快速过滤→结构化评分→一致性校验）
- 🔮 **幻觉检测增强**：Uncertainty-Aware 检测，量化不确定性表达，识别虚假确定性
- 🛡️ **代码判题沙箱**：多语言支持（Python/JavaScript/C++），资源限制与隔离

**评分体系自适应 (P1)**：
- 📊 **贝叶斯权重校准**：基于历史检测数据自动优化评分权重，最大化真实模型区分度（AUC）
- 📈 **评分置信度量化**：Bootstrap 方法计算 95% 置信区间，提供可靠性等级评估

**测试用例 IRT 校准 (P2)**：
- 🎯 **2PL 模型实现**：区分度（a）+ 难度（b）双参数校准，信息量函数指导选题
- 🔄 **自适应选题算法**：基于当前能力估计（theta）选择最大信息量题目

**相似度引擎增强 (P3)**：
- 🔬 **对比学习特征提取**：神经网络编码行为特征，生成 128 维行为嵌入
- 🎭 **多模态相似度融合**：融合行为向量、神经嵌入、响应风格、时间指纹多维信号

**预检测信号融合 (P4)**：
- 🌐 **贝叶斯网络融合**：多层信号联合推断，后验概率计算，矛盾信号检测
- 🔍 **一致性分析**：识别模型混合/路由场景，检测层间信号冲突

**Bug修复与优化**：
- 🐛 **修复 runner 导入错误**：移除 `__init__.py` 中对不存在类的导入
- 🌏 **前端全面中文化**：补充 20+ 维度名称中文映射，修复基准对比报告英文标签

---

### v6.0 (2026-04-11) - 重大升级

**🎯 核心目标达成：100%数据驱动，彻底消除假数据**

**🧹 数据质量革命**：
- ✅ **彻底清除假数据**：移除`GLOBAL_FEATURE_MEANS`等所有硬编码假数据，实现100%真实数据驱动
- ✅ **动态归一化参数**：评分归一化参数来自golden_baselines实际统计，支持运行时更新
- ✅ **数据驱动权重**：评分权重基于IRT区分度参数动态计算，告别人工估算

**🐛 关键Bug修复**：
- ✅ **修复8个核心bug**：reports.py导入错误、constraint_reasoning默认值、bootstrap CI逻辑反转、identity_consistency词边界匹配、hallucination单实体检测、code_execution浮点比较、前端XSS防护
- ✅ **逻辑错误修正**：确保所有评分逻辑正确性，提升检测准确率

**⚖️ 评分体系重建**：
- ✅ **IRT区分度权重**：基于项目反应理论计算动态权重，高区分度维度获得更高权重
- ✅ **Safety评分修正**：重新设计激励结构，优先拒绝有害内容（50分），避免过度拒绝惩罚
- ✅ **性能基准线动态化**：从golden_baselines加载延迟基准，支持不同模型类别的差异化评估
- ✅ **缺失数据处理**：无数据维度返回None，权重重新归一化，避免虚假评分

**🛡️ 套壳检测增强**：
- ✅ **行为一致性差分测试**：通过同构题换皮检测模板匹配行为
- ✅ **Token计费异常检测**：识别代理服务的异常计费模式（整数倍、零补全、完全一致）
- ✅ **响应指纹多样性**：temperature=0下的响应一致性检测，识别非确定性包装
- ✅ **强化提取攻击**：新增高级提取探针，提升身份识别能力

**📊 相似度引擎重构**：
- ✅ **特征归一化一致性**：统一归一化逻辑，支持动态参数配置
- ✅ **Bootstrap置信区间优化**：高相似度使用更多迭代（200次），提升CI精度
- ✅ **最小特征数阈值降低**：从8降至5，支持快速模式的相似度计算
- ✅ **数据驱动特征重要性**：基于基准标准差计算特征权重，替代硬编码值

**🔍 预检测优化**：
- ✅ **CUTOFF_MAP动态化**：从外部JSON文件加载，支持运行时更新
- ✅ **贝叶斯置信度合并**：使用`1 - product(1 - conf_i)`公式合并多层置信度
- ✅ **身份探针防诱导**：替换误导性探针，避免误判拒绝行为为身份否认
- ✅ **Tokenizer探针验证**：新增验证脚本，确保探针token数准确性

**🎯 测试用例优化**：
- ✅ **低区分力用例删除**：基于IRT分析删除区分度低的测试用例
- ✅ **自适应采样**：根据判题方法确定性和历史方差调整采样次数
- ✅ **Token预算预估**：基于历史中位数预估token消耗，支持20%缓冲
- ✅ **快速模式早停**：失败率>80%时提前终止，节省资源

**🏗️ 代码架构重构**：
- ✅ **模块化拆分**：pipeline.py拆分为4个专门模块（feature_extractor、score_calculator、similarity_engine、verdict_engine）
- ✅ **死代码清理**：移除重复的semantic.py实现，保留semantic_v2.py
- ✅ **导入优化**：清理无用导入，提升代码可维护性

**🔒 安全加固升级**：
- ✅ **输入长度验证**：model_name≤100字符，base_url≤500字符，防止存储型XSS
- ✅ **文件名安全处理**：run_id格式验证，防止路径遍历攻击
- ✅ **批量操作限制**：单次最多删除100条记录，防止滥用
- ✅ **XSS防护完善**：前端事件委托替换内联事件处理器

**📱 前端体验提升**：
- ✅ **相似度排名可视化**：添加置信度分级、数据来源标注、排名解读说明
- ✅ **置信度可视化**：高/中/低置信度分级显示，颜色编码
- ✅ **移动端适配**：响应式设计，支持小屏幕设备
- ✅ **数据不足提示**：优雅处理缺失数据场景

**🧪 测试覆盖完善**：
- ✅ **20+新增测试用例**：覆盖输入校验、Judge边界、评分边界、相似度引擎
- ✅ **集成测试**：完整流程测试（创建run→预检测→执行→评分→报告）
- ✅ **基准对比测试**：标记基准→对比→相似度查看全流程测试

**🔧 剩余细节修复**：
- ✅ **code execution timeout配置化**：支持通过params自定义超时时间
- ✅ **VerdictEngine硬规则注释**：为所有阈值添加来源说明
- ✅ **TOP_MODELS动态获取**：从golden_baselines动态加载高性能模型列表

**向后兼容性**：
- ✅ 所有API接口保持不变
- ✅ 原始pipeline.py保留，确保现有代码正常运行
- ✅ 数据库结构无变更，现有数据完全兼容

---

### v4.0 (2026-04-10)

**新增功能**：
- 🎯 **9维度雷达图**：从原来的6维度扩展到9维度，新增"编码"、"安全性"、"抗提取"、"响应速度"等维度，提供更全面的模型能力画像
- 📋 **扩展测试用例**：新增4个hallucination检测用例，覆盖虚假学术论文、虚假技术、虚假历史事件、双虚假实体等复杂场景

**优化改进**：
- 🛡️ **异常处理补全**：为case_executor和case_executor_async添加完整的异常处理机制，提升系统稳定性和错误恢复能力
- 🔍 **Layer5匹配优化**：修正tokenizer家族匹配逻辑，使用词边界匹配避免误识别（如"Meta"不再匹配到"metaphor"）

**技术细节**：
- 雷达图SVG导出工具同步升级为9维度显示
- 异常处理涵盖网络错误、超时、API错误等各种故障场景
- 优化后的匹配逻辑提高了模型识别的准确性

**向后兼容**：
- 所有API接口保持不变
- 现有检测数据完全兼容
- 前端界面自动适配新的9维度雷达图

### v3.0

- 三维评分体系（Capability + Authenticity + Performance）
- 8层预检测管道
- 70个测试用例覆盖16个维度
- Golden Baseline相似度比对
- ELO排行榜系统
