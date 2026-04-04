# LLM Inspector v1.0

LLM 套壳检测工具 —— 检测 LLM API 是否真正提供其声称的模型服务，还是在背后代理/包装了其他模型。

## 工作原理

LLM Inspector 通过多层行为指纹分析，识别 API 背后的真实模型：

1. **预检测（Pre-detection）**：5 层渐进式指纹识别管道（HTTP 头信息、自我报告、身份探测、知识截止日期、偏好/格式分析），以最小 token 消耗快速锁定候选模型
2. **全量测试**：v1 套件 19 个用例（6 类别）+ v2 套件 33 个用例（11 类别），覆盖协议合规、指令遵循、推理、编码、反欺骗等维度
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
│   │   ├── main.py                # HTTP 服务器入口（16 个 API 端点）
│   │   ├── core/
│   │   │   ├── config.py          # 环境配置
│   │   │   ├── db.py              # SQLite 数据库层（12 张表）
│   │   │   ├── schemas.py         # 数据模型（dataclasses）
│   │   │   ├── security.py        # 加密 + SSRF 防护
│   │   │   └── logging.py         # 结构化 JSON 日志
│   │   ├── adapters/
│   │   │   └── openai_compat.py   # OpenAI 兼容 API 适配器
│   │   ├── predetect/
│   │   │   └── pipeline.py        # 5 层预检测管道
│   │   ├── runner/
│   │   │   ├── orchestrator.py    # 测试编排器
│   │   │   └── case_executor.py   # 单用例执行器
│   │   ├── judge/
│   │   │   └── methods.py         # 9 种判定方法
│   │   ├── analysis/
│   │   │   └── pipeline.py        # 特征提取 + v2 评分 + 报告生成
│   │   ├── repository/
│   │   │   └── repo.py            # 数据库 CRUD
│   │   ├── tasks/
│   │   │   ├── worker.py          # 线程池后台 Worker（4 线程）
│   │   │   └── seeder.py          # 数据库初始化种子
│   │   └── fixtures/
│   │       ├── suite_v1.json      # 19 个测试用例（v1）
│   │       ├── suite_v2.json      # 33 个测试用例（v2，11 类别）
│   │       └── benchmarks/
│   │           └── default_profiles.json  # 108 个基准模型画像
│   └── tests/
│       └── test_all.py            # 测试套件（~60+ 用例）
└── sandbox-test/
    ├── .env.example               # 环境变量模板
    └── setup.ps1                  # PowerShell 一键部署脚本
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
cp sandbox-test/.env.example sandbox-test/.env
# 编辑 .env 文件，填入必要配置

# 启动服务
cd backend
python -m app.main
```

服务默认运行在 `http://localhost:8000`。

**Windows 快捷启动**：直接运行 `start.bat`。

**PowerShell 部署**：

```powershell
cd sandbox-test
.\setup.ps1               # 启动（自动安装依赖、生成密钥、健康检查）
.\setup.ps1 -Port 9000    # 指定端口
.\setup.ps1 -NoBrowser    # 不自动打开浏览器
.\setup.ps1 -Stop          # 停止服务
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
| GET | `/api/v1/runs/:id/report` | 获取分析报告 |
| GET | `/api/v1/runs/:id/responses` | 获取测试响应详情 |
| GET | `/api/v1/runs/:id/scorecard` | 获取 v2 评分卡 |

### 比对与排行

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/compare-runs` | 创建 A/B 对比任务 |
| GET | `/api/v1/compare-runs` | 列出所有对比任务 |
| GET | `/api/v1/compare-runs/:id` | 获取对比结果 |
| GET | `/api/v1/benchmarks` | 列出基准模型 |
| GET | `/api/v1/models/:name/trend` | 模型历史评分趋势 |
| GET | `/api/v1/leaderboard` | 模型排行榜 |

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
  → Orchestrator: PreDetect(5层) → CaseExecutor → Judge(9种) → Analysis Pipeline
```

### 预检测 5 层管道

| 层级 | 名称 | 探测内容 |
|------|------|----------|
| 1 | HTTP Header | 响应头信息指纹 |
| 2 | Self-report | 模型自我身份报告 |
| 3 | Identity Probe | 深层身份探测 |
| 4 | Knowledge Cutoff | 知识截止日期 |
| 5 | Preference/Format | 偏好与格式分析 |

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

### 信任判定等级

| 评分条件 | 等级 | 含义 |
|----------|------|------|
| auth≥85 且 score≥75 | `trusted` | 可信 |
| auth≥70 或 score≥65 | `suspicious` | 可疑 |
| auth≥50 或 score≥45 | `high_risk` | 高风险 |
| 低于上述阈值 | `fake` | 疑似套壳 |

## 检测模式

| 模式 | 说明 |
|------|------|
| `quick` | 快速检测，跳过风格测试，支持提前终止 |
| `standard` | 标准检测，运行完整测试套件 |
| `full` | 全量检测，所有测试用例 + 多次采样 |

## 测试类别（v2 · 33 用例 · 11 类别）

| 类别 | 用例数 | 说明 |
|------|--------|------|
| Protocol | 2 | 基础聊天、usage 字段 |
| Instruction | 5 | 格式约束、语言切换、字数限制 |
| System | 4 | 系统提示遵从、角色扮演 |
| Reasoning | 4 | 数学推理、逻辑问题 |
| Coding | 3 | 代码生成与执行 |
| Consistency | 2 | 多次采样一致性 |
| Antispoof | 3 | 身份探测、矛盾检测 |
| Parameter | 3 | temperature、max_tokens 行为 |
| Style | 3 | Markdown 偏好、格式习惯 |
| Refusal | 2 | 敏感内容处理、拒绝措辞 |
| Performance | 2 | 响应速度、吞吐量 |

### 9 种判定方法

`exact_match` · `regex_match` · `json_schema` · `line_count` · `refusal_detect` · `heuristic_style` · `code_execution` · `identity_consistency` · `any_text`

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

## 运行测试

```bash
cd llm-inspector
PYTHONPATH=backend python backend/tests/test_all.py
```

自制测试框架（非 pytest），覆盖 config / security / db / seeder / judge / analysis / repo / predetect / executor / http 共 ~60+ 用例。

## 许可证

本项目仅供学习和安全研究使用。
