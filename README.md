# LLM Inspector v1.0

LLM 套壳检测工具 —— 检测 LLM API 是否真正提供其声称的模型服务，还是在背后代理/包装了其他模型。

## 工作原理

LLM Inspector 通过多层行为指纹分析，识别 API 背后的真实模型：

1. **预检测（Pre-detection）**：5 层渐进式指纹识别管道（HTTP 头信息、自我报告、身份探测、知识截止日期、偏好/格式分析），以最小 token 消耗快速锁定候选模型
2. **全量测试**：19 个测试用例，覆盖 6 大类别（协议合规性、指令遵循、系统提示遵从、参数合规性、风格分析、拒绝行为）
3. **特征提取与比对**：从响应中提取行为特征向量，与内置基准模型（GPT-4o、GPT-4o-mini、Claude-3.5-Sonnet、DeepSeek-V3、Qwen2.5-72B）进行余弦相似度比对
4. **风险评估**：输出 low / medium / high / very_high 四级风险等级，指示 API 套壳的可能性

## 技术栈

- **后端**：Python 3.12，零外部框架依赖（纯标准库实现 HTTP 服务器、数据库、任务队列）
- **前端**：单文件 SPA（纯 HTML/CSS/JS，无构建步骤）
- **数据库**：SQLite（WAL 模式）
- **加密**：AES-256-GCM（API 密钥静态加密），唯一外部依赖 `cryptography`

## 项目结构

```
llm-inspector/
├── start.bat                      # Windows 一键启动
├── frontend/
│   └── index.html                 # 前端单文件应用
├── backend/
│   ├── app/
│   │   ├── main.py                # HTTP 服务器入口
│   │   ├── core/
│   │   │   ├── config.py          # 环境配置
│   │   │   ├── db.py              # SQLite 数据库层（9 张表）
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
│   │   │   └── methods.py         # 7 种判定方法
│   │   ├── analysis/
│   │   │   └── pipeline.py        # 特征提取 + 评分 + 报告生成
│   │   ├── repository/
│   │   │   └── repo.py            # 数据库 CRUD
│   │   ├── tasks/
│   │   │   ├── worker.py          # 线程池后台 Worker（4 线程）
│   │   │   └── seeder.py          # 数据库初始化种子
│   │   └── fixtures/
│   │       ├── suite_v1.json      # 19 个测试用例定义
│   │       └── benchmarks/
│   │           └── default_profiles.json  # 5 个基准模型特征向量
│   └── tests/
│       └── test_all.py            # 测试套件
└── sandbox-test/
    ├── .env.example               # 环境变量模板
    └── setup.ps1                  # PowerShell 一键部署脚本
```

## 快速开始

### 环境要求

- Python 3.12+
- `cryptography` 包

### 安装与运行

```bash
cd llm-inspector

# 创建并激活虚拟环境
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# 安装依赖
pip install cryptography

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
.\setup.ps1               # 启动
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
| `INTER_REQUEST_DELAY_MS` | `500` | 请求间隔（防限流） |
| `PREDETECT_CONFIDENCE_THRESHOLD` | `0.85` | 预检测置信度阈值 |
| `DEFAULT_REQUEST_TIMEOUT_SEC` | `60` | 请求超时时间 |
| `RAW_RESPONSE_TTL_DAYS` | `7` | 原始响应保留天数 |

生成加密密钥：

```bash
python3 -c "import secrets,base64; print(base64.b64encode(secrets.token_bytes(32)).decode())"
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/health` | 健康检查 + Worker 状态 |
| POST | `/api/v1/runs` | 创建检测任务 |
| GET | `/api/v1/runs` | 列出所有检测任务 |
| GET | `/api/v1/runs/:id` | 获取任务状态与进度 |
| DELETE | `/api/v1/runs/:id` | 删除检测任务 |
| GET | `/api/v1/runs/:id/report` | 获取分析报告 |
| GET | `/api/v1/runs/:id/responses` | 获取测试响应详情 |
| GET | `/api/v1/benchmarks` | 列出基准模型 |

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

## 检测模式

| 模式 | 说明 |
|------|------|
| `quick` | 快速检测，跳过风格测试，支持提前终止 |
| `standard` | 标准检测，运行完整测试套件 |
| `full` | 全量检测，所有测试用例 + 多次采样 |

## 测试类别

| 类别 | 说明 | 示例 |
|------|------|------|
| Protocol | 协议合规性 | 基础聊天、usage 字段、流式传输 |
| Instruction | 指令遵循 | 格式约束、语言切换、字数限制 |
| System | 系统提示遵从 | 角色扮演、输出限制 |
| Parameter | 参数合规性 | temperature、max_tokens 行为 |
| Style | 风格分析 | Markdown 偏好、列表格式、代码风格 |
| Refusal | 拒绝行为 | 敏感内容处理、拒绝措辞模式 |

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

## 许可证

本项目仅供学习和安全研究使用。
