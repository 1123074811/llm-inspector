# LLM Inspector v11.0 (Next-Gen)

LLM Inspector 是一款用于大语言模型（LLM）套壳检测与能力评估的权威工具 —— 旨在检测 LLM API 是否真正提供其声称的模型服务，还是在背后代理/包装了其他基座模型，并对其真实能力进行精准定级。

**🚀 v11.0 演进计划 (In Progress)**：
- **认知诊断模型 (CDM)**：从 MDIRT 升级至基于 DINA 的认知诊断，输出具体微观技能（如反事实推理）的掌握概率与贝叶斯置信区间。
- **Token 极致压缩与动态 Few-Shot**：引入 DSPy 提示词自动编译与 FAISS 向量检索，仅提取最匹配的 Few-Shot，大幅节省 Token 且不降精度。
- **100% 进度与透明度保障**：集成 `pyfailsafe` 断路器应对 API 熔断，引入 OpenTelemetry 实现全链路 Token 与耗时的 100% 透明追踪。
- **高阶对抗与题库提纯**：利用 IIF (项目信息函数) 自动剔除无区分度的假数据，引入 GPQA、SWE-bench 终极测试，以及多语言翻译与 Base64 注入攻击，彻底击穿套壳模型的浅层防御。
- **Shapley Value 归因报告**：引入博弈论特征归因分析，通过 ECharts 动态雷达图明确指出模型失分的根本原因。

**v10.0 核心特性**（从“静态广度测试”进化为“自适应深度探测”）：
- **多维能力评分 (MDIRT)**：引入多维项目反应理论 (MDIRT) 及 Glicko-2 动态 K 因子调整，取代传统的单一百分制，提供高颗粒度的能力测算。
- **计算机化自适应测试 (CAT)**：结合信息瓶颈理论（LLMLingua 提示词压缩）与基于测量标准误 (SEM) 的提前终止策略，确保 100% 精准度下节省高达 40% 的 Token 开销。
- **高难度与对抗性测试集**：集成 AIME 2024、USAMO、MATH 等高阶竞赛题库测试 Deep Thinking 能力；引入 JailbreakBench 等对抗提示词（格式强制越狱、多轮认知陷阱），有效剥离套壳模型的浅层安全对齐。
- **透明判题与链式校验**：升级 `transparent_judge.py` 引入 Chain-of-Verification 机制，消除模型在长逻辑推理判题时的幻觉误判。
- **DBpedia 知识图谱深度核查**：升级知识验证模块，直接通过 SPARQL 查询 DBpedia 获取结构化三元组数据，取代单一维基查询。
- **实时透明日志与高可用架构**：新增 SSE (Server-Sent Events) 接口支持 100% 执行过程可视化；引入指数退避抖动重试 (Exponential Backoff with Jitter) 并逐步向 Celery+Redis 分布式异步架构迁移。

## 🎯 工作原理

LLM Inspector 通过多层行为指纹分析与动态探测，精准定位 API 背后的真实模型：

1. **预检测管道（Pre-detection）**：8 层渐进式指纹识别（HTTP 头信息、自我报告、知识截止日期、Tokenizer 指纹、偏好分析等），以最小消耗快速锁定候选模型。
2. **自适应深度探测（Adaptive Probing）**：基于 `suite_v10.json` 测试集，系统动态根据当前题目的 IRT 区分度与模型作答情况，实时决定下一步的探测方向。
3. **特征提取与基线比对**：从响应中提取行为特征向量，与本地/云端的 Golden Baseline 进行余弦相似度比对。
4. **多维评分与信任判定**：输出各维度的 $\theta_d$ 标准分，并给出 `trusted`（可信）、`suspicious`（可疑）、`high_risk`（高风险）等分级判定。

## 🛠 技术栈

- **后端**：Python 3.12+，零外部 Web 框架依赖（纯标准库实现 HTTP 服务器、SQLite 数据库引擎）。
- **任务分发**：支持内存级队列与 Celery + Redis 分布式队列扩展。
- **前端**：多文件 SPA（纯 HTML/CSS/JS），无复杂构建步骤，支持 SSE 实时日志渲染。
- **关键依赖**：`cryptography`, `numpy`, `scikit-learn`, `SPARQLWrapper`, `tiktoken`, `celery`, `redis`, `pytest`。

## 📁 项目结构

```text
llm-inspector/
├── start.bat / start.sh           # 一键启动脚本
├── backend/
│   ├── app/
│   │   ├── main.py                # HTTP 服务器入口
│   │   ├── core/                  # 配置、数据库、加密、SSE推送中心
│   │   ├── adapters/              # OpenAI 兼容 API 适配器
│   │   ├── predetect/             # 预检测管道与指纹识别
│   │   ├── runner/                # 动态编排器 (CAT) 与用例执行
│   │   ├── judge/                 # 透明判题器与链式校验逻辑
│   │   ├── analysis/              # MDIRT 评分引擎、相似度与特征提取
│   │   ├── knowledge/             # DBpedia SPARQL 知识图谱验证
│   │   ├── tasks/                 # Celery Worker / Seeder
│   │   └── fixtures/              # suite_v10 测试用例集、探针数据
│   └── tests/                     # 单元测试与集成测试
├── docs/                          # 历史升级方案与架构文档
└── frontend/                      # 纯前端 UI 代码
```

## 🚀 快速开始

### 环境要求
- Python 3.12+
- 可选：Redis Server（如需启用 Celery 分布式队列）

### 安装与运行

```bash
cd llm-inspector

# 创建并激活虚拟环境
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# 安装依赖
pip install -r requirements.txt

# 启动服务 (Windows 可直接运行 start.bat)
cd backend
python -m app.main
```

> 服务默认运行在 `http://localhost:8000`。

## ⚙️ 检测模式

本系统通过智能预算控制（SmartBudget）自动在以下模式间进行调度：

| 模式 | 探测深度 | Token 预算 (默认) | 并发 | 适用场景 |
|------|--------|-----------|------|------|
| **Quick (快速)** | L1 被动探针 + 基础能力 | 15K | 12 | API 可用性验证、基础模型分类、粗筛真伪 |
| **Standard (标准)** | L2 主动探针 + 扩展能力 | 40K | 8 | 完整能力定级、可靠真伪判定、行为指纹比对 |
| **Deep (深度)** | L3 对抗提取 + 高阶竞赛题 | 100K | 3 | 攻防越狱对抗、系统提示词提取、深层逻辑测算 |

*可通过修改 `.env` 或 `config.py` 中的 `TOKEN_BUDGET_*` 变量来自定义各模式的预算上限。*

## 📊 核心评价指标说明

- **Theta (θ) 标准分**：取代传统的百分制，基于多维项目反应理论计算。均值设为 500，标准差 100，客观反映模型在各个细分维度（如 Reasoning, Coding, Safety 等）上的绝对能力。
- **置信区间 (Confidence Interval)**：通过 Fisher 信息量与测量标准误 (SEM) 给出评分的置信范围。
- **Baseline Match Score**：候选模型与已知基座模型行为特征的余弦相似度匹配得分。

## 📝 证书与开源协议

本项目遵循 MIT 开源协议。欢迎提交 Issue 与 Pull Request 共同完善！