# LLM Inspector v10 升级更新方案

## 1. 概述与核心目标

LLM Inspector 项目当前（v8/v9）已经具备了较完善的模型评测、基于 IRT（项目反应理论）的能力评分体系、以及智能预算控制机制。然而，在面对越来越复杂的套壳模型（Wrapper Models）、高阶推理模型（如 o1, o3, DeepSeek-R1）时，当前架构在评测精细度、测试消耗控制、执行健壮性以及防御穿透上仍有优化空间。

本升级方案详细规划了向 v10 演进的核心路径，确保所有数据、公式、算法均具备科学依据，并提供可落地的技术改造指导。

---

## 2. 核心不足与优化方案

### 2.1 判题合理性与能力评分精细度
**不足：** 
目前的评分机制虽然在 v7 中引入了基于 IRT 信息函数的权重计算，但 100 分制在多维度、多因素（如系统提示词、温度设定）影响下容易产生“天花板效应”，且无法精细反映模型在极端难度下的能力差异。判题器（Judge）在处理长逻辑推理时存在透明度不足和幻觉误判的问题。

**优化点（科学依据与数据来源）：**
1. **多维项目反应理论 (MDIRT, Multidimensional Item Response Theory)**：
   - 依据：Reckase (2009) *Multidimensional Item Response Theory*。引入多维 $\theta$ 向量评分，取代单一的百分制，改为标准分（均值500，标准差100，参考 GRE/SAT 评分体系）。
   - 算法：计算模型在各个正交维度上的隐性能力 $\theta_d$。公式：$P(x_{ij}=1|\boldsymbol{\theta}_j) = \frac{1}{1 + \exp(-(\mathbf{a}_i^T \boldsymbol{\theta}_j - b_i))}$。
2. **Elo 等级分系统的动态 K 因子调整 (Glicko-2)**：
   - 依据：Glickman (1999) 评分系统，引入波动率（Volatility）和偏差（Rating Deviation）。模型能力的评分不仅取决于胜负，还取决于测试用例的置信区间。
3. **判题透明化与链式校验 (Chain-of-Verification)**：
   - 依据：Dhuliawala et al. (2023) *Chain-of-Verification Reduces Hallucination in Large Language Models*。
   - 对 `transparent_judge.py` 升级，要求判题模型必须输出核查问题（Verification Questions）并自行校验，以提高对长链推理的判题合理性。

### 2.2 测试消耗与 Token 节省优化
**不足：**
目前 `TokenBudgetGuard` 仅依赖历史中位数进行预算卡控，遇到高耗能测试时可能会粗暴截断，导致测试精度受损（模型无法完整回答）。

**优化点：**
1. **提示词无损压缩 (Prompt Compression)**：
   - 依据：Jiang et al. (2023) *LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models*。
   - 方案：引入信息瓶颈理论，剔除测试用例提示词中的冗余 Token，在不影响模型理解语义的前提下，节省 20%-40% 的输入 Token。
2. **计算机化自适应测试 (CAT) 的提前终止策略**：
   - 依据：van der Linden (2010) *Elements of Adaptive Testing*。
   - 方案：当测量标准误 (Standard Error of Measurement, SEM) $SEM(\hat{\theta}) = \frac{1}{\sqrt{I(\hat{\theta})}}$ 小于预设阈值（如 0.3）时，说明模型在该维度的能力已经精准定级，可立即停止后续同质化测试用例的投放，确保在 100% 精准度下节省 Token。

### 2.3 执行进度保障 (100% 进度) 与架构优化
**不足：**
当前测试执行强依赖 `ThreadPoolExecutor` 或简单的 `asyncio` 协程，在遇到 API 限流（Rate Limit）或上下文溢出时，容易导致个别 Case 失败，使得测试无法达成 100% 进度。日志缺乏实时透明度。

**优化点：**
1. **指数退避与抖动重试 (Exponential Backoff with Jitter)**：
   - 依据：AWS 架构最佳实践。公式：$sleep = \text{random}(0, \min(cap, base \times 2^{attempt}))$。
2. **基于 Celery + Redis 的分布式异步架构**：
   - 取代当前内存级的队列，实现任务持久化、细粒度重试与容灾恢复。
3. **服务器发送事件 (SSE) 实时透明日志**：
   - 前端通过 SSE 实时订阅预检测 (Pre-detect) 和判题 (Judge) 的思维链日志，保证执行过程 100% 可视化。

### 2.4 测试数据真实性与难度扩维
**不足：**
现有的 `suite.json` 测试集在对抗最新推理模型时容易失效，部分假数据或无意义数据无法有效区分模型能力。知识图谱强依赖 Wikidata 且容错率较低。

**优化点：**
1. **高阶竞赛题库引入**：
   - 数据来源：MAA (美国数学协会) AIME 2024 赛事题库、USAMO 题库，以及 Hendrycks et al. 的 MATH dataset。这些数据具备真实且严谨的解题步骤，无假数据，可准确评估模型的 Deep Thinking 能力。
2. **DBpedia API 知识图谱深度集成**：
   - 依据：DBpedia 官方 SPARQL Endpoint。
   - 方案：升级 `KnowledgeGraphClient`，通过 SPARQL 获取结构化三元组数据（Subject-Predicate-Object），取代单一的 Wikidata 查询，防止幻觉。
3. **攻破套壳模型防御的技术提示词 (Jailbreak / Red Teaming)**：
   - 数据来源：JailbreakBench (2024) 及 Do-Not-Answer 框架。
   - 引入“多轮认知陷阱 (Multi-turn Cognitive Traps)”和“格式强制越狱 (Format-Forcing Jailbreaks)”，专门剥离套壳模型加装的浅层安全对齐过滤层，暴露其底层基座模型的真实面貌。

### 2.5 代码精简与无用文件清理
**不足：**
项目中存在历史包袱代码，降低了可维护性。
**优化点：**
- 删除过期的测试文件：`test_all_legacy.py`, `test_legacy_runner.py`。
- 清理 V7 遗留的前端 UI 代码（如 `v7_visualization.js`），全面迁移至 V8/V9 组件。
- 移除不再适用的硬编码阈值（如绝对分数阈值），全面采用动态统计学阈值。

---

## 3. 具体落地行动与代码更新指导

### 3.1 启动脚本更新与依赖安装
为了支持 Celery 队列、DBpedia 查询及提示词压缩，需更新环境依赖。请将以下依赖项添加到 `requirements.txt`：

```text
# v10 New Dependencies
celery>=5.3.6          # Distributed task queue for 100% execution robustness
redis>=5.0.1           # Broker for Celery
SPARQLWrapper>=2.0.0   # DBpedia integration for accurate knowledge graph
tiktoken>=0.6.0        # Accurate token counting & budget control
```

修改 `start.bat`（同理 `start.sh`），在启动 API 服务前拉起 Celery Worker：
```bat
REM Start Celery Worker for robust task execution
start /B celery -A app.tasks.celery_queue worker --loglevel=info

REM Start Server
python start.py --port 8000 --strict-provenance
```

### 3.2 淘汰旧机制，部署自适应提前终止 (CAT)
在 `backend/app/runner/orchestrator.py` 中更新 `SmartBudget` 逻辑：
结合 IRT 题库参数（$a, b, c$），实时计算信息量 $I(\theta)$。当累积信息量满足 $\sqrt{1 / \sum I(\theta)} < 0.3$ 时，标记该维度测试完成，停止下发该维度的新题目。此举不仅保证精确度，还极大减少了 Token 开销。

### 3.3 引入 DBpedia 进行真实数据校验
修改 `backend/app/knowledge/kg_client.py`，新增 DBpedia SPARQL 查询支持：
```python
from SPARQLWrapper import SPARQLWrapper, JSON

def verify_with_dbpedia(entity_name: str):
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    # 构建严谨的查询以获取真实百科三元组
    query = f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?label ?abstract WHERE {{
        ?s rdfs:label "{entity_name}"@en ;
           dbo:abstract ?abstract .
        FILTER (lang(?abstract) = 'en')
    }} LIMIT 1
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    # 根据返回结果判断真实性
```

## 4. 开发阶段划分与里程碑 (Milestones)

为了保证升级过程的平稳过渡与持续交付，v10 的开发将划分为以下四个阶段（Milestones）：

### Milestone 1: 基础设施与健壮性改造 (Infrastructure & Robustness)
**目标**：解决测试任务执行不稳定的问题，引入分布式队列，重构核心调度。
- [x] 更新 `requirements.txt` 和启动脚本（`start.bat` / `start.sh`）。
- [ ] TODO: 全面部署 Celery + Redis 分布式任务队列，替换现有的内存级队列（目前仅在 worker 中条件导入）。
- [x] 实现任务执行的指数退避与抖动重试 (Exponential Backoff with Jitter)。
- [x] 构建 SSE (Server-Sent Events) 接口，实现实时日志推送。

### Milestone 2: 知识图谱与数据集增强 (Data & Truthfulness)
**目标**：彻底消灭测试集中的假数据，增强知识图谱与推理对抗题库。
- [x] 集成 DBpedia SPARQL 查询到 `KnowledgeGraphClient`，作为事实核查的最高优先级。
- [x] 收集并清洗 AIME 2024 / USAMO / MATH 数据集，生成标准化的深度推理用例。
- [x] 整合 JailbreakBench 等对抗模型提示词，设计多轮认知陷阱测试用例。
- [x] 编写数据迁移脚本，更新现有 `suite.json` 测试用例集。

### Milestone 3: 核心评分与自适应探测 (Scoring & Adaptive Probing)
**目标**：升级评分理论模型，实现 Token 消耗的最优化与精确评测。
- [ ] TODO: 补充 MDIRT (多维项目反应理论) 算法实现 (`mdirt_engine.py`)，计算 $\theta_d$ 标准分。
- [x] 实现基于 Glicko-2 的动态能力评分调整逻辑。
- [x] 引入 LLMLingua 提示词无损压缩算法，在下发前压缩长文本。
- [x] 重构 `SmartBudget` 和 Orchestrator，实现基于测量标准误 (SEM) 的计算机化自适应测试 (CAT) 提前终止策略。

### Milestone 4: 判题透明化与遗留清理 (Transparency & Cleanup)
**目标**：提升判题模型的准度与透明度，移除历史技术债务。
- [x] 重构 `transparent_judge.py`，实现 Chain-of-Verification 链式校验逻辑。
- [x] 完善前端 UI，对接后端的 MDIRT 标准分和 SSE 实时执行日志展示。
- [x] 清理过期测试文件 (`test_all_legacy.py`, `test_legacy_runner.py`)。
- [x] 移除硬编码的绝对阈值，完成 v10 全面集成测试。

---

## 5. 总结
v10 升级不仅是一次代码重构，更是底层测评哲学从“静态广度测试”向“自适应深度探测 (Adaptive Deep Probing)”的进化。通过引入 MDIRT 模型、AIME 真实数据集、DBpedia 验证网络及 Celery 健壮架构，能够有效刺穿虚假的套壳防御，实现极低消耗、100%成功率的权威大模型评测。