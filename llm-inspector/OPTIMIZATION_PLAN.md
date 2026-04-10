# LLM Inspector v3.1 — 优化计划

> 编写日期: 2026-04-10
> 当前版本: v3.1 | 52 个测试全部通过

---

## 一、已修复的 Bug

### 1.1 parse_error 大量出现（features.py 属性访问错误）

**根因**: `features.py` 中直接访问 `s.error_type` 和 `s.judge_confidence`，但 `SampleResult` 没有这两个属性。

- `error_type` 在 `SampleResult.response.error_type`（LLMResponse 上）
- `judge_confidence` 在 `SampleResult.judge_detail["confidence"]` 中

**影响**: 
- 错误分类统计（`_failure_attribution`）全部失效，所有失败都被归为 `api_error`
- 置信度聚合返回空列表，`avg_judge_confidence` 恒为 0

**修复**: 4 处属性访问修正
```python
# Before (错误)
s.error_type → s.response.error_type
s.judge_confidence → s.judge_detail.get("confidence")
```

### 1.2 NoneType.__format__ 格式化错误

**根因**: `judge/methods.py` 中语义评判返回 `{"keyword_coverage": None}`（显式 None），而 `pipeline.py:2734` 使用 `detail.get("keyword_coverage", 0)` — 当 key 存在但值为 None 时，`dict.get()` 不会使用默认值。

**影响**: 任何使用 `semantic_judge` 且未通过的用例在生成失败解释时崩溃。

**修复**: `detail.get("keyword_coverage") or 0`

### 1.3 预检测日志增强

**改进**: 在 `PreDetectionPipeline` 的每层执行后增加详细日志：
- 每层结果摘要（identified_as, confidence, tokens_used, evidence_count）
- 逐条 evidence 输出
- 最终合并结果摘要（layers_run, merged_confidence, best_layer, identified_as）

---

## 二、架构优化建议

### 2.1 并发模型优化

**现状**:
- 顶层 Worker 池硬编码为 4 线程（`worker.py:21`）
- 每阶段创建新的 `ThreadPoolExecutor`，有线程启动开销
- 使用 `urllib.request.urlopen()` 做阻塞 HTTP 调用
- 用例间固定 300ms 延迟防止限流

**建议**:

#### 方案 A: 保持线程模型，优化参数（低成本改动）
```python
# worker.py - 使 worker 池可配置
MAX_CONCURRENT_RUNS = int(os.getenv("MAX_CONCURRENT_RUNS", "4"))
_pool = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_RUNS)

# orchestrator.py - 复用线程池而非每阶段新建
# 在 run_pipeline 开始时创建一个 pool，所有阶段共享
```

#### 方案 B: 迁移到 asyncio + aiohttp（中等改动）
```python
# 好处: 
# - 真正的非阻塞 I/O，一个线程可管理数百并发请求
# - 更精确的超时控制（asyncio.wait_for）
# - 更低的内存占用

# 迁移路径:
# 1. adapter.chat() → async adapter.chat()
# 2. ThreadPoolExecutor → asyncio.gather()
# 3. time.sleep(0.3) → asyncio.sleep(0.3)
```

#### 方案 C: 自适应延迟（立即可做）
```python
# case_executor.py - 将固定 300ms 改为自适应
# 正常时: 100ms
# 收到 429 后: 指数退避 (1s, 2s, 4s)
# 连续成功 10 次后: 降回 100ms
INTER_SAMPLE_DELAY_BASE = 0.1  # 从 0.3 降到 0.1
```

### 2.2 超时策略优化

**现状**:
- 全局固定 60s 超时
- 超时不重试（仅 429/5xx 重试）
- 预检测硬编码 15s

**建议**:

```python
# 按用例复杂度分级超时
TIMEOUT_MAP = {
    "protocol": 15,      # 简单协议检查
    "instruction": 30,   # 指令遵循
    "reasoning": 60,     # 推理题需要更长思考
    "coding": 90,        # 编码题（尤其 DP/复杂算法）
    "extraction": 30,    # 对抗提取
    "consistency": 30,   # 一致性检测
}

# case_executor.py 中根据 case.category 选择超时
timeout = TIMEOUT_MAP.get(case.category, 60)
```

```python
# 超时重试（至少 1 次）
# adapters/openai_compat.py - 将 timeout 加入可重试列表
_RETRYABLE_CODES = {429, 500, 502, 503, 504}
_RETRY_ON_TIMEOUT = True  # 新增: 超时也重试一次
```

### 2.3 数据库写入优化

**现状**: 每个用例结果立即写入数据库（`_save_case_result`），同步阻塞。

**建议**: 批量写入
```python
# 方案: 每阶段结束后批量写入
# _run_cases_concurrent 内部先暂存到内存
# 阶段完成后一次性 executemany()

# 或: 使用后台写入线程
write_queue = Queue()
def _db_writer():
    while True:
        batch = []
        batch.append(write_queue.get())
        # 收集 50ms 内的所有待写入项
        deadline = time.monotonic() + 0.05
        while time.monotonic() < deadline:
            try:
                batch.append(write_queue.get_nowait())
            except Empty:
                break
        _batch_save(batch)
```

### 2.4 提前终止策略

**现状**: 高错误率检查仅在全部完成后执行。

**建议**: 阶段内动态检查
```python
# 每完成 10 个用例检查一次错误率
if len(case_results) % 10 == 0:
    error_rate = failed_count / len(case_results)
    if error_rate > 0.8:
        logger.warning("Error rate >80%, aborting phase early")
        break
```

---

## 三、测试效率提升方案

### 3.1 Quick 模式加速（目标: <30s）

| 策略 | 预计收益 |
|------|---------|
| 降低 inter-sample 延迟至 100ms | -40% 等待时间 |
| Quick 模式并发提升至 12 | -30% 执行时间 |
| 简单题超时降至 15s | 避免长时间阻塞 |
| 预检测 + 测试并行启动 | 节省预检测等待时间 |

### 3.2 用例优先级调度

```python
# 高区分力用例优先执行
# 如果前 N 个高区分力用例结果足够明确，可跳过低区分力用例
PRIORITY_CATEGORIES = [
    "reasoning",    # 高区分力
    "coding",       # 高区分力
    "extraction",   # 真伪判定关键
    "consistency",  # 套壳检测关键
    "instruction",  # 中等区分力
    "knowledge",    # 中等区分力（区分模型代际）
    "protocol",     # 低区分力（但执行快）
]
```

### 3.3 渐进式结果输出

当前前端需等待整个 pipeline 完成才能看到结果。建议：

1. **阶段性结果推送**: 每阶段完成后计算临时评分并更新前端
2. **WebSocket/SSE**: 实时推送进度（当前用例/总用例、临时评分）
3. **增量报告**: 预检测结果立即可查看，不必等待完整测试

---

## 四、能力测试与模型区分策略

### 4.1 当前项目的区分能力

项目已具备较好的多维度区分框架：

| 维度 | 区分目标 | 当前用例 | 覆盖度 |
|------|---------|---------|--------|
| 推理能力 | 区分强弱模型 | 20 题（含梯度难度 0.3-0.95） | 良好 |
| 编码能力 | 区分编码能力 | 9 题（函数/调试/重构/DP） | 良好 |
| 知识与幻觉 | 区分模型代际 | 10 题（含时效性知识） | 良好 |
| 指令遵循 | 区分精调质量 | 8 题 | 良好 |
| 身份一致性 | 检测套壳 | 5 题 + 预检测 7 层 | 优秀 |
| 对抗提取 | 检测系统提示泄露 | 12 题 | 优秀 |
| 指纹匹配 | 识别底层模型 | 14 个家族行为库 | 优秀 |
| Tool Use | 区分工具调用能力 | 6 题 | 中等 |

### 4.2 提升区分度的建议

#### A. 增加"签名题"（Signature Questions）
某些题目对特定模型有独特的回答模式，可作为"指纹"：
```yaml
- question: "请用中文回答：什么是 transformer 架构？"
  # Claude 倾向于用"本质上"开头，GPT 倾向于用"Transformer 是一种..."
  # DeepSeek 倾向于更学术化的表述
  fingerprint_features:
    - opening_style    # 开头习惯
    - list_preference  # 是否喜欢列表
    - markdown_density # Markdown 使用密度
```

#### B. 增加 Few-shot 一致性测试
同一问题在 0-shot 和 3-shot 下的表现差异，不同模型差异显著。

#### C. 增加 token 生成速度指纹
不同模型的 TPS (tokens per second) 分布有明显差异：
- GPT-4o: ~80-120 TPS
- Claude 3.5 Sonnet: ~60-90 TPS
- DeepSeek-V3: ~40-60 TPS

### 4.3 主流 AI 评测项目对比

| 框架 | 重点方向 | 核心特点 | 与本项目差异 |
|------|---------|---------|-------------|
| **LMSYS Chatbot Arena** | 用户偏好排名 | 人类盲测打分 (ELO) | 本项目为自动化评测 |
| **HELM (Stanford)** | 全面基准 | 42 个场景、7 个指标 | 覆盖面更广，但不含真伪检测 |
| **OpenCompass** | 中文评测 | 70+ 数据集、能力圈图 | 更偏学术基准，不含套壳检测 |
| **lm-evaluation-harness** | 标准化评测 | HuggingFace 生态、标准 prompt 模板 | 面向开放权重模型，不含 API 检测 |
| **AlpacaEval** | 指令遵循 | GPT-4 自动评判 | 单维度评测 |
| **SuperCLUE** | 中文能力 | 多维度中文评测 | 不含真伪/套壳检测 |

**本项目的独特价值**: 结合了能力评估 + 真伪鉴别 + 套壳检测，这是上述主流框架都不涵盖的。这是一个差异化优势，建议继续强化。

### 4.4 借鉴主流框架的可行改进

1. **ELO 排名体系**: 参考 LMSYS Arena，为检测过的模型维护一个能力 ELO 排名
2. **标准化 Prompt 模板**: 参考 lm-eval-harness，确保同一题目对不同模型使用完全相同的 prompt
3. **多评判者共识**: 参考 AlpacaEval，对主观题使用多个评判方法取共识
4. **结果可复现性**: 记录完整的请求/响应对，支持离线重新评判
5. **Logprobs 级指纹探测**: 若 API 返回 logprobs，可通过特定 token 概率分布进行更精确的模型识别（学术前沿方向）
6. **自适应测试（参考 OpenCompass CircularEval）**: 根据前序结果动态调整后续题目，跳过已确认能力范围的题目
7. **Prompt 污染检测**: 检测模型是否在训练数据中见过测试题，避免分数虚高（主流框架的重要关注点）
8. **请求缓存与去重**: 参考 lm-eval-harness，缓存相同 prompt 的请求结果，对比测试时避免重复调用

---

## 五、整体优先级排序

| 优先级 | 改动 | 复杂度 | 预期收益 |
|--------|------|--------|---------|
| P0 | ~~修复 parse_error（features.py 属性访问）~~ ✅ | 低 | 错误分类恢复正常 |
| P0 | ~~修复 NoneType.__format__~~ ✅ | 低 | 消除运行时崩溃 |
| P0 | ~~增强预检测日志~~ ✅ | 低 | 可观测性提升 |
| P1 | 自适应延迟 + 分级超时 | 低 | 测试速度提升 30-50% |
| P1 | 超时重试 | 低 | 减少假性失败 |
| P2 | 用例优先级调度 | 中 | Quick 模式加速 |
| P2 | 渐进式结果输出 | 中 | 用户体验提升 |
| P2 | 数据库批量写入 | 中 | 高并发性能提升 |
| P3 | 迁移到 asyncio | 高 | 根本性能提升 |
| P3 | 签名题 + Few-shot 测试 | 中 | 区分度提升 |
| P3 | ELO 排名体系 | 中 | 竞品差异化 |

---

## 六、总结

LLM Inspector 当前的核心架构（预检测 7 层 → 分阶段并发测试 → 28 种评判方法 → 三维评分）已经相当成熟。本次修复了 3 个影响测试准确性的 bug，并提出了分阶段的优化路线。

**短期重点（P1）**: 自适应延迟 + 分级超时可立即实现，预计提速 30-50%。

**中期重点（P2）**: 渐进式输出 + 用例优先级调度将大幅改善用户体验。

**长期方向（P3）**: asyncio 迁移 + 签名题扩展将使项目具备与主流评测框架竞争的能力，同时保持"真伪鉴别"的独特优势。
