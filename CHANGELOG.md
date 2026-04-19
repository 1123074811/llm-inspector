# Changelog

All notable changes to LLM Inspector are documented here.

## [v14.0.0-phase1] — 2026-04-19

### Added
- `/api/v14/health` — v14 命名空间健康检查占位端点（Phase 3+ 扩展完整 v14 路由）
- `core/db_migrations.py Migration003` — 安全删除历史遗留 `benchmark_profiles` 表（`DROP TABLE IF EXISTS`）
- `suite_v13.json` 补齐数据链：90 道题全部新增 `source_ref.url`（指向 HuggingFace/AoPS 原始数据集）+ `answer_provenance.verified_at`
- `backend/tools/debug/` — 调试脚本统一目录
- `backend/tests/archive/` — legacy 回归测试归档目录
- `pyproject.toml [tool.pytest.ini_options] norecursedirs` — 排除 archive/ 和 tools/ 防止 pytest 误采集

### Changed
- v8 API 路由响应头新增 `X-API-Deprecated: true` + `X-API-Successor: /api/v1/`，并写 warning 日志
- `InspectorHandler._send()` 新增 `extra_headers` 可选参数
- `frontend/index.html` 预检测说明更新为「20 层渐进探针」（含 v14 Phase 3/5 的 L17-L20）
- `frontend/app.js` — 移除 `toggleAdvancedSettings()` / `previewIsomorphicCases()` / `applyIsomorphicCases()` 废弃函数

### Removed
- **高级设置面板**（`index.html` 原 L72-89 / v2 时代题库维护工具，已无对应后端功能）
- `frontend/v8_components.js` / `frontend/v8_styles.css` → 移至 `frontend/archive/`（index.html 不再引用）
- 根目录 v3 残留：`E:/code/llm-inspector/test_inspector.db` / `v8_phase4_demo.py` / `examples/` / `frontend/v7_visualization.js`
- `backend/tests/legacy/` → 整体归档至 `backend/tests/archive/`（10 个 v8/v9/v11 回归测试文件）
- `backend/debug_e2e.py` / `debug_features.py` / `debug_params.py` → 移至 `backend/tools/debug/`
- `backend/scripts/archive/*.json` → 移至 `docs/archive/`

### Test Coverage
- **268 passed, 4 skipped**（archive/ 已从 pytest 采集中排除）
- v14 Phase 1 验收：6/6 项目通过 grep 检查

---

## [v13.0.0] — 2026-04-18

### Breaking Changes
- `ScoreCard` JSON 返回新增 `v13` 块（含 `stanine`/`percentile`/`theta`/`theta_ci95`/`judge_kappa`），旧字段 `total_score` 保留不变
- 测试套件默认版本升级为 `suite_v13`（suite_v10 仍可通过 `suite_version` 参数指定）
- `HARD_RULES` 阈值现在从 `SOURCES.yaml` 读取（有 fallback，不影响已部署实例）

### New Features
- **数据溯源骨架**：`SOURCES.yaml` 注册表，40+ 条记录，含 `source_url`/`license`/`retrieved_at`；启动时 `ProvenanceGuard` 校验
- **双刻度评分**：新增 Stanine-9（Canfield 1951）+ 百分位（HELM 参考分布）+ θ 逻辑分
- **双盲判题**：`dual_judge.py`，并发 rule + semantic，Cohen's κ 监控，κ < 0.60 触发 transparent_judge
- **SRC 驱动阈值**：`VerdictEngine.HARD_RULES` 所有阈值从 `SRC["verdict.*"]` 读取，fallback 保护
- **NNLS 权重拟合**：`fit_weights.py` 从 HELM v1.10 + LMSYS Arena 数据拟合能力维度权重
- **题库 v13**：90 道真实基准题（GPQA/AIME/MATH-500/SWE-bench/MMLU-Pro/CMMLU/JailbreakBench），含 `source_ref` + `license`
- **Layer 15 — ASCII Art Attack**（Jiang et al. 2024，arXiv:2402.11753）
- **Layer 16 — Indirect Prompt Injection**（Greshake et al. 2023，arXiv:2302.12173）
- **多语言攻击扩展**：L14 从 4 种语言扩展到 13 种低资源语言（Yong et al. 2023）
- **结构化事件总线**：`core/events.py`，21 种 `EventKind`，SSE / LOG / Tracer 三路分发
- **B-03 修复**：`execute_full()` `try/finally` 保证进度永达终态
- **Run Watchdog**：守护线程每 5 分钟扫描超时 running 任务并标记 `partial_failed`
- **Trace JSONL 持久化**：每个 run 的事件写入 `data/traces/{run_id}.jsonl`
- **`GET /api/v1/runs/{id}/timeline.svg`**：纯 Python 服务端 SVG 时间轴
- **DBpedia 双源交叉验证**：`dbpedia_client.py` + `kg_client.py` 并发 fan-out，冲突时标记 `conflicting`
- **SQLite KG 缓存**：`data/kg_cache.sqlite`，TTL 30 天
- **参考嵌入网络**：14 个模型参考向量（`reference_embeddings.json`），无用户基准时自动 fallback
- **Token 缓存提示**：`PromptOptimizer` 新增 `CacheStats` + `get_cache_control_headers()`
- **前端文案修正**：Quick/Standard/Deep 题数修正，移除硬编码成本估算，预检测说明更新为 16 层
- **雷达图升级**：ECharts 5（CDN），支持百分制/Stanine/θ 三刻度切换

### Test Coverage
- Phase 1: +34 tests (provenance + dual_judge + scoring_v13)
- Phase 2: +20 tests (weights + stanine + dual_judge)
- Phase 3: +34 tests (Layer 15/16 + multilingual)
- Phase 4: +16 tests (progress completeness + event bus)
- Phase 5: +10 tests (KG consistency + reference embeddings)
- **Total: 268 passed, 4 skipped**

### Removed
- `backend/app/judge/semantic_v3.py` (死代码，无引用)
- `backend/app/judge/hallucination_v3.py` (死代码，无引用)
- `backend/app/fixtures/suite_v1.json / v2.json / v3.json / suite_extraction.json` (归档到 docs/archive/)
- 阶段性测试文件 `test_v8_phase*.py / test_v9_phase*.py / test_v11_phase*.py` 归入 `tests/legacy/`

### Migration Guide
无破坏性迁移要求。`ScoreCard.to_dict()` 新增 `"v13"` 子块，旧字段保持原位不变。

---

## [v12.0.0] — 2026-01-xx

（历史版本，参见 docs/archive/）
