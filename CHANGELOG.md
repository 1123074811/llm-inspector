# Changelog

All notable changes to LLM Inspector are documented here.

## [v14.0.0-phase4] — 2026-04-20

### Added
- `backend/app/judge/numeric_tolerance.py` — 数值容差判题器：支持科学记数法、百分比、单位剥离；相对误差 ≤ 5%（NIST SP 330-2019）；绝对值 < 1e-9 时切换为绝对误差 ≤ 1e-6
- `backend/app/judge/multi_choice_verified.py` — 严格选择题判题器：7 种提取模式（单字母/英文声明/中文/CMMLU 格式）；双答案歧义检测；引用 Hendrycks et al. 2021 MMLU "strict letter match" 协议
- `backend/app/judge/semantic_entailment.py` — 本地 NLI 语义蕴含判题器：3 级降级链（sentence-transformers cross-encoder/nli-deberta-v3-base → word-overlap Jaccard → semantic_v2 规则）；引用 Reimers & Gurevych 2019
- `judge/transparent_judge.py` `JudgeChainRunner` — 4 级判题降级链（外部 LLM → 本地 NLI → semantic_v2 规则 → hallucination_v2 规则），含完整 `judge_chain` 日志
- `judge/transparent_judge.py` `run_judge_chain()` — 模块级便捷函数
- `judge/consensus.py` `fleiss_kappa()` — Fleiss's κ（Fleiss 1971，Psychological Bulletin 76:378）支持 ≥3 判题器同意度评估
- `GET /api/v14/runs/{id}/judge-chain` — 返回指定 run 中所有用例的判题路径日志
- `handlers/v14_handlers.py` `handle_judge_chain` — 对应 handler
- `tests/test_v14_phase4.py` — 28 条验收测试（全部通过）

### Changed
- `judge/hallucination_v2.py` `_check_against_knowledge_graph()` — **B3 修复**：从占位符改为真实 DBpediaClient 调用（entity 提取 → DBpedia 验证 → 冲突标记 `conflict=true`），支持离线回退；`_calibrate_hallucination_verdict` 新增 `fake_entity_confirmed` / `conflict` 信号权重
- `judge/methods.py` — 注册 3 个新判题方法：`numeric_tolerance` / `multi_choice_verified` / `semantic_entailment`
- `main.py` — 注册 `GET /api/v14/runs/{id}/judge-chain` 路由

### Test Coverage
- **346 passed, 4 skipped**（+28 vs Phase 3 的 318）

---

## [v14.0.0-phase3] — 2026-04-19

### Added
- `backend/app/_data/model_taxonomy.yaml` — 16 家族模型分类表（claude/gpt/gemini/qwen/deepseek/glm/doubao/ernie/kimi/kiro/minimax/baichuan/yi/iflytek/mistral/llama），每条含 `official_names`/`internal_codenames`/`refusal_signatures`/`style_keywords`/`source_url`/`license`
- `backend/app/predetect/identity_exposure.py` — 真实模型暴露引擎：`analyze_responses()` 贝叶斯后验推断（softmax over raw signal scores），`Layer17IdentityExposure` 零 Token 预检测层，碰撞阈值 0.80，信号权重 official_names:3.0 / internal_codenames:2.0 / refusal_signatures:2.5 / style_keywords:1.0
- `backend/app/predetect/system_prompt_harvester.py` — 系统提示词抽取：Tier1（强单模式匹配）+ Tier2（≥2 结构模式匹配）双级检测，`harvest()` 返回 `HarvestResult`，自动脱敏 URL/Base64/API Key/UUID
- `core/db_migrations.py Migration004V14IdentityExposureColumn` — 安全添加 `identity_exposure_result TEXT` 列（`ALTER TABLE test_runs ADD COLUMN IF NOT EXISTS`）
- `repository/repo.py` — 新增 `save_identity_exposure(run_id, report_dict)` + `get_identity_exposure(run_id)`
- `GET /api/v14/model-taxonomy` — 返回完整 model_taxonomy.yaml 内容（JSON）
- `GET /api/v14/runs/{id}/identity-exposure` — 返回指定 run 的 `IdentityExposureReport`，支持 lazy backfill（对 Phase 3 前的历史 run 自动补分析）
- `GET /api/v14/runs/{id}/system-prompt` — 返回从 run 中提取的系统提示词（已脱敏）
- `IdentityExposureReport` dataclass — 新增至 `core/schemas.py`：`claimed_model`/`claimed_family`/`identity_collision`/`collision_confidence`/`top_families`/`extracted_system_prompt`/`total_responses_scanned`
- `frontend/index.html` — 新增"疑似实际模型"卡片 `<div id="identity-exposure-card">`
- `frontend/app.js` — 新增 `renderIdentityExposure()` / `toggleIdentityCard()` / `escapeHtml()`，挂载至 `loadReport()`
- `tests/test_v14_phase3.py` — 24 条验收测试（全部通过）

### Changed
- `predetect/pipeline.py` — 新增 L17 Identity Exposure 层（Deep 模式，零 Token，复用前层证据）
- `runner/report_assembly.py` — 测试完成后自动运行身份暴露分析 + 系统提示词抽取，结果持久化至 DB（非致命，失败仅 warning）
- `handlers/v14_handlers.py` — 新增 `handle_model_taxonomy` / `handle_identity_exposure` / `handle_system_prompt`
- `main.py` — 注册 3 条新路由，`_handle_v14_health` 更新提及 Phase 3

### Test Coverage
- **318 passed, 4 skipped**（+24 vs Phase 2 的 294）

---

## [v14.0.0-phase2] — 2026-04-19

### Added
- `ScoreCard.completeness` — v14 字段：当轮测试中非 None 能力维度数 / 总维度数（0-1），`to_dict()` 写入 `v13.completeness`
- `GET /api/v14/bt-leaderboard` — Bradley-Terry 强度排行榜（MM 算法，Bradley & Terry 1952 Biometrika 39:324），从 compare_runs 数据拟合，无数据时 fallback 到 ELO 分
- `handlers/v14_handlers.py` — v14 专属处理器模块（`_compute_bradley_terry` + `handle_bt_leaderboard`）
- `scripts/fit_weights_v14.py` — NNLS 能力维度权重拟合脚本（内嵌 HELM v1.10 top-8 参考数据，R²=0.9576）
- `SOURCES.yaml` 新增 6 条 verdict cap 溯源记录：`verdict.difficulty_cap` / `behavioral_invariant_cap` / `coding_zero_cap` / `identity_exposed_cap` / `extraction_weak_cap` / `fingerprint_mismatch_cap`
- `tests/test_v14_phase2.py` — 26 条验收测试（全部通过）

### Changed
- **消除 19 处 `return 50.0` 假数据兜底**（scoring.py ×6、score_calculator.py ×4、shapley_attribution.py ×2、attribution.py ×1、adaptive_scoring.py ×2、estimation.py ×1、_speed_score fallback ×1、_extraction_resistance zero-weight ×1）；空输入时统一返回 `None`，调用方重新归一化权重
- `ScoreCard` 字段类型修正为 `float | None`：`reasoning_score`、`adversarial_reasoning_score`、`coding_score`、`similarity_to_claimed`、`speed_score`、`stability_score`（默认 `None` 而非 `0.0`）
- `ScoreCard.to_dict()` 现对 `None` 维度输出 `null`（而非 `0`），前端可显示 "N/A"
- `total_score` 计算改为归一化加权（仅对非 None 的顶层维度归一化），消除零填充偏差
- `authenticity_score` 计算同步改为字典式归一化（处理 `similarity_to_claimed=None`）
- `SOURCES.yaml` 8 条 `capability.weight.*.default` 更新为 NNLS 拟合值（reasoning: 0.0000, adversarial: 0.0968, instruction: 0.2492, coding: 0.2571, safety: 0.0190, protocol: 0.0690, knowledge: 0.0781, tool_use: 0.2307）；`phase2_replace: false`
- `SOURCES.yaml` 3 条 `scorecard.weight.*` 更新 `phase2_replace: false`（值 0.45/0.30/0.25 经 NNLS 确认）
- `verdicts.py` `_SRC_KEY_MAP` 补全 6 条 cap 映射，移除所有 `# TODO: derived cap` 注释
- `verdicts.py` `_RULE_FALLBACKS` 注释更新为 `# SRC: verdict.<key>`（溯源完整）

### Test Coverage
- **294 passed, 4 skipped**（+26 vs Phase 1 的 268）

---

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
