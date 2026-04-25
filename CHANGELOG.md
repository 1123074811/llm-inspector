# Changelog

All notable changes to LLM Inspector are documented here.

## [v15.0.0] — 2026-04-25

### Added
- `docs/MIGRATION_v14_to_v15.md` — v14→v15 迁移指南（API 变更 / 新模块清单 / DB 迁移说明 / 测试文件索引）

### Changed
- `CLAUDE.md` / `README.md` / `CHANGELOG.md` — 同步所有 Phase 0–13 结构变更
- 全量回归：`pytest backend/tests/ -q` → **825 passed, 4 skipped** ✅

---

## [v15.0.0-phase13] — 2026-04-25

### Added
- `backend/app/preflight/error_taxonomy.py` — 预检错误分类：`ErrorCode` StrEnum（15 个代码）、`ErrorDetail` dataclass、`_ERROR_DETAILS` 映射表（含 retryable 标志 + 中英双语消息）、`make_error()` 工厂函数（支持 `raw_status`/`raw_body` 摘录/`source_layer`）
- `backend/app/preflight/connection_check.py` — 预检连通五步执行器：`PreflightStep`/`PreflightReport` dataclass（含 `to_dict()`）、`_check_inputs()` URL/Key/Model 静态校验、`_check_schema()` 响应 JSON 结构校验、`_note_capabilities()` 能力探针记录、`_http_status_to_error_code()` HTTP 状态→错误码映射（含 400 body 内容嗅探）、`run_preflight()` 全流程执行（超时控制、短路传播）
- `backend/tests/test_v15_phase13.py` — 31 条验收测试（error_taxonomy + connection_check，全部通过）

### Test Coverage
- **825 passed, 4 skipped**（+31 vs Phase 12 的 794）

---

## [v15.0.0-phase12] — 2026-04-25

### Added
- `backend/app/_data/judge_registry.yaml` — 评判方法注册表 YAML：含 `exact_match`/`regex`/`semantic_v2` 等方法的 mode/description/applicable_for/biases/references 字段
- `backend/app/analysis/judge_registry.py` — 注册中心模块：`list_methods()` / `get_method(name)` / `methods_by_mode(mode)` / `applicable_for(question_type)` / `registry_summary()`；PyYAML 加载 + stdlib fallback；`@lru_cache(maxsize=1)` 单例懒加载
- `GET /api/v15/judge-registry` — 列出所有注册评判方法（`registry_summary()` 响应）
- `GET /api/v15/judge-registry/{method}` — 单个方法详情（404 on unknown）
- `backend/tests/test_v15_phase12.py` — 33 条验收测试（YAML 内容/模块函数/handler，全部通过）

### Changed
- `handlers/v15_handlers.py` — 新增 `handle_judge_registry` + `handle_judge_registry_method`
- `main.py` — 注册两条新路由

### Test Coverage
- **794 passed, 4 skipped**（+33 vs Phase 11 的 761）

---

## [v15.0.0-phase11] — 2026-04-25

### Added
- `backend/app/runner/import_dataset.py` — 数据集导入管线：`ImportReport` dataclass（total_imported / skipped_duplicates / validation_errors / source_datasets / new_categories / to_dict()）、`DatasetImporter` 类（`SUITE_DIR` 类变量、`generate_id()` slug 生成、`validate_case()` 字段+类别+judge_method 校验、`load_suite()` / `save_suite()` 读写 suite_v*.json、`import_cases()` 支持 overwrite + 自动补全默认值）
- `POST /api/v15/dataset/import` — 批量导入测试用例；返回 `ImportReport.to_dict()`
- `POST /api/v15/dataset/validate` — 验证单条用例格式；返回 `{valid, error}`
- `backend/tests/test_v15_phase11.py` — 37 条验收测试（全部通过）

### Changed
- `handlers/v15_handlers.py` — 新增 `handle_import_dataset` + `handle_validate_case`
- `main.py` — 注册两条新路由

### Test Coverage
- **761 passed, 4 skipped**（+37 vs Phase 10 的 724）

---

## [v15.0.0-phase10] — 2026-04-25

### Added
- `backend/app/runner/cache_strategy.py` — 响应缓存策略：`CacheStrategy`（SHA-256 key = base_url+payload hash，仅缓存 `temperature=0`，按 category 配置 TTL，SQLite 持久化，`evict_expired()` 惰性清理）、`CacheMetrics` dataclass + `snapshot()`、`build_key()` / `get()` / `set()` 接口
- `backend/app/analysis/narrative_builder.py` — 纯规则叙事生成器：`NarrativeBuilder.build(report_dict)` 从结构化报告生成中文摘要文本，零 Token 消耗
- `GET /api/v15/cache-stats` — 全局缓存指标快照
- `POST /api/v15/cache/evict` — 手动驱逐过期缓存条目
- `GET /api/v15/runs/{id}/token-audit` — Token 效率审计（token_audit + cache_metrics）
- `core/db_migrations.py Migration007V15CacheTable` — 幂等创建 `llm_response_cache` 表（cache_key / response_json / created_at / expires_at）
- `backend/tests/test_v15_phase10.py` — 18 条验收测试（全部通过）

### Changed
- `runner/case_executor.py` — CacheStrategy 接入：temperature=0 时自动查缓存/写缓存（non-fatal try/except）
- `tasks/seeder.py` — 新增加载 `suite_v13.json` 和 `suite_v15.json`
- `_data/version.json` — 版本号更新为 `v15.0.0`，phases_complete 列表补全 phase10-13

### Test Coverage
- **724 passed, 4 skipped**（+18 vs Phase 9 的 706）

---

## [v15.0.0-phase9] — 2026-04-25

### Added
- `backend/app/analysis/judge_calibration.py` — 评判校准工具：`compute_fleiss_kappa(rating_matrix)` 多评判器 Fleiss's κ（Fleiss 1971）、`compute_cohen_kappa(a, b)` 双评判器 Cohen's κ（Cohen 1960）、`judge_bias_detection(predictions, labels)` 系统性偏差检测（precision/recall/F1 + 偏差类型分类）
- `backend/tests/test_v15_phase9.py` — 24 条验收测试（全部通过）

### Test Coverage
- **706 passed, 4 skipped**（+24 vs Phase 8 的 682）

---

## [v15.0.0-phase8] — 2026-04-25

### Added
- `backend/app/analysis/uncertainty.py` — 不确定性量化：`bootstrap_ci(data, n_bootstrap, ci)` Bootstrap 置信区间、`sem(data)` 测量标准误、`hdi(data, credible_mass)` 最高密度区间（HDI）、`weighted_ci(data, weights, ci)` 加权置信区间
- `backend/tests/test_v15_phase8.py` — 27 条验收测试（全部通过）

### Test Coverage
- **682 passed, 4 skipped**（+27 vs Phase 7 的 655）

---

## [v15.0.0-phase7] — 2026-04-25

### Added
- `backend/app/analysis/calibration_metrics.py` — 校准指标计算：`brier_score(probs, labels)` Brier 分（越低越好）、`log_loss(probs, labels)` 对数损失、`ece(probs, labels, n_bins)` 期望校准误差（ECE）、`reliability_curve(probs, labels, n_bins)` 可靠性曲线（bin 中点/平均概率/平均准确率）
- `backend/tests/test_v15_phase7.py` — 25 条验收测试（全部通过）

### Test Coverage
- **655 passed, 4 skipped**（+25 vs Phase 6 的 630）

---

## [v15.0.0-phase6] — 2026-04-25

### Added
- `backend/app/predetect/layer_l20_self_paradox.py` — L20 自我矛盾探针：多轮对话诱导矛盾声明，贝叶斯后验估计一致性置信度；仅 Deep 模式运行
- `backend/app/predetect/layer_l21_multistep_drift.py` — L21 多步漂移检测：上下文积压后检测答案语义偏移（余弦相似度 + Jaccard 重叠）；仅 Deep 模式
- `backend/app/predetect/layer_l22_prompt_reconstruct.py` — L22 提示词重构：反推系统提示词结构关键词；仅 Deep 模式
- `backend/app/predetect/layer_l23_adversarial_tools.py` — L23 对抗性 Tool-Call：注入恶意 tool schema 探测函数调用拦截能力；仅 Deep 模式；`extra_params={"tool_choice":"auto"}` 传递非标准字段
- `backend/tests/test_v15_phase6.py` — 27 条验收测试（全部通过）

### Changed
- `predetect/pipeline.py` — Deep 模式新增 L20-L23 执行块（运行于 L19 之后）

### Test Coverage
- **630 passed, 4 skipped**（+27 vs Phase 5 的 603）

---

## [v15.0.0-phase5] — 2026-04-24

### Added
- `backend/app/preflight/__init__.py` — preflight 包初始化
- `backend/app/preflight/connection_check.py`（初版）— A1-A5 预检连通五步基础框架
- `GET /api/v15/runs/{id}/preflight` — 预检结果查询端点
- `handlers/v15_handlers.py` `handle_get_preflight_result` — 对应 handler

### Test Coverage
- **603 passed, 4 skipped**（基础，含 Phase 0-4 测试）

---

## [v15.0.0-phase4] — 2026-04-24

### Changed / Fixed
- `predetect/layer_l23_adversarial_tools.py` — **Bug 修复**：`LLMRequest()` 错误传入 `tool_choice="auto"` 非法字段，改为 `extra_params={"tool_choice": "auto"}`
- `core/schemas.py` `ScoreCard` — 多处 `None` 语义修正，消除 v14 遗留假数据兜底
- 多处兼容性修复确保 Phase 0-3 测试在 v15 baseline 上全部通过

### Test Coverage
- **603 passed, 4 skipped**

---

## [v15.0.0-phase3] — 2026-04-24

### Added
- `backend/app/analysis/narrative_builder.py`（初版）— 纯规则叙事生成器核心逻辑

### Test Coverage
- **603 passed, 4 skipped**

---

## [v15.0.0-phase2] — 2026-04-24

### Added
- `backend/app/authenticity/model_card_diff.py` — 模型卡差异对比：`ModelCardDiff.build()` 对比声称模型 vs 疑似模型的能力声明差异，`risk_level`（low/medium/high/critical）、`wrapper_probability`、`diff_items` 列表
- `GET /api/v15/runs/{id}/model-card-diff` — 返回 `ModelCardDiff.to_dict()`

### Changed
- `handlers/v15_handlers.py` — 新增 `handle_get_model_card_diff`
- `main.py` — 注册新路由

### Test Coverage
- **603 passed, 4 skipped**

---

## [v15.0.0-phase1] — 2026-04-24

### Added
- `backend/app/authenticity/__init__.py` — authenticity 包初始化
- `backend/app/authenticity/evidence_ledger.py` — 贝叶斯证据台账：`EvidenceItem` dataclass（signal/weight/confidence/source_layer）、`EvidenceLedger`（`add_evidence()` / `wrapper_probability()` 贝叶斯奇数融合 / `risk_level()` / `suspected_actual_model()` / `to_dict()`）、`extract_evidence_from_predetect()` 从 PreDetectionResult 提取证据
- `GET /api/v15/runs/{id}/evidence-ledger` — 返回 `EvidenceLedger.to_dict()`

### Changed
- `handlers/v15_handlers.py` — 新增 `handle_get_evidence_ledger`
- `main.py` — 注册新路由

### Test Coverage
- **603 passed, 4 skipped**

---

## [v15.0.0-phase0] — 2026-04-24

### Added
- `backend/app/fixtures/suite_v15.json` — v15 基础测试套件（初始 8 条用例，通过导入管线持续追加）
- `backend/app/_data/version.json` — 版本文件（`{"version":"v15.0.0","phases_complete":[...],"built_at":"..."}`）
- `GET /api/v15/health` — v15 命名空间健康检查（返回 `api_version: "v15"` + version.json 信息）

### Changed
- `core/db_migrations.py` — 注册 `Migration007V15CacheTable`（`llm_response_cache` 表，幂等建表）
- `tasks/seeder.py` — 加载循环新增 `suite_v13.json` + `suite_v15.json`

### Test Coverage
- 基础建立，后续阶段累积至 **825 passed, 4 skipped**

---

## [v14.0.0-phase9] — 2026-04-20

### Added
- `docs/MIGRATION_v13_to_v14.md` — v13→v14 迁移指南（API 变更 / ScoreCard null 化 / 前端 N/A 适配 / 判题方法变化 / DB 迁移说明）

### Changed
- `CLAUDE.md` / `README.md` / `CHANGELOG.md` — 同步所有 Phase 1–8 结构变更
- 全量回归：`pytest backend/tests/ -q` → **471 passed, 4 skipped** ✅

---

## [v14.0.0-phase8] — 2026-04-20

### Added
- `frontend/app.js` `safeFetch(url, options)` — 全局错误边界：404/500/网络断开统一触发 Toast 提示，消除白屏
- `frontend/app.js` `showToast(message, type)` — 类型化 Toast（info/error/warn），3.5s 自动消除
- `frontend/app.js` `fmtScore(v, digits)` — null/undefined → `'N/A'`，消除假 0 / 假 50 展示
- `frontend/app.js` `renderV14Cards(report, runId)` — Token 分析卡片 + 数据完整性进度条 + PreDetect 20 层状态点
- `frontend/app.js` `renderBarChartFallback(dims)` — 雷达图维度 < 5 时降级为条形图展示
- `frontend/styles.css` — Toast 动画样式（`.toast`/`.toast-info`/`.toast-error`/`.toast-warn`/`@keyframes fadeInUp`）+ `.v14-progress-bar`
- `frontend/index.html` — 新增 `<div id="toast-container">`
- `backend/tests/test_v14_phase8.py` — 16 条验收测试（ScoreCard v14 字段 / null 发射 / 排行榜分页 / 健康端点）

### Changed
- `frontend/app.js` — 排行榜分页（20 条/页，Prev/Next + 客户端模型名搜索）
- `frontend/app.js` — 雷达图渲染前过滤 null 维度，< 5 维触发条形图降级
- `frontend/app.js` — `loadReport()` / `loadLeaderboard()` 改用 `safeFetch()`
- `handlers/models.py` — `/api/v1/leaderboard` 支持 `offset` 分页参数
- 模式描述"预计消耗"→"最大预算"

### Test Coverage
- **471 passed, 4 skipped**（+16 vs Phase 7 的 455）

---

## [v14.0.0-phase7] — 2026-04-20

### Added
- `backend/app/runner/retry_policy.py` — 分级重试策略：`RetryConfig`（max_retries_network=3 / max_retries_429=2 / 指数退避 0.5s→8s）；`with_retry()` 自动区分网络超时/HTTP 429/不可重试错误；每次重试事件写 `data/traces/{run_id}/errors.jsonl`；引用 Google SRE Book Ch.22 + RFC 7231 §7.1.3
- `ScoreCard.skipped_cases: list[str]` — 被跳过用例的 ID 明细列表；`to_dict()` 同步输出
- `GET /api/v14/circuit-breaker/history` — 返回断路器最近 100 条状态转换事件（open/close/half_open/reset）
- `handlers/v14_handlers.py` `handle_circuit_breaker_history`
- `tests/test_v14_phase7.py` — 32 条验收测试（全部通过）

### Changed / Fixed
- `repository/repo.py` — **B7 修复**：`update_run_progress(completed, total, skipped=0)` 进度公式改为 `round(100 * completed / max(1, total - skipped))`，消除因合法跳过导致进度卡在 95%~98% 的问题；新增 `list_stale_runs(limit, after_id)` 游标分页接口
- `tasks/watchdog.py` — **B8 修复**：用游标分页循环替换原 `limit=500` 单次查询，全库扫描不遗漏
- `core/circuit_breaker.py` — 新增 `_cb_event_history` 环形缓冲区（max=100）+ `_record_cb_event()` + `get_cb_event_history()`，所有状态转换自动记录
- `core/config.py` — 新增 `RUN_MAX_DURATION_SEC`（默认 3600s，可通过环境变量热更新）
- `main.py` — 注册 `GET /api/v14/circuit-breaker/history` 路由

### Test Coverage
- **455 passed, 4 skipped**（+32 vs Phase 6 的 423）

---

## [v14.0.0-phase6] — 2026-04-20

### Added
- `backend/app/runner/adaptive_sampling.py` — IRT 2PL 信息量驱动动态采样数：`item_information(θ,a,b,c)`（Fisher 信息公式）；`adaptive_n_samples()` 阈值规则（I>1.0→n=1 / I>0.5→n=2 / I≤0.5→n=3）；`get_adaptive_n_samples(case, theta)` 便捷封装；引用 van der Linden & Glas 2010 + Weiss & Kingsbury 1984
- `backend/app/runner/token_counter.py` — Token 精确计数：优先 tiktoken（o200k/cl100k 自动检测），fallback `len(text)//4`；`count_tokens(text, model)` 返回 `(count, method)` 元组；`count_messages_tokens(messages)` 含 3-token/message 格式开销；引用 OpenAI Cookbook
- `ScoreCard` 新增字段：`prompt_optimizer_used: bool`、`tokens_saved_estimate: int | None`、`token_counting_method: str`
- `ScoreCard.to_dict()` 新增 `"token_analysis"` 子块（optimizer_used / savings / counting_method）
- `GET /api/v14/runs/{id}/token-analysis` — 返回 run 的 Token 使用分析（预算 vs 实际 + 优化器节省）
- `handlers/v14_handlers.py` `handle_token_analysis` — 对应 handler
- `tests/test_v14_phase6.py` — 37 条验收测试（全部通过）

### Changed / Fixed
- `runner/case_executor.py` — **B5 修复**：PromptOptimizer.compile_prompt() 入链（non-fatal try/except），动态 TF-IDF Few-Shot 示例注入；自适应采样入链（non-fatal），高信息量用例自动减少 n_samples
- `runner/prompt_optimizer.py` — 修复 `_retrieve_tfidf_with_scores()` 返回值 bug（原本 strip 了 score tuple，导致 `_retrieve_hybrid()` TypeError）
- `main.py` — 注册 `GET /api/v14/runs/[^/]+/token-analysis` 路由

### Test Coverage
- **423 passed, 4 skipped**（+37 vs Phase 5 的 386）

---

## [v14.0.0-phase5] — 2026-04-20

### Added
- `backend/app/predetect/layers_l18_l19.py` — L18 Response Timing Side-Channel（零 Token；重分析前层 TTFT/TPS 数据；Gaussian KL 散度对比 6 家族参考分布；置信度上限 0.50；引用 Yu et al. 2024 timing fingerprinting）+ L19 Token Distribution Side-Channel（零 Token；4-gram 重复率；响应长度 Wasserstein 距离；置信度上限 0.45；引用 Carlini et al. 2023 arXiv:2403.06634）
- `core/events.py` 新增 `EventKind.PREDETECT_LAYER_TRACE` — 每层预检测完成后发射 SSE 事件
- `GET /api/v14/runs/{id}/predetect-trace?offset=N&limit=N` — 分页读取预检测 JSONL 日志
- `handlers/v14_handlers.py` `handle_predetect_trace` — 对应 handler
- `repository/repo.py` `get_predetect_trace_path()` + `read_predetect_trace()` — JSONL 读取工具
- `tests/test_v14_phase5.py` — 40 条验收测试（全部通过）

### Changed
- `predetect/pipeline.py` — 新增 `_write_predetect_trace()` JSONL sink（非致命；每层写入 `data/traces/{run_id}/predetect.jsonl`；同步发射 SSE `PREDETECT_LAYER_TRACE` 事件）；Deep 模式新增 L18 + L19 执行块（运行于 L17 之后）；`run()` 接受可选 `run_id` 参数用于日志落盘
- `main.py` — 注册 `GET /api/v14/runs/[^/]+/predetect-trace` 路由

### Test Coverage
- **386 passed, 4 skipped**（+40 vs Phase 4 的 346）

---

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
