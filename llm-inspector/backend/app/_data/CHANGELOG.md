# Changelog

All notable changes to LLM Inspector are documented in this file.
Format follows [Keep a Changelog 1.1.0](https://keepachangelog.com/).

## [v16.0.0] - 2026-04-26

### Added

- **Phase 0**: Code audit and cleanup — removed dead imports, unified version to v16
- **Phase 1**: Preflight error taxonomy with 9 new ErrorCode entries, upstream error transparency (error_payload, http_status in LLMResponse)
- **Phase 1.5**: OfficialEndpoint triple-consistency check (URL + TLS + response headers)
- **Phase 2**: Layered retry (5xx/truncation/JSON decode), case-level remediation rounds, excluded_from_scoring flag
- **Phase 3**: Weighted ECE, Bradley-Terry weight fitting, ScoreCard coverage/weight_provenance_trace/weighted_ece/excluded_case_count fields
- **Phase 4**: `/v1/models` probe (ModelDiscovery), system prompt enhanced extraction (Repeat-back/JSON-mode/Token-economy/Multi-turn), _SECRET_PATTERNS sanitization, RealModelCard + Evidence data structures
- **Phase 5**: IRT cold-start prior table (6 categories × 4 difficulty levels), dataset license validation, expand_to_v16 with GPQA/AIME/LiveCodeBench/HumanEval+/MMLU-Pro/TruthfulQA
- **Phase 6**: Dual-source KG verification (Wikidata + DBpedia), dynamic question generation, kg_conflict/degraded fields, verify_with_degradation graceful fallback
- **Phase 7**: Prompt compression (LLMLingua-2 lightweight), cache key includes model_name, TokenAuditTracker with JSONL output
- **Phase 8**: 8 new SSE event types, TraceWriter for standardized JSONL traces (preflight/predetect/judge_chain/errors/token_audit)
- **Phase 9**: Real Model Card UI template, toast notification upgrade (hint/actions/fatal level), mobile breakpoint, risk-badge styling
- **Phase 10**: Migration guide, CHANGELOG, version.json update
- **Phase 11**: Bayesian evidence-weighted VerdictEngine (5 up + 5 down symmetric rules), VerdictReport with P(fake)/CI/borderline/inconclusive, discrimination_audit.py (Spearman/kappa/discrimination_index), EWMA reference updater (stale_after_days=90/discard=180), Verdict Explainer frontend panel

### Changed

- `RetryConfig` now includes `jitter_ratio` (default 0.1) for exponential backoff jitter
- `WikidataClient` User-Agent updated to v16 per Wikimedia policy
- `cache_strategy.build_key()` now accepts `model_name` parameter
- `showToast()` in frontend now supports `hint` and `actions` parameters
- Replaced `identity-exposure-tpl` template with `report-real-model-card` template

### Removed

- Duplicate `ErrorCode.E_MODEL_NOT_FOUND` definition
- Old v14 `identity-exposure-tpl` template

### Fixed

- `fit_weights.py` Bradley-Terry empty input edge case (zero-size array)
- `fit_weights.py` `global DIMS` syntax error replaced with local `active_dims`
- `test_v14_phase7.py` jitter-induced assertion failures (set `jitter_ratio=0.0`)
- `test_v15_phase0.py` and `test_v15_phase10.py` version string assertions updated to v16
- **SSL UNEXPECTED_EOF_WHILE_READING**: adapters now use certifi CA bundle (not Python's outdated built-in certs); `ssl.SSLError` and `URLError`-wrapped SSL errors are classified as `ssl_error` and retried
- **v16 suite missing cases**: `load_cases()` now recognizes v16 as composite suite (v10+v13+v15+v16_test_comm+v16_test_nc), standard mode gets 177 cases instead of 39
- **Truncated responses skipped by judge**: `finish_reason=length` no longer auto-skips judging for content-rich methods (constraint_reasoning, semantic, etc.); only format-strict judges (exact_match, regex_match, etc.) skip
- **DB path CWD-dependence**: `DATABASE_URL` default changed from relative `./llm_inspector.db` to absolute path anchored to `backend/_data/`, eliminating stale DB files scattered by different CWDs
- **identity_consistency regex double-escape**: `r'\\b'` (literal backslash-b) changed to `r'\b'` (word boundary); consistency cases like "Jupiter"/"4" were always failing
- **identity_consistency negation detection**: "I'm not Claude" no longer passes `expected='claude'`; added negation pattern matching for English and Chinese
- **semantic/semantic_match judge aliases**: v16 suite GPQA/TruthfulQA cases using `judge_method: "semantic"` and `"semantic_match"` now correctly dispatch to `semantic_entailment_judge` and `_semantic_judge`
- **OfficialEndpoint not wired to VerdictEngine**: `VerdictEngine.assess()` now reads `predetect.routing_info.official_endpoint`; when verified, boosts `confidence_real` by up to 15 points and relaxes hard-rule caps (behavioral_invariant 55→70, extraction_weak 65→80)
- **Extraction resistance asymmetric None handling**: `judge_passed=None` samples now skipped in `_extraction_resistance()` instead of being counted as "failed resistance" for non-leak judges
- **reason_rope_001 missing target_pattern**: Added `target_pattern="无解|无法"` in DB so it pairs with `reason_rope_iso_007` for behavioral invariance scoring
- **OfficialEndpoint TLS cert parse crash**: `dict(peercert.get("subject", []))` fails on tuple-of-tuples format; added try/except with safe fallback, TLS check now returns `True` for valid SSL connections
- **OfficialEndpoint evasion_indicators YAML parse crash**: YAML list-of-dicts not converted to flat dict before `.get()` calls; added list→dict merge logic (was already partially present but applied after the crash point)
- **PreDetect failure drops official_endpoint result**: `_step_predetect()` exception handler returned empty `PreDetectionResult` without saving to DB or attempting official endpoint check; now saves fallback result with `routing_info` containing official endpoint verification

### Acceptance audit (2026-04-27)

- **OfficialEndpoint TLS-latency false-positive**: `evasion_indicators.response_latency_anomaly_ms` raised from 500 ms to 2000 ms (RFC 8446 §2 + Cloudflare Radar normal trans-continental TLS handshake 600–1100 ms); per-signal penalty lowered from −0.15 to −0.05 so a single soft anomaly can no longer flip a 3-factor pass below `OFFICIAL_ENDPOINT_MIN_CONFIDENCE=0.85`. Verified: `https://api.deepseek.com/v1` now returns `verified=True confidence=1.00` from China network (previously `verified=False` due to 953 ms handshake).
- **Frontend OFFICIAL badge missing**: `predetectHtml()` did not render `routing_info.official_endpoint` despite backend populating it; added `officialHtml` block above `routingHtml` showing OFFICIAL ✅ badge with provider, confidence %, four-factor breakdown and soft-signal list. Partial match (URL ✓ but other checks fail) renders an amber "reverse-proxy suspected" warning.

- **CRITICAL — v16 Phase 5 stub-prompt regression (110 cases)**: `suite_v16_test_comm.json` and `suite_v16_test_nc.json` shipped with placeholder prompts (`"GPQA #1"`, `"LCB #1"`, `"MMLU #1"`, `"TQ #1"` etc.) instead of the real GPQA Diamond / LiveCodeBench / MMLU-Pro / TruthfulQA questions, causing **0% pass on coding/safety/knowledge/reasoning** for any v16 standard run that included them (the model's response is uniformly "请提供更多上下文"). Triple mitigation:
  1. `repo.load_cases("v16", ...)` composite hard-pinned to `("v10","v13","v15")` — stub suites no longer pulled into v16.
  2. `tasks.seeder._seed_test_cases` force-sets `enabled=False` for any case from a stub suite file at seed time, so a fresh boot can never re-enable them.
  3. All 60 already-seeded `v16_test_nc` rows in DB updated to `enabled=0`.
  Regression guard: `backend/tests/test_v16_no_stub_prompts.py` (4 tests, all pass) checks the composite tuple, the seeder gate, and asserts `repo.load_cases` returns zero stub-style prompts.
  Follow-up: a real importer (network-aware, license-aware) is required before re-enabling these suites — tracked separately, NOT shipping in v16.0.0.

### Security

- `_SECRET_PATTERNS` redaction for GitHub tokens, AWS keys, JWTs, Bearer tokens, Google API keys, Slack tokens
- System prompt sanitization now applies secondary `_SECRET_PATTERNS` pass
