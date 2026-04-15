"""
runner/orchestrator.py — main pipeline entry points

Contains only the three public entry points:
  run_pipeline()          — full sync pipeline
  continue_pipeline()     — resume a pre_detected run
  skip_testing_pipeline() — skip testing, generate report from predetect only

All helper functions/classes have been extracted to:
  budget_control.py  — SmartBudget, SmartModeStrategy, TokenBudgetGuard
  case_prep.py       — _load_suite, _load_benchmarks, _prepare_cases, etc.
  execution.py       — _run_cases_concurrent, _checkpoint_should_stop, etc.
  report_assembly.py — _build_and_save_report
  compare.py         — run_compare_pipeline, A/B significance
  async_runner.py    — run_pipeline_async, _run_cases_async
"""
from __future__ import annotations

import time

from app.core.schemas import (
    TestCase, CaseResult, PreDetectionResult,
)
from app.core.eval_schemas import EvalTestCase
from app.core.circuit_breaker import circuit_breaker, CircuitState
from app.core.tracer import get_tracer, remove_tracer
from app.core.logging import get_logger
from app.core.security import get_key_manager
from app.core.config import settings
from app.adapters.openai_compat import OpenAICompatibleAdapter
from app.predetect.pipeline import PreDetectionPipeline
from app.repository import repo

# ── Re-exports from split modules (for backward compatibility) ─────────────────
from app.runner.budget_control import (                    # noqa: F401
    FAMILY_DISCRIMINATORS, SmartBudget, SmartModeStrategy, TokenBudgetGuard,
)
from app.runner.case_prep import (                         # noqa: F401
    _FIXTURES_DIR, _benchmark_cache, _load_suite, _load_benchmarks,
    _save_case_results_batch, _mode_concurrency, _case_value,
    _adaptive_samples, _CONFIRMATORY_CATEGORIES, _select_confirmatory_cases,
    _prepare_cases,
)
from app.runner.execution import (                         # noqa: F401
    _adaptive_pause, _update_backoff, _checkpoint_should_stop,
    _run_cases_concurrent,
)
from app.runner.report_assembly import _build_and_save_report  # noqa: F401
from app.runner.compare import (                           # noqa: F401
    run_compare_pipeline, _compute_ab_significance, _paired_bootstrap,
)
from app.runner.async_runner import (                      # noqa: F401
    _run_cases_async, run_pipeline_async,
)

logger = get_logger(__name__)


def run_pipeline(run_id: str) -> None:
    """
    Full pipeline (v11 — with circuit breaker + tracing):
      1. Pre-detection (0-token identification attempt)
      2. Connectivity check
      3. Test case execution (with early-stop, circuit breaker)
      4. Feature extraction + scoring
      5. Similarity + risk assessment
      6. Report generation
    """
    logger.info("Pipeline starting", run_id=run_id)

    # ── v11: Initialize tracer ────────────────────────────────────────────────
    tracer = get_tracer(run_id)
    tracer.start()

    # ── Load run metadata ────────────────────────────────────────────────────
    run = repo.get_run(run_id)
    if not run:
        logger.error("Run not found", run_id=run_id)
        remove_tracer(run_id)
        return

    km = get_key_manager()
    try:
        api_key = km.decrypt(run["api_key_encrypted"])
    except Exception as e:
        repo.update_run_status(run_id, "failed", error_message=f"Key decrypt failed: {e}")
        remove_tracer(run_id)
        return

    adapter = OpenAICompatibleAdapter(run["base_url"], api_key)
    base_url = run["base_url"]
    suite_version = run.get("suite_version", "v1")
    test_mode = run.get("test_mode", "standard")

    # ── v11: Check circuit breaker before proceeding ──────────────────────────
    if circuit_breaker.is_open(base_url):
        logger.warning("Circuit breaker is OPEN, suspending run", run_id=run_id, base_url=base_url)
        repo.update_run_status(run_id, "suspended", error_message="API circuit breaker open — run suspended for retry")
        tracer.add_event("circuit_open", base_url=base_url, action="suspended")
        remove_tracer(run_id)
        return

    # ── Step 1: Pre-detection ────────────────────────────────────────────────
    repo.update_run_status(run_id, "pre_detecting")
    logger.info("Running pre-detection", run_id=run_id)
    extraction_mode = test_mode == "deep"
    try:
        with tracer.span("predetect", mode=test_mode) as span:
            pre_result: PreDetectionResult = PreDetectionPipeline().run(
                adapter, run["model_name"],
                extraction_mode=extraction_mode,
                run_id=run_id,
            )
            span.set_attribute("identified_as", pre_result.identified_as)
            span.set_attribute("confidence", pre_result.confidence)
            span.set_attribute("tokens_used", pre_result.total_tokens_used)
            tracer.record_tokens("predetect", pre_result.total_tokens_used)
        repo.save_predetect_result(run_id, pre_result.to_dict())
        logger.info(
            "Pre-detection complete",
            run_id=run_id,
            identified=pre_result.identified_as,
            confidence=pre_result.confidence,
            tokens=pre_result.total_tokens_used,
        )
        # v11: Record success for circuit breaker
        circuit_breaker.record_success(base_url)
    except Exception as e:
        logger.warning("Pre-detection failed, continuing to full test", error=str(e))
        # v11: Record failure for circuit breaker
        circuit_breaker.record_failure(base_url, str(e))
        pre_result = PreDetectionResult(
            success=False, identified_as=None, confidence=0.0,
            layer_stopped=None, should_proceed_to_testing=True,
        )

    # If pre-detection is highly confident AND not extraction mode,
    # pause and let the user decide whether to continue full testing.
    # The user can POST /api/v1/runs/{id}/continue or /skip-testing.
    if not pre_result.should_proceed_to_testing and not extraction_mode:
        logger.info("Pre-detection sufficient, pausing for user decision", run_id=run_id)
        repo.update_run_status(run_id, "pre_detected")
        remove_tracer(run_id)
        return

    # ── Step 2: Connectivity check ───────────────────────────────────────────
    repo.update_run_status(run_id, "running")
    with tracer.span("connectivity", base_url=base_url) as conn_span:
        conn_check = adapter.list_models()
    conn_status = conn_check.get("status_code")
    conn_error = conn_check.get("error")

    # Hard fail: network-level error (no status_code means DNS/TCP failure)
    if conn_error and not conn_status:
        msg = (
            f"无法连接到 API：{conn_error}。"
            f"请检查 base_url 是否正确（当前：{run['base_url']}），"
            f"网络是否可达，以及 API Key 是否有效。"
        )
        repo.update_run_status(run_id, "failed", error_message=msg)
        logger.error("Connectivity failed (network)", run_id=run_id, error=msg)
        circuit_breaker.record_failure(base_url, f"network: {conn_error}")
        tracer.add_event("connectivity_failed", reason="network", error=conn_error)
        remove_tracer(run_id)
        return

    # Soft fail on /models (401/403/404): many providers (e.g. Baidu Qianfan
    # Coding Plan) don't expose a /models endpoint but /chat/completions works.
    # Fallback: send a minimal chat request with the actual model name.
    if conn_status and conn_status in (401, 403, 404):
        logger.info(
            "list_models returned error, probing chat endpoint as fallback",
            run_id=run_id,
            models_status=conn_status,
        )
        from app.core.schemas import LLMRequest, Message
        probe_req = LLMRequest(
            model=run["model_name"],
            messages=[Message(role="user", content="hi")],
            max_tokens=1,
            temperature=0.0,
            timeout_sec=15,
        )
        probe_resp = adapter.chat(probe_req)

        # Network-level failure on the chat endpoint
        if probe_resp.error_type and not probe_resp.status_code:
            msg = (
                f"无法连接到 API chat 端点：{probe_resp.error_message}。"
                f"请检查 base_url 是否正确（当前：{run['base_url']}），"
                f"网络是否可达，以及 API Key 是否有效。"
            )
            repo.update_run_status(run_id, "failed", error_message=msg)
            logger.error("Connectivity failed (chat probe network)", run_id=run_id, error=msg)
            circuit_breaker.record_failure(base_url, f"chat_probe_network: {probe_resp.error_message}")
            remove_tracer(run_id)
            return

        # Chat endpoint returns 401 → genuine auth failure
        if probe_resp.status_code == 401:
            msg = (
                f"API 鉴权失败：API Key 无效或未授权（HTTP 401）。"
                f"请检查 API Key 是否正确。base_url：{run['base_url']}"
            )
            repo.update_run_status(run_id, "failed", error_message=msg)
            logger.error("Connectivity failed (auth 401)", run_id=run_id)
            circuit_breaker.record_failure(base_url, "auth_401")
            remove_tracer(run_id)
            return

        # Chat endpoint returns 404 → endpoint path is wrong
        if probe_resp.status_code == 404:
            msg = (
                f"API 端点不存在（HTTP 404）。当前 base_url：{run['base_url']}，"
                f"系统将请求发送到 {run['base_url']}/chat/completions。"
                f"请确认该路径是否正确。例如百度千帆 Coding Plan 应填写 "
                f"https://qianfan.baidubce.com/v2/coding"
            )
            repo.update_run_status(run_id, "failed", error_message=msg)
            logger.error("Connectivity failed (404 on chat endpoint)", run_id=run_id)
            circuit_breaker.record_failure(base_url, "endpoint_404")
            remove_tracer(run_id)
            return

        # Any response (even 400/403/422/500) from the chat endpoint means it's
        # reachable. 403 with the real model name could mean rate-limit, model
        # access restriction, or parameter validation — not necessarily bad auth.
        # Continue to testing; the high-error-rate guard will catch persistent failures.
        logger.info(
            "Chat endpoint probe: endpoint reachable",
            run_id=run_id,
            probe_status=probe_resp.status_code,
            probe_ok=probe_resp.ok,
            probe_error=probe_resp.error_type,
        )

    # ── Step 3: Load test cases + execute ────────────────────────────────────
    with tracer.span("load_cases", suite=suite_version, mode=test_mode) as load_span:
        cases = _load_suite(suite_version, test_mode)
        load_span.set_attribute("total_cases", len(cases))
    if not cases:
        logger.warning("No test cases found", suite_version=suite_version)
        repo.update_run_status(run_id, "failed", error_message="No test cases loaded")
        remove_tracer(run_id)
        return

    # Targeted confirmation: when predetect has moderate confidence (0.60-0.84),
    # select only the cases with highest discriminative power for the candidate
    # model family, instead of running the full suite. This saves ~40% tokens
    # without losing detection precision — the selected cases are specifically
    # chosen to confirm or deny the predetect hypothesis.
    if (
        test_mode != "full"
        and pre_result.success
        and 0.60 <= pre_result.confidence < settings.PREDETECT_CONFIDENCE_THRESHOLD
        and pre_result.identified_as
    ):
        cases = _select_confirmatory_cases(cases, pre_result.identified_as)
        logger.info(
            "Targeted confirmation mode",
            run_id=run_id,
            candidate=pre_result.identified_as,
            selected_cases=len(cases),
        )

    phase1_cases, phase2_cases = _prepare_cases(cases, test_mode)

    # Resume support: skip already completed cases with at least one response
    existing_responses = repo.get_responses(run_id)
    completed_case_ids = {r.get("case_id") for r in existing_responses if r.get("case_id")}
    if completed_case_ids:
        phase1_cases = [c for c in phase1_cases if c.id not in completed_case_ids]
        phase2_cases = [c for c in phase2_cases if c.id not in completed_case_ids]

    case_results: list[CaseResult] = []
    failed_count_ref = {"count": 0}
    backoff_state = {"delay_ms": settings.INTER_REQUEST_DELAY_MS}

    # Token budget guard — initialized based on test mode
    budget_map = {
        "quick":      settings.TOKEN_BUDGET_QUICK,
        "standard":   settings.TOKEN_BUDGET_STANDARD,
        "deep":       settings.TOKEN_BUDGET_DEEP,
        "full":       settings.TOKEN_BUDGET_FULL,
        "extraction": settings.TOKEN_BUDGET_FULL,
        "smart":      13_000,  # Smart mode default budget
    }
    if test_mode == "smart" and pre_result.confidence > 0:
        smart_strategy = SmartModeStrategy()
        smart_budget = smart_strategy.decide_budget(pre_result)
        budget_guard = TokenBudgetGuard(smart_budget.token_budget)
        logger.info(
            "Smart mode budget allocated",
            run_id=run_id,
            budget=smart_budget.token_budget,
            description=smart_budget.description,
            phase1=smart_budget.phase1_size,
            phase2=smart_budget.phase2_size,
            phase3=smart_budget.phase3_size,
        )
    else:
        budget_guard = TokenBudgetGuard(budget_map.get(test_mode, settings.TOKEN_BUDGET_STANDARD))
    logger.info(
        "Executing test cases",
        run_id=run_id,
        total=len(phase1_cases) + len(phase2_cases),
        phase1=len(phase1_cases),
        phase2=len(phase2_cases),
        concurrency=_mode_concurrency(test_mode),
        token_budget=budget_guard.budget,
    )

    # v11: Phase 1 execution with tracing and circuit breaker
    with tracer.span("phase1", case_count=len(phase1_cases)) as p1_span:
        cancelled = _run_cases_concurrent(
            adapter=adapter,
            model_name=run["model_name"],
            cases=phase1_cases,
            test_mode=test_mode,
            run_id=run_id,
            phase_label="phase1",
            case_results=case_results,
            failed_count_ref=failed_count_ref,
            backoff_state=backoff_state,
            budget_guard=budget_guard,
            base_url=base_url,
        )
        p1_span.set_attribute("completed", len(case_results))
        p1_span.set_attribute("failed", failed_count_ref["count"])
        p1_span.set_attribute("tokens_used", budget_guard.used if budget_guard else 0)
        if budget_guard:
            tracer.record_tokens("phase1", budget_guard.used)

    if cancelled:
        repo.update_run_status(run_id, "failed", error_message="Cancelled by user")
        logger.info("Pipeline cancelled", run_id=run_id)
        remove_tracer(run_id)
        return

    stop_now, cached_features, cached_sims, _ = _checkpoint_should_stop(
        test_mode=test_mode,
        case_results=case_results,
        features_cache=None,
        sims_cache=None,
        scorecard_cache=None,
    )

    if stop_now:
        logger.info("Early stop triggered", run_id=run_id, test_mode=test_mode)
        tracer.add_event("early_stop", test_mode=test_mode)
        _build_and_save_report(
            run_id,
            run,
            pre_result,
            case_results,
            cached_features or {},
            suite_version,
            precomputed_similarities=cached_sims,
        )
        final_status = "partial_failed" if failed_count_ref["count"] > 0 else "completed"
        repo.update_run_status(run_id, final_status)
        logger.info("Pipeline complete", run_id=run_id, status=final_status)
        tracer.finish()
        remove_tracer(run_id)
        return

    if phase2_cases:
        # P2: Progressive result output - generate partial report after phase 1
        try:
            logger.info("Generating partial report after Phase 1", run_id=run_id)
            _build_and_save_report(
                run_id,
                run,
                pre_result,
                case_results,
                {"partial": 1.0},
                suite_version,
                precomputed_similarities=cached_sims,
            )
        except Exception as e:
            logger.warning("Failed to generate partial report", error=str(e))

        # Stage B targeted expansion is already value-ranked.
        with tracer.span("phase2", case_count=len(phase2_cases)) as p2_span:
            cancelled = _run_cases_concurrent(
                adapter=adapter,
                model_name=run["model_name"],
                cases=phase2_cases,
                test_mode=test_mode,
                run_id=run_id,
                phase_label="phase2",
                case_results=case_results,
                failed_count_ref=failed_count_ref,
                backoff_state=backoff_state,
                budget_guard=budget_guard,
                base_url=base_url,
            )
            p2_span.set_attribute("failed", failed_count_ref["count"])
            if budget_guard:
                tracer.record_tokens("phase2", budget_guard.used)
        if cancelled:
            repo.update_run_status(run_id, "failed", error_message="Cancelled by user")
            logger.info("Pipeline cancelled", run_id=run_id)
            remove_tracer(run_id)
            return

        # Stage C arbitration: only for standard mode when decision remains unstable.
        # Deep mode does not arbitrate — it runs all cases fully.
        if test_mode == "standard":
            stop2, f2, s2, sc2 = _checkpoint_should_stop(
                test_mode=test_mode,
                case_results=case_results,
                features_cache=None,
                sims_cache=None,
                scorecard_cache=None,
            )
            if not stop2:
                # pick arbitration cases with highest value among remaining set
                seen_ids = {c.case.id for c in case_results}
                remaining = [c for c in cases if c.id not in seen_ids]
                remaining.sort(key=_case_value, reverse=True)
                arbitration = remaining[:settings.ARBITRATION_MAX]
                if arbitration:
                    with tracer.span("phase3_arbitration", case_count=len(arbitration)) as p3_span:
                        cancelled = _run_cases_concurrent(
                            adapter=adapter,
                            model_name=run["model_name"],
                            cases=arbitration,
                            test_mode=test_mode,
                            run_id=run_id,
                            phase_label="phase3",
                            case_results=case_results,
                            failed_count_ref=failed_count_ref,
                            backoff_state=backoff_state,
                            budget_guard=budget_guard,
                            base_url=base_url,
                        )
                    if cancelled:
                        repo.update_run_status(run_id, "failed", error_message="Cancelled by user")
                        logger.info("Pipeline cancelled", run_id=run_id)
                        remove_tracer(run_id)
                        return

    # ── 检查：是否几乎所有请求均失败（说明是连接/配置问题而非模型问题）────────────
    if case_results:
        total_samples = sum(len(r.samples) for r in case_results)
        error_samples = sum(
            1 for r in case_results for s in r.samples if s.response.error_type
        )
        if total_samples > 0:
            error_rate = error_samples / total_samples
            if error_rate >= 0.9:
                # 收集最常见的错误类型作为提示
                error_types: dict[str, int] = {}
                error_messages: list[str] = []
                for r in case_results:
                    for s in r.samples:
                        et = s.response.error_type
                        if et:
                            error_types[et] = error_types.get(et, 0) + 1
                        em = s.response.error_message
                        if em and em not in error_messages:
                            error_messages.append(em)
                top_error = max(error_types, key=error_types.get) if error_types else "未知错误"
                sample_msg = error_messages[0][:150] if error_messages else ""
                diag_msg = (
                    f"API 连接/配置失败：{error_rate:.0%} 的请求均出错（错误类型：{top_error}）。"
                    f" 示例错误：{sample_msg}。"
                    f" 请检查：① base_url 路径是否正确（如千帆需使用 /v2/chat/completions，"
                    f"当前 base_url：{run['base_url']}）；② API Key 是否有效；③ 网络是否可达。"
                    f" 本次结果不代表模型真实能力，请修正配置后重试。"
                )
                repo.update_run_status(run_id, "failed", error_message=diag_msg)
                logger.error(
                    "Pipeline aborted: near-total request failure indicates config error",
                    run_id=run_id,
                    error_rate=error_rate,
                    top_error=top_error,
                )
                # v11: Record circuit breaker failure and trace event
                circuit_breaker.record_failure(base_url, f"high_error_rate:{error_rate:.0%}:{top_error}")
                tracer.add_event("high_error_rate_abort", error_rate=error_rate, top_error=top_error)
                remove_tracer(run_id)
                return

    # ── Steps 4-6: Analysis + report ─────────────────────────────────────────
    with tracer.span("analysis") as analysis_span:
        _build_and_save_report(run_id, run, pre_result, case_results, {}, suite_version)
        analysis_span.set_attribute("case_count", len(case_results))

    final_status = "partial_failed" if failed_count_ref["count"] > 0 else "completed"
    repo.update_run_status(run_id, final_status)
    logger.info("Pipeline complete", run_id=run_id, status=final_status)

    # v11: Finalize trace and cleanup
    trace = tracer.finish()
    remove_tracer(run_id)


def continue_pipeline(run_id: str) -> None:
    """Resume a pre_detected run: skip predetect, go straight to connectivity check + testing."""
    logger.info("Continue pipeline (from pre_detected)", run_id=run_id)
    run = repo.get_run(run_id)
    if not run:
        logger.error("Run not found", run_id=run_id)
        return

    km = get_key_manager()
    try:
        api_key = km.decrypt(run["api_key_encrypted"])
    except Exception as e:
        repo.update_run_status(run_id, "failed", error_message=f"Key decrypt failed: {e}")
        return

    adapter = OpenAICompatibleAdapter(run["base_url"], api_key)
    base_url = run["base_url"]
    suite_version = run.get("suite_version", "v1")
    test_mode = run.get("test_mode", "standard")

    # v11: Initialize tracer for continue_pipeline
    tracer = get_tracer(run_id)
    tracer.start()

    # v11: Check circuit breaker
    if circuit_breaker.is_open(base_url):
        logger.warning("Circuit breaker OPEN, cannot continue", run_id=run_id, base_url=base_url)
        repo.update_run_status(run_id, "suspended", error_message="API circuit breaker open — run suspended for retry")
        remove_tracer(run_id)
        return

    # Reload predetect result
    pre_dict = run.get("predetect_result") or {}
    pre_result = PreDetectionResult(
        success=pre_dict.get("success", False),
        identified_as=pre_dict.get("identified_as"),
        confidence=pre_dict.get("confidence", 0.0),
        layer_stopped=pre_dict.get("layer_stopped"),
        total_tokens_used=pre_dict.get("total_tokens_used", 0),
        should_proceed_to_testing=True,  # Force continue
        routing_info=pre_dict.get("routing_info", {}),
    )

    # Jump to Step 2: connectivity check + testing (same as run_pipeline from line 589)
    repo.update_run_status(run_id, "running")
    with tracer.span("connectivity", base_url=base_url) as conn_span:
        conn_check = adapter.list_models()
    conn_status = conn_check.get("status_code")
    conn_error = conn_check.get("error")

    if conn_error and not conn_status:
        msg = (
            f"无法连接到 API：{conn_error}。"
            f"请检查 base_url 是否正确（当前：{run['base_url']}），"
            f"网络是否可达，以及 API Key 是否有效。"
        )
        repo.update_run_status(run_id, "failed", error_message=msg)
        circuit_breaker.record_failure(base_url, f"network: {conn_error}")
        remove_tracer(run_id)
        return

    if conn_status and conn_status in (401, 403, 404):
        from app.core.schemas import LLMRequest as _LR, Message as _M
        probe_resp = adapter.chat(_LR(
            model=run["model_name"],
            messages=[_M(role="user", content="hi")],
            max_tokens=1, temperature=0.0, timeout_sec=15,
        ))
        if probe_resp.error_type and not probe_resp.status_code:
            repo.update_run_status(run_id, "failed",
                                   error_message=f"无法连接到 API chat 端点：{probe_resp.error_message}")
            circuit_breaker.record_failure(base_url, f"chat_probe: {probe_resp.error_message}")
            remove_tracer(run_id)
            return
        if probe_resp.status_code == 401:
            repo.update_run_status(run_id, "failed",
                                   error_message=f"API 鉴权失败（HTTP 401）。base_url：{run['base_url']}")
            circuit_breaker.record_failure(base_url, "auth_401")
            remove_tracer(run_id)
            return
        if probe_resp.status_code == 404:
            repo.update_run_status(run_id, "failed",
                                   error_message=f"API 端点不存在（HTTP 404）。base_url：{run['base_url']}")
            circuit_breaker.record_failure(base_url, "endpoint_404")
            remove_tracer(run_id)
            return

    # Load and execute test cases
    cases = _load_suite(suite_version, test_mode)
    if not cases:
        repo.update_run_status(run_id, "failed", error_message="No test cases loaded")
        return

    if (
        test_mode != "full"
        and pre_result.success
        and 0.60 <= pre_result.confidence < settings.PREDETECT_CONFIDENCE_THRESHOLD
        and pre_result.identified_as
    ):
        cases = _select_confirmatory_cases(cases, pre_result.identified_as)

    phase1_cases, phase2_cases = _prepare_cases(cases, test_mode)

    existing_responses = repo.get_responses(run_id)
    completed_case_ids = {r.get("case_id") for r in existing_responses if r.get("case_id")}
    if completed_case_ids:
        phase1_cases = [c for c in phase1_cases if c.id not in completed_case_ids]
        phase2_cases = [c for c in phase2_cases if c.id not in completed_case_ids]

    case_results: list[CaseResult] = []
    failed_count_ref = {"count": 0}
    backoff_state = {"delay_ms": settings.INTER_REQUEST_DELAY_MS}

    budget_map = {
        "quick": settings.TOKEN_BUDGET_QUICK,
        "standard": settings.TOKEN_BUDGET_STANDARD,
        "full": settings.TOKEN_BUDGET_FULL,
    }
    budget_guard = TokenBudgetGuard(budget_map.get(test_mode, settings.TOKEN_BUDGET_STANDARD))

    with tracer.span("phase1", case_count=len(phase1_cases)) as p1_span:
        cancelled = _run_cases_concurrent(
            adapter=adapter, model_name=run["model_name"],
            cases=phase1_cases, test_mode=test_mode, run_id=run_id,
            phase_label="phase1", case_results=case_results,
            failed_count_ref=failed_count_ref, backoff_state=backoff_state,
            budget_guard=budget_guard, base_url=base_url,
        )
        p1_span.set_attribute("completed", len(case_results))
        p1_span.set_attribute("failed", failed_count_ref["count"])
    if cancelled:
        repo.update_run_status(run_id, "failed", error_message="Cancelled by user")
        remove_tracer(run_id)
        return

    stop_now, cached_features, cached_sims, _ = _checkpoint_should_stop(
        test_mode=test_mode, case_results=case_results,
        features_cache=None, sims_cache=None, scorecard_cache=None,
    )
    if stop_now:
        _build_and_save_report(run_id, run, pre_result, case_results,
                               cached_features or {}, suite_version,
                               precomputed_similarities=cached_sims)
        final_status = "partial_failed" if failed_count_ref["count"] > 0 else "completed"
        repo.update_run_status(run_id, final_status)
        tracer.finish()
        remove_tracer(run_id)
        return

    if phase2_cases:
        with tracer.span("phase2", case_count=len(phase2_cases)) as p2_span:
            cancelled = _run_cases_concurrent(
                adapter=adapter, model_name=run["model_name"],
                cases=phase2_cases, test_mode=test_mode, run_id=run_id,
                phase_label="phase2", case_results=case_results,
                failed_count_ref=failed_count_ref, backoff_state=backoff_state,
                budget_guard=budget_guard, base_url=base_url,
            )
            p2_span.set_attribute("failed", failed_count_ref["count"])
        if cancelled:
            repo.update_run_status(run_id, "failed", error_message="Cancelled by user")
            remove_tracer(run_id)
            return

    # High error rate guard
    if case_results:
        total_samples = sum(len(r.samples) for r in case_results)
        error_samples = sum(1 for r in case_results for s in r.samples if s.response.error_type)
        if total_samples > 0 and (error_samples / total_samples) >= 0.9:
            error_types: dict[str, int] = {}
            for r in case_results:
                for s in r.samples:
                    if s.response.error_type:
                        error_types[s.response.error_type] = error_types.get(s.response.error_type, 0) + 1
            top_error = max(error_types, key=error_types.get) if error_types else "unknown"
            repo.update_run_status(run_id, "failed",
                                   error_message=f"API 连接/配置失败：90%+ 请求出错（{top_error}）。请修正配置后重试。")
            circuit_breaker.record_failure(base_url, f"high_error_rate:{top_error}")
            tracer.add_event("high_error_rate_abort", top_error=top_error)
            remove_tracer(run_id)
            return

    with tracer.span("analysis") as analysis_span:
        _build_and_save_report(run_id, run, pre_result, case_results, {}, suite_version)
        analysis_span.set_attribute("case_count", len(case_results))
    final_status = "partial_failed" if failed_count_ref["count"] > 0 else "completed"
    repo.update_run_status(run_id, final_status)
    logger.info("Continue pipeline complete", run_id=run_id, status=final_status)
    tracer.finish()
    remove_tracer(run_id)


def skip_testing_pipeline(run_id: str) -> None:
    """Skip full testing for a pre_detected run, generate report from predetect only."""
    logger.info("Skip testing pipeline", run_id=run_id)
    run = repo.get_run(run_id)
    if not run:
        logger.error("Run not found", run_id=run_id)
        return

    pre_dict = run.get("predetect_result") or {}
    pre_result = PreDetectionResult(
        success=pre_dict.get("success", False),
        identified_as=pre_dict.get("identified_as"),
        confidence=pre_dict.get("confidence", 0.0),
        layer_stopped=pre_dict.get("layer_stopped"),
        total_tokens_used=pre_dict.get("total_tokens_used", 0),
        should_proceed_to_testing=False,
        routing_info=pre_dict.get("routing_info", {}),
    )
    suite_version = run.get("suite_version", "v1")

    _build_and_save_report(run_id, run, pre_result, [], {}, suite_version)
    repo.update_run_status(run_id, "completed")
    logger.info("Skip testing pipeline complete", run_id=run_id)
