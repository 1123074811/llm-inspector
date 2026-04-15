"""
runner/async_runner.py — async case execution and async pipeline

_run_cases_async: asyncio-native case runner
run_pipeline_async: full async pipeline using AsyncOpenAICompatibleAdapter

Extracted from orchestrator.py to keep individual files under ~280 lines.
"""
from __future__ import annotations

import time

from app.core.schemas import TestCase, CaseResult, PreDetectionResult
from app.core.config import settings
from app.core.logging import get_logger
from app.core.security import get_key_manager
from app.runner.budget_control import TokenBudgetGuard
from app.runner.case_prep import (
    _load_suite, _load_benchmarks, _prepare_cases, _select_confirmatory_cases,
    _save_case_results_batch,
)
from app.runner.report_assembly import _build_and_save_report
from app.repository import repo
from app.predetect.pipeline import PreDetectionPipeline
# AsyncOpenAICompatibleAdapter is imported lazily inside run_pipeline_async (it lives in openai_compat_async)

logger = get_logger(__name__)

import asyncio as _asyncio


async def _run_cases_async(
    adapter,
    model_name: str,
    cases: list,
    test_mode: str,
    run_id: str,
    phase_label: str,
    case_results: list,
    failed_count_ref: dict,
    budget_guard=None,
) -> bool:
    """
    Async replacement for _run_cases_concurrent.

    Uses asyncio.Semaphore to cap concurrency (same as ThreadPoolExecutor
    max_workers) but without thread creation overhead. All case coroutines
    run in the same event loop thread — I/O waits yield the loop to peers.

    Returns True if the run was cancelled.
    """
    from app.runner.case_executor_async import async_execute_case

    if not cases:
        return False

    from app.runner.orchestrator import _mode_concurrency
    max_workers = _mode_concurrency(test_mode)
    semaphore = _asyncio.Semaphore(max_workers)
    budget_threshold = (budget_guard.budget * 0.05) if budget_guard else 0
    pending_saves: list = []
    budget_exhausted = False

    async def _run_one(case) -> tuple:
        if budget_exhausted:
            return case, None, None
        async with semaphore:
            try:
                result = await async_execute_case(adapter, model_name, case)
                return case, result, None
            except Exception as exc:
                return case, None, exc

    # Build and gather all tasks
    tasks = [_asyncio.create_task(_run_one(c)) for c in cases]
    check_early_abort = False

    for completed in _asyncio.as_completed(tasks):
        # Check for cancellation
        cancelled = await _asyncio.to_thread(repo.is_run_cancel_requested, run_id)
        if cancelled:
            for t in tasks:
                t.cancel()
            return True

        case, result, exc = await completed

        if exc is not None:
            failed_count_ref["count"] += 1
            logger.warning("Async case failed", run_id=run_id,
                           case_id=getattr(case, "id", "?"), error=str(exc))
            check_early_abort = True
            continue

        if result is None:
            continue  # budget_exhausted skip

        case_results.append(result)
        pending_saves.append(result)

        # Budget tracking
        if budget_guard:
            budget_guard.record_result(result)
            if budget_guard.remaining < budget_threshold:
                budget_exhausted = True
                logger.info(
                    "Token budget exhausted (async), skipping remaining",
                    run_id=run_id, phase=phase_label,
                    used=budget_guard.used, budget=budget_guard.budget,
                )

        # Batch save every 5 results (DB write offloaded to thread)
        if len(pending_saves) >= 5:
            saves_copy = pending_saves[:]
            pending_saves.clear()
            await _asyncio.to_thread(_save_case_results_batch, run_id, saves_copy)
            check_early_abort = True

        logger.info(
            "Async case done",
            run_id=run_id, case_id=case.id,
            pass_rate=round(result.pass_rate, 2), phase=phase_label,
        )

        # Early abort on >80% failure rate after 10+ results
        if check_early_abort and len(case_results) >= 10:
            total = len(case_results)
            failed = failed_count_ref["count"]
            if total > 0 and (failed / total) > 0.8:
                logger.warning("Async error rate >80%, aborting phase early",
                               run_id=run_id, phase=phase_label)
                for t in tasks:
                    t.cancel()
                break
            check_early_abort = False

    if pending_saves:
        await _asyncio.to_thread(_save_case_results_batch, run_id, pending_saves)

    return False


async def run_pipeline_async(run_id: str) -> None:
    """
    Phase D: asyncio-native full test pipeline.

    Mirrors run_pipeline() but replaces _run_cases_concurrent with
    _run_cases_async so all HTTP I/O is non-blocking.

    Entry point: call asyncio.run(run_pipeline_async(run_id)) from
    the worker thread, or await it from an existing event loop.
    """
    from app.adapters.openai_compat_async import AsyncOpenAICompatibleAdapter

    logger.info("Async pipeline starting", run_id=run_id)

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

    adapter = AsyncOpenAICompatibleAdapter(run["base_url"], api_key)
    # Sync adapter for operations that are still sync (predetect, conn check)
    sync_adapter = adapter.to_sync_adapter()

    suite_version = run.get("suite_version", "v1")
    test_mode = run.get("test_mode", "standard")

    # ── Step 1: Pre-detection (sync, stays blocking for now) ─────────────────
    repo.update_run_status(run_id, "pre_detecting")
    extraction_mode = test_mode == "deep"
    try:
        pre_result = await _asyncio.to_thread(
            lambda: PreDetectionPipeline().run(
                sync_adapter, run["model_name"],
                extraction_mode=extraction_mode,
                run_id=run_id,
            )
        )
        repo.save_predetect_result(run_id, pre_result.to_dict())
        logger.info("Async pre-detection complete", run_id=run_id,
                    identified=pre_result.identified_as, confidence=pre_result.confidence)
    except Exception as e:
        logger.warning("Async pre-detection failed, continuing", error=str(e))
        pre_result = PreDetectionResult(
            success=False, identified_as=None, confidence=0.0,
            layer_stopped=None, should_proceed_to_testing=True,
        )

    if not pre_result.should_proceed_to_testing and not extraction_mode:
        repo.update_run_status(run_id, "pre_detected")
        return

    # ── Step 2: Connectivity check (no async needed — fast probe) ────────────
    repo.update_run_status(run_id, "running")
    conn_check = await adapter.alist_models()
    conn_status = conn_check.get("status_code")
    conn_error = conn_check.get("error")

    if conn_error and not conn_status:
        msg = (
            f"无法连接到 API：{conn_error}。"
            f"请检查 base_url 是否正确（当前：{run['base_url']}），"
            f"网络是否可达，以及 API Key 是否有效。"
        )
        repo.update_run_status(run_id, "failed", error_message=msg)
        logger.error("Async connectivity failed", run_id=run_id, error=msg)
        return

    # ── Step 3: Load and execute test cases ──────────────────────────────────
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
    budget_guard = TokenBudgetGuard(
        {
            "quick": settings.TOKEN_BUDGET_QUICK,
            "standard": settings.TOKEN_BUDGET_STANDARD,
            "deep": settings.TOKEN_BUDGET_DEEP,
            "full": settings.TOKEN_BUDGET_FULL,
        }.get(test_mode, settings.TOKEN_BUDGET_STANDARD)
    )

    logger.info("Async executing test cases", run_id=run_id,
                total=len(phase1_cases) + len(phase2_cases),
                phase1=len(phase1_cases), phase2=len(phase2_cases),
                concurrency=_mode_concurrency(test_mode))

    cancelled = await _run_cases_async(
        adapter=adapter, model_name=run["model_name"],
        cases=phase1_cases, test_mode=test_mode, run_id=run_id,
        phase_label="phase1", case_results=case_results,
        failed_count_ref=failed_count_ref, budget_guard=budget_guard,
    )
    if cancelled:
        repo.update_run_status(run_id, "failed", error_message="Cancelled by user")
        return

    from app.runner.orchestrator import _checkpoint_should_stop
    stop_now, cached_features, cached_sims, _ = _checkpoint_should_stop(
        test_mode=test_mode, case_results=case_results,
        features_cache=None, sims_cache=None, scorecard_cache=None,
    )
    if stop_now:
        logger.info("Async early stop triggered", run_id=run_id, test_mode=test_mode)
        _build_and_save_report(run_id, run, pre_result, case_results,
                               cached_features or {}, suite_version,
                               precomputed_similarities=cached_sims)
        final_status = "partial_failed" if failed_count_ref["count"] > 0 else "completed"
        repo.update_run_status(run_id, final_status)
        return

    if phase2_cases:
        # Progressive partial report after phase 1
        try:
            _build_and_save_report(run_id, run, pre_result, case_results,
                                   {"partial": 1.0}, suite_version,
                                   precomputed_similarities=cached_sims)
        except Exception as e:
            logger.warning("Async partial report failed", error=str(e))

        cancelled = await _run_cases_async(
            adapter=adapter, model_name=run["model_name"],
            cases=phase2_cases, test_mode=test_mode, run_id=run_id,
            phase_label="phase2", case_results=case_results,
            failed_count_ref=failed_count_ref, budget_guard=budget_guard,
        )
        if cancelled:
            repo.update_run_status(run_id, "failed", error_message="Cancelled by user")
            return

    # ── Steps 4-6: Analysis + report (CPU-bound, stays sync) ─────────────────
    await _asyncio.to_thread(
        _build_and_save_report, run_id, run, pre_result, case_results, {}, suite_version
    )
    final_status = "partial_failed" if failed_count_ref["count"] > 0 else "completed"
    repo.update_run_status(run_id, final_status)
    logger.info("Async pipeline complete", run_id=run_id, status=final_status)


def _mode_concurrency(test_mode: str) -> int:
    from app.runner.orchestrator import _mode_concurrency as _orig
    return _orig(test_mode)
