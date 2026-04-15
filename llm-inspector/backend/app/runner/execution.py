"""
runner/execution.py — case execution, backoff, checkpoint, concurrency

_adaptive_pause, _update_backoff: rate-limiting helpers
_checkpoint_should_stop: CAT / early-exit logic
_run_cases_concurrent: ThreadPoolExecutor case runner

Extracted from orchestrator.py to keep individual files under ~280 lines.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

from app.core.schemas import TestCase, CaseResult
from app.core.config import settings
from app.core.logging import get_logger
from app.runner.case_executor import execute_case
from app.runner.case_prep import _save_case_results_batch
from app.analysis.pipeline import FeatureExtractor, SimilarityEngine, ScoreCardCalculator
from app.runner.budget_control import TokenBudgetGuard

logger = get_logger(__name__)


def _adaptive_pause(backoff_state: dict) -> None:
    delay_ms = backoff_state.get("delay_ms", settings.INTER_REQUEST_DELAY_MS)
    if delay_ms > 0:
        time.sleep(delay_ms / 1000)


def _update_backoff(backoff_state: dict, result: CaseResult | None = None,
                    error: Exception | None = None) -> None:
    base_ms = settings.INTER_REQUEST_DELAY_MS
    current = int(backoff_state.get("delay_ms", base_ms))

    should_backoff = False
    if error is not None:
        should_backoff = True
    elif result is not None:
        for s in result.samples:
            status = s.response.status_code or 0
            if status == 429 or status >= 500:
                should_backoff = True
                break

    if should_backoff:
        backoff_state["delay_ms"] = min(max(base_ms, 200), max(200, current) * 2)
    else:
        backoff_state["delay_ms"] = base_ms


def _checkpoint_should_stop(test_mode: str, case_results: list[CaseResult],
                            features_cache: dict | None, sims_cache: list | None,
                            scorecard_cache=None) -> tuple[bool, dict | None, list | None, object | None]:
    if not case_results:
        return False, features_cache, sims_cache, scorecard_cache

    # v10: Computerized Adaptive Testing (CAT) early stopping
    # If standard error of measurement (SEM) is low enough, we can stop early
    total_cases = len(case_results)
    if total_cases >= 10:
        # Check CAT threshold
        from app.analysis.irt_params import get_calibrated_params

        dim_info = {}
        for cr in case_results:
            dim = getattr(cr.case, 'dimension', 'unknown')
            params = get_calibrated_params(cr.case.id)
            if params:
                # We don't know exact theta here without full recalculation,
                # so we use expected information at theta=0 as a heuristic proxy
                info = params.calculate_information(0.0)
                dim_info[dim] = dim_info.get(dim, 0.0) + info

        # If all major dimensions have SEM < 0.3 (Info > 11.1)
        cat_satisfied = True
        major_dims = [d for d in dim_info.keys() if d != "unknown"]

        if len(major_dims) >= 3:
            for dim in major_dims:
                info = dim_info[dim]
                sem = 1.0 / (info ** 0.5) if info > 0 else 999.0
                if sem > 0.3:
                    cat_satisfied = False
                    break

            if cat_satisfied and test_mode != "deep":
                logger.info(
                    "v10 CAT early stopping triggered (SEM < 0.3 for all dims)",
                    test_mode=test_mode,
                    cases_run=total_cases
                )
                return True, features_cache, sims_cache, scorecard_cache

    # v6: Failure rate early stop for all modes
    # If error rate > 80% after 10 cases, abort early (API likely broken)
    if total_cases >= 10:
        failed_cases = sum(1 for r in case_results if r.pass_rate == 0.0)
        fail_rate = failed_cases / total_cases
        if fail_rate > 0.80:
            logger.warning(
                "High failure rate detected, aborting early",
                fail_rate=round(fail_rate, 2),
                total_cases=total_cases,
                test_mode=test_mode,
            )
            return True, features_cache, sims_cache, scorecard_cache

    # v6: Quick mode 2-point early stop (minimum 2 cases)
    if test_mode == "quick":
        # After just 2 core cases, check if model is clearly identifiable
        if len(case_results) >= 2:
            extractor = FeatureExtractor()
            features = extractor.extract(case_results)
            similarities: list = sims_cache or []
            if not similarities:
                from app.runner.orchestrator import _load_benchmarks
                run_suite = case_results[0].case.suite_version if case_results else "v2"
                similarities = SimilarityEngine().compare(features, _load_benchmarks(run_suite))

            if similarities and len(similarities) >= 2:
                top = similarities[0]
                second = similarities[1]
                # If similarity gap >= 0.15, we have strong identification
                if top.similarity_score >= 0.70 and (top.similarity_score - second.similarity_score) >= 0.15:
                    return True, features, similarities, scorecard_cache
        # Otherwise continue to 6-case threshold for quick mode
        if len(case_results) < 6:
            return False, features_cache, sims_cache, scorecard_cache

    if len(case_results) < 6:
        return False, features_cache, sims_cache, scorecard_cache

    # Deep mode: no early stopping — always run full suite
    if test_mode == "deep":
        return False, features_cache, sims_cache, scorecard_cache

    extractor = FeatureExtractor()
    features = extractor.extract(case_results)
    if not features:
        return False, features_cache, sims_cache, scorecard_cache

    similarities: list = sims_cache or []
    if not similarities:
        from app.runner.orchestrator import _load_benchmarks
        run_suite = case_results[0].case.suite_version if case_results else "v2"
        similarities = SimilarityEngine().compare(features, _load_benchmarks(run_suite))

    if not similarities:
        return False, features, similarities, scorecard_cache

    top = similarities[0]
    second = similarities[1] if len(similarities) > 1 else None
    delta = (top.similarity_score - second.similarity_score) if second else 1.0

    if test_mode == "quick":
        return (top.similarity_score >= 0.78 and delta >= 0.10), features, similarities, scorecard_cache

    if test_mode == "standard":
        sc = scorecard_cache
        if sc is None:
            sc = ScoreCardCalculator().calculate(
                features=features,
                case_results=case_results,
                similarities=similarities,
                predetect=None,
                claimed_model=None,
            )

        if top.similarity_score >= 0.85 and delta >= 0.12:
            near_boundary = abs(sc.total_score - 70.0) <= 5.0 or abs(sc.authenticity_score - 70.0) <= 6.0
            unstable = delta < 0.05
            if not near_boundary and not unstable:
                return True, features, similarities, sc

        if top.similarity_score >= 0.92 and delta >= 0.08:
            return True, features, similarities, sc

    return False, features, similarities, scorecard_cache


def _run_cases_concurrent(adapter, model_name: str, cases: list[TestCase],
                          test_mode: str, run_id: str, phase_label: str,
                          case_results: list[CaseResult], failed_count_ref: dict,
                          backoff_state: dict,
                          budget_guard: TokenBudgetGuard | None = None,
                          base_url: str | None = None) -> bool:
    if not cases:
        return False

    from app.runner.orchestrator import _mode_concurrency
    from app.core.circuit_breaker import circuit_breaker
    from app.repository import repo

    max_workers = _mode_concurrency(test_mode)
    idx = 0
    in_flight = {}
    lock = threading.Lock()

    # Token budget thresholds: stop dispatching new work when < 5% budget remains
    budget_threshold = (budget_guard.budget * 0.05) if budget_guard else 0
    pending_saves = []

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=f"case-{phase_label}") as pool:
        budget_exhausted = False
        while idx < len(cases) or in_flight:
            if repo.is_run_cancel_requested(run_id):
                logger.warning("Run cancel requested", run_id=run_id, phase=phase_label)
                break

            while len(in_flight) < max_workers and idx < len(cases) and not budget_exhausted:
                # v11: Check circuit breaker before submitting new cases
                if base_url and circuit_breaker.is_open(base_url):
                    logger.warning(
                        "Circuit breaker OPEN, suspending case dispatch",
                        run_id=run_id, phase=phase_label, base_url=base_url,
                    )
                    # Don't submit new cases; wait for in-flight to complete
                    break

                # Check token budget before submitting
                if budget_guard and budget_guard.remaining < budget_threshold:
                    logger.info(
                        "Token budget exhausted, skipping remaining phase cases",
                        run_id=run_id, phase=phase_label,
                        used=budget_guard.used, budget=budget_guard.budget,
                    )
                    budget_exhausted = True
                    break

                case = cases[idx]
                idx += 1
                _adaptive_pause(backoff_state)
                fut = pool.submit(execute_case, adapter, model_name, case)
                in_flight[fut] = case

            if not in_flight:
                break
            done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)

            check_early_abort = False
            for fut in done:
                case = in_flight.pop(fut)
                try:
                    result = fut.result()
                    with lock:
                        case_results.append(result)
                        pending_saves.append(result)
                        if len(pending_saves) >= 5:
                            _save_case_results_batch(run_id, pending_saves)
                            pending_saves.clear()
                            check_early_abort = True

                    # Track token consumption for budget guard
                    if budget_guard:
                        budget_guard.record_result(result)

                    # v11: Record circuit breaker success
                    if base_url:
                        circuit_breaker.record_success(base_url)

                    _update_backoff(backoff_state, result=result)
                    logger.info(
                        "Case done",
                        run_id=run_id,
                        case_id=case.id,
                        pass_rate=round(result.pass_rate, 2),
                        phase=phase_label,
                        tokens_remaining=budget_guard.remaining if budget_guard else None,
                    )
                except Exception as e:
                    with lock:
                        failed_count_ref["count"] += 1
                    # v11: Record circuit breaker failure
                    if base_url:
                        circuit_breaker.record_failure(base_url, str(e)[:200])
                    _update_backoff(backoff_state, error=e)
                    logger.warning("Case failed", run_id=run_id, case_id=case.id, error=str(e))
                    check_early_abort = True

            if check_early_abort:
                with lock:
                    total = len(case_results)
                    failed = failed_count_ref["count"]
                if total >= 10 and total > 0 and (failed / total) > 0.8:
                    logger.warning("Error rate >80%, aborting phase early", run_id=run_id, phase=phase_label)
                    break

    if pending_saves:
        _save_case_results_batch(run_id, pending_saves)
        pending_saves.clear()
