"""
RunOrchestrator — executes the full test pipeline for a single run.
Called from the task worker (thread pool or Celery).
"""
from __future__ import annotations

import json
import pathlib
import time
from app.core.schemas import (
    TestCase, CaseResult, PreDetectionResult,
)
from app.core.logging import get_logger
from app.core.security import get_key_manager
from app.core.config import settings
from app.adapters.openai_compat import OpenAICompatibleAdapter
from app.predetect.pipeline import PreDetectionPipeline
from app.runner.case_executor import execute_case
from app.analysis.pipeline import (
    FeatureExtractor, ScoreCalculator,
    SimilarityEngine, RiskEngine, ReportBuilder,
)
from app.repository import repo

logger = get_logger(__name__)

_FIXTURES_DIR = pathlib.Path(__file__).parent.parent / "fixtures"


def _load_suite(suite_version: str, test_mode: str) -> list[TestCase]:
    """Load test cases from DB (seeded from fixture JSON)."""
    raw_cases = repo.load_cases(suite_version, test_mode)
    cases = []
    for c in raw_cases:
        cases.append(TestCase(
            id=c["id"],
            category=c["category"],
            name=c["name"],
            user_prompt=c["user_prompt"],
            expected_type=c["expected_type"],
            judge_method=c["judge_method"],
            system_prompt=c.get("system_prompt"),
            params=c.get("params", {}),
            max_tokens=c.get("max_tokens", 5),
            n_samples=c.get("n_samples", 1),
            temperature=c.get("temperature", 0.0),
            weight=c.get("weight", 1.0),
            enabled=bool(c.get("enabled", 1)),
            suite_version=c.get("suite_version", "v1"),
        ))
    return cases


def _load_benchmarks(suite_version: str) -> list[dict]:
    return repo.get_benchmarks(suite_version)


def _save_case_result(run_id: str, result: CaseResult) -> None:
    for sample in result.samples:
        r = sample.response
        repo.save_response(
            run_id=run_id,
            case_id=result.case.id,
            sample_index=sample.sample_index,
            resp_data={
                "response_text": r.content,
                "raw_response": r.raw_json,
                "raw_headers": r.headers,
                "status_code": r.status_code,
                "latency_ms": r.latency_ms,
                "first_token_ms": r.first_token_ms,
                "finish_reason": r.finish_reason,
                "usage_prompt_tokens": r.usage_prompt_tokens,
                "usage_completion_tokens": r.usage_completion_tokens,
                "usage_total_tokens": r.usage_total_tokens,
                "error_type": r.error_type,
                "error_message": r.error_message,
                "judge_passed": sample.judge_passed,
                "judge_detail": sample.judge_detail,
            },
        )


def _should_early_stop(case_results: list[CaseResult],
                       benchmarks: list[dict], test_mode: str) -> bool:
    """Check if similarity is already confident enough to stop early."""
    if test_mode != "quick":
        return False
    if len(case_results) < 5:
        return False
    extractor = FeatureExtractor()
    features = extractor.extract(case_results)
    if not features:
        return False
    engine = SimilarityEngine()
    sims = engine.compare(features, benchmarks)
    if not sims:
        return False
    top = sims[0]
    second = sims[1] if len(sims) > 1 else None
    delta = (top.similarity_score - second.similarity_score) if second else 1.0
    return top.similarity_score >= 0.85 and delta >= 0.15


def run_pipeline(run_id: str) -> None:
    """
    Full pipeline:
      1. Pre-detection (0-token identification attempt)
      2. Connectivity check
      3. Test case execution (with early-stop)
      4. Feature extraction + scoring
      5. Similarity + risk assessment
      6. Report generation
    """
    logger.info("Pipeline starting", run_id=run_id)

    # ── Load run metadata ────────────────────────────────────────────────────
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
    suite_version = run.get("suite_version", "v1")
    test_mode = run.get("test_mode", "standard")

    # ── Step 1: Pre-detection ────────────────────────────────────────────────
    repo.update_run_status(run_id, "pre_detecting")
    logger.info("Running pre-detection", run_id=run_id)
    try:
        pre_result: PreDetectionResult = PreDetectionPipeline().run(
            adapter, run["model_name"]
        )
        repo.save_predetect_result(run_id, pre_result.to_dict())
        logger.info(
            "Pre-detection complete",
            run_id=run_id,
            identified=pre_result.identified_as,
            confidence=pre_result.confidence,
            tokens=pre_result.total_tokens_used,
        )
    except Exception as e:
        logger.warning("Pre-detection failed, continuing to full test", error=str(e))
        pre_result = PreDetectionResult(
            success=False, identified_as=None, confidence=0.0,
            layer_stopped=None, should_proceed_to_testing=True,
        )

    # If pre-detection is highly confident, skip full test
    if not pre_result.should_proceed_to_testing:
        logger.info("Pre-detection sufficient, skipping full test", run_id=run_id)
        _build_and_save_report(
            run_id, run, pre_result, [], {}, suite_version
        )
        return

    # ── Step 2: Connectivity check ───────────────────────────────────────────
    repo.update_run_status(run_id, "running")
    conn_check = adapter.list_models()
    if conn_check.get("error") and not conn_check.get("status_code"):
        msg = f"Cannot reach API: {conn_check.get('error')}"
        repo.update_run_status(run_id, "failed", error_message=msg)
        logger.error("Connectivity failed", run_id=run_id, error=msg)
        return

    # ── Step 3: Load test cases + execute ────────────────────────────────────
    cases = _load_suite(suite_version, test_mode)
    if not cases:
        logger.warning("No test cases found", suite_version=suite_version)
        repo.update_run_status(run_id, "failed", error_message="No test cases loaded")
        return

    benchmarks = _load_benchmarks(suite_version)
    case_results: list[CaseResult] = []
    failed_count = 0

    logger.info("Executing test cases", run_id=run_id, total=len(cases))

    for i, case in enumerate(cases):
        try:
            result = execute_case(adapter, run["model_name"], case)
            case_results.append(result)
            _save_case_result(run_id, result)
            logger.info(
                "Case done",
                run_id=run_id,
                case_id=case.id,
                pass_rate=round(result.pass_rate, 2),
                progress=f"{i+1}/{len(cases)}",
            )
        except Exception as e:
            failed_count += 1
            logger.warning("Case failed", run_id=run_id, case_id=case.id, error=str(e))

        # Inter-request delay
        if i < len(cases) - 1:
            time.sleep(settings.INTER_REQUEST_DELAY_MS / 1000)

        # Early stop check every 5 cases (quick mode only)
        if (i + 1) % 5 == 0:
            if _should_early_stop(case_results, benchmarks, test_mode):
                logger.info("Early stop triggered", run_id=run_id, cases_done=i+1)
                break

    # ── Steps 4-6: Analysis + report ─────────────────────────────────────────
    _build_and_save_report(run_id, run, pre_result, case_results, {}, suite_version)

    final_status = "partial_failed" if failed_count > 0 else "completed"
    repo.update_run_status(run_id, final_status)
    logger.info("Pipeline complete", run_id=run_id, status=final_status)


def _build_and_save_report(
    run_id: str,
    run: dict,
    pre_result: PreDetectionResult,
    case_results: list[CaseResult],
    extra_features: dict,
    suite_version: str,
) -> None:
    """Extract features, score, similarity, risk, build report."""
    # Feature extraction
    extractor = FeatureExtractor()
    features = extractor.extract(case_results)
    features.update(extra_features)
    if features:
        repo.save_features(run_id, features)

    # Scoring
    scorer = ScoreCalculator()
    scores = scorer.calculate(features)

    # Similarity
    benchmarks = _load_benchmarks(suite_version)
    similarity_engine = SimilarityEngine()
    similarities = similarity_engine.compare(features, benchmarks)
    if similarities:
        repo.save_similarities(run_id, [
            {
                "benchmark": s.benchmark_name,
                "score": s.similarity_score,
                "ci_95_low": s.ci_95_low,
                "ci_95_high": s.ci_95_high,
                "rank": s.rank,
            }
            for s in similarities
        ])

    # Risk assessment
    risk_engine = RiskEngine()
    risk = risk_engine.assess(features, similarities, pre_result)

    # Build final report
    builder = ReportBuilder()
    report = builder.build(
        run_id=run_id,
        base_url=run["base_url"],
        model_name=run["model_name"],
        test_mode=run.get("test_mode", "standard"),
        predetect=pre_result,
        case_results=case_results,
        features=features,
        scores=scores,
        similarities=similarities,
        risk=risk,
    )
    repo.save_report(run_id, report)
