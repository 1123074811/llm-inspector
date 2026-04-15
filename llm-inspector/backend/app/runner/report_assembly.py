"""
runner/report_assembly.py — _build_and_save_report

Extracts features, scores, similarity, risk, generates the final report dict,
and persists it to the repository.

Extracted from orchestrator.py to keep individual files under ~260 lines.
"""
from __future__ import annotations

from app.core.schemas import CaseResult, PreDetectionResult
from app.core.config import settings
from app.core.logging import get_logger
from app.analysis.pipeline import (
    FeatureExtractor, ScoreCalculator, SimilarityEngine, RiskEngine,
    ReportBuilder, ScoreCardCalculator, VerdictEngine,
    ThetaEstimator, UncertaintyEstimator, PercentileMapper, PairwiseEngine,
)
from app.analysis.cdm_engine import cdm_engine
from app.analysis.shapley_attribution import shapley_attributor
from app.repository import repo

logger = get_logger(__name__)


def _build_and_save_report(
    run_id: str,
    run: dict,
    pre_result: PreDetectionResult,
    case_results: list[CaseResult],
    extra_features: dict,
    suite_version: str,
    precomputed_similarities: list | None = None,
) -> dict:
    """Extract features, score, similarity, risk, build report."""
    extractor = FeatureExtractor()
    features = extractor.extract(case_results)
    features.update(extra_features)
    if features:
        repo.save_features(run_id, features)

    scoring_profile_version = run.get("scoring_profile_version", settings.CALIBRATION_VERSION)
    calibration_tag = run.get("calibration_tag")

    # Scoring
    scorer = ScoreCalculator()
    scores = scorer.calculate(features)

    # Similarity (allow checkpoint cache injection)
    similarities = precomputed_similarities
    if similarities is None:
        from app.runner.orchestrator import _load_benchmarks
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

    # v2 scorecard + verdict
    scorecard_calc = ScoreCardCalculator()
    scorecard = scorecard_calc.calculate(
        features=features,
        case_results=case_results,
        similarities=similarities,
        predetect=pre_result,
        claimed_model=run.get("model_name"),
    )
    verdict_engine = VerdictEngine()
    verdict = verdict_engine.assess(
        scorecard=scorecard,
        similarities=similarities,
        predetect=pre_result,
        features=features,
        case_results=case_results,
    )

    breakdowns = {
        "total": scorecard.total_score,
        "capability": scorecard.capability_score,
        "authenticity": scorecard.authenticity_score,
        "performance": scorecard.performance_score,
        "reasoning": scorecard.reasoning_score,
        "adversarial_reasoning": scorecard.adversarial_reasoning_score,
        "instruction": scorecard.instruction_score,
        "coding": scorecard.coding_score,
        "safety": scorecard.safety_score,
        "protocol": scorecard.protocol_score,
        "consistency": scorecard.consistency_score,
        "speed": scorecard.speed_score,
        "stability": scorecard.stability_score,
        "cost_efficiency": scorecard.cost_efficiency,
    }

    # Backward-compatible write path: if running mixed code version,
    # gracefully fallback to single-row API instead of failing the run.
    if hasattr(repo, "save_score_breakdowns"):
        repo.save_score_breakdowns(run_id, breakdowns)
    else:
        for dim, val in breakdowns.items():
            repo.save_score_breakdown(run_id, dim, val)

    repo.save_score_history(
        model_name=run["model_name"],
        base_url=run["base_url"],
        run_id=run_id,
        total=scorecard.total_score,
        capability=scorecard.capability_score,
        authenticity=scorecard.authenticity_score,
        performance=scorecard.performance_score,
    )

    # Theta report (relative scale)
    item_stats_rows = repo.list_item_stats()
    item_stats = {r.get("item_id"): r for r in item_stats_rows}

    theta_estimator = ThetaEstimator()
    theta_report = theta_estimator.estimate(case_results, item_stats)
    theta_report = UncertaintyEstimator().apply_ci(theta_report, case_results, theta_estimator, item_stats)

    hist = repo.get_model_theta_trend(run["model_name"], limit=200)
    theta_report = PercentileMapper().map_percentiles(theta_report, hist)

    theta_dims_payload = {d.dimension: d.to_dict() for d in theta_report.dimensions}
    pct_dims_payload = {d.dimension: d.percentile for d in theta_report.dimensions}

    repo.save_theta_history(
        run_id=run_id,
        model_name=run["model_name"],
        base_url=run["base_url"],
        theta_global=theta_report.global_theta,
        theta_global_ci_low=theta_report.global_ci_low,
        theta_global_ci_high=theta_report.global_ci_high,
        theta_dims=theta_dims_payload,
        percentile_global=theta_report.global_percentile,
        percentile_dims=pct_dims_payload,
        calibration_version=theta_report.calibration_version,
        method=theta_report.method,
    )

    baseline_theta = None
    if similarities:
        top_benchmark = similarities[0].benchmark_name
        baseline_hist = repo.get_model_theta_trend(top_benchmark, limit=1)
        if baseline_hist:
            baseline_theta = float(baseline_hist[0].get("theta_global", 0.0) or 0.0)

    pairwise = PairwiseEngine().compare_to_baseline(theta_report, baseline_theta)
    if pairwise:
        model_b = similarities[0].benchmark_name if similarities else "baseline"
        repo.save_pairwise_result(
            run_id=run_id,
            model_a=run["model_name"],
            model_b=model_b,
            delta_theta=pairwise["delta_theta"],
            win_prob_a=pairwise["win_prob"],
            method=pairwise.get("method", "bradley_terry"),
            details=pairwise,
        )
        # Update ELO standings
        from app.analysis.elo import EloLeaderboard
        try:
            elo_board = EloLeaderboard()
            new_a, new_b = elo_board.update_from_pairwise(
                model_a=run["model_name"],
                display_a=run["model_name"],
                model_b=model_b,
                display_b=model_b.title(),
                win_prob_a=pairwise["win_prob"],
                run_id=run_id,
            )
            pairwise["elo_rating_a"] = new_a
            pairwise["elo_rating_b"] = new_b
        except Exception as e:
            logger.error("Failed to update ELO rankings", error=str(e))

    # Build final report
    # v11: CDM (Cognitive Diagnostic Model) — per-skill mastery diagnosis
    cdm_report = None
    try:
        cdm_report = cdm_engine.diagnose(case_results, theta_report)
        logger.info(
            "CDM diagnosis complete",
            run_id=run_id,
            n_skills=cdm_report.n_skills,
            overall_mastery=round(cdm_report.overall_mastery_rate, 3),
            weakest=cdm_report.weakest_skills[:3],
        )
    except Exception as e:
        logger.warning("CDM diagnosis failed, continuing without it", error=str(e))

    # v11: Shapley Value attribution — score decomposition
    attribution_report = None
    try:
        attribution_report = shapley_attributor.attribute(
            scorecard=scorecard,
            verdict=verdict,
            features=features,
        )
        logger.info(
            "Shapley attribution complete",
            run_id=run_id,
            score_delta=round(attribution_report.score_delta, 1),
            top_positive=attribution_report.top_positive[:3],
            top_negative=attribution_report.top_negative[:3],
        )
    except Exception as e:
        logger.warning("Shapley attribution failed, continuing without it", error=str(e))

    # v11 Phase 3: Suite pruning analysis — mark non-discriminative cases
    pruning_report = None
    try:
        from app.analysis.suite_pruner import suite_pruner
        case_dicts = [
            {
                "id": cr.case.id,
                "irt_a": cr.case.irt_a,
                "irt_b": cr.case.irt_b,
                "irt_c": cr.case.irt_c if hasattr(cr.case, 'irt_c') else 0.25,
                "weight": cr.case.weight,
                "max_tokens": cr.case.max_tokens,
            }
            for cr in case_results
        ]
        pruning_report = suite_pruner.analyze_suite(case_dicts)
        logger.info(
            "Suite pruning analysis complete",
            run_id=run_id,
            total=pruning_report.total_cases,
            non_discriminative=pruning_report.non_discriminative_cases,
            token_savings=f"{pruning_report.estimated_token_savings_pct:.1f}%",
        )
    except Exception as e:
        logger.warning("Suite pruning analysis failed, continuing without it", error=str(e))

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
        scorecard=scorecard,
        verdict=verdict,
        theta_report=theta_report,
        pairwise=pairwise,
        scoring_profile_version=scoring_profile_version,
        calibration_tag=calibration_tag,
    )

    # v11: Append CDM and Shapley reports to the final report
    if cdm_report is not None:
        report["cdm"] = cdm_report.to_dict()
    if attribution_report is not None:
        report["attribution"] = attribution_report.to_dict()
    if pruning_report is not None:
        report["suite_pruning"] = pruning_report.to_dict()

    repo.save_report(run_id, report)
    return report
