from __future__ import annotations
from app.core.schemas import CaseResult, PreDetectionResult
from app.core.config import settings
from app.core.logging import get_logger
from app.analysis.pipeline import (
    FeatureExtractor, SimilarityEngine, RiskEngine, ReportBuilder,
)
from app.repository import repo
from app.orchestration.scoring_flow import ScoringFlow
from app.validation.lineage_guard import LineageGuard

logger = get_logger(__name__)

class ReportFlow:
    """Handles feature extraction, similarity calculation, and report aggregation."""

    def __init__(self):
        self.scoring_flow = ScoringFlow()

    def assemble_and_save_report(
        self,
        run_id: str,
        run_metadata: dict,
        pre_result: PreDetectionResult,
        case_results: list[CaseResult],
        extra_features: dict,
        suite_version: str,
        precomputed_similarities: list | None = None,
    ) -> dict:
        """Assembles and persists the final V12 report."""
        logger.info("Executing report flow", run_id=run_id)

        # 1. Feature Extraction
        extractor = FeatureExtractor()
        features = extractor.extract(case_results)
        features.update(extra_features)
        if features:
            repo.save_features(run_id, features)

        # 2. Similarity Engine
        similarities = precomputed_similarities
        if similarities is None:
            # We import here to avoid circular dependencies if any
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

        # 3. Scoring Flow
        scoring_results = self.scoring_flow.calculate_scores(
            run_id=run_id,
            run_metadata=run_metadata,
            case_results=case_results,
            features=features,
            similarities=similarities,
            pre_result=pre_result,
        )

        # 4. Risk Assessment
        risk_engine = RiskEngine()
        risk = risk_engine.assess(features, similarities, pre_result)

        # 5. Metadata and Provenance
        scoring_profile_version = run_metadata.get("scoring_profile_version", settings.CALIBRATION_VERSION)
        calibration_tag = run_metadata.get("calibration_tag")
        
        breakdowns = self.scoring_flow.register_provenance(
            run_id=run_id,
            scorecard=scoring_results["scorecard"],
            case_results=case_results
        )

        # 6. Save Base Metrics
        if hasattr(repo, "save_score_breakdowns"):
            repo.save_score_breakdowns(run_id, breakdowns)
        
        repo.save_score_history(
            model_name=run_metadata["model_name"],
            base_url=run_metadata["base_url"],
            run_id=run_id,
            total=scoring_results["scorecard"].total_score,
            capability=scoring_results["scorecard"].capability_score,
            authenticity=scoring_results["scorecard"].authenticity_score,
            performance=scoring_results["scorecard"].performance_score,
        )

        # Save Theta history
        theta_report = scoring_results["theta_report"]
        theta_dims_payload = {d.dimension: d.to_dict() for d in theta_report.dimensions}
        pct_dims_payload = {d.dimension: d.percentile for d in theta_report.dimensions}

        repo.save_theta_history(
            run_id=run_id,
            model_name=run_metadata["model_name"],
            base_url=run_metadata["base_url"],
            theta_global=theta_report.global_theta,
            theta_global_ci_low=theta_report.global_ci_low,
            theta_global_ci_high=theta_report.global_ci_high,
            theta_dims=theta_dims_payload,
            percentile_global=theta_report.global_percentile,
            percentile_dims=pct_dims_payload,
            calibration_version=theta_report.calibration_version,
            method=theta_report.method,
        )

        # 7. Final Report Building
        builder = ReportBuilder()
        report = builder.build(
            run_id=run_id,
            base_url=run_metadata["base_url"],
            model_name=run_metadata["model_name"],
            test_mode=run_metadata.get("test_mode", "standard"),
            predetect=pre_result,
            case_results=case_results,
            features=features,
            scores=scoring_results["base_scores"],
            similarities=similarities,
            risk=risk,
            scorecard=scoring_results["scorecard"],
            verdict=scoring_results["verdict"],
            theta_report=theta_report,
            pairwise=scoring_results["pairwise"],
            scoring_profile_version=scoring_profile_version,
            calibration_tag=calibration_tag,
        )

        # Optional segments
        if scoring_results["cdm_report"]:
            report["cdm"] = scoring_results["cdm_report"].to_dict()
        if scoring_results["attribution_report"]:
            report["attribution"] = scoring_results["attribution_report"].to_dict()

        # 8. Suite Pruning Analysis
        try:
            from app.analysis.suite_pruner import suite_pruner
            case_dicts = [
                {
                    "id": cr.case.id,
                    "irt_a": cr.case.irt_a,
                    "irt_b": cr.case.irt_b,
                    "irt_c": getattr(cr.case, 'irt_c', 0.25),
                    "weight": cr.case.weight,
                    "max_tokens": cr.case.max_tokens,
                }
                for cr in case_results
            ]
            pruning_report = suite_pruner.analyze_suite(case_dicts)
            report["suite_pruning"] = pruning_report.to_dict()
        except Exception as e:
            logger.warning("Suite pruning analysis skipped", error=str(e))

        # 9. Lineage Guard Validation
        guard = LineageGuard(strict_mode=not settings.DEBUG)
        lineage_results = guard.validate_report_data(breakdowns)
        report["lineage"] = lineage_results
        
        if not lineage_results["is_valid"]:
            logger.warning("Lineage validation failed", issues=lineage_results["issues"])
            report["verdict"]["warning"] = "Data lineage integrity issues detected."
            if not settings.DEBUG:
                report["verdict"]["rating"] = "PENDING_VERIFICATION"

        repo.save_report(run_id, report)
        return report
