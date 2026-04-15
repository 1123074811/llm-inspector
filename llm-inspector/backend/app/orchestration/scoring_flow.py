from __future__ import annotations
from datetime import datetime
from app.core.schemas import CaseResult, PreDetectionResult
from app.core.config import settings
from app.core.logging import get_logger
from app.analysis.pipeline import (
    ScoreCalculator, ScoreCardCalculator, VerdictEngine,
    ThetaEstimator, UncertaintyEstimator, PercentileMapper, PairwiseEngine,
)
from app.analysis.cdm_engine import cdm_engine
from app.analysis.shapley_attribution import shapley_attributor
from app.repository import repo

logger = get_logger(__name__)

class ScoringFlow:
    """Handles the V12 scoring logic: Theta, CDM, Shapley, ELO, and Verdicts."""

    def calculate_scores(
        self,
        run_id: str,
        run_metadata: dict,
        case_results: list[CaseResult],
        features: dict,
        similarities: list,
        pre_result: PreDetectionResult,
    ) -> dict:
        """
        Calculates all score-related metrics.
        Returns a dictionary containing scorecard, verdict, theta_report, etc.
        """
        logger.info("Starting scoring flow", run_id=run_id)

        # 1. Base Scores (Feature-based)
        scorer = ScoreCalculator()
        base_scores = scorer.calculate(features)

        # 2. Theta Estimation (IRT-based)
        item_stats_rows = repo.list_item_stats()
        item_stats = {r.get("item_id"): r for r in item_stats_rows}

        theta_estimator = ThetaEstimator()
        theta_report = theta_estimator.estimate(case_results, item_stats)
        theta_report = UncertaintyEstimator().apply_ci(theta_report, case_results, theta_estimator, item_stats)

        hist = repo.get_model_theta_trend(run_metadata["model_name"], limit=200)
        theta_report = PercentileMapper().map_percentiles(theta_report, hist)

        # 3. Scorecard & Verdict (Mapped from Theta)
        scorecard_calc = ScoreCardCalculator()
        scorecard = scorecard_calc.calculate(
            features=features,
            case_results=case_results,
            similarities=similarities,
            predetect=pre_result,
            claimed_model=run_metadata.get("model_name"),
            theta_report=theta_report,
        )

        verdict_engine = VerdictEngine()
        verdict = verdict_engine.assess(
            scorecard=scorecard,
            similarities=similarities,
            predetect=pre_result,
            features=features,
            case_results=case_results,
        )

        # 4. Pairwise & ELO
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
                model_a=run_metadata["model_name"],
                model_b=model_b,
                delta_theta=pairwise["delta_theta"],
                win_prob_a=pairwise["win_prob"],
                method=pairwise.get("method", "bradley_terry"),
                details=pairwise,
            )
            
            from app.analysis.elo import EloLeaderboard
            try:
                elo_board = EloLeaderboard()
                new_a, new_b = elo_board.update_from_pairwise(
                    model_a=run_metadata["model_name"],
                    display_a=run_metadata["model_name"],
                    model_b=model_b,
                    display_b=model_b.title(),
                    win_prob_a=pairwise["win_prob"],
                    run_id=run_id,
                )
                pairwise["elo_rating_a"] = new_a
                pairwise["elo_rating_b"] = new_b
            except Exception as e:
                logger.error("Failed to update ELO rankings", error=str(e))

        # 5. Advanced Analysis (CDM, Shapley)
        cdm_report = None
        try:
            cdm_report = cdm_engine.diagnose(case_results, theta_report)
        except Exception as e:
            logger.warning("CDM diagnosis failed", error=str(e))

        attribution_report = None
        try:
            attribution_report = shapley_attributor.attribute(
                scorecard=scorecard,
                verdict=verdict,
                features=features,
            )
        except Exception as e:
            logger.warning("Shapley attribution failed", error=str(e))

        return {
            "base_scores": base_scores,
            "theta_report": theta_report,
            "scorecard": scorecard,
            "verdict": verdict,
            "pairwise": pairwise,
            "cdm_report": cdm_report,
            "attribution_report": attribution_report,
        }

    def register_provenance(self, run_id: str, scorecard: any, case_results: list):
        """Register data lineage for calculated scores."""
        from app.core.provenance import get_provenance_tracker, DataProvenance
        tracker = get_provenance_tracker()
        timestamp = datetime.utcnow().isoformat()
        
        breakdowns = {
            "total": scorecard.total_score,
            "capability": scorecard.capability_score,
            "authenticity": scorecard.authenticity_score,
            "performance": scorecard.performance_score,
        }
        
        for dim, val in breakdowns.items():
            source_type = "derived" if dim == "total" else "irt_calibration"
            tracker.register(f"{dim}_score", DataProvenance(
                source_type=source_type,
                source_id=f"v12_run_{run_id}_{dim}",
                collected_at=timestamp,
                sample_size=len(case_results),
                confidence=scorecard.confidence_level,
                verified=True,
                notes=f"v12 automated scoring for {dim}"
            ))
        return breakdowns
