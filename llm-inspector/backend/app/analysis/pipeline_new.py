"""
Refactored Analysis Pipeline (V6)
Main analysis orchestrator using modular components.

This is the refactored version of pipeline.py that uses the new modular
architecture for better code organization and maintainability.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from app.core.logging import get_logger
from .feature_extractor import FeatureExtractor
from .score_calculator import ScoreCardCalculator
from .similarity_engine import SimilarityEngine
from .verdict_engine import VerdictEngine

logger = get_logger(__name__)


# Data classes (keep these in pipeline for now)
class CaseResult:
    """Result of running a test case."""
    def __init__(
        self,
        case: Any,
        samples: List[Any],
        pass_rate: float,
        mean_latency_ms: Optional[float] = None,
        breakdown: Optional[Dict] = None,
    ):
        self.case = case
        self.samples = samples
        self.pass_rate = pass_rate
        self.mean_latency_ms = mean_latency_ms
        self.breakdown = breakdown or {}


class SimilarityResult:
    """Similarity comparison result."""
    def __init__(
        self,
        benchmark: str,
        score: float,
        ci_95_low: Optional[float],
        ci_95_high: Optional[float],
        valid_features: int,
        run_id: Optional[str] = None,
        overall_score: Optional[float] = None,
    ):
        self.benchmark = benchmark
        self.score = score
        self.ci_95_low = ci_95_low
        self.ci_95_high = ci_95_high
        self.valid_features = valid_features
        self.run_id = run_id
        self.overall_score = overall_score


class ScoreCard:
    """Score card containing all dimension scores."""
    def __init__(
        self,
        overall_score: float,
        reasoning_score: Optional[float],
        adversarial_reasoning_score: Optional[float],
        instruction_score: Optional[float],
        coding_score: Optional[float],
        safety_score: Optional[float],
        protocol_score: Optional[float],
        knowledge_score: Optional[float],
        tool_use_score: Optional[float],
        performance_score: float,
        speed_score: float,
        stability_score: float,
        cost_efficiency: float,
        confidence_level: float,
        breakdown: Optional[Dict] = None,
    ):
        self.overall_score = overall_score
        self.reasoning_score = reasoning_score
        self.adversarial_reasoning_score = adversarial_reasoning_score
        self.instruction_score = instruction_score
        self.coding_score = coding_score
        self.safety_score = safety_score
        self.protocol_score = protocol_score
        self.knowledge_score = knowledge_score
        self.tool_use_score = tool_use_score
        self.performance_score = performance_score
        self.speed_score = speed_score
        self.stability_score = stability_score
        self.cost_efficiency = cost_efficiency
        self.confidence_level = confidence_level
        self.breakdown = breakdown or {}


class PreDetectionResult:
    """Pre-detection pipeline result."""
    def __init__(
        self,
        success: bool,
        identified_as: Optional[str],
        confidence: float,
        layer_stopped: str,
        layer_results: List[Any],
        total_tokens_used: int,
        should_proceed_to_testing: bool,
        routing_info: Optional[Dict] = None,
    ):
        self.success = success
        self.identified_as = identified_as
        self.confidence = confidence
        self.layer_stopped = layer_stopped
        self.layer_results = layer_results
        self.total_tokens_used = total_tokens_used
        self.should_proceed_to_testing = should_proceed_to_testing
        self.routing_info = routing_info or {}


class AnalysisPipeline:
    """
    V6 Refactored Analysis Pipeline
    
    Uses modular components for better code organization:
    - FeatureExtractor: Extract features from case results
    - ScoreCardCalculator: Calculate dimension scores and overall scorecard
    - SimilarityEngine: Calculate similarity with baselines
    - VerdictEngine: Assess trustworthiness with multi-signal analysis
    """
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.score_calculator = ScoreCardCalculator()
        self.similarity_engine = SimilarityEngine()
        self.verdict_engine = VerdictEngine()
        logger.info("V6 Analysis Pipeline initialized with modular components")

    def analyze(
        self,
        case_results: List[CaseResult],
        predetect_result: Optional[PreDetectionResult] = None,
        claimed_model: Optional[str] = None,
        baseline_comparison: bool = True,
        item_stats: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Run complete analysis pipeline.
        
        Args:
            case_results: Results from test cases
            predetect_result: Optional pre-detection result
            claimed_model: Claimed model name (for family-based weights)
            baseline_comparison: Whether to perform baseline comparison
            item_stats: Optional item statistics for data-driven weights
            
        Returns:
            Complete analysis results
        """
        logger.info("Starting V6 analysis pipeline", case_count=len(case_results))
        
        # Step 1: Feature extraction
        logger.info("Step 1: Extracting features from case results")
        features = self.feature_extractor.extract(case_results)
        
        # Step 2: Score calculation
        logger.info("Step 2: Calculating dimension scores")
        scorecard = self.score_calculator.calculate(
            features=features,
            case_results=case_results,
            similarities=[],  # Will be filled in step 3
            predetect=predetect_result,
            claimed_model=claimed_model,
            item_stats=item_stats,
        )
        
        # Step 3: Similarity analysis (optional)
        similarities = []
        if baseline_comparison:
            logger.info("Step 3: Performing baseline similarity analysis")
            similarities = self._calculate_similarities(features)
        
        # Step 4: Trust assessment
        logger.info("Step 4: Assessing trustworthiness")
        verdict = self.verdict_engine.assess(
            scorecard=scorecard,
            similarities=similarities,
            predetect=predetect_result,
            features=features,
            case_results=case_results,
        )
        
        # Step 5: Build final report
        logger.info("Step 5: Building final analysis report")
        report = self._build_report(
            features=features,
            scorecard=scorecard,
            similarities=similarities,
            verdict=verdict,
            predetect_result=predetect_result,
            case_results=case_results,
        )
        
        logger.info("V6 analysis pipeline completed", 
                   overall_score=scorecard.overall_score,
                   confidence=verdict.confidence)
        
        return report

    def _calculate_similarities(self, features: Dict[str, float]) -> List[SimilarityResult]:
        """Calculate similarity with available baselines."""
        try:
            from app.repository.repo import get_repository
            repo = get_repository()
            baselines = repo.list_baselines(limit=20)
            
            if not baselines:
                logger.info("No baselines available for similarity comparison")
                return []
            
            # Calculate feature importance from baselines
            feature_importance = self.similarity_engine.compute_feature_importance_from_baselines(baselines)
            
            # Rank similarities
            similarity_results = self.similarity_engine.rank_similarities(
                run_features=features,
                baselines=baselines,
                feature_importance=feature_importance,
                limit=10,
            )
            
            # Convert to SimilarityResult objects
            similarities = []
            for result in similarity_results:
                similarities.append(SimilarityResult(
                    benchmark=result["benchmark"],
                    score=result["similarity"],
                    ci_95_low=result["ci_95_low"],
                    ci_95_high=result["ci_95_high"],
                    valid_features=result["valid_features"],
                    run_id=result.get("run_id"),
                    overall_score=result.get("overall_score"),
                ))
            
            logger.info(f"Calculated {len(similarities)} similarity comparisons")
            return similarities
            
        except Exception as e:
            logger.error("Failed to calculate similarities", error=str(e))
            return []

    def _build_report(
        self,
        features: Dict[str, float],
        scorecard: ScoreCard,
        similarities: List[SimilarityResult],
        verdict: Any,
        predetect_result: Optional[PreDetectionResult],
        case_results: List[CaseResult],
    ) -> Dict[str, Any]:
        """Build comprehensive analysis report."""
        report = {
            # Core results
            "scorecard": {
                "overall_score": scorecard.overall_score,
                "dimension_scores": {
                    "reasoning": scorecard.reasoning_score,
                    "adversarial": scorecard.adversarial_reasoning_score,
                    "instruction": scorecard.instruction_score,
                    "coding": scorecard.coding_score,
                    "safety": scorecard.safety_score,
                    "protocol": scorecard.protocol_score,
                    "knowledge": scorecard.knowledge_score,
                    "tool_use": scorecard.tool_use_score,
                },
                "performance_scores": {
                    "performance": scorecard.performance_score,
                    "speed": scorecard.speed_score,
                    "stability": scorecard.stability_score,
                    "cost_efficiency": scorecard.cost_efficiency,
                },
                "confidence_level": scorecard.confidence_level,
                "breakdown": scorecard.breakdown,
            },
            
            # Trust assessment
            "trust_verdict": {
                "is_real": verdict.is_real,
                "confidence": verdict.confidence,
                "risk_level": verdict.risk_level,
                "reasoning": verdict.reasoning,
                "final_score": verdict.final_score,
                "evidence_chain": verdict.evidence_chain,
            },
            
            # Similarity analysis
            "similarities": [
                {
                    "benchmark": s.benchmark,
                    "similarity": s.score,
                    "ci_95_low": s.ci_95_low,
                    "ci_95_high": s.ci_95_high,
                    "valid_features": s.valid_features,
                    "run_id": s.run_id,
                    "overall_score": s.overall_score,
                }
                for s in similarities
            ],
            
            # Pre-detection results
            "predetect": {
                "success": predetect_result.success if predetect_result else False,
                "identified_as": predetect_result.identified_as if predetect_result else None,
                "confidence": predetect_result.confidence if predetect_result else 0.0,
                "layer_stopped": predetect_result.layer_stopped if predetect_result else None,
                "total_tokens_used": predetect_result.total_tokens_used if predetect_result else 0,
                "routing_info": predetect_result.routing_info if predetect_result else {},
            },
            
            # Feature analysis
            "features": features,
            
            # Raw case results (for detailed analysis)
            "case_results": [
                {
                    "case_id": r.case.id if hasattr(r.case, 'id') else r.case.name,
                    "category": r.case.category,
                    "pass_rate": r.pass_rate,
                    "mean_latency_ms": r.mean_latency_ms,
                    "sample_count": len(r.samples),
                    "breakdown": r.breakdown,
                }
                for r in case_results
            ],
            
            # Metadata
            "metadata": {
                "pipeline_version": "v6_refactored",
                "analysis_timestamp": json.dumps({"timestamp": "now"}),  # Placeholder
                "total_cases": len(case_results),
                "total_samples": sum(len(r.samples) for r in case_results),
                "feature_count": len(features),
                "similarity_count": len(similarities),
            },
        }
        
        return report

    def analyze_quick_mode(
        self,
        case_results: List[CaseResult],
        predetect_result: Optional[PreDetectionResult] = None,
        claimed_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Quick mode analysis with reduced computation.
        Skips detailed similarity analysis for faster results.
        """
        logger.info("Starting V6 quick mode analysis")
        
        # Extract features
        features = self.feature_extractor.extract(case_results)
        
        # Calculate scores
        scorecard = self.score_calculator.calculate(
            features=features,
            case_results=case_results,
            similarities=[],
            predetect=predetect_result,
            claimed_model=claimed_model,
        )
        
        # Quick similarity check (only top 3 baselines)
        similarities = []
        try:
            from app.repository.repo import get_repository
            repo = get_repository()
            baselines = repo.list_baselines(limit=5)  # Reduced limit for quick mode
            
            if baselines:
                similarity_results = self.similarity_engine.rank_similarities(
                    run_features=features,
                    baselines=baselines,
                    limit=3,  # Only top 3 for quick mode
                )
                
                for result in similarity_results:
                    similarities.append(SimilarityResult(
                        benchmark=result["benchmark"],
                        score=result["similarity"],
                        ci_95_low=result["ci_95_low"],
                        ci_95_high=result["ci_95_high"],
                        valid_features=result["valid_features"],
                    ))
        except Exception as e:
            logger.warning("Quick mode similarity analysis failed", error=str(e))
        
        # Trust assessment
        verdict = self.verdict_engine.assess(
            scorecard=scorecard,
            similarities=similarities,
            predetect=predetect_result,
            features=features,
            case_results=case_results,
        )
        
        # Build simplified report
        report = self._build_report(
            features=features,
            scorecard=scorecard,
            similarities=similarities,
            verdict=verdict,
            predetect_result=predetect_result,
            case_results=case_results,
        )
        
        report["metadata"]["analysis_mode"] = "quick"
        
        logger.info("V6 quick mode analysis completed")
        return report


# Convenience function for backward compatibility
def analyze_case_results(
    case_results: List[CaseResult],
    predetect_result: Optional[PreDetectionResult] = None,
    claimed_model: Optional[str] = None,
    test_mode: str = "standard",
    item_stats: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Backward compatibility function for analyzing case results.
    
    Args:
        case_results: Results from test cases
        predetect_result: Optional pre-detection result
        claimed_model: Claimed model name
        test_mode: Analysis mode (quick, standard, deep)
        item_stats: Item statistics for data-driven weights
        
    Returns:
        Analysis results
    """
    pipeline = AnalysisPipeline()
    
    if test_mode == "quick":
        return pipeline.analyze_quick_mode(
            case_results=case_results,
            predetect_result=predetect_result,
            claimed_model=claimed_model,
        )
    else:
        return pipeline.analyze(
            case_results=case_results,
            predetect_result=predetect_result,
            claimed_model=claimed_model,
            baseline_comparison=True,
            item_stats=item_stats,
        )
