"""Theta Scale Scoring System for LLM Inspector v8.0.

Implements IRT-based theta score conversion with confidence intervals
and multi-dimensional score synthesis.

Reference:
- Embretson & Reise (2000): Item Response Theory for Psychologists
- van der Linden (2010): Elements of Adaptive Testing
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone

from app.analysis.irt_params import IRTParameters
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ThetaScore:
    """Theta ability estimate with confidence interval."""
    
    theta: float  # Ability estimate (-4 to 4)
    standard_error: float
    confidence_level: float = 0.95
    
    # Confidence interval
    ci_lower: float = field(init=False)
    ci_upper: float = field(init=False)
    
    # Source information
    dimension: str = ""
    item_count: int = 0
    test_length: int = 0
    
    # Quality metrics
    reliability: float = 0.0
    information: float = 0.0
    
    def __post_init__(self):
        """Calculate confidence interval."""
        z_score = 1.96 if self.confidence_level == 0.95 else 1.645  # 95% or 90%
        margin = z_score * self.standard_error
        self.ci_lower = max(-4.0, self.theta - margin)
        self.ci_upper = min(4.0, self.theta + margin)
    
    @property
    def percentile(self) -> float:
        """Convert theta to percentile score (0-100)."""
        try:
            from scipy.stats import norm
            return norm.cdf(self.theta) * 100
        except ImportError:
            # Fallback using error function
            return 100 * (0.5 * (1 + math.erf(self.theta / math.sqrt(2))))
    
    @property
    def precision(self) -> str:
        """Get precision level based on standard error."""
        if self.standard_error < 0.2:
            return "high"
        elif self.standard_error < 0.4:
            return "medium"
        else:
            return "low"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "theta": round(self.theta, 3),
            "standard_error": round(self.standard_error, 3),
            "confidence_level": self.confidence_level,
            "confidence_interval": {
                "lower": round(self.ci_lower, 3),
                "upper": round(self.ci_upper, 3),
            },
            "percentile": round(self.percentile, 1),
            "dimension": self.dimension,
            "item_count": self.item_count,
            "test_length": self.test_length,
            "reliability": round(self.reliability, 3),
            "information": round(self.information, 3),
            "precision": self.precision,
        }


@dataclass
class CompositeScore:
    """Multi-dimensional composite score."""
    
    score: float  # Composite theta or percentile
    standard_error: float
    
    # Component scores
    dimension_scores: Dict[str, ThetaScore] = field(default_factory=dict)
    dimension_weights: Dict[str, float] = field(default_factory=dict)
    
    # Synthesis method
    synthesis_method: str = "information_weighted"
    
    # Quality metrics
    reliability: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "score": round(self.score, 3),
            "standard_error": round(self.standard_error, 3),
            "synthesis_method": self.synthesis_method,
            "reliability": round(self.reliability, 3),
            "dimensions": {
                dim: {
                    "theta": round(s.theta, 3),
                    "weight": round(self.dimension_weights.get(dim, 0), 3),
                    "percentile": round(s.percentile, 1),
                }
                for dim, s in self.dimension_scores.items()
            },
        }


class ThetaScoringEngine:
    """
    Enhanced theta scoring engine with confidence intervals.
    
    Features:
    - Maximum likelihood estimation (Newton-Raphson)
    - Weighted likelihood estimation (WLE)
    - Expected a posteriori (EAP) with prior
    - Multi-dimensional synthesis
    """
    
    def __init__(self, estimation_method: str = "mle"):
        """
        Initialize scoring engine.
        
        Args:
            estimation_method: "mle", "wle", or "eap"
        """
        self.estimation_method = estimation_method
        
        # Convergence settings
        self.max_iterations = 50
        self.convergence_threshold = 0.001
        
        logger.info(f"ThetaScoringEngine initialized (method={estimation_method})")
    
    def calculate_theta(
        self,
        responses: List[bool],  # True = correct, False = incorrect
        item_params: List[IRTParameters],
        dimension: str = "",
    ) -> ThetaScore:
        """
        Calculate theta estimate from item responses.
        
        Args:
            responses: List of correct/incorrect responses
            item_params: IRT parameters for each item
            dimension: Dimension name
            
        Returns:
            ThetaScore with confidence interval
        """
        if not responses or not item_params:
            return ThetaScore(
                theta=0.0,
                standard_error=999.0,
                dimension=dimension,
            )
        
        if len(responses) != len(item_params):
            raise ValueError("responses and item_params must have same length")
        
        # Calculate based on estimation method
        if self.estimation_method == "mle":
            theta, se = self._mle_estimate(responses, item_params)
        elif self.estimation_method == "wle":
            theta, se = self._wle_estimate(responses, item_params)
        elif self.estimation_method == "eap":
            theta, se = self._eap_estimate(responses, item_params)
        else:
            theta, se = self._mle_estimate(responses, item_params)
        
        # Calculate information and reliability
        total_info = sum(
            p.calculate_information(theta) for p in item_params
        )
        reliability = total_info / (total_info + 1) if total_info > 0 else 0.0
        
        return ThetaScore(
            theta=theta,
            standard_error=se,
            dimension=dimension,
            item_count=sum(1 for r in responses if r is not None),
            test_length=len(responses),
            reliability=reliability,
            information=total_info,
        )
    
    def _mle_estimate(
        self,
        responses: List[bool],
        item_params: List[IRTParameters],
    ) -> Tuple[float, float]:
        """
        Maximum likelihood estimation (Newton-Raphson).
        
        Reference: Embretson & Reise (2000), Eq. 5.4
        """
        theta = 0.0  # Initial estimate
        
        for iteration in range(self.max_iterations):
            # First derivative (score function)
            score = 0.0
            # Second derivative (information)
            info = 0.0
            
            for response, param in zip(responses, item_params):
                if response is None:
                    continue
                    
                prob = param.probability_correct(theta)
                # Avoid division by zero
                prob = max(0.001, min(0.999, prob))
                
                # Score contribution
                score += param.a * (int(response) - prob)
                
                # Information contribution
                info += param.calculate_information(theta)
            
            # Newton-Raphson update
            if info > 0:
                delta = score / info
                theta += delta
                
                if abs(delta) < self.convergence_threshold:
                    break
            else:
                break
        
        # Standard error
        final_info = sum(
            p.calculate_information(theta) for p in item_params
        )
        se = 1.0 / math.sqrt(final_info) if final_info > 0 else 999.0
        
        # Clamp to valid range
        theta = max(-4.0, min(4.0, theta))
        
        return (theta, se)

    def calculate_mdirt_theta(
        self,
        responses: List[bool],
        item_params: List[IRTParameters],
        dimensions: List[str]
    ) -> Dict[str, ThetaScore]:
        """
        v10: Multidimensional Item Response Theory (MDIRT) estimation.
        Calculates orthogonal theta vector.
        
        Reference: Reckase (2009) Multidimensional Item Response Theory
        
        Returns:
            Dict mapping dimension to ThetaScore (Standardized: Mean=500, SD=100)
        """
        # Map items to dimensions
        dim_items = {}
        for resp, param, dim in zip(responses, item_params, dimensions):
            if resp is not None:
                dim_items.setdefault(dim, []).append((resp, param))
        
        results = {}
        for dim, items in dim_items.items():
            dim_resps = [i[0] for i in items]
            dim_params = [i[1] for i in items]
            
            # Estimate raw theta (-4 to 4)
            raw_theta, se = self._mle_estimate(dim_resps, dim_params)
            
            # Convert to Standard Score (Mean 500, SD 100)
            # theta=0 -> 500, theta=1 -> 600
            standard_score = 500 + (raw_theta * 100)
            standard_se = se * 100
            
            # Calculate info and reliability
            total_info = sum(p.calculate_information(raw_theta) for p in dim_params)
            reliability = total_info / (total_info + 1) if total_info > 0 else 0.0
            
            results[dim] = ThetaScore(
                theta=standard_score,
                standard_error=standard_se,
                dimension=dim,
                item_count=len(dim_resps),
                test_length=len(dim_resps),
                reliability=reliability,
                information=total_info
            )
            # Override CI for standard score
            results[dim].ci_lower = standard_score - (1.96 * standard_se)
            results[dim].ci_upper = standard_score + (1.96 * standard_se)
            
        return results

    def _wle_estimate(
        self,
        responses: List[bool],
        item_params: List[IRTParameters],
    ) -> Tuple[float, float]:
        """
        Weighted likelihood estimation (bias-corrected MLE).
        
        Reference: Warm (1989) Weighted Likelihood Estimation
        """
        # Start with MLE
        theta, _ = self._mle_estimate(responses, item_params)
        
        # Apply bias correction (simplified)
        n_items = len([r for r in responses if r is not None])
        if n_items > 0:
            # Simple bias correction for extreme scores
            if theta > 3.0:
                theta = 3.0 - 1.0 / n_items
            elif theta < -3.0:
                theta = -3.0 + 1.0 / n_items
        
        # Recalculate SE
        info = sum(
            p.calculate_information(theta) for p in item_params
        )
        se = 1.0 / math.sqrt(info) if info > 0 else 999.0
        
        return (theta, se)
    
    def _eap_estimate(
        self,
        responses: List[bool],
        item_params: List[IRTParameters],
    ) -> Tuple[float, float]:
        """
        Expected a posteriori estimation (with normal prior).
        
        Reference: Bock & Mislevy (1982)
        """
        # Quadrature points for numerical integration
        quadrature_points = [-3, -2, -1, 0, 1, 2, 3]
        quadrature_weights = [0.05, 0.15, 0.25, 0.25, 0.15, 0.05, 0.05]
        
        # Calculate likelihood at each quadrature point
        likelihoods = []
        for qp in quadrature_points:
            log_likelihood = 0.0
            for response, param in zip(responses, item_params):
                if response is None:
                    continue
                prob = param.probability_correct(qp)
                prob = max(0.001, min(0.999, prob))
                log_likelihood += (
                    int(response) * math.log(prob) +
                    (1 - int(response)) * math.log(1 - prob)
                )
            likelihoods.append(math.exp(log_likelihood))
        
        # Weight by prior (normal distribution)
        weighted_likelihoods = [
            l * w for l, w in zip(likelihoods, quadrature_weights)
        ]
        
        # Normalize
        total_weight = sum(weighted_likelihoods)
        if total_weight == 0:
            return (0.0, 999.0)
        
        posteriors = [w / total_weight for w in weighted_likelihoods]
        
        # EAP estimate (mean of posterior)
        theta = sum(p * qp for p, qp in zip(posteriors, quadrature_points))
        
        # Posterior variance
        variance = sum(
            p * (qp - theta) ** 2 
            for p, qp in zip(posteriors, quadrature_points)
        )
        se = math.sqrt(variance)
        
        return (theta, se)
    
    def synthesize_dimensions(
        self,
        dimension_scores: Dict[str, ThetaScore],
        method: str = "information_weighted",
    ) -> CompositeScore:
        """
        Synthesize multi-dimensional scores into composite.
        
        Args:
            dimension_scores: Dict of dimension -> ThetaScore
            method: "information_weighted", "equal", or "custom"
            
        Returns:
            CompositeScore
        """
        if not dimension_scores:
            return CompositeScore(score=0.0, standard_error=999.0)
        
        # Calculate weights based on method
        if method == "information_weighted":
            # Weight by Fisher information (inverse variance weighting)
            total_info = sum(s.information for s in dimension_scores.values())
            weights = {
                dim: s.information / total_info if total_info > 0 else 1.0 / len(dimension_scores)
                for dim, s in dimension_scores.items()
            }
        elif method == "equal":
            n = len(dimension_scores)
            weights = {dim: 1.0 / n for dim in dimension_scores}
        else:
            # Custom weights would be passed in
            weights = {dim: 1.0 / len(dimension_scores) for dim in dimension_scores}
        
        # Weighted average of theta scores
        weighted_theta = sum(
            s.theta * weights[dim] 
            for dim, s in dimension_scores.items()
        )
        
        # Standard error of composite
        # Formula: SE_composite = sqrt(sum(w_i^2 * SE_i^2))
        composite_variance = sum(
            (weights[dim] ** 2) * (s.standard_error ** 2)
            for dim, s in dimension_scores.items()
        )
        composite_se = math.sqrt(composite_variance)
        
        # Reliability of composite
        # Formula: reliability = 1 - (SE^2 / variance)
        # Assuming population variance of 1 for theta scores
        reliability = max(0, 1 - composite_variance)
        
        return CompositeScore(
            score=weighted_theta,
            standard_error=composite_se,
            dimension_scores=dimension_scores,
            dimension_weights=weights,
            synthesis_method=method,
            reliability=reliability,
        )
    
    def percent_to_theta(
        self,
        percentile: float,
        item_params: List[IRTParameters],
    ) -> Tuple[float, float]:
        """
        Convert percentile score to theta (backward compatibility).
        
        Args:
            percentile: 0-100 score
            item_params: IRT parameters
            
        Returns:
            Tuple of (theta, standard_error)
        """
        # Treat percentile as proportion correct
        n_items = len(item_params)
        n_correct = int(n_items * percentile / 100)
        
        # Create synthetic responses
        responses = [True] * n_correct + [False] * (n_items - n_correct)
        
        theta_score = self.calculate_theta(responses, item_params)
        return (theta_score.theta, theta_score.standard_error)


def get_theta_engine(estimation_method: str = "mle") -> ThetaScoringEngine:
    """Get theta scoring engine instance."""
    return ThetaScoringEngine(estimation_method)
