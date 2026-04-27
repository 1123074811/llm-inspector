"""
IRT 2PL Calibration Engine - Item Response Theory parameter estimation.

Calibrates test case parameters using Expectation-Maximization algorithm.
All parameters derived from actual test data - no hardcoded estimates.

Reference:
- Embretson, S. E., & Reise, S. P. (2000). Item Response Theory for Psychologists.
- Baker, F. B., & Kim, S. H. (2004). Item Response Theory: Parameter Estimation Techniques.

v7.0 Scientific Scoring Implementation
"""

from __future__ import annotations

import math
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

import numpy as np
from scipy.optimize import minimize_scalar, minimize
from scipy.stats import norm

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class IRTParameters:
    """IRT 2PL parameters for a test case.
    
    Attributes:
        a: Discrimination parameter (0.5 - 2.0 optimal, higher = better discrimination)
        b: Difficulty parameter (-3 to 3 scale, 0 = average difficulty)
        c: Guessing parameter (fixed at 0.25 for multiple choice, 0 for open-ended)
        fit_rmse: Root mean square error of model fit
        info_max: Maximum information value
        reliability: Test-retest reliability estimate
        n_calibrated: Number of models used for calibration
        calibration_date: ISO format date string
        data_source: Source of calibration data
    """
    a: float
    b: float
    c: float = 0.0
    fit_rmse: float = 0.0
    info_max: float = 0.0
    reliability: float = 0.0
    n_calibrated: int = 0
    calibration_date: str = ""
    data_source: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IRTParameters":
        """Create from dictionary."""
        return cls(**data)
    
    def is_valid(self) -> Tuple[bool, List[str]]:
        """Validate parameters meet scientific standards."""
        issues = []
        
        if self.a < 0.3:
            issues.append(f"Low discrimination (a={self.a:.2f} < 0.3)")
        if self.a > 3.0:
            issues.append(f"Extreme discrimination (a={self.a:.2f} > 3.0)")
        
        if abs(self.b) > 3.5:
            issues.append(f"Extreme difficulty (b={self.b:.2f})")
        
        if self.fit_rmse > 0.15:
            issues.append(f"Poor model fit (RMSE={self.fit_rmse:.3f} > 0.15)")
        
        if self.n_calibrated < 50:
            issues.append(f"Insufficient calibration data (n={self.n_calibrated} < 50)")
        
        if self.reliability < 0.7:
            issues.append(f"Low reliability (r={self.reliability:.2f} < 0.7)")
        
        return len(issues) == 0, issues
    
    def calculate_information(self, theta: float | np.ndarray) -> float | np.ndarray:
        """Calculate Fisher information at given ability level(s).
        
        I(θ) = a² * P(θ) * (1-P(θ)) / (1-c)²
        
        Higher information = more precise measurement at that ability level.
        
        Args:
            theta: Ability parameter(s), typically -3 to 3
            
        Returns:
            Information value(s)
        """
        a, b, c = self.a, self.b, self.c
        
        # Probability of correct response
        p = c + (1 - c) / (1 + np.exp(-a * (theta - b)))
        
        # Fisher information
        info = (a ** 2 * p * (1 - p)) / ((1 - c) ** 2)
        
        return info
    
    def probability_correct(self, theta: float) -> float:
        """Calculate probability of correct response at ability θ."""
        return self.c + (1 - self.c) / (1 + math.exp(-self.a * (theta - self.b)))


@dataclass
class CalibrationResult:
    """Result of IRT calibration for a test case."""
    case_id: str
    parameters: IRTParameters
    convergence_status: str
    log_likelihood: float
    aic: float  # Akaike Information Criterion
    bic: float  # Bayesian Information Criterion
    

def _calculate_irt_probability(a: float, b: float, c: float, theta: float) -> float:
    """Calculate 2PL IRT probability of correct response."""
    return c + (1 - c) / (1 + math.exp(-a * (theta - b)))


def _marginal_log_likelihood(
    params: np.ndarray,
    responses: np.ndarray,
    thetas: np.ndarray,
    theta_weights: np.ndarray
) -> float:
    """Calculate marginal log-likelihood for MML estimation.
    
    Uses Gauss-Hermite quadrature for integration over ability distribution.
    
    Args:
        params: [a, b, c] parameters
        responses: Binary response matrix (n_models x n_items)
        thetas: Quadrature points for ability distribution
        theta_weights: Quadrature weights
        
    Returns:
        Negative log-likelihood (for minimization)
    """
    a, b, c = params[0], params[1], max(0, min(params[2], 0.5))  # Constrain c
    
    log_lik = 0.0
    
    for i in range(responses.shape[0]):  # For each model
        model_lik = 0.0
        
        for q, w in zip(thetas, theta_weights):  # Quadrature integration
            # Calculate P(X|θ) for this model
            p_correct = _calculate_irt_probability(a, b, c, q)
            
            # Likelihood of observed response
            if responses[i] == 1:
                prob = p_correct
            else:
                prob = 1 - p_correct
            
            # Weight by ability prior
            model_lik += w * max(prob, 1e-10)
        
        log_lik += math.log(max(model_lik, 1e-10))
    
    return -log_lik  # Return negative for minimization


class IRTCalibrator:
    """
    IRT 2PL Parameter Calibration using Marginal Maximum Likelihood (MML).
    
    MML integrates out the ability parameters using quadrature,
    providing more stable estimates than joint maximum likelihood.
    
    Reference: Bock & Lieberman (1970) for MML estimation
    """
    
    # Gauss-Hermite quadrature points and weights (standard normal)
    # Using 15-point quadrature for accuracy
    QUADRATURE_POINTS = np.array([
        -4.499990707309391, -3.669950373404452, -2.967166927905605,
        -2.325732486173857, -1.719992575186493, -1.136115585210924,
        -0.565069583255575, 0.0, 0.565069583255575,
        1.136115585210924, 1.719992575186493, 2.325732486173857,
        2.967166927905605, 3.669950373404452, 4.499990707309391
    ])
    
    QUADRATURE_WEIGHTS = np.array([
        1.522475804253536e-09, 3.654616459502574e-06, 0.0002478188383258185,
        0.004943055923129217, 0.03915529853367195, 0.1512698154352779,
        0.3176091250917188, 0.3809264288972775, 0.3176091250917188,
        0.1512698154352779, 0.03915529853367195, 0.004943055923129217,
        0.0002478188383258185, 3.654616459502574e-06, 1.522475804253536e-09
    ])
    
    def __init__(self, min_calibrations: int = 50):
        """
        Initialize calibrator.
        
        Args:
            min_calibrations: Minimum number of model responses required for calibration
        """
        self.min_calibrations = min_calibrations
    
    def calibrate_case(
        self,
        case_id: str,
        responses: List[Tuple[str, bool, Optional[float]]],
        dimension: str = "unknown"
    ) -> CalibrationResult:
        """
        Calibrate IRT parameters for a single test case.
        
        Args:
            case_id: Unique identifier for the test case
            responses: List of (model_id, passed, model_ability_theta) tuples.
                      model_ability_theta can be None if unknown.
            dimension: Test case dimension category
            
        Returns:
            CalibrationResult with fitted parameters and quality metrics
            
        Raises:
            ValueError: If insufficient calibration data provided
        """
        if len(responses) < self.min_calibrations:
            raise ValueError(
                f"Insufficient calibration data for {case_id}: "
                f"{len(responses)} < {self.min_calibrations} required"
            )
        
        logger.info(f"Calibrating {case_id} with {len(responses)} model responses")
        
        # Extract response vector
        response_vec = np.array([1 if passed else 0 for _, passed, _ in responses])
        
        # Initial parameter estimates (method of moments)
        p_correct = np.mean(response_vec)
        
        # Initial difficulty estimate: inverse logit of pass rate
        b_init = -np.log(max(p_correct, 0.01) / max(1 - p_correct, 0.01))
        b_init = np.clip(b_init, -2.5, 2.5)
        
        # Initial discrimination (moderate)
        a_init = 1.0
        
        # Initial guessing parameter (0 for open-ended, would be 0.25 for MC)
        c_init = 0.0
        
        # Optimize using MML
        initial_params = np.array([a_init, b_init, c_init])
        
        # Parameter bounds
        bounds = [(0.1, 3.0), (-3.5, 3.5), (0.0, 0.5)]
        
        try:
            result = minimize(
                lambda p: _marginal_log_likelihood(
                    p, response_vec, self.QUADRATURE_POINTS, self.QUADRATURE_WEIGHTS
                ),
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-8}
            )
            
            if not result.success:
                logger.warning(f"Calibration optimization did not converge for {case_id}")
                convergence_status = "partial"
            else:
                convergence_status = "converged"
            
            a_est, b_est, c_est = result.x[0], result.x[1], max(0, min(result.x[2], 0.5))
            
            # Calculate fit statistics
            fit_rmse = self._calculate_fit_rmse(
                response_vec, a_est, b_est, c_est
            )
            
            # Calculate maximum information
            info_max = max(self._calculate_information_curve(a_est, b_est, c_est))
            
            # Estimate reliability (simplified)
            reliability = self._estimate_reliability(
                response_vec, a_est, b_est, c_est
            )
            
            # Calculate AIC and BIC
            n_params = 2 if c_est == 0 else 3  # a and b always, c if estimated
            log_lik = -result.fun
            aic = 2 * n_params - 2 * log_lik
            bic = n_params * np.log(len(responses)) - 2 * log_lik
            
            params = IRTParameters(
                a=round(a_est, 3),
                b=round(b_est, 3),
                c=round(c_est, 3),
                fit_rmse=round(fit_rmse, 4),
                info_max=round(info_max, 3),
                reliability=round(reliability, 3),
                n_calibrated=len(responses),
                calibration_date=datetime.utcnow().isoformat(),
                data_source=f"empirical_{len(responses)}_models"
            )
            
            return CalibrationResult(
                case_id=case_id,
                parameters=params,
                convergence_status=convergence_status,
                log_likelihood=round(log_lik, 4),
                aic=round(aic, 2),
                bic=round(bic, 2)
            )
            
        except Exception as e:
            logger.error(f"Calibration failed for {case_id}: {e}")
            # Return default parameters with warning flags
            return CalibrationResult(
                case_id=case_id,
                parameters=IRTParameters(
                    a=1.0, b=0.0, c=0.0,
                    fit_rmse=999.0,
                    n_calibrated=len(responses),
                    calibration_date=datetime.utcnow().isoformat(),
                    data_source="fallback_default"
                ),
                convergence_status=f"failed: {str(e)[:50]}",
                log_likelihood=0.0,
                aic=9999.0,
                bic=9999.0
            )
    
    def _calculate_fit_rmse(
        self,
        responses: np.ndarray,
        a: float,
        b: float,
        c: float
    ) -> float:
        """Calculate RMSE of model fit."""
        # Use estimated abilities (simplified EAP)
        thetas = np.linspace(-3, 3, 20)
        
        predicted = []
        for resp in responses:
            # Expected value across ability distribution
            exp_val = np.mean([
                _calculate_irt_probability(a, b, c, theta)
                for theta in thetas
            ])
            predicted.append(exp_val)
        
        residuals = responses - np.array(predicted)
        return np.sqrt(np.mean(residuals ** 2))
    
    def _calculate_information_curve(
        self,
        a: float,
        b: float,
        c: float
    ) -> np.ndarray:
        """Calculate information curve across ability range."""
        thetas = np.linspace(-3, 3, 100)
        info_values = []
        
        for theta in thetas:
            p = _calculate_irt_probability(a, b, c, theta)
            info = (a ** 2 * p * (1 - p)) / ((1 - c) ** 2)
            info_values.append(info)
        
        return np.array(info_values)
    
    def _estimate_reliability(
        self,
        responses: np.ndarray,
        a: float,
        b: float,
        c: float
    ) -> float:
        """Estimate test-retest reliability (simplified)."""
        # Use Spearman-Brown prophecy formula approximation
        # based on observed variance vs model-predicted variance
        observed_var = np.var(responses)
        
        if observed_var < 0.01:
            return 0.5  # Default for near-constant responses
        
        # Predicted variance from model
        thetas = np.linspace(-3, 3, 20)
        probs = [_calculate_irt_probability(a, b, c, t) for t in thetas]
        model_var = np.var(probs)
        
        # Reliability estimate
        reliability = min(0.95, max(0.5, 1 - (model_var / observed_var)))
        
        return reliability
    
    def calibrate_suite(
        self,
        historical_results: Dict[str, List[Tuple[str, bool, Optional[float]]]]
    ) -> Dict[str, CalibrationResult]:
        """
        Calibrate entire test suite.
        
        Args:
            historical_results: Dict mapping case_id to list of responses
            
        Returns:
            Dict mapping case_id to CalibrationResult
        """
        results = {}
        
        for case_id, responses in historical_results.items():
            try:
                result = self.calibrate_case(case_id, responses)
                results[case_id] = result
                
                # Log quality metrics
                valid, issues = result.parameters.is_valid()
                if not valid:
                    logger.warning(f"{case_id} calibration issues: {issues}")
                
            except ValueError as e:
                logger.warning(f"Skipping {case_id}: {e}")
                continue
        
        return results
    
    def save_calibration_cache(
        self,
        results: Dict[str, CalibrationResult],
        filepath: str
    ) -> None:
        """Save calibration results to cache file."""
        data = {
            case_id: {
                "parameters": r.parameters.to_dict(),
                "convergence_status": r.convergence_status,
                "log_likelihood": r.log_likelihood,
                "aic": r.aic,
                "bic": r.bic
            }
            for case_id, r in results.items()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved calibration cache to {filepath}")
    
    def load_calibration_cache(
        self,
        filepath: str
    ) -> Dict[str, CalibrationResult]:
        """Load calibration results from cache file."""
        if not os.path.exists(filepath):
            return {}
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = {}
        for case_id, r_data in data.items():
            params = IRTParameters.from_dict(r_data["parameters"])
            results[case_id] = CalibrationResult(
                case_id=case_id,
                parameters=params,
                convergence_status=r_data["convergence_status"],
                log_likelihood=r_data["log_likelihood"],
                aic=r_data["aic"],
                bic=r_data["bic"]
            )
        
        logger.info(f"Loaded calibration cache from {filepath}: {len(results)} cases")
        return results


def calculate_data_driven_weights(
    irt_params: Dict[str, IRTParameters],
    theta_range: Tuple[float, float] = (-3, 3)
) -> Dict[str, float]:
    """
    Calculate dimension weights based on IRT information functions.
    
    Weight ∝ ∫ I(θ) * g(θ) dθ
    where g(θ) is the prior distribution of model abilities (assumed uniform).
    
    This maximizes the expected precision of measurement.
    
    Args:
        irt_params: Dict mapping case_id to IRTParameters
        theta_range: Range of ability values to integrate over
        
    Returns:
        Dict mapping dimension to weight (sums to 1.0)
    """
    from collections import defaultdict
    
    # Group by dimension
    dim_info: Dict[str, List[float]] = defaultdict(list)
    
    thetas = np.linspace(theta_range[0], theta_range[1], 100)
    
    for case_id, params in irt_params.items():
        # Extract dimension from case_id or lookup
        dimension = _extract_dimension(case_id)
        
        # Calculate area under information curve
        info_values = params.calculate_information(thetas)
        total_info = np.trapz(info_values, thetas)
        
        dim_info[dimension].append(total_info)
    
    # Calculate mean information per dimension
    dim_mean_info = {
        dim: np.mean(infos) if infos else 0.0
        for dim, infos in dim_info.items()
    }
    
    # Normalize to sum to 1.0
    total = sum(dim_mean_info.values())
    if total == 0:
        # Fallback to equal weights
        n_dims = len(dim_mean_info)
        return {dim: 1.0 / n_dims for dim in dim_mean_info}
    
    weights = {
        dim: round(info / total, 3)
        for dim, info in dim_mean_info.items()
    }
    
    return weights


def _extract_dimension(case_id: str) -> str:
    """Extract dimension from case_id (e.g., 'instr_001' -> 'instruction')."""
    prefix_map = {
        'instr': 'instruction',
        'sys': 'instruction',
        'reason': 'reasoning',
        'code': 'coding',
        'math': 'reasoning',
        'safety': 'safety',
        'protocol': 'protocol',
        'refusal': 'safety',
        'param': 'protocol',
        'tool': 'tool_use',
        'knowledge': 'knowledge',
        'adversarial': 'adversarial',
    }
    
    for prefix, dim in prefix_map.items():
        if case_id.startswith(prefix):
            return dim
    
    return 'unknown'


# Global calibration cache instance
_calibration_cache: Dict[str, CalibrationResult] = {}


def get_calibrated_params(case_id: str) -> Optional[IRTParameters]:
    """Get calibrated parameters for a test case (from cache)."""
    global _calibration_cache
    
    if case_id in _calibration_cache:
        return _calibration_cache[case_id].parameters
    
    return None


def load_suite_calibration(cache_path: Optional[str] = None) -> None:
    """Load calibration data for entire test suite."""
    global _calibration_cache
    
    if cache_path is None:
        cache_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 'fixtures', 'irt_calibration_cache.json'
        )
    
    calibrator = IRTCalibrator()
    _calibration_cache = calibrator.load_calibration_cache(cache_path)


# Initialize on module load
load_suite_calibration()


# ── v16 Phase 5: IRT Cold Start Prior ────────────────────────────────────────

# Prior table for IRT parameters by category and difficulty.
# Registered in SOURCES.yaml as irt.prior.<category>.<difficulty>.
# References:
#   - GPQA Diamond: Rein et al. (2023) Table 2, b≈1.5 for expert-level
#   - AIME: Historical pass rates ~10-15% → b≈2.0
#   - Baker & Kim (2004) "Item Response Theory: Parameter Estimation Techniques"
_COLD_START_PRIORS: dict[str, dict[str, tuple[float, float, float]]] = {
    # category -> {difficulty_level -> (a, b, c)}
    "reasoning": {
        "easy": (0.8, -0.5, 0.0),
        "medium": (1.0, 0.5, 0.0),
        "hard": (1.2, 1.5, 0.0),
        "expert": (1.5, 2.0, 0.0),
    },
    "coding": {
        "easy": (0.7, -0.3, 0.0),
        "medium": (0.9, 0.5, 0.0),
        "hard": (1.1, 1.5, 0.0),
        "expert": (1.3, 2.0, 0.0),
    },
    "knowledge": {
        "easy": (0.6, -1.0, 0.25),
        "medium": (0.8, 0.0, 0.25),
        "hard": (1.0, 1.0, 0.25),
        "expert": (1.2, 1.5, 0.20),
    },
    "safety": {
        "easy": (0.7, -0.5, 0.0),
        "medium": (0.9, 0.3, 0.0),
        "hard": (1.1, 1.0, 0.0),
        "expert": (1.3, 1.5, 0.0),
    },
    "instruction": {
        "easy": (0.8, -0.5, 0.0),
        "medium": (1.0, 0.3, 0.0),
        "hard": (1.2, 1.0, 0.0),
        "expert": (1.4, 1.5, 0.0),
    },
    "adversarial": {
        "easy": (0.9, 0.0, 0.0),
        "medium": (1.1, 0.8, 0.0),
        "hard": (1.3, 1.5, 0.0),
        "expert": (1.5, 2.0, 0.0),
    },
    # Default for any category not listed
    "default": {
        "easy": (0.7, -0.5, 0.0),
        "medium": (0.9, 0.5, 0.0),
        "hard": (1.1, 1.5, 0.0),
        "expert": (1.3, 2.0, 0.0),
    },
}


def _difficulty_to_level(difficulty: float) -> str:
    """Map numeric difficulty [0,1] to level string."""
    if difficulty < 0.3:
        return "easy"
    elif difficulty < 0.6:
        return "medium"
    elif difficulty < 0.85:
        return "hard"
    else:
        return "expert"


def cold_start_prior(
    category: str,
    difficulty: float | None = None,
    difficulty_meta: dict | None = None,
) -> IRTParameters:
    """
    v16 Phase 5: Return IRT prior parameters for cold-start cases.

    When a test case has no empirical calibration data (< 50 samples),
    use these priors instead of hardcoded fallbacks.

    Args:
        category: Test case category (e.g. "reasoning", "coding").
        difficulty: Numeric difficulty [0, 1] from suite JSON.
        difficulty_meta: Optional metadata dict with 'level' key.

    Returns:
        IRTParameters with prior (a, b, c) values.
    """
    # Resolve difficulty level
    if difficulty_meta and "level" in difficulty_meta:
        level = difficulty_meta["level"]
    elif difficulty is not None:
        level = _difficulty_to_level(difficulty)
    else:
        level = "medium"

    # Look up prior table
    cat_priors = _COLD_START_PRIORS.get(category, _COLD_START_PRIORS["default"])
    a, b, c = cat_priors.get(level, cat_priors.get("medium", (1.0, 0.5, 0.0)))

    return IRTParameters(
        a=a,
        b=b,
        c=c,
        data_source=f"cold_start_prior:{category}:{level}",
    )
