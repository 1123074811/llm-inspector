"""
Factor Analysis Module for Dimension Validation.

⚠️  SCRIPT-ONLY — Not part of the production server pipeline.
    Only referenced by scripts/validate_phase1.py for offline CFA validation.

Implements Confirmatory Factor Analysis (CFA) to validate that scoring dimensions
are statistically independent constructs with proper convergent and discriminant validity.

Reference:
- Hu & Bentler (1999). Cutoff criteria for fit indexes in covariance structure analysis
- Kline (2015). Principles and Practice of Structural Equation Modeling
- Brown (2015). Confirmatory Factor Analysis for Applied Research

v7.0 Scientific Validation Implementation
"""

from __future__ import annotations

import math
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum

import numpy as np
from scipy import linalg
from scipy.stats import chi2

from app.core.logging import get_logger

logger = get_logger(__name__)


class ValidityStatus(Enum):
    """Validity assessment status."""
    VALID = "valid"           # Meets all criteria
    MARGINAL = "marginal"       # Acceptable but borderline
    INVALID = "invalid"       # Fails to meet criteria


@dataclass
class CFAResult:
    """Confirmatory Factor Analysis results."""
    # Absolute fit indices
    chi_square: float
    df: int  # Degrees of freedom
    p_value: float
    rmsea: float  # Root Mean Square Error of Approximation
    rmsea_ci_lower: float
    rmsea_ci_upper: float
    srmr: float  # Standardized Root Mean Square Residual
    
    # Incremental fit indices
    cfi: float  # Comparative Fit Index
    tli: float  # Tucker-Lewis Index (NNFI)
    
    # Model parameters
    factor_loadings: Dict[str, Dict[str, float]]  # dimension -> item -> loading
    factor_correlations: Dict[str, Dict[str, float]]  # dimension correlations
    
    # Validity assessment
    convergent_validity: Dict[str, float]  # AVE for each dimension
    discriminant_validity: Dict[str, Dict[str, Tuple[float, bool]]]  # HTMT ratios
    
    # Overall assessment
    fit_acceptable: bool
    validity_acceptable: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "absolute_fit": {
                "chi_square": round(self.chi_square, 2),
                "df": self.df,
                "p_value": round(self.p_value, 4),
                "rmsea": round(self.rmsea, 3),
                "rmsea_ci": [round(self.rmsea_ci_lower, 3), round(self.rmsea_ci_upper, 3)],
                "srmr": round(self.srmr, 3),
            },
            "incremental_fit": {
                "cfi": round(self.cfi, 3),
                "tli": round(self.tli, 3),
            },
            "factor_loadings": {
                dim: {k: round(v, 3) for k, v in items.items()}
                for dim, items in self.factor_loadings.items()
            },
            "factor_correlations": {
                dim1: {dim2: round(corr, 3) for dim2, corr in corrs.items()}
                for dim1, corrs in self.factor_correlations.items()
            },
            "convergent_validity": {
                k: round(v, 3) for k, v in self.convergent_validity.items()
            },
            "discriminant_validity": self.discriminant_validity,
            "assessment": {
                "fit_acceptable": self.fit_acceptable,
                "validity_acceptable": self.validity_acceptable,
            }
        }


class DimensionValidator:
    """
    Validate dimension structure using Confirmatory Factor Analysis.
    
    Ensures that scoring dimensions:
    1. Have convergent validity (items load strongly on their dimension)
    2. Have discriminant validity (dimensions are not too highly correlated)
    3. Meet established model fit criteria (Hu & Bentler, 1999)
    """
    
    # Hu & Bentler (1999) recommended cutoffs
    RMSEA_CUTOFF = 0.06  # Good fit: < 0.06, Acceptable: < 0.08
    SRMR_CUTOFF = 0.08   # Good fit: < 0.08
    CFI_CUTOFF = 0.95    # Good fit: > 0.95
    TLI_CUTOFF = 0.95    # Good fit: > 0.95
    
    # Convergent validity: Average Variance Extracted (AVE) > 0.5
    AVE_CUTOFF = 0.50
    
    # Discriminant validity: Heterotrait-Monotrait (HTMT) ratio < 0.85
    HTMT_CUTOFF = 0.85
    
    # Factor loading minimum
    LOADING_MIN = 0.50
    LOADING_GOOD = 0.70
    
    def __init__(self):
        self.results: List[CFAResult] = []
    
    def validate_dimensions(
        self,
        dimension_scores: Dict[str, List[float]],
        item_mappings: Optional[Dict[str, List[str]]] = None
    ) -> CFAResult:
        """
        Perform CFA on dimension scores to validate factor structure.
        
        Args:
            dimension_scores: Dict mapping dimension name to list of scores
                (each list should have same length = n_models)
            item_mappings: Optional mapping of items to dimensions for detailed CFA
            
        Returns:
            CFAResult with fit indices and validity assessments
        """
        # Convert to data matrix
        dimensions = list(dimension_scores.keys())
        n_models = len(dimension_scores[dimensions[0]])
        
        # Create data matrix (n_models x n_dimensions)
        data = np.array([dimension_scores[dim] for dim in dimensions]).T
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(data.T)
        
        logger.info(f"CFA on {n_models} models across {len(dimensions)} dimensions")
        
        # Simplified CFA estimation
        # In full implementation, would use SEM library like semopy or lavaan via rpy2
        
        # Estimate factor loadings (simplified)
        factor_loadings = self._estimate_factor_loadings(corr_matrix, dimensions)
        
        # Estimate model-implied covariance matrix
        sigma = self._calculate_model_implied_cov(factor_loadings, corr_matrix)
        
        # Calculate fit indices
        fit_indices = self._calculate_fit_indices(corr_matrix, sigma, n_models, len(dimensions))
        
        # Calculate factor correlations
        factor_correlations = self._estimate_factor_correlations(corr_matrix, dimensions)
        
        # Assess convergent validity (AVE)
        convergent_validity = self._calculate_ave(factor_loadings)
        
        # Assess discriminant validity (HTMT)
        discriminant_validity = self._calculate_htmt(
            corr_matrix, dimensions, factor_loadings
        )
        
        # Determine overall acceptability
        fit_acceptable = (
            fit_indices['rmsea'] < self.RMSEA_CUTOFF and
            fit_indices['srmr'] < self.SRMR_CUTOFF and
            fit_indices['cfi'] > self.CFI_CUTOFF and
            fit_indices['tli'] > self.TLI_CUTOFF
        )
        
        validity_acceptable = all(
            ave > self.AVE_CUTOFF for ave in convergent_validity.values()
        ) and all(
            htmt_info[1] for dim_htmt in discriminant_validity.values()
            for htmt_info in dim_htmt.values()
        )
        
        result = CFAResult(
            chi_square=fit_indices['chi_square'],
            df=fit_indices['df'],
            p_value=fit_indices['p_value'],
            rmsea=fit_indices['rmsea'],
            rmsea_ci_lower=fit_indices['rmsea_ci_lower'],
            rmsea_ci_upper=fit_indices['rmsea_ci_upper'],
            srmr=fit_indices['srmr'],
            cfi=fit_indices['cfi'],
            tli=fit_indices['tli'],
            factor_loadings=factor_loadings,
            factor_correlations=factor_correlations,
            convergent_validity=convergent_validity,
            discriminant_validity=discriminant_validity,
            fit_acceptable=fit_acceptable,
            validity_acceptable=validity_acceptable
        )
        
        self.results.append(result)
        
        return result
    
    def _estimate_factor_loadings(
        self,
        corr_matrix: np.ndarray,
        dimensions: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Estimate standardized factor loadings.
        
        Simplified estimation using correlation with dimension score
        as proxy for factor loading.
        """
        n_dims = len(dimensions)
        loadings = {}
        
        for i, dim in enumerate(dimensions):
            # Items loading on this factor
            # In simplified model, each dimension is its own factor
            dim_loadings = {}
            
            for j, other_dim in enumerate(dimensions):
                # Loading is correlation with factor
                loading = corr_matrix[i, j] if i != j else 1.0
                dim_loadings[other_dim] = loading
            
            loadings[dim] = dim_loadings
        
        return loadings
    
    def _calculate_model_implied_cov(
        self,
        factor_loadings: Dict[str, Dict[str, float]],
        sample_corr: np.ndarray
    ) -> np.ndarray:
        """
        Calculate model-implied covariance (correlation) matrix.
        
        Σ = Λ Φ Λ' + Θ
        where Λ = factor loadings, Φ = factor correlations, Θ = error variances
        """
        n_dims = len(factor_loadings)
        
        # Construct Lambda (loading matrix)
        lambda_matrix = np.zeros((n_dims, n_dims))
        for i, (dim, loads) in enumerate(factor_loadings.items()):
            for j, (item, loading) in enumerate(loads.items()):
                lambda_matrix[j, i] = loading
        
        # Factor correlation matrix (identity in orthogonal model)
        phi = np.eye(n_dims)
        
        # Error variances (1 - communality)
        theta = np.eye(n_dims)
        for i in range(n_dims):
            communality = sum(lambda_matrix[i, :] ** 2)
            theta[i, i] = max(0.1, 1 - communality)
        
        # Model-implied covariance
        sigma = lambda_matrix @ phi @ lambda_matrix.T + theta
        
        # Standardize to correlation matrix
        # Ensure diagonal is positive to avoid math domain error
        diag_sigma = np.diag(sigma)
        diag_sigma = np.maximum(diag_sigma, 1e-10)  # Floor at small positive value
        d = np.sqrt(diag_sigma)
        sigma_corr = sigma / np.outer(d, d)
        
        # Clip to valid correlation range
        sigma_corr = np.clip(sigma_corr, -1, 1)
        # Ensure diagonal is exactly 1
        np.fill_diagonal(sigma_corr, 1.0)
        
        return sigma_corr
    
    def _calculate_fit_indices(
        self,
        sample_corr: np.ndarray,
        model_corr: np.ndarray,
        n_obs: int,
        n_vars: int
    ) -> Dict[str, float]:
        """
        Calculate CFA model fit indices.
        """
        # Residuals
        residuals = sample_corr - model_corr
        
        # Chi-square (simplified - not accurate for small samples)
        # Would use proper likelihood calculation in full implementation
        chi_sq = np.sum(residuals ** 2) * n_obs
        df = int(n_vars * (n_vars - 1) / 2)  # Unique elements minus estimated parameters
        p_value = 1 - chi2.cdf(chi_sq, df) if df > 0 else 1.0
        
        # RMSEA
        if df > 0 and n_obs > 1:
            rmsea = math.sqrt(max(0, chi_sq - df) / (df * (n_obs - 1)))
        else:
            rmsea = 0.0
        
        # RMSEA 90% CI (simplified)
        rmsea_se = rmsea / math.sqrt(n_obs) if n_obs > 1 else 0.05
        rmsea_ci_lower = max(0, rmsea - 1.645 * rmsea_se)
        rmsea_ci_upper = rmsea + 1.645 * rmsea_se
        
        # SRMR
        srmr = math.sqrt(np.mean(residuals ** 2))
        
        # CFI (Comparative Fit Index)
        # Simplified calculation using null model chi-square
        chi_sq_null = n_obs * n_vars  # Rough approximation
        if chi_sq_null > chi_sq:
            cfi = 1 - (chi_sq - df) / max(1, chi_sq_null - df)
        else:
            cfi = 1.0
        cfi = max(0, min(1, cfi))
        
        # TLI (Tucker-Lewis Index)
        if chi_sq_null > 0 and df > 0:
            tli = ((chi_sq_null / (n_vars * (n_vars - 1) / 2)) - (chi_sq / df)) / \
                  ((chi_sq_null / (n_vars * (n_vars - 1) / 2)) - 1)
        else:
            tli = 1.0
        tli = max(0, min(1, tli))
        
        return {
            'chi_square': chi_sq,
            'df': df,
            'p_value': p_value,
            'rmsea': rmsea,
            'rmsea_ci_lower': rmsea_ci_lower,
            'rmsea_ci_upper': rmsea_ci_upper,
            'srmr': srmr,
            'cfi': cfi,
            'tli': tli,
        }
    
    def _estimate_factor_correlations(
        self,
        corr_matrix: np.ndarray,
        dimensions: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Extract factor correlations from correlation matrix."""
        correlations = {}
        
        for i, dim1 in enumerate(dimensions):
            correlations[dim1] = {}
            for j, dim2 in enumerate(dimensions):
                if i != j:
                    correlations[dim1][dim2] = corr_matrix[i, j]
        
        return correlations
    
    def _calculate_ave(
        self,
        factor_loadings: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Calculate Average Variance Extracted (AVE) for each dimension.
        
        AVE = (Σ λ²) / n
        where λ = standardized factor loading
        
        Fornell & Larcker (1981) criterion: AVE > 0.5 indicates convergent validity
        """
        ave_scores = {}
        
        for dim, loadings in factor_loadings.items():
            # Square loadings and average
            squared_loadings = [l ** 2 for l in loadings.values()]
            ave = sum(squared_loadings) / len(squared_loadings) if squared_loadings else 0
            ave_scores[dim] = ave
        
        return ave_scores
    
    def _calculate_htmt(
        self,
        corr_matrix: np.ndarray,
        dimensions: List[str],
        factor_loadings: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, Tuple[float, bool]]]:
        """
        Calculate Heterotrait-Monotrait (HTMT) ratio of correlations.
        
        HTMT = mean(heterotrait correlations) / mean(monotrait correlations)
        
        Henseler et al. (2015): HTMT < 0.85 indicates discriminant validity
        """
        htmt_matrix = {}
        n = len(dimensions)
        
        for i, dim1 in enumerate(dimensions):
            htmt_matrix[dim1] = {}
            
            for j, dim2 in enumerate(dimensions):
                if i >= j:
                    continue
                
                # Get monotrait correlations (correlations of items within same factor)
                mono_corr_1 = []
                mono_corr_2 = []
                
                loadings_1 = list(factor_loadings[dim1].values())
                loadings_2 = list(factor_loadings[dim2].values())
                
                # Calculate geometric mean of monotrait correlations
                for k in range(len(loadings_1)):
                    for l in range(k + 1, len(loadings_1)):
                        mono_corr_1.append(loadings_1[k] * loadings_1[l])
                
                for k in range(len(loadings_2)):
                    for l in range(k + 1, len(loadings_2)):
                        mono_corr_2.append(loadings_2[k] * loadings_2[l])
                
                # Calculate heterotrait correlations
                hetero_corr = []
                for k in range(len(loadings_1)):
                    for l in range(len(loadings_2)):
                        hetero_corr.append(
                            factor_loadings[dim1][dim2] * 
                            factor_loadings[dim2][dim1]
                        )
                
                # HTMT calculation
                if mono_corr_1 and mono_corr_2 and hetero_corr:
                    mean_1 = sum(mono_corr_1) / len(mono_corr_1)
                    mean_2 = sum(mono_corr_2) / len(mono_corr_2)
                    
                    # Ensure non-negative for sqrt
                    product = max(0.0, mean_1 * mean_2)
                    mono_mean = math.sqrt(product)
                    
                    hetero_mean = sum(hetero_corr) / len(hetero_corr)
                    htmt = abs(hetero_mean) / mono_mean if mono_mean > 0.001 else 0
                else:
                    htmt = abs(corr_matrix[i, j])  # Fallback to correlation
                
                is_valid = htmt < self.HTMT_CUTOFF
                htmt_matrix[dim1][dim2] = (round(htmt, 3), is_valid)
        
        return htmt_matrix
    
    def interpret_results(self, result: CFAResult) -> List[str]:
        """
        Generate human-readable interpretation of CFA results.
        
        Returns:
            List of interpretation statements
        """
        interpretations = []
        
        # Model fit interpretation
        if result.fit_acceptable:
            interpretations.append(
                "✓ Model fit is acceptable (meets Hu & Bentler, 1999 criteria)"
            )
        else:
            interpretations.append(
                "✗ Model fit is not acceptable - review factor structure"
            )
            
            if result.rmsea > self.RMSEA_CUTOFF:
                interpretations.append(
                    f"  - RMSEA ({result.rmsea:.3f}) exceeds cutoff ({self.RMSEA_CUTOFF})"
                )
            if result.cfi < self.CFI_CUTOFF:
                interpretations.append(
                    f"  - CFI ({result.cfi:.3f}) below cutoff ({self.CFI_CUTOFF})"
                )
        
        # Convergent validity
        low_ave = [
            dim for dim, ave in result.convergent_validity.items()
            if ave < self.AVE_CUTOFF
        ]
        
        if low_ave:
            interpretations.append(
                f"✗ Convergent validity issues in: {', '.join(low_ave)}"
            )
        else:
            interpretations.append(
                "✓ All dimensions show adequate convergent validity (AVE > 0.5)"
            )
        
        # Discriminant validity
        high_htmt = [
            (dim1, dim2, htmt[0])
            for dim1, corrs in result.discriminant_validity.items()
            for dim2, htmt in corrs.items()
            if not htmt[1]
        ]
        
        if high_htmt:
            interpretations.append("✗ Discriminant validity issues:")
            for dim1, dim2, htmt in high_htmt:
                interpretations.append(
                    f"  - {dim1} and {dim2} (HTMT = {htmt:.3f}) may be measuring similar constructs"
                )
        else:
            interpretations.append(
                "✓ All dimensions show adequate discriminant validity (HTMT < 0.85)"
            )
        
        return interpretations


def run_dimension_validation(
    historical_scores: Dict[str, Dict[str, List[float]]]
) -> CFAResult:
    """
    Convenience function to run dimension validation on historical data.
    
    Args:
        historical_scores: Dict mapping model_id -> dimension -> list of scores
        
    Returns:
        CFAResult with validation assessment
    """
    # Aggregate scores by dimension
    dimension_scores: Dict[str, List[float]] = {}
    
    for model_id, dims in historical_scores.items():
        for dim, scores in dims.items():
            if dim not in dimension_scores:
                dimension_scores[dim] = []
            dimension_scores[dim].extend(scores)
    
    # Ensure equal length (take minimum)
    min_len = min(len(scores) for scores in dimension_scores.values())
    dimension_scores = {
        dim: scores[:min_len] 
        for dim, scores in dimension_scores.items()
    }
    
    validator = DimensionValidator()
    result = validator.validate_dimensions(dimension_scores)
    
    return result


if __name__ == '__main__':
    # Example usage
    example_data = {
        'model_1': {'reasoning': [85, 82, 88], 'coding': [90, 87, 92], 'instruction': [78, 80, 75]},
        'model_2': {'reasoning': [70, 68, 72], 'coding': [75, 73, 77], 'instruction': [65, 67, 63]},
    }
    
    result = run_dimension_validation(example_data)
    print(json.dumps(result.to_dict(), indent=2))
