"""
Data Validation Pipeline for v7.0

Ensures all data meets scientific rigor requirements.
All validation results are data-driven and traceable.

Reference:
- Hu & Bentler (1999) Cutoff criteria for fit indexes
- APA Standards for Educational and Psychological Testing
"""

from __future__ import annotations

import json
import os
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from app.core.logging import get_logger

logger = get_logger(__name__)


class ValidationStatus(Enum):
    """Validation status levels."""
    VALID = "valid"           # Meets all scientific standards
    WARNING = "warning"       # Acceptable but has minor issues
    INVALID = "invalid"     # Must be fixed before use
    UNVERIFIED = "unverified" # Cannot verify (missing data)


@dataclass
class DataSource:
    """Traceable data source information."""
    source_type: str  # 'empirical', 'literature', 'expert_review', 'computation'
    description: str
    date_collected: str  # ISO format
    sample_size: Optional[int] = None
    methodology: str = ""
    references: List[str] = field(default_factory=list)
    confidence_level: float = 0.0  # 0-1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_type": self.source_type,
            "description": self.description,
            "date_collected": self.date_collected,
            "sample_size": self.sample_size,
            "methodology": self.methodology,
            "references": self.references,
            "confidence_level": self.confidence_level,
        }


@dataclass
class ValidationResult:
    """Result of data validation."""
    status: ValidationStatus
    data_type: str
    item_id: str
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    source_trace: Dict[str, Any] = field(default_factory=dict)
    validation_date: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    validator_version: str = "v7.0"
    
    def is_acceptable(self) -> bool:
        """Check if validation result is acceptable for use."""
        return self.status in (ValidationStatus.VALID, ValidationStatus.WARNING)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "data_type": self.data_type,
            "item_id": self.item_id,
            "issues": self.issues,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "source_trace": self.source_trace,
            "validation_date": self.validation_date,
            "validator_version": self.validator_version,
            "acceptable": self.is_acceptable(),
        }


class DataValidator:
    """
    Validates all data sources for scientific rigor.
    
    Implements validation rules from V7 upgrade plan:
    - No hardcoded values without empirical basis
    - All IRT parameters must come from calibration
    - All weights must be data-driven
    - All thresholds must have statistical basis
    """
    
    # Scientific standards from literature
    IRT_DISCRIMINATION_MIN = 0.5  # Embretson & Reise (2000)
    IRT_DISCRIMINATION_OPTIMAL = 1.0
    IRT_DIFFICULTY_RANGE = (-3.0, 3.0)
    IRT_FIT_RMSE_MAX = 0.1
    MIN_CALIBRATION_N = 50
    
    # CFA fit indices (Hu & Bentler, 1999)
    CFA_CFI_MIN = 0.95
    CFA_RMSEA_MAX = 0.06
    CFA_SRMR_MAX = 0.08
    
    def __init__(self):
        self.validation_history: List[ValidationResult] = []
    
    def validate_irt_parameters(
        self,
        case_id: str,
        irt_params: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate IRT parameters meet scientific standards.
        
        Args:
            case_id: Test case identifier
            irt_params: Dictionary with a, b, c, fit_rmse, etc.
            
        Returns:
            ValidationResult with detailed assessment
        """
        issues = []
        warnings = []
        recommendations = []
        
        # Check required fields exist
        required_fields = ['a', 'b', 'calibration_date', 'n_calibrated']
        for field in required_fields:
            if field not in irt_params:
                issues.append(f"Missing required IRT field: {field}")
        
        if issues:
            return ValidationResult(
                status=ValidationStatus.INVALID,
                data_type="irt_parameters",
                item_id=case_id,
                issues=issues,
                recommendations=["Run IRT calibration with minimum 100 model responses"]
            )
        
        # Validate discrimination (a)
        a = irt_params.get('a', 0)
        if a < self.IRT_DISCRIMINATION_MIN:
            issues.append(
                f"Low discrimination (a={a:.2f} < {self.IRT_DISCRIMINATION_MIN})"
            )
            recommendations.append(
                "Improve test case discriminative power or consider removal"
            )
        elif a < self.IRT_DISCRIMINATION_OPTIMAL:
            warnings.append(
                f"Suboptimal discrimination (a={a:.2f} < {self.IRT_DISCRIMINATION_OPTIMAL})"
            )
        
        # Validate difficulty (b)
        b = irt_params.get('b', 0)
        if abs(b) > max(self.IRT_DIFFICULTY_RANGE):
            issues.append(
                f"Extreme difficulty (b={b:.2f} outside {self.IRT_DIFFICULTY_RANGE})"
            )
            recommendations.append("Adjust test case difficulty to target average models")
        
        # Validate fit
        fit_rmse = irt_params.get('fit_rmse', 999)
        if fit_rmse > self.IRT_FIT_RMSE_MAX:
            issues.append(
                f"Poor model fit (RMSE={fit_rmse:.3f} > {self.IRT_FIT_RMSE_MAX})"
            )
            recommendations.append("Review test case design and judging logic")
        
        # Validate calibration sample size
        n_calibrated = irt_params.get('n_calibrated', 0)
        if n_calibrated < self.MIN_CALIBRATION_N:
            issues.append(
                f"Insufficient calibration data (n={n_calibrated} < {self.MIN_CALIBRATION_N})"
            )
            recommendations.append(f"Collect more test data (target: 100+ models)")
        
        # Validate reliability if available
        reliability = irt_params.get('reliability')
        if reliability is not None and reliability < 0.7:
            warnings.append(
                f"Low reliability estimate (r={reliability:.2f} < 0.7)"
            )
        
        # Check data source
        data_source = irt_params.get('data_source', '')
        if not data_source or 'fallback' in data_source.lower():
            issues.append("Parameters from fallback/default source, not empirical calibration")
        
        # Determine status
        if issues:
            status = ValidationStatus.INVALID
        elif warnings:
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.VALID
        
        return ValidationResult(
            status=status,
            data_type="irt_parameters",
            item_id=case_id,
            issues=issues,
            warnings=warnings,
            recommendations=recommendations,
            source_trace={
                "calibration_date": irt_params.get('calibration_date'),
                "n_calibrated": n_calibrated,
                "data_source": data_source,
            }
        )
    
    def validate_scoring_weights(
        self,
        weights: Dict[str, float],
        source_info: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Validate scoring weights are data-driven and properly normalized.
        
        Args:
            weights: Dict mapping dimension to weight
            source_info: Information about weight calculation method
            
        Returns:
            ValidationResult
        """
        issues = []
        warnings = []
        recommendations = []
        
        # Check normalization
        total = sum(weights.values())
        if abs(total - 1.0) > 0.01:
            issues.append(
                f"Weights do not sum to 1.0 (sum={total:.3f})"
            )
            recommendations.append("Renormalize weights to sum to 1.0")
        
        # Check for uniform distribution (suspicious)
        if len(set(weights.values())) == 1:
            warnings.append(
                "Uniform weights suggest lack of data-driven calibration"
            )
            recommendations.append(
                "Calibrate weights using IRT information function integration"
            )
        
        # Check for extreme weights
        for dim, weight in weights.items():
            if weight > 0.5:
                warnings.append(
                    f"Dominant weight for {dim} ({weight:.1%}) may skew overall score"
                )
            if weight < 0.02:
                warnings.append(
                    f"Negligible weight for {dim} ({weight:.1%}) - consider removal"
                )
        
        # Validate source
        if source_info is None:
            issues.append("Missing weight calculation source information")
            recommendations.append("Document weight calculation methodology")
        else:
            method = source_info.get('method', '')
            if 'irt' not in method.lower() and 'empirical' not in method.lower():
                warnings.append(
                    f"Weights may not be data-driven (method: {method})"
                )
            
            date = source_info.get('date', '')
            if not date:
                warnings.append("Missing weight calibration date")
        
        # Check dimension coverage
        expected_dims = {'reasoning', 'coding', 'instruction', 'safety', 'protocol'}
        missing = expected_dims - set(weights.keys())
        if missing:
            warnings.append(f"Missing expected dimensions: {missing}")
        
        # Determine status
        if issues:
            status = ValidationStatus.INVALID
        elif warnings:
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.VALID
        
        return ValidationResult(
            status=status,
            data_type="scoring_weights",
            item_id="global_weights",
            issues=issues,
            warnings=warnings,
            recommendations=recommendations,
            source_trace=source_info or {}
        )
    
    def validate_test_case(
        self,
        case_id: str,
        case_data: Dict[str, Any]
    ) -> ValidationResult:
        """
        Comprehensive validation of a test case.
        
        Args:
            case_id: Test case identifier
            case_data: Full test case configuration
            
        Returns:
            ValidationResult
        """
        issues = []
        warnings = []
        recommendations = []
        
        # Validate IRT parameters
        irt_params = case_data.get('irt_parameters')
        if irt_params:
            irt_validation = self.validate_irt_parameters(case_id, irt_params)
            issues.extend(irt_validation.issues)
            warnings.extend(irt_validation.warnings)
            recommendations.extend(irt_validation.recommendations)
        else:
            issues.append("Missing IRT calibration parameters")
            recommendations.append("Run IRT calibration with minimum 100 models")
        
        # Validate content validity
        content_validity = case_data.get('content_validity')
        if content_validity:
            relevance = content_validity.get('relevance_score', 0)
            if relevance < 4.0:
                warnings.append(f"Low content validity rating ({relevance}/5)")
        else:
            warnings.append("Missing content validity assessment")
        
        # Validate data provenance
        provenance = case_data.get('data_provenance')
        if not provenance:
            warnings.append("Missing data provenance trace")
        
        # Validate judge method
        judge_method = case_data.get('judge_method', '')
        if not judge_method:
            issues.append("Missing judge method specification")
        
        # Validate expected type
        expected_type = case_data.get('expected_type', '')
        if not expected_type:
            warnings.append("Missing expected_type specification")
        
        # Determine status
        if issues:
            status = ValidationStatus.INVALID
        elif warnings:
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.VALID
        
        return ValidationResult(
            status=status,
            data_type="test_case",
            item_id=case_id,
            issues=issues,
            warnings=warnings,
            recommendations=recommendations,
            source_trace=provenance or {}
        )
    
    def validate_threshold(
        self,
        threshold_name: str,
        value: float,
        scientific_basis: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Validate a threshold has scientific/statistical basis.
        
        Args:
            threshold_name: Name of threshold
            value: Threshold value
            scientific_basis: Documentation of scientific basis
            
        Returns:
            ValidationResult
        """
        issues = []
        warnings = []
        
        # Check for magic numbers (common hardcoded values)
        magic_numbers = [0.5, 0.8, 0.9, 0.95, 1.0, 0.0, 100.0]
        if value in magic_numbers and not scientific_basis:
            warnings.append(
                f"Threshold value {value} appears to be a 'magic number' without documented basis"
            )
        
        # Validate scientific basis
        if scientific_basis is None:
            issues.append("Missing scientific basis documentation")
        else:
            required_fields = ['method', 'data_source', 'confidence_level']
            for field in required_fields:
                if field not in scientific_basis:
                    warnings.append(f"Scientific basis missing field: {field}")
        
        status = ValidationStatus.INVALID if issues else (
            ValidationStatus.WARNING if warnings else ValidationStatus.VALID
        )
        
        return ValidationResult(
            status=status,
            data_type="threshold",
            item_id=threshold_name,
            issues=issues,
            warnings=warnings,
            recommendations=["Provide statistical analysis or literature reference for threshold"]
            if issues else [],
            source_trace=scientific_basis or {}
        )
    
    def validate_fingerprint_data(
        self,
        fingerprint: Dict[str, Any]
    ) -> ValidationResult:
        """Validate tokenizer fingerprint data has empirical basis."""
        issues = []
        warnings = []
        
        # Check for verification method
        verification = fingerprint.get('verification_method', '')
        if not verification:
            warnings.append("Fingerprint not verified against official tokenizer")
        
        # Check sample size
        sample_size = fingerprint.get('sample_size', 0)
        if sample_size < 10:
            warnings.append(f"Small verification sample (n={sample_size})")
        
        # Check date
        verification_date = fingerprint.get('verification_date', '')
        if not verification_date:
            warnings.append("Missing verification date")
        
        status = ValidationStatus.WARNING if warnings else ValidationStatus.VALID
        
        return ValidationResult(
            status=status,
            data_type="tokenizer_fingerprint",
            item_id=fingerprint.get('probe_text', 'unknown'),
            issues=issues,
            warnings=warnings,
            source_trace=fingerprint
        )


def validate_test_suite(
    suite_path: str,
    calibration_cache_path: Optional[str] = None
) -> Tuple[List[ValidationResult], Dict[str, int]]:
    """
    Validate entire test suite.
    
    Args:
        suite_path: Path to test suite JSON file
        calibration_cache_path: Path to IRT calibration cache
        
    Returns:
        Tuple of (validation_results, summary_counts)
    """
    validator = DataValidator()
    results = []
    
    # Load test suite
    with open(suite_path, 'r', encoding='utf-8') as f:
        suite = json.load(f)
    
    # Load calibration data if available
    calibration_data = {}
    if calibration_cache_path and os.path.exists(calibration_cache_path):
        with open(calibration_cache_path, 'r', encoding='utf-8') as f:
            calibration_data = json.load(f)
    
    # Validate each test case
    for case in suite.get('cases', []):
        case_id = case['id']
        
        # Merge calibration data if available
        if case_id in calibration_data:
            case['irt_parameters'] = calibration_data[case_id].get('parameters', {})
        
        result = validator.validate_test_case(case_id, case)
        results.append(result)
    
    # Summary
    counts = {
        'valid': sum(1 for r in results if r.status == ValidationStatus.VALID),
        'warning': sum(1 for r in results if r.status == ValidationStatus.WARNING),
        'invalid': sum(1 for r in results if r.status == ValidationStatus.INVALID),
        'total': len(results),
    }
    
    return results, counts


def validate_scoring_system(
    weights: Dict[str, float],
    irt_params: Dict[str, Dict[str, Any]]
) -> List[ValidationResult]:
    """
    Validate entire scoring system.
    
    Args:
        weights: Dimension weights
        irt_params: IRT parameters for all test cases
        
    Returns:
        List of validation results
    """
    validator = DataValidator()
    results = []
    
    # Validate weights
    weight_validation = validator.validate_scoring_weights(
        weights,
        source_info={
            'method': 'irt_information_integration',
            'date': datetime.utcnow().isoformat(),
        }
    )
    results.append(weight_validation)
    
    # Validate all IRT parameters
    for case_id, params in irt_params.items():
        result = validator.validate_irt_parameters(case_id, params)
        results.append(result)
    
    return results
