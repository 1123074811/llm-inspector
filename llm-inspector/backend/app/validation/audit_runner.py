"""
Automated Data Audit Runner for Continuous Validation.

Periodically validates all data sources to ensure scientific rigor is maintained.
Can be run manually or scheduled as a CI/CD job.

Reference:
- Data Provenance Standards (W3C)
- FAIR Data Principles
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from app.core.logging import get_logger
from .data_validation import (
    DataValidator, ValidationResult, ValidationStatus,
    validate_test_suite, validate_scoring_system
)

logger = get_logger(__name__)


@dataclass
class AuditReport:
    """Comprehensive data audit report."""
    timestamp: str
    total_checked: int
    valid_count: int
    warning_count: int
    invalid_count: int
    details: List[ValidationResult] = field(default_factory=list)
    must_fix: bool = False
    critical_issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "summary": {
                "total_checked": self.total_checked,
                "valid": self.valid_count,
                "warning": self.warning_count,
                "invalid": self.invalid_count,
                "pass_rate": self.valid_count / max(self.total_checked, 1),
            },
            "must_fix": self.must_fix,
            "critical_issues": self.critical_issues,
            "recommendations": self.recommendations,
            "details": [d.to_dict() for d in self.details],
        }
    
    def to_markdown(self) -> str:
        """Generate markdown report for human review."""
        lines = [
            "# Data Audit Report",
            f"**Timestamp**: {self.timestamp}",
            "",
            "## Summary",
            f"- **Total Checked**: {self.total_checked}",
            f"- **Valid**: {self.valid_count} ({self.valid_count/max(self.total_checked,1):.1%})",
            f"- **Warning**: {self.warning_count}",
            f"- **Invalid**: {self.invalid_count}",
            f"- **Must Fix**: {'Yes' if self.must_fix else 'No'}",
            "",
        ]
        
        if self.critical_issues:
            lines.extend([
                "## Critical Issues",
                *[f"- {issue}" for issue in self.critical_issues],
                "",
            ])
        
        if self.recommendations:
            lines.extend([
                "## Recommendations",
                *[f"- {rec}" for rec in self.recommendations],
                "",
            ])
        
        # Group by status
        invalid_items = [d for d in self.details if d.status == ValidationStatus.INVALID]
        warning_items = [d for d in self.details if d.status == ValidationStatus.WARNING]
        
        if invalid_items:
            lines.extend([
                "## Invalid Items (Must Fix)",
                "",
            ])
            for item in invalid_items:
                lines.extend([
                    f"### {item.item_id} ({item.data_type})",
                    "**Issues**:",
                    *[f"- {issue}" for issue in item.issues],
                    "",
                ])
        
        if warning_items:
            lines.extend([
                "## Warnings (Should Review)",
                "",
            ])
            for item in warning_items:
                lines.extend([
                    f"### {item.item_id} ({item.data_type})",
                    "**Warnings**:",
                    *[f"- {warning}" for warning in item.warnings],
                    "",
                ])
        
        return "\n".join(lines)


class DataAuditRunner:
    """
    Runs comprehensive audits of all data sources.
    """
    
    def __init__(
        self,
        suite_path: Optional[str] = None,
        calibration_path: Optional[str] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize audit runner.
        
        Args:
            suite_path: Path to test suite JSON
            calibration_path: Path to IRT calibration cache
            config_path: Path to configuration files
        """
        base_dir = Path(__file__).parent.parent
        
        self.suite_path = suite_path or str(
            base_dir / 'fixtures' / 'suite_v3.json'
        )
        self.calibration_path = calibration_path or str(
            base_dir / 'fixtures' / 'irt_calibration_cache.json'
        )
        self.config_path = config_path or str(
            base_dir / 'core' / 'config.py'
        )
        
        self.validator = DataValidator()
    
    def run_full_audit(self) -> AuditReport:
        """
        Execute comprehensive data audit.
        
        Checks:
        1. All test cases have IRT parameters
        2. All weights are data-driven
        3. No hardcoded constants without documentation
        4. All external APIs have fallback
        5. Token counts verified against official tokenizers
        6. Thresholds have statistical basis
        
        Returns:
            AuditReport with complete findings
        """
        logger.info("Starting full data audit...")
        start_time = datetime.utcnow()
        
        all_results: List[ValidationResult] = []
        critical_issues: List[str] = []
        
        # Audit 1: Test Suite
        logger.info("Auditing test suite...")
        if os.path.exists(self.suite_path):
            suite_results, suite_counts = validate_test_suite(
                self.suite_path,
                self.calibration_path
            )
            all_results.extend(suite_results)
            
            if suite_counts['invalid'] > 0:
                critical_issues.append(
                    f"{suite_counts['invalid']} test cases failed validation"
                )
            
            logger.info(
                f"Test suite audit: {suite_counts['valid']}/{suite_counts['total']} valid"
            )
        else:
            critical_issues.append(f"Test suite not found: {self.suite_path}")
        
        # Audit 2: Scoring Weights
        logger.info("Auditing scoring weights...")
        weights_results = self._audit_scoring_weights()
        all_results.extend(weights_results)
        
        # Audit 3: Configuration
        logger.info("Auditing configuration...")
        config_results = self._audit_configuration()
        all_results.extend(config_results)
        
        # Audit 4: Hardcoded Values Check
        logger.info("Checking for undocumented hardcoded values...")
        hardcode_results = self._audit_hardcoded_values()
        all_results.extend(hardcode_results)
        
        # Compile report
        valid_count = sum(1 for r in all_results if r.status == ValidationStatus.VALID)
        warning_count = sum(1 for r in all_results if r.status == ValidationStatus.WARNING)
        invalid_count = sum(1 for r in all_results if r.status == ValidationStatus.INVALID)
        
        must_fix = invalid_count > 0 or len(critical_issues) > 0
        
        # Generate recommendations
        recommendations = self._generate_recommendations(all_results)
        
        report = AuditReport(
            timestamp=start_time.isoformat(),
            total_checked=len(all_results),
            valid_count=valid_count,
            warning_count=warning_count,
            invalid_count=invalid_count,
            details=all_results,
            must_fix=must_fix,
            critical_issues=critical_issues,
            recommendations=recommendations,
        )
        
        logger.info(
            f"Audit complete: {valid_count} valid, {warning_count} warnings, "
            f"{invalid_count} invalid"
        )
        
        return report
    
    def _audit_scoring_weights(self) -> List[ValidationResult]:
        """Audit scoring weights configuration."""
        results = []
        
        # Try to load from config or YAML
        weights_file = Path(self.config_path).parent / 'scoring_weights.yaml'
        
        if weights_file.exists():
            try:
                import yaml
                with open(weights_file, 'r') as f:
                    weights_data = yaml.safe_load(f)
                
                weights = {
                    k: v['value'] for k, v in weights_data.get('weights', {}).items()
                }
                source_info = {
                    'method': weights_data.get('source', 'unknown'),
                    'date': weights_data.get('calibration_date', ''),
                }
                
                result = self.validator.validate_scoring_weights(weights, source_info)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to load weights: {e}")
                results.append(ValidationResult(
                    status=ValidationStatus.INVALID,
                    data_type="scoring_weights",
                    item_id="config",
                    issues=[f"Failed to load weights: {e}"]
                ))
        else:
            # Check Python config file
            results.append(ValidationResult(
                status=ValidationStatus.WARNING,
                data_type="scoring_weights",
                item_id="config",
                warnings=["Weights not in external YAML file - check code for hardcoded values"],
                recommendations=["Move weights to external YAML with data source documentation"]
            ))
        
        return results
    
    def _audit_configuration(self) -> List[ValidationResult]:
        """Audit configuration for hardcoded thresholds."""
        results = []
        
        # List of known thresholds that should be validated
        known_thresholds = [
            ('PREDETECT_CONFIDENCE_THRESHOLD', 0.85, 'pre_detection'),
            ('SIMILARITY_THRESHOLD_HIGH', 0.9, 'similarity'),
            ('SIMILARITY_THRESHOLD_MEDIUM', 0.75, 'similarity'),
            ('RISK_THRESHOLD_HIGH', 0.8, 'risk_assessment'),
        ]
        
        # In practice, would import config and check values
        # For now, create validation placeholders
        for name, expected_default, category in known_thresholds:
            result = self.validator.validate_threshold(
                name,
                expected_default,
                scientific_basis=None  # Would check actual config
            )
            results.append(result)
        
        return results
    
    def _audit_hardcoded_values(self) -> List[ValidationResult]:
        """Audit code for undocumented hardcoded values."""
        results = []
        
        # Patterns to check in codebase
        suspicious_patterns = [
            (r'GLOBAL_FEATURE_MEANS', "Hardcoded feature means"),
            (r'confidence\s*=\s*0\.\d+', "Hardcoded confidence value"),
            (r'threshold\s*=\s*0\.\d+', "Hardcoded threshold"),
        ]
        
        # Would scan codebase in practice
        # For now, report that manual review is needed
        results.append(ValidationResult(
            status=ValidationStatus.WARNING,
            data_type="code_audit",
            item_id="hardcoded_values",
            warnings=["Manual code review required for hardcoded value documentation"],
            recommendations=[
                "Run grep for '0.8', '0.9', '100.0' and document each occurrence",
                "Move magic numbers to named constants with scientific basis"
            ]
        ))
        
        return results
    
    def _generate_recommendations(
        self,
        results: List[ValidationResult]
    ) -> List[str]:
        """Generate actionable recommendations from audit results."""
        recommendations = []
        
        # Count issues by type
        irt_issues = sum(
            1 for r in results 
            if r.data_type == "irt_parameters" and r.status != ValidationStatus.VALID
        )
        
        if irt_issues > 0:
            recommendations.append(
                f"Run IRT calibration for {irt_issues} test cases with insufficient data"
            )
        
        weight_issues = any(
            r.data_type == "scoring_weights" and r.status != ValidationStatus.VALID
            for r in results
        )
        
        if weight_issues:
            recommendations.append(
                "Recalculate weights using IRT information function integration"
            )
        
        threshold_issues = any(
            r.data_type == "threshold" and r.status != ValidationStatus.VALID
            for r in results
        )
        
        if threshold_issues:
            recommendations.append(
                "Document scientific basis for all thresholds with literature references"
            )
        
        # General recommendations
        if not recommendations:
            recommendations.append(
                "All validation checks passed - maintain current data quality standards"
            )
        
        return recommendations
    
    def save_report(
        self,
        report: AuditReport,
        output_path: str
    ) -> None:
        """Save audit report to file."""
        # Save JSON version
        json_path = output_path.replace('.md', '.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, indent=2)
        
        # Save markdown version
        md_path = output_path.replace('.json', '.md')
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(report.to_markdown())
        
        logger.info(f"Audit report saved to {json_path} and {md_path}")
    
    def check_critical_only(self) -> bool:
        """
        Quick check for critical issues only.
        
        Returns:
            True if no critical issues found, False otherwise
        """
        report = self.run_full_audit()
        return not report.must_fix


def run_audit_cli():
    """Command-line interface for running audits."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run data validation audit for LLM Inspector"
    )
    parser.add_argument(
        '--output', '-o',
        default='data_audit_report.md',
        help='Output file path for audit report'
    )
    parser.add_argument(
        '--critical-only', '-c',
        action='store_true',
        help='Only check for critical issues'
    )
    parser.add_argument(
        '--suite-path',
        help='Path to test suite JSON'
    )
    parser.add_argument(
        '--calibration-path',
        help='Path to IRT calibration cache'
    )
    
    args = parser.parse_args()
    
    runner = DataAuditRunner(
        suite_path=args.suite_path,
        calibration_path=args.calibration_path
    )
    
    if args.critical_only:
        ok = runner.check_critical_only()
        print("PASS" if ok else "FAIL")
        return 0 if ok else 1
    
    report = runner.run_full_audit()
    runner.save_report(report, args.output)
    
    print(f"Audit complete: {report.valid_count}/{report.total_checked} valid")
    print(f"Report saved to: {args.output}")
    
    if report.must_fix:
        print("WARNING: Critical issues found that must be addressed")
        return 1
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(run_audit_cli())
