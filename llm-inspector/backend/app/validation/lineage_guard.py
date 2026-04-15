"""
lineage_guard.py — Data Lineage & Scientific Rigor Guard for LLM Inspector v12.

Enforces data chain integrity by validating that all reported metrics have 
proper provenance, sufficient sample sizes, and scientific basis.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from app.core.provenance import DataProvenance, get_provenance_tracker
from app.core.logging import get_logger

logger = get_logger(__name__)

class LineageGuard:
    """
    Enforces data lineage and scientific rigor policies during report generation.
    
    Policies:
    1. Every metric must have a provenance record.
    2. Fallback values are marked as 'invalid' for final conclusions.
    3. Minimum sample size (n) must be met for 'measured' status.
    4. Derived metrics must trace back to parent metrics.
    """

    MIN_SAMPLE_SIZE = 30
    MIN_CONFIDENCE = 0.5

    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self.tracker = get_provenance_tracker()

    def validate_report_data(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate report data and annotate with lineage status.
        
        Annotates each major section with:
        - data_status: measured | derived | insufficient_evidence | blocked
        - lineage_valid: bool
        """
        validation_results = {
            "is_valid": True,
            "issues": [],
            "annotations": {},
            "validation_timestamp": datetime.utcnow().isoformat()
        }

        # Validate core dimensions
        dimensions = ["reasoning", "coding", "instruction", "safety", "protocol"]
        for dim in dimensions:
            score_key = f"{dim}_score"
            if score_key in report:
                status = self._check_metric_lineage(score_key, report[score_key])
                validation_results["annotations"][score_key] = status
                if status["status"] == "blocked" and self.strict_mode:
                    validation_results["is_valid"] = False
                    validation_results["issues"].append(f"Metric {score_key} blocked: {status['reason']}")

        # Validate theta report if available (v12 specific)
        if "theta_report" in report:
            theta_validation = self._validate_theta_report(report["theta_report"])
            validation_results["annotations"]["theta_report"] = theta_validation
            if not theta_validation["lineage_valid"] and self.strict_mode:
                validation_results["is_valid"] = False
                validation_results["issues"].append("Theta report lineage validation failed")

        # Validate composite scores
        composite_scores = ["total_score", "capability_score", "authenticity_score", "performance_score"]
        for score in composite_scores:
            if score in report:
                status = self._check_composite_score_lineage(score, report[score], report)
                validation_results["annotations"][score] = status
                if status["status"] == "blocked" and self.strict_mode:
                    validation_results["is_valid"] = False
                    validation_results["issues"].append(f"Composite score {score} blocked: {status['reason']}")

        return validation_results

    def _check_metric_lineage(self, metric_id: str, value: Any) -> Dict[str, Any]:
        """Check lineage for a single metric."""
        provenance = self.tracker.get(metric_id)
        
        if not provenance:
            return {
                "status": "blocked",
                "reason": "Missing provenance record",
                "lineage_valid": False
            }

        if provenance.source_type == "fallback":
            return {
                "status": "insufficient_evidence",
                "reason": "Using fallback/default value",
                "lineage_valid": False
            }

        if provenance.sample_size < self.MIN_SAMPLE_SIZE:
            if provenance.source_type != "literature":
                return {
                    "status": "insufficient_evidence",
                    "reason": f"Insufficient sample size (n={provenance.sample_size} < {self.MIN_SAMPLE_SIZE})",
                    "lineage_valid": True # Still valid but weak
                }

        if provenance.confidence < self.MIN_CONFIDENCE:
            return {
                "status": "insufficient_evidence",
                "reason": f"Low confidence ({provenance.confidence} < {self.MIN_CONFIDENCE})",
                "lineage_valid": True
            }

        return {
            "status": "measured" if provenance.source_type != "derived" else "derived",
            "reason": None,
            "lineage_valid": True,
            "provenance": provenance.to_dict()
        }

    def _validate_theta_report(self, theta_report: Dict[str, Any]) -> Dict[str, Any]:
        """Validate theta report lineage (v12 specific)."""
        validation_result = {
            "lineage_valid": True,
            "status": "measured",
            "issues": []
        }

        # Check global theta provenance
        global_theta_key = f"theta_global_{theta_report.get('calibration_version', 'unknown')}"
        global_provenance = self.tracker.get(global_theta_key)
        
        if not global_provenance:
            validation_result["lineage_valid"] = False
            validation_result["status"] = "blocked"
            validation_result["issues"].append("Missing global theta provenance")
        else:
            # Validate calibration version
            if global_provenance.source_type != "irt_calibration":
                validation_result["lineage_valid"] = False
                validation_result["status"] = "insufficient_evidence"
                validation_result["issues"].append("Global theta not from IRT calibration")

        # Check dimension theta provenance
        dimensions = theta_report.get("dimensions", [])
        for dim in dimensions:
            dim_name = dim.get("dimension")
            dim_theta_key = f"theta_{dim_name}_{theta_report.get('calibration_version', 'unknown')}"
            dim_provenance = self.tracker.get(dim_theta_key)
            
            if not dim_provenance:
                validation_result["lineage_valid"] = False
                validation_result["status"] = "insufficient_evidence"
                validation_result["issues"].append(f"Missing dimension theta provenance for {dim_name}")

        return validation_result

    def _check_composite_score_lineage(self, score_id: str, value: Any, report: Dict[str, Any]) -> Dict[str, Any]:
        """Check lineage for composite scores."""
        # Composite scores are derived from component scores
        composite_provenance = self.tracker.get(score_id)
        
        if not composite_provenance:
            # Create derived provenance on-the-fly if components exist
            if score_id == "total_score":
                components = ["reasoning_score", "coding_score", "instruction_score", "safety_score", "protocol_score"]
            elif score_id == "capability_score":
                components = ["reasoning_score", "coding_score", "instruction_score"]
            elif score_id == "authenticity_score":
                components = ["safety_score", "protocol_score"]
            elif score_id == "performance_score":
                components = ["reasoning_score", "coding_score"]
            else:
                components = []

            # Check if all components have valid provenance
            component_valid = True
            for comp in components:
                comp_provenance = self.tracker.get(comp)
                if not comp_provenance or comp_provenance.source_type == "fallback":
                    component_valid = False
                    break

            if component_valid and components:
                # Create derived provenance
                from app.core.provenance import DataProvenance
                derived_provenance = DataProvenance(
                    source_type="derived",
                    source_id=f"derived_{score_id}",
                    collected_at=datetime.utcnow().isoformat(),
                    sample_size=sum(self.tracker.get(comp).sample_size for comp in components if self.tracker.get(comp)),
                    confidence=min(self.tracker.get(comp).confidence for comp in components if self.tracker.get(comp)),
                    verified=True,
                    verified_by="lineage_guard",
                    notes=f"Derived from components: {', '.join(components)}"
                )
                self.tracker.register(score_id, derived_provenance)
                composite_provenance = derived_provenance

        if not composite_provenance:
            return {
                "status": "blocked",
                "reason": "Missing composite score provenance and insufficient component data",
                "lineage_valid": False
            }

        if composite_provenance.source_type == "fallback":
            return {
                "status": "insufficient_evidence",
                "reason": "Using fallback value for composite score",
                "lineage_valid": False
            }

        return {
            "status": "derived",
            "reason": None,
            "lineage_valid": True,
            "provenance": composite_provenance.to_dict()
        }

    def verify_config_integrity(self, config_data: Dict[str, Any], manifest: Dict[str, Any]) -> bool:
        """
        Verify that configuration constants match the manifest (checksums/versions).
        Part of Task 4 in Phase 1.
        """
        # Implementation for v12 Phase 2
        try:
            # Check version consistency
            config_version = config_data.get("generation", {}).get("version")
            manifest_version = manifest.get("version")
            
            if config_version != manifest_version:
                logger.warning(f"Version mismatch: config={config_version}, manifest={manifest_version}")
                return False
            
            # Validate checksums if available
            config_checksum = config_data.get("generation", {}).get("source_data_hash")
            manifest_checksum = manifest.get("checksum")
            
            if config_checksum and manifest_checksum:
                if config_checksum != manifest_checksum:
                    logger.warning(f"Checksum mismatch: config={config_checksum}, manifest={manifest_checksum}")
                    return False
            
            # Verify calibration data integrity
            calibration_info = config_data.get("generation", {})
            if not calibration_info.get("tool") or not calibration_info.get("command"):
                logger.warning("Missing calibration tool information")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Config integrity verification failed: {e}")
            return False
