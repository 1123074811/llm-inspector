"""Data provenance tracking system for LLM Inspector v8.0.

All scoring-related data must carry provenance information to ensure
data chain integrity and scientific rigor.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any
import hashlib
import json


@dataclass(frozen=True)
class DataProvenance:
    """Data provenance tracking - immutable object.
    
    All scoring-related data must carry provenance information.
    """
    
    # Data source type
    source_type: str  # "irt_calibration", "literature", "experiment", "fallback", "derived"
    
    # Specific source identifier
    source_id: str  # DOI/experiment ID/calibration version/algorithm name
    
    # Collection timestamp
    collected_at: str  # ISO 8601 format
    
    # Sample size
    sample_size: int
    
    # Confidence level (0-1)
    confidence: float
    
    # Verification status
    verified: bool
    
    # Verifier/system
    verified_by: Optional[str] = None
    
    # Parent provenance (for derived data)
    parent_provenance: Optional['DataProvenance'] = None
    
    # Notes
    notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "source_type": self.source_type,
            "source_id": self.source_id,
            "collected_at": self.collected_at,
            "sample_size": self.sample_size,
            "confidence": self.confidence,
            "verified": self.verified,
            "verified_by": self.verified_by,
            "notes": self.notes,
        }
        if self.parent_provenance:
            result["parent_provenance"] = self.parent_provenance.to_dict()
        return result
    
    def compute_hash(self) -> str:
        """Compute data fingerprint."""
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    @staticmethod
    def create_fallback(value_name: str, reason: str = "No data available") -> 'DataProvenance':
        """Create fallback value provenance."""
        return DataProvenance(
            source_type="fallback",
            source_id=f"fallback_{value_name}",
            collected_at=datetime.utcnow().isoformat(),
            sample_size=0,
            confidence=0.0,
            verified=False,
            verified_by="system",
            notes=f"Fallback value: {reason}"
        )
    
    @staticmethod
    def from_irt_calibration(
        case_id: str, 
        calibration_version: str,
        sample_size: int,
        confidence: float
    ) -> 'DataProvenance':
        """Create provenance from IRT calibration."""
        return DataProvenance(
            source_type="irt_calibration",
            source_id=f"irt_{calibration_version}_{case_id}",
            collected_at=datetime.utcnow().isoformat(),
            sample_size=sample_size,
            confidence=confidence,
            verified=True,
            verified_by="irt_calibration_system",
            notes=f"IRT 2PL calibration for {case_id}"
        )
    
    @staticmethod
    def from_literature(
        title: str,
        authors: str,
        year: int,
        doi: Optional[str] = None,
        confidence: float = 0.9
    ) -> 'DataProvenance':
        """Create provenance from academic literature."""
        source_id = doi if doi else f"{authors}_{year}"
        return DataProvenance(
            source_type="literature",
            source_id=source_id,
            collected_at=datetime.utcnow().isoformat(),
            sample_size=0,  # Literature-based, no direct sampling
            confidence=confidence,
            verified=True,
            verified_by="reference_database",
            notes=f"Source: {title} ({year})"
        )


class ProvenanceTracker:
    """Data provenance tracker.
    
    Records the source and change history of all data.
    """
    
    def __init__(self):
        self._provenance_log: Dict[str, DataProvenance] = {}
    
    def register(self, data_id: str, provenance: DataProvenance) -> None:
        """Register data provenance."""
        self._provenance_log[data_id] = provenance
    
    def get(self, data_id: str) -> Optional[DataProvenance]:
        """Get data provenance."""
        return self._provenance_log.get(data_id)
    
    def validate_chain(self, data_id: str) -> Dict[str, Any]:
        """Validate data provenance chain."""
        provenance = self.get(data_id)
        if not provenance:
            return {"valid": False, "error": "Provenance not found"}
        
        issues = []
        
        # Check confidence level
        if provenance.confidence < 0.5:
            issues.append("Low confidence (< 0.5)")
        
        # Check sample size
        if provenance.sample_size < 30 and provenance.source_type != "fallback":
            if provenance.source_type != "literature":  # Literature doesn't require sample size
                issues.append("Small sample size (< 30)")
        
        # Check verification status
        if not provenance.verified:
            issues.append("Not verified")
        
        # Validate parent chain
        if provenance.parent_provenance:
            parent_validation = self._validate_parent(provenance.parent_provenance)
            if not parent_validation["valid"]:
                issues.extend([f"Parent: {i}" for i in parent_validation["issues"]])
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "provenance": provenance.to_dict()
        }
    
    def _validate_parent(self, parent: DataProvenance) -> Dict[str, Any]:
        """Validate parent provenance."""
        issues = []
        
        if parent.confidence < 0.5:
            issues.append("Low confidence")
        
        if not parent.verified:
            issues.append("Not verified")
        
        return {"valid": len(issues) == 0, "issues": issues}
    
    def get_all_with_source_type(self, source_type: str) -> Dict[str, DataProvenance]:
        """Get all provenance entries with a specific source type."""
        return {
            k: v for k, v in self._provenance_log.items() 
            if v.source_type == source_type
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get provenance statistics."""
        if not self._provenance_log:
            return {"total": 0}
        
        source_types = {}
        confidence_sum = 0
        verified_count = 0
        fallback_count = 0
        
        for p in self._provenance_log.values():
            source_types[p.source_type] = source_types.get(p.source_type, 0) + 1
            confidence_sum += p.confidence
            if p.verified:
                verified_count += 1
            if p.source_type == "fallback":
                fallback_count += 1
        
        return {
            "total": len(self._provenance_log),
            "source_types": source_types,
            "average_confidence": confidence_sum / len(self._provenance_log),
            "verified_count": verified_count,
            "fallback_count": fallback_count,
            "data_quality_score": (verified_count / len(self._provenance_log)) * 100
        }


# Global tracker instance
_provenance_tracker = ProvenanceTracker()


def get_provenance_tracker() -> ProvenanceTracker:
    """Get global provenance tracker."""
    return _provenance_tracker


def reset_provenance_tracker() -> None:
    """Reset global provenance tracker (mainly for testing)."""
    global _provenance_tracker
    _provenance_tracker = ProvenanceTracker()
