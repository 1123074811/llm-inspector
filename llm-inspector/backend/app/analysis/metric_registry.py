"""
Metric/threshold provenance registry for v9 Phase A.

Purpose:
- Eliminate undocumented magic numbers in scoring/verdict pipeline
- Provide auditable source chain for each threshold/constant
- Support strict runtime validation in production or CI
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from app.core.config import settings


@dataclass(frozen=True)
class MetricSource:
    """Source metadata for a metric/threshold constant."""

    metric_name: str
    value: float
    source_type: str  # paper | official_doc | benchmark_run | policy
    source_ref: str
    rationale: str


# Phase A: start with high-impact thresholds used in verdict/config path.
METRIC_SOURCES: Dict[str, MetricSource] = {
    "VERDICT_TRUSTED_THRESHOLD": MetricSource(
        metric_name="VERDICT_TRUSTED_THRESHOLD",
        value=float(settings.VERDICT_TRUSTED_THRESHOLD),
        source_type="policy",
        source_ref="internal/scoring-policy/v9-phase-a",
        rationale="Trusted cutoff for high-confidence authenticity verdict.",
    ),
    "VERDICT_SUSPICIOUS_THRESHOLD": MetricSource(
        metric_name="VERDICT_SUSPICIOUS_THRESHOLD",
        value=float(settings.VERDICT_SUSPICIOUS_THRESHOLD),
        source_type="policy",
        source_ref="internal/scoring-policy/v9-phase-a",
        rationale="Boundary between suspicious and high-risk levels.",
    ),
    "VERDICT_HIGH_RISK_THRESHOLD": MetricSource(
        metric_name="VERDICT_HIGH_RISK_THRESHOLD",
        value=float(settings.VERDICT_HIGH_RISK_THRESHOLD),
        source_type="policy",
        source_ref="internal/scoring-policy/v9-phase-a",
        rationale="Minimum confidence_real that is still above likely-fake range.",
    ),
    "PREDETECT_CONFIDENCE_THRESHOLD": MetricSource(
        metric_name="PREDETECT_CONFIDENCE_THRESHOLD",
        value=float(settings.PREDETECT_CONFIDENCE_THRESHOLD),
        source_type="benchmark_run",
        source_ref="predetect-calibration/v8-v9-transition",
        rationale="Predetect confidence cutoff for pausing full testing.",
    ),
}


REQUIRED_METRICS: List[str] = [
    "VERDICT_TRUSTED_THRESHOLD",
    "VERDICT_SUSPICIOUS_THRESHOLD",
    "VERDICT_HIGH_RISK_THRESHOLD",
    "PREDETECT_CONFIDENCE_THRESHOLD",
]


def validate_required_metric_sources(strict: bool = False) -> List[str]:
    """
    Validate required metric source registrations.

    Args:
        strict: when True, raises ValueError on missing/invalid entries.

    Returns:
        List[str]: validation issues (empty means pass).
    """
    issues: List[str] = []

    for name in REQUIRED_METRICS:
        src = METRIC_SOURCES.get(name)
        if not src:
            issues.append(f"missing source registration: {name}")
            continue
        if not src.source_ref or src.source_ref.strip() == "":
            issues.append(f"missing source_ref: {name}")
        if src.source_type not in {"paper", "official_doc", "benchmark_run", "policy"}:
            issues.append(f"invalid source_type for {name}: {src.source_type}")

    if strict and issues:
        raise ValueError("metric source validation failed: " + "; ".join(issues))

    return issues


def get_metric_source(metric_name: str) -> MetricSource | None:
    """Get source metadata by metric name."""
    return METRIC_SOURCES.get(metric_name)
