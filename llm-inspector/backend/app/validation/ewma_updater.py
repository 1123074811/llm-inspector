"""
validation/ewma_updater.py — v16 Phase 11: EWMA Reference Distribution Updater

Automatically updates reference distributions (timing_refs, token_dist_refs,
reference_embeddings) using Exponentially Weighted Moving Average.

Prevents systematic fingerprint_match degradation when model versions update.

Reference: Hunter 1986, Journal of Quality Technology 18(4)

Rules:
- retrieved_at距今 > stale_after_days (90) → weight ×0.3
- retrieved_at距今 > 180 days → discard + alert
- New official run data → EWMA merge into reference
"""
from __future__ import annotations

import json
import pathlib as _pl
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

from app.core.logging import get_logger

logger = get_logger(__name__)

_STALE_AFTER_DAYS = 90
_DISCARD_AFTER_DAYS = 180
_EWMA_ALPHA = 0.3  # Smoothing factor (higher = more weight on new data)


@dataclass
class StalenessReport:
    """Report on reference distribution staleness."""
    total_entries: int = 0
    stale_entries: int = 0      # > 90 days
    discarded_entries: int = 0  # > 180 days
    fresh_entries: int = 0      # <= 90 days
    staleness_ratio: float = 0.0
    entries: list[dict] = None

    def __post_init__(self):
        if self.entries is None:
            self.entries = []

    def to_dict(self) -> dict:
        return {
            "total_entries": self.total_entries,
            "stale_entries": self.stale_entries,
            "discarded_entries": self.discarded_entries,
            "fresh_entries": self.fresh_entries,
            "staleness_ratio": round(self.staleness_ratio, 4),
        }


def check_staleness(
    refs: dict,
    now: datetime | None = None,
) -> StalenessReport:
    """
    Check staleness of reference distributions.

    Args:
        refs: Dict with 'retrieved_at' ISO timestamp per entry.
        now: Current time (defaults to UTC now).

    Returns:
        StalenessReport with counts and ratio.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    report = StalenessReport()
    entries = refs if isinstance(refs, list) else [refs]

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        report.total_entries += 1
        retrieved = entry.get("retrieved_at", "")
        if not retrieved:
            report.stale_entries += 1
            continue

        try:
            retrieved_dt = datetime.fromisoformat(retrieved.replace("Z", "+00:00"))
            age_days = (now - retrieved_dt).days
        except (ValueError, TypeError):
            report.stale_entries += 1
            continue

        if age_days > _DISCARD_AFTER_DAYS:
            report.discarded_entries += 1
        elif age_days > _STALE_AFTER_DAYS:
            report.stale_entries += 1
        else:
            report.fresh_entries += 1

    if report.total_entries > 0:
        report.staleness_ratio = (report.stale_entries + report.discarded_entries) / report.total_entries

    return report


def ewma_merge(
    old_value: float,
    new_value: float,
    alpha: float = _EWMA_ALPHA,
) -> float:
    """
    EWMA merge: new = alpha * new_value + (1 - alpha) * old_value.

    Args:
        old_value: Existing reference value.
        new_value: New observation value.
        alpha: Smoothing factor (0-1).

    Returns:
        Merged value.
    """
    return alpha * new_value + (1 - alpha) * old_value


def apply_staleness_weight(
    entry: dict,
    now: datetime | None = None,
) -> dict:
    """
    Apply staleness weight to a reference entry.

    - > 90 days: weight × 0.3
    - > 180 days: mark for discard
    - <= 90 days: full weight
    """
    if now is None:
        now = datetime.now(timezone.utc)

    retrieved = entry.get("retrieved_at", "")
    if not retrieved:
        entry["_staleness_weight"] = 0.3
        entry["_should_discard"] = False
        return entry

    try:
        retrieved_dt = datetime.fromisoformat(retrieved.replace("Z", "+00:00"))
        age_days = (now - retrieved_dt).days
    except (ValueError, TypeError):
        entry["_staleness_weight"] = 0.3
        entry["_should_discard"] = False
        return entry

    if age_days > _DISCARD_AFTER_DAYS:
        entry["_staleness_weight"] = 0.0
        entry["_should_discard"] = True
    elif age_days > _STALE_AFTER_DAYS:
        entry["_staleness_weight"] = 0.3
        entry["_should_discard"] = False
    else:
        entry["_staleness_weight"] = 1.0
        entry["_should_discard"] = False

    return entry


def update_reference_file(
    ref_path: _pl.Path,
    new_data: dict,
    alpha: float = _EWMA_ALPHA,
) -> bool:
    """
    EWMA-merge new official run data into a reference JSON file.

    Args:
        ref_path: Path to reference JSON file.
        new_data: New observation data to merge.
        alpha: EWMA smoothing factor.

    Returns:
        True if update succeeded.
    """
    if not ref_path.exists():
        logger.warning("Reference file not found", path=str(ref_path))
        return False

    try:
        with open(ref_path, "r", encoding="utf-8") as f:
            refs = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load reference file", error=str(e))
        return False

    # Apply EWMA merge to numeric fields
    model_name = new_data.get("model_name", "")
    if model_name in refs:
        old = refs[model_name]
        if isinstance(old, dict) and isinstance(new_data, dict):
            for key in old:
                if key in new_data and isinstance(old[key], (int, float)):
                    old[key] = ewma_merge(old[key], new_data[key], alpha)
            old["retrieved_at"] = datetime.now(timezone.utc).isoformat()
            old["ewma_alpha"] = alpha
    else:
        refs[model_name] = {
            **new_data,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "ewma_alpha": alpha,
        }

    try:
        with open(ref_path, "w", encoding="utf-8") as f:
            json.dump(refs, f, ensure_ascii=False, indent=2)
        logger.info("Reference updated", path=str(ref_path), model=model_name)
        return True
    except OSError as e:
        logger.warning("Failed to write reference file", error=str(e))
        return False
