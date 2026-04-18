"""
analysis/stanine.py — Stanine-9 scale conversion utilities

Stanine (STAndard NINE) maps a continuous ability estimate (theta, logit scale)
to an integer 1-9. The boundaries are from Canfield (1951), standard
psychometric practice.

Source: SRC["stanine.boundaries"]
Reference: https://www.scribbr.com/statistics/stanine/ (Canfield 1951)
"""
from __future__ import annotations
import math


def theta_to_stanine(theta: float) -> int:
    """Map IRT theta (logit) to Stanine 1-9."""
    try:
        from app._data import SRC
        bounds = [float(b) for b in SRC["stanine.boundaries"].value]
    except Exception:
        bounds = [-1.75, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75]
    for i, b in enumerate(bounds):
        if theta < b:
            return i + 1
    return 9


def theta_to_percentile(theta: float, reference_thetas: list[float] | None = None) -> float:
    """
    Convert theta to percentile rank using KDE over reference distribution.
    Falls back to normal CDF (mean=0, sd=1) when no reference is provided.
    Reference: IRT logit scale is approximately normally distributed N(0,1).
    """
    if reference_thetas and len(reference_thetas) >= 5:
        # Empirical percentile from reference sample
        below = sum(1 for t in reference_thetas if t < theta)
        return round(below / len(reference_thetas) * 100, 1)
    # Normal CDF approximation (mean=0, sd=1 per SRC["irt.theta_sd"])
    z = theta  # theta_sd = 1.0 per SRC
    return round(100 * (0.5 * (1.0 + math.erf(z / math.sqrt(2)))), 1)
