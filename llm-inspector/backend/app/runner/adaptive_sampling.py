"""
runner/adaptive_sampling.py — IRT-driven adaptive sample count selection.

Implements dynamic n_samples selection based on the Fisher Information
from IRT 2PL item parameters.

Formula: I(θ) = a² × P(θ) × (1 - P(θ))
where P(θ) = 1 / (1 + exp(-a × (θ - b)))

Reference:
    van der Linden, W.J. & Glas, C.A.W. (2010).
    "Elements of Adaptive Testing". Springer.
    URL: https://link.springer.com/book/10.1007/978-1-4419-0742-4

    Weiss, D.J. & Kingsbury, G.G. (1984).
    "Application of computerized adaptive testing to educational problems."
    Journal of Educational Measurement, 21(4), 361-375.
    DOI: 10.1111/j.1745-3984.1984.tb01040.x
"""
from __future__ import annotations

import math

from app.core.logging import get_logger

logger = get_logger(__name__)

# Information thresholds for n_samples decisions.
# Source: heuristic derived from van der Linden & Glas (2010), Chapter 3.
_HIGH_INFO_THRESHOLD = 1.0   # I > 1.0  → n=1 (highly informative item)
_MED_INFO_THRESHOLD = 0.5    # I > 0.5  → n=2 (moderately informative)
# I <= 0.5 → n=3 (low information, more samples needed)

# Clip range to avoid numerical blow-up at extremes.
_INFO_CLIP_MAX = 10.0
_INFO_CLIP_MIN = 0.0


def item_information(
    theta: float,
    a: float,
    b: float,
    c: float = 0.0,
) -> float:
    """
    Compute the Fisher information of an IRT 3PL item at ability level theta.

    For 3PL model (Lord 1980):
        P(θ) = c + (1 - c) / (1 + exp(-a * (θ - b)))
        I(θ) = a² × (P - c)² / ((1 - c)² × P × (1 - P))

    Degenerates to standard 2PL when c=0:
        I(θ) = a² × P × (1 - P)

    Args:
        theta: ability estimate (logit scale)
        a:     discrimination parameter (typically 0.1–3.0)
        b:     difficulty parameter (logit scale, typically -3 to 3)
        c:     pseudo-guessing parameter (0–1, typically 0 for 2PL)

    Returns:
        Fisher information clipped to [0.0, 10.0]

    Reference:
        Lord, F.M. (1980). Applications of item response theory to practical
        testing problems. Erlbaum. ISBN: 978-0-89859-006-7
    """
    # Guard: keep c in [0, 1) to avoid division by zero in denominator
    c = max(0.0, min(c, 0.9999))

    try:
        exponent = -a * (theta - b)
        # Guard against extreme exponent values for numerical stability
        exponent = max(-500.0, min(500.0, exponent))
        p = c + (1.0 - c) / (1.0 + math.exp(exponent))
    except (OverflowError, ZeroDivisionError):
        return 0.0

    p_minus_c = p - c
    one_minus_c = 1.0 - c
    p_times_q = p * (1.0 - p)

    if p_times_q <= 0.0 or one_minus_c <= 0.0:
        return 0.0

    info = (a ** 2) * (p_minus_c ** 2) / ((one_minus_c ** 2) * p_times_q)

    # Clip to avoid numerical blow-up
    return float(max(_INFO_CLIP_MIN, min(_INFO_CLIP_MAX, info)))


def adaptive_n_samples(
    theta: float,
    irt_a: float,
    irt_b: float,
    irt_c: float = 0.0,
) -> int:
    """
    Determine the optimal number of samples based on IRT item information.

    Decision rule (heuristic, van der Linden & Glas 2010):
        I > 1.0  → 1 sample  (item is highly informative at this theta)
        I > 0.5  → 2 samples (moderately informative)
        I <= 0.5 → 3 samples (low information; repeat for stability)

    Args:
        theta:  current ability estimate (None treated as 0.0)
        irt_a:  item discrimination
        irt_b:  item difficulty
        irt_c:  guessing parameter

    Returns:
        int in {1, 2, 3}
    """
    if theta is None:
        theta = 0.0

    info = item_information(theta, irt_a, irt_b, irt_c)

    if info > _HIGH_INFO_THRESHOLD:
        n = 1
    elif info > _MED_INFO_THRESHOLD:
        n = 2
    else:
        n = 3

    logger.debug(
        "adaptive_n_samples computed",
        theta=theta,
        irt_a=irt_a,
        irt_b=irt_b,
        irt_c=irt_c,
        info=round(info, 4),
        n_samples=n,
    )
    return n


def get_adaptive_n_samples(
    case: dict,
    current_theta: float | None = None,
) -> int:
    """
    Convenience wrapper: read IRT params from a case dict and return n_samples.

    Falls back to n_samples=2 if IRT parameters are missing or invalid.

    Args:
        case:          test case dict (expects keys irt_a, irt_b, irt_c)
        current_theta: current ability estimate for the run (None → 0.0)

    Returns:
        Recommended number of samples (1, 2, or 3)
    """
    try:
        irt_a = float(case.get("irt_a") or 1.0)
        irt_b = float(case.get("irt_b") or 0.0)
        irt_c = float(case.get("irt_c") or 0.0)
        return adaptive_n_samples(
            theta=current_theta if current_theta is not None else 0.0,
            irt_a=irt_a,
            irt_b=irt_b,
            irt_c=irt_c,
        )
    except Exception as exc:
        logger.debug("get_adaptive_n_samples fallback", error=str(exc))
        return 2  # Safe default
