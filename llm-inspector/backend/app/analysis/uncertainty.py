"""
analysis/uncertainty.py — v15 Phase 8: Uncertainty computation for scores.

Provides:
  - bootstrap_ci(samples, stat_fn, B=200, ci=0.95)  — Bootstrap CI (Efron 1979)
  - sem(theta, info_sum)                             — IRT standard error
  - hdi(posterior_samples, prob=0.95)               — Highest Density Interval
  - weighted_ci(values, weights, ci=0.95)            — Weighted bootstrap CI

All functions return (lower, upper) or (lower, upper, method_name).
"""
from __future__ import annotations

import random
from statistics import mean, stdev
from typing import Callable


def bootstrap_ci(
    samples: list[float],
    stat_fn: Callable[[list[float]], float] | None = None,
    B: int = 200,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap confidence interval for any statistic.

    Args:
        samples: Observed data points.
        stat_fn: Function that computes the statistic from a resample.
                 Defaults to mean.
        B: Number of bootstrap replicates.
        ci: Confidence level (e.g. 0.95).

    Returns:
        (lower_bound, upper_bound).

    Reference: Efron (1979) "Bootstrap Methods", Ann. Stat.
    """
    if not samples:
        return (float("nan"), float("nan"))

    stat_fn = stat_fn or (lambda x: mean(x))
    n = len(samples)
    estimates: list[float] = []

    for _ in range(B):
        resample = [random.choice(samples) for _ in range(n)]
        try:
            estimates.append(stat_fn(resample))
        except Exception:
            continue

    if not estimates:
        return (float("nan"), float("nan"))

    estimates.sort()
    tail = (1.0 - ci) / 2.0
    lo_idx = max(0, int(tail * len(estimates)))
    hi_idx = min(len(estimates) - 1, int((1.0 - tail) * len(estimates)))
    return (estimates[lo_idx], estimates[hi_idx])


def sem(theta: float, info_sum: float) -> float | None:
    """Standard error of measurement for IRT theta.

    SEM = 1 / sqrt(I) where I is the total test information.

    Args:
        theta: IRT theta estimate (logit scale).
        info_sum: Sum of item information at theta.

    Returns:
        SEM value, or None if info_sum is invalid.

    Reference: Lord (1980) "Applications of Item Response Theory".
    """
    if info_sum is None or info_sum <= 0:
        return None
    return 1.0 / (info_sum ** 0.5)


def hdi(posterior_samples: list[float], prob: float = 0.95) -> tuple[float, float]:
    """Highest Density Interval from posterior samples.

    Uses the shortest interval method: sorts samples, then slides a
    window of width `prob` and picks the narrowest.

    Args:
        posterior_samples: MCMC or bootstrap posterior samples.
        prob: Probability mass to cover (default 0.95).

    Returns:
        (lower, upper).

    Reference: Hyndman (1996) "Computing and graphing highest density regions".
    """
    if not posterior_samples:
        return (float("nan"), float("nan"))

    sorted_s = sorted(posterior_samples)
    n = len(sorted_s)
    n_in = max(1, int(prob * n))
    best_lo = sorted_s[0]
    best_hi = sorted_s[n_in - 1]
    best_width = best_hi - best_lo

    for i in range(1, n - n_in + 1):
        lo = sorted_s[i]
        hi = sorted_s[i + n_in - 1]
        width = hi - lo
        if width < best_width:
            best_width = width
            best_lo = lo
            best_hi = hi

    return (best_lo, best_hi)


def weighted_ci(
    values: list[float],
    weights: list[float],
    B: int = 200,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Weighted bootstrap confidence interval.

    Resamples with probability proportional to weights.

    Args:
        values: Data points.
        weights: Weight for each data point (does not need to sum to 1).
        B: Number of bootstrap replicates.
        ci: Confidence level.

    Returns:
        (lower_bound, upper_bound).
    """
    if not values or not weights or len(values) != len(weights):
        return (float("nan"), float("nan"))

    total_w = sum(weights)
    if total_w <= 0:
        return (float("nan"), float("nan"))

    probs = [w / total_w for w in weights]

    def weighted_mean(_samples: list[float]) -> float:
        return mean(_samples) if _samples else 0.0

    estimates: list[float] = []
    n = len(values)
    for _ in range(B):
        resample = random.choices(values, weights=probs, k=n)
        try:
            estimates.append(weighted_mean(resample))
        except Exception:
            continue

    if not estimates:
        return (float("nan"), float("nan"))

    estimates.sort()
    tail = (1.0 - ci) / 2.0
    lo_idx = max(0, int(tail * len(estimates)))
    hi_idx = min(len(estimates) - 1, int((1.0 - tail) * len(estimates)))
    return (estimates[lo_idx], estimates[hi_idx])
