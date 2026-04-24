"""
analysis/calibration_metrics.py — v15 Phase 8: Calibration metrics.

Provides proper scoring rules and calibration measures for probability
predictions from the judgment system:

  - brier_score(probs, outcomes)
  - log_loss(probs, outcomes)
  - ece(probs, outcomes, n_bins=10)
  - reliability_curve(probs, outcomes, n_bins=10)

References:
  Brier (1950) "Verification of forecasts expressed in terms of probability"
  Guo et al. (2017) "On Calibration of Modern Neural Networks", ICML
  Naeini et al. (2015) "Obtaining Well Calibrated Probabilities Using
    Bayesian Binning", AAAI
"""
from __future__ import annotations

import math


def brier_score(probs: list[float], outcomes: list[int]) -> float | None:
    """Brier score (mean squared error between predictions and outcomes).

    Args:
        probs: Predicted probabilities (0-1) for the positive class.
        outcomes: Binary outcomes (0 or 1).

    Returns:
        Brier score (0 = perfect, 1 = worst), or None if inputs are empty.

    Reference: Brier (1950) Monthly Weather Review.
    """
    if not probs or not outcomes or len(probs) != len(outcomes):
        return None
    n = len(probs)
    total = sum((p - o) ** 2 for p, o in zip(probs, outcomes))
    return total / n


def log_loss(probs: list[float], outcomes: list[int], eps: float = 1e-15) -> float | None:
    """Log loss / cross-entropy loss.

    Args:
        probs: Predicted probabilities (0-1).
        outcomes: Binary outcomes (0 or 1).
        eps: Small epsilon to avoid log(0).

    Returns:
        Log loss value, or None if inputs are empty.

    Reference: Good (1952) "Rational Decisions".
    """
    if not probs or not outcomes or len(probs) != len(outcomes):
        return None
    n = len(probs)
    total = 0.0
    for p, o in zip(probs, outcomes):
        p = max(eps, min(1 - eps, p))
        total += o * math.log(p) + (1 - o) * math.log(1 - p)
    return -total / n


def ece(
    probs: list[float],
    outcomes: list[int],
    n_bins: int = 10,
) -> float | None:
    """Expected Calibration Error.

    Partitions predictions into n_bins equal-width bins and computes
    the weighted average of |accuracy - confidence| per bin.

    Args:
        probs: Predicted probabilities (0-1).
        outcomes: Binary outcomes (0 or 1).
        n_bins: Number of bins.

    Returns:
        ECE value, or None if inputs are empty.

    Reference: Naeini et al. (2015) AAAI; Guo et al. (2017) ICML.
    """
    if not probs or not outcomes or len(probs) != len(outcomes):
        return None
    if n_bins < 2:
        n_bins = 10

    n = len(probs)
    bin_boundaries = [i / n_bins for i in range(n_bins + 1)]
    ece_sum = 0.0

    for i in range(n_bins):
        lo = bin_boundaries[i]
        hi = bin_boundaries[i + 1]
        in_bin = [
            j for j in range(n)
            if (lo <= probs[j] < hi) or (i == n_bins - 1 and probs[j] == 1.0)
        ]
        if not in_bin:
            continue

        bin_conf = sum(probs[j] for j in in_bin) / len(in_bin)
        bin_acc = sum(outcomes[j] for j in in_bin) / len(in_bin)
        ece_sum += (len(in_bin) / n) * abs(bin_acc - bin_conf)

    return ece_sum


def reliability_curve(
    probs: list[float],
    outcomes: list[int],
    n_bins: int = 10,
) -> tuple[list[float], list[float], list[int]]:
    """Compute reliability diagram data.

    Returns:
        (bin_confidences, bin_accuracies, bin_counts) for plotting.

    Reference: Guo et al. (2017) "On Calibration of Modern Neural Networks".
    """
    if not probs or not outcomes or len(probs) != len(outcomes):
        return ([], [], [])

    if n_bins < 2:
        n_bins = 10

    n = len(probs)
    bin_bounds = [i / n_bins for i in range(n_bins + 1)]
    confs: list[float] = []
    accs: list[float] = []
    counts: list[int] = []

    for i in range(n_bins):
        lo = bin_bounds[i]
        hi = bin_bounds[i + 1]
        in_bin = [
            j for j in range(n)
            if (lo <= probs[j] < hi) or (i == n_bins - 1 and probs[j] == 1.0)
        ]
        if not in_bin:
            continue
        confs.append(sum(probs[j] for j in in_bin) / len(in_bin))
        accs.append(sum(outcomes[j] for j in in_bin) / len(in_bin))
        counts.append(len(in_bin))

    return (confs, accs, counts)
