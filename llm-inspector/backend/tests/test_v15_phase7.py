"""
Tests for v15 Phase 7: Calibration Metrics.

Covers analysis/calibration_metrics.py:
  - brier_score(probs, outcomes)
  - log_loss(probs, outcomes, eps)
  - ece(probs, outcomes, n_bins)
  - reliability_curve(probs, outcomes, n_bins)

References: Brier (1950), Guo et al. (2017) ICML, Naeini et al. (2015) AAAI.
"""
from __future__ import annotations
import math
import pytest


# ---------------------------------------------------------------------------
# brier_score
# ---------------------------------------------------------------------------

def test_brier_score_perfect_predictions():
    from app.analysis.calibration_metrics import brier_score
    probs = [1.0, 1.0, 0.0, 0.0]
    outcomes = [1, 1, 0, 0]
    score = brier_score(probs, outcomes)
    assert score == pytest.approx(0.0)


def test_brier_score_worst_predictions():
    from app.analysis.calibration_metrics import brier_score
    probs = [0.0, 0.0, 1.0, 1.0]
    outcomes = [1, 1, 0, 0]
    score = brier_score(probs, outcomes)
    assert score == pytest.approx(1.0)


def test_brier_score_uniform_half():
    from app.analysis.calibration_metrics import brier_score
    probs = [0.5] * 4
    outcomes = [1, 1, 0, 0]
    score = brier_score(probs, outcomes)
    assert score == pytest.approx(0.25)


def test_brier_score_empty_returns_none():
    from app.analysis.calibration_metrics import brier_score
    assert brier_score([], []) is None


def test_brier_score_mismatched_lengths_returns_none():
    from app.analysis.calibration_metrics import brier_score
    assert brier_score([0.5, 0.5], [1]) is None


def test_brier_score_single_element():
    from app.analysis.calibration_metrics import brier_score
    score = brier_score([0.8], [1])
    assert score == pytest.approx((0.8 - 1) ** 2)


def test_brier_score_range():
    from app.analysis.calibration_metrics import brier_score
    probs = [0.3, 0.7, 0.6, 0.4]
    outcomes = [0, 1, 1, 0]
    score = brier_score(probs, outcomes)
    assert score is not None
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# log_loss
# ---------------------------------------------------------------------------

def test_log_loss_perfect_predictions():
    from app.analysis.calibration_metrics import log_loss
    probs = [1.0 - 1e-15, 1.0 - 1e-15, 1e-15, 1e-15]
    outcomes = [1, 1, 0, 0]
    loss = log_loss(probs, outcomes)
    assert loss is not None
    assert loss < 0.001  # near zero


def test_log_loss_uniform():
    from app.analysis.calibration_metrics import log_loss
    probs = [0.5] * 4
    outcomes = [1, 1, 0, 0]
    loss = log_loss(probs, outcomes)
    assert loss is not None
    assert loss == pytest.approx(math.log(2), rel=1e-3)


def test_log_loss_empty_returns_none():
    from app.analysis.calibration_metrics import log_loss
    assert log_loss([], []) is None


def test_log_loss_mismatched_returns_none():
    from app.analysis.calibration_metrics import log_loss
    assert log_loss([0.5], [1, 0]) is None


def test_log_loss_nonnegative():
    from app.analysis.calibration_metrics import log_loss
    probs = [0.3, 0.7, 0.9, 0.1]
    outcomes = [0, 1, 0, 1]
    loss = log_loss(probs, outcomes)
    assert loss is not None
    assert loss >= 0.0


def test_log_loss_clips_extreme_probs():
    """Extreme probs should be clipped, not cause log(0)."""
    from app.analysis.calibration_metrics import log_loss
    loss = log_loss([0.0, 1.0], [1, 0])  # worst case but clipped
    assert loss is not None
    assert math.isfinite(loss)


# ---------------------------------------------------------------------------
# ece
# ---------------------------------------------------------------------------

def test_ece_perfect_calibration():
    from app.analysis.calibration_metrics import ece
    # Model always predicts 0.9 and is right 90% of the time — well calibrated
    probs = [0.9] * 10
    outcomes = [1] * 9 + [0]  # 90% correct
    result = ece(probs, outcomes)
    assert result is not None
    assert result < 0.15  # near zero for perfectly calibrated


def test_ece_empty_returns_none():
    from app.analysis.calibration_metrics import ece
    assert ece([], []) is None


def test_ece_mismatched_returns_none():
    from app.analysis.calibration_metrics import ece
    assert ece([0.5, 0.5], [1]) is None


def test_ece_range():
    from app.analysis.calibration_metrics import ece
    probs = [0.1, 0.2, 0.8, 0.9, 0.5, 0.6, 0.4, 0.7]
    outcomes = [0, 0, 1, 1, 1, 1, 0, 1]
    result = ece(probs, outcomes)
    assert result is not None
    assert 0.0 <= result <= 1.0


def test_ece_low_bins_clamped():
    from app.analysis.calibration_metrics import ece
    # n_bins < 2 should be treated as 10
    probs = [0.5] * 4
    outcomes = [1, 1, 0, 0]
    result = ece(probs, outcomes, n_bins=1)
    assert result is not None


def test_ece_boundary_bin():
    from app.analysis.calibration_metrics import ece
    # probability exactly 1.0 goes into the last bin
    probs = [1.0, 0.5, 0.0]
    outcomes = [1, 1, 0]
    result = ece(probs, outcomes)
    assert result is not None


# ---------------------------------------------------------------------------
# reliability_curve
# ---------------------------------------------------------------------------

def test_reliability_curve_returns_three_lists():
    from app.analysis.calibration_metrics import reliability_curve
    probs = [0.1, 0.3, 0.5, 0.7, 0.9]
    outcomes = [0, 0, 1, 1, 1]
    confs, accs, counts = reliability_curve(probs, outcomes)
    assert isinstance(confs, list)
    assert isinstance(accs, list)
    assert isinstance(counts, list)


def test_reliability_curve_empty_returns_empty():
    from app.analysis.calibration_metrics import reliability_curve
    confs, accs, counts = reliability_curve([], [])
    assert confs == []
    assert accs == []
    assert counts == []


def test_reliability_curve_lengths_match():
    from app.analysis.calibration_metrics import reliability_curve
    probs = [0.1, 0.4, 0.6, 0.9, 0.95, 0.2, 0.7]
    outcomes = [0, 0, 1, 1, 1, 0, 1]
    confs, accs, counts = reliability_curve(probs, outcomes)
    assert len(confs) == len(accs) == len(counts)


def test_reliability_curve_counts_sum_to_n():
    from app.analysis.calibration_metrics import reliability_curve
    probs = [0.1, 0.4, 0.6, 0.9]
    outcomes = [0, 0, 1, 1]
    confs, accs, counts = reliability_curve(probs, outcomes)
    assert sum(counts) == len(probs)


def test_reliability_curve_accuracies_bounded():
    from app.analysis.calibration_metrics import reliability_curve
    probs = [0.1, 0.5, 0.9, 0.3, 0.7]
    outcomes = [0, 1, 1, 0, 1]
    _, accs, _ = reliability_curve(probs, outcomes)
    for acc in accs:
        assert 0.0 <= acc <= 1.0


def test_reliability_curve_confidence_in_bin_range():
    from app.analysis.calibration_metrics import reliability_curve
    probs = [0.15, 0.25, 0.75, 0.85]
    outcomes = [0, 1, 1, 1]
    confs, _, _ = reliability_curve(probs, outcomes)
    for conf in confs:
        assert 0.0 <= conf <= 1.0
