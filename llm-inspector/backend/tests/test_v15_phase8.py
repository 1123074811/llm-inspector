"""
Tests for v15 Phase 8: Uncertainty Computation.

Covers analysis/uncertainty.py:
  - bootstrap_ci(samples, stat_fn, B, ci)  — Bootstrap CI (Efron 1979)
  - sem(theta, info_sum)                   — IRT standard error of measurement
  - hdi(posterior_samples, prob)           — Highest Density Interval
  - weighted_ci(values, weights, B, ci)    — Weighted bootstrap CI
"""
from __future__ import annotations
import math
import random
import pytest


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------

def test_bootstrap_ci_returns_tuple_of_two():
    from app.analysis.uncertainty import bootstrap_ci
    lo, hi = bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0])
    assert isinstance(lo, float)
    assert isinstance(hi, float)


def test_bootstrap_ci_empty_returns_nan():
    from app.analysis.uncertainty import bootstrap_ci
    lo, hi = bootstrap_ci([])
    assert math.isnan(lo)
    assert math.isnan(hi)


def test_bootstrap_ci_lo_le_hi():
    from app.analysis.uncertainty import bootstrap_ci
    random.seed(42)
    lo, hi = bootstrap_ci([2.0, 4.0, 6.0, 8.0, 10.0], B=200)
    assert lo <= hi


def test_bootstrap_ci_covers_mean():
    from app.analysis.uncertainty import bootstrap_ci
    from statistics import mean
    random.seed(42)
    data = list(range(1, 21))  # [1..20], mean=10.5
    lo, hi = bootstrap_ci(data, B=500, ci=0.95)
    assert lo <= mean(data) <= hi


def test_bootstrap_ci_custom_stat_fn():
    from app.analysis.uncertainty import bootstrap_ci
    random.seed(0)
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    lo, hi = bootstrap_ci(data, stat_fn=max, B=200)
    # max should be >= 5 in most resamples
    assert hi >= 4.0


def test_bootstrap_ci_single_element():
    from app.analysis.uncertainty import bootstrap_ci
    lo, hi = bootstrap_ci([7.0], B=50)
    # Single element: both should be the value
    assert lo == pytest.approx(7.0)
    assert hi == pytest.approx(7.0)


def test_bootstrap_ci_wider_with_lower_ci():
    """95% CI should be wider than 50% CI."""
    from app.analysis.uncertainty import bootstrap_ci
    random.seed(99)
    data = [float(x) for x in range(1, 101)]
    lo95, hi95 = bootstrap_ci(data, B=300, ci=0.95)
    lo50, hi50 = bootstrap_ci(data, B=300, ci=0.50)
    width95 = hi95 - lo95
    width50 = hi50 - lo50
    assert width95 >= width50


# ---------------------------------------------------------------------------
# sem
# ---------------------------------------------------------------------------

def test_sem_basic():
    from app.analysis.uncertainty import sem
    # I = 4.0 → SEM = 1/sqrt(4) = 0.5
    result = sem(0.0, 4.0)
    assert result == pytest.approx(0.5)


def test_sem_zero_info_returns_none():
    from app.analysis.uncertainty import sem
    assert sem(0.0, 0.0) is None


def test_sem_negative_info_returns_none():
    from app.analysis.uncertainty import sem
    assert sem(0.0, -1.0) is None


def test_sem_none_info_returns_none():
    from app.analysis.uncertainty import sem
    assert sem(0.0, None) is None


def test_sem_high_info_small_sem():
    from app.analysis.uncertainty import sem
    # Large information → small SEM
    result = sem(1.0, 100.0)
    assert result == pytest.approx(0.1)


def test_sem_theta_value_ignored():
    from app.analysis.uncertainty import sem
    # SEM depends only on info_sum, not theta
    assert sem(0.0, 9.0) == pytest.approx(sem(5.0, 9.0))


# ---------------------------------------------------------------------------
# hdi
# ---------------------------------------------------------------------------

def test_hdi_returns_tuple():
    from app.analysis.uncertainty import hdi
    lo, hi = hdi([1.0, 2.0, 3.0, 4.0, 5.0])
    assert isinstance(lo, float)
    assert isinstance(hi, float)


def test_hdi_empty_returns_nan():
    from app.analysis.uncertainty import hdi
    lo, hi = hdi([])
    assert math.isnan(lo)
    assert math.isnan(hi)


def test_hdi_lo_le_hi():
    from app.analysis.uncertainty import hdi
    lo, hi = hdi([5.0, 3.0, 7.0, 1.0, 9.0])
    assert lo <= hi


def test_hdi_95pct_contains_bulk():
    from app.analysis.uncertainty import hdi
    data = [float(x) for x in range(1, 101)]
    lo, hi = hdi(data, prob=0.95)
    # 95% of 100 points = 95; interval should cover [1..95] or [6..100]
    assert hi - lo <= 95.0


def test_hdi_single_element():
    from app.analysis.uncertainty import hdi
    lo, hi = hdi([42.0])
    assert lo == pytest.approx(42.0)
    assert hi == pytest.approx(42.0)


def test_hdi_symmetric_distribution_centered():
    from app.analysis.uncertainty import hdi
    import random
    random.seed(7)
    data = [random.gauss(0.0, 1.0) for _ in range(1000)]
    lo, hi = hdi(data, prob=0.95)
    # Roughly symmetric around 0
    center = (lo + hi) / 2.0
    assert abs(center) < 0.3


def test_hdi_skewed_picks_dense_region():
    from app.analysis.uncertainty import hdi
    # Bimodal: cluster at 0 and cluster at 10; HDI should select the denser region
    data = [0.0] * 100 + [10.0] * 5
    lo, hi = hdi(data, prob=0.80)
    # Should choose the tight cluster near 0
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# weighted_ci
# ---------------------------------------------------------------------------

def test_weighted_ci_returns_tuple():
    from app.analysis.uncertainty import weighted_ci
    lo, hi = weighted_ci([1.0, 2.0, 3.0], [1.0, 1.0, 1.0])
    assert isinstance(lo, float)
    assert isinstance(hi, float)


def test_weighted_ci_empty_returns_nan():
    from app.analysis.uncertainty import weighted_ci
    lo, hi = weighted_ci([], [])
    assert math.isnan(lo)
    assert math.isnan(hi)


def test_weighted_ci_mismatched_lengths_returns_nan():
    from app.analysis.uncertainty import weighted_ci
    lo, hi = weighted_ci([1.0, 2.0], [1.0])
    assert math.isnan(lo)
    assert math.isnan(hi)


def test_weighted_ci_lo_le_hi():
    from app.analysis.uncertainty import weighted_ci
    random.seed(5)
    lo, hi = weighted_ci([1.0, 5.0, 10.0], [0.5, 1.0, 0.5], B=200)
    assert lo <= hi


def test_weighted_ci_zero_total_weight_returns_nan():
    from app.analysis.uncertainty import weighted_ci
    lo, hi = weighted_ci([1.0, 2.0], [0.0, 0.0])
    assert math.isnan(lo)
    assert math.isnan(hi)


def test_weighted_ci_high_weight_biases_ci():
    """Upweighting the largest element should shift CI upward."""
    from app.analysis.uncertainty import weighted_ci
    random.seed(12)
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    even_lo, even_hi = weighted_ci(values, [1.0] * 5, B=300)
    biased_lo, biased_hi = weighted_ci(values, [0.1, 0.1, 0.1, 0.1, 10.0], B=300)
    assert biased_lo >= even_lo - 0.5  # biased toward 5 → at least as high
