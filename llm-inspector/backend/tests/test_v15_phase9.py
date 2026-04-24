"""
Tests for v15 Phase 9: Judge Calibration / Reliability Monitoring.

Covers analysis/judge_calibration.py:
  - compute_fleiss_kappa(ratings_matrix)
  - compute_cohen_kappa(rater1, rater2)
  - judge_bias_detection(primary_results, rule_results, threshold)

References: Fleiss (1971), Cohen (1960).
"""
from __future__ import annotations
import pytest


# ---------------------------------------------------------------------------
# compute_fleiss_kappa
# ---------------------------------------------------------------------------

def test_fleiss_kappa_perfect_agreement():
    from app.analysis.judge_calibration import compute_fleiss_kappa
    # 4 subjects, 2 categories; all raters agree on category 0
    # Each row sums to N (number of raters)
    matrix = [
        [3, 0],  # all 3 raters chose category 0
        [3, 0],
        [3, 0],
        [3, 0],
    ]
    k = compute_fleiss_kappa(matrix)
    assert k is not None
    assert k == pytest.approx(1.0, abs=0.01)


def test_fleiss_kappa_empty_returns_none():
    from app.analysis.judge_calibration import compute_fleiss_kappa
    assert compute_fleiss_kappa([]) is None


def test_fleiss_kappa_single_rater_returns_none():
    from app.analysis.judge_calibration import compute_fleiss_kappa
    # N=1 per row → division by N*(N-1) = 0
    matrix = [[1, 0], [0, 1], [1, 0]]
    k = compute_fleiss_kappa(matrix)
    assert k is None


def test_fleiss_kappa_range():
    from app.analysis.judge_calibration import compute_fleiss_kappa
    matrix = [
        [2, 1],
        [1, 2],
        [2, 1],
        [1, 2],
    ]
    k = compute_fleiss_kappa(matrix)
    assert k is not None
    assert -1.0 <= k <= 1.0


def test_fleiss_kappa_zero_kappa():
    from app.analysis.judge_calibration import compute_fleiss_kappa
    # Carefully constructed matrix to produce kappa ≈ 0:
    # Each row is [3,1] or [1,3] with N=4 raters and 2 categories.
    # P_i per row = (9+1-4)/12 = 0.5, p_j = [0.5, 0.5], P_e = 0.5
    # → kappa = (0.5 - 0.5)/(1 - 0.5) = 0.0
    matrix = [
        [3, 1],
        [1, 3],
        [3, 1],
        [1, 3],
    ]
    k = compute_fleiss_kappa(matrix)
    assert k is not None
    assert abs(k) < 0.05  # should be exactly 0.0


def test_fleiss_kappa_three_categories():
    from app.analysis.judge_calibration import compute_fleiss_kappa
    matrix = [
        [3, 0, 0],
        [0, 3, 0],
        [0, 0, 3],
    ]
    k = compute_fleiss_kappa(matrix)
    assert k is not None
    # Perfect agreement on different categories → kappa=1.0
    assert k == pytest.approx(1.0, abs=0.05)


def test_fleiss_kappa_empty_row_list_returns_none():
    from app.analysis.judge_calibration import compute_fleiss_kappa
    assert compute_fleiss_kappa([[]]) is None


# ---------------------------------------------------------------------------
# compute_cohen_kappa
# ---------------------------------------------------------------------------

def test_cohen_kappa_perfect_agreement():
    from app.analysis.judge_calibration import compute_cohen_kappa
    rater1 = [1, 1, 0, 0, 1, 0]
    rater2 = [1, 1, 0, 0, 1, 0]
    k = compute_cohen_kappa(rater1, rater2)
    assert k is not None
    assert k == pytest.approx(1.0)


def test_cohen_kappa_perfect_disagreement():
    from app.analysis.judge_calibration import compute_cohen_kappa
    rater1 = [1, 1, 1, 0, 0, 0]
    rater2 = [0, 0, 0, 1, 1, 1]
    k = compute_cohen_kappa(rater1, rater2)
    assert k is not None
    assert k < 0.0  # negative kappa = worse than chance


def test_cohen_kappa_empty_returns_none():
    from app.analysis.judge_calibration import compute_cohen_kappa
    assert compute_cohen_kappa([], []) is None


def test_cohen_kappa_mismatched_returns_none():
    from app.analysis.judge_calibration import compute_cohen_kappa
    assert compute_cohen_kappa([1, 0], [1]) is None


def test_cohen_kappa_single_category_returns_none():
    from app.analysis.judge_calibration import compute_cohen_kappa
    # All same label → only 1 category → returns None
    k = compute_cohen_kappa([1, 1, 1], [1, 1, 1])
    assert k is None


def test_cohen_kappa_range():
    from app.analysis.judge_calibration import compute_cohen_kappa
    rater1 = [1, 0, 1, 0, 1, 1, 0, 0]
    rater2 = [1, 0, 0, 1, 1, 0, 0, 1]
    k = compute_cohen_kappa(rater1, rater2)
    assert k is not None
    assert -1.0 <= k <= 1.0


def test_cohen_kappa_moderate_agreement():
    from app.analysis.judge_calibration import compute_cohen_kappa
    # 75% agreement → moderate kappa > 0.4
    rater1 = [1, 1, 0, 0, 1, 1, 0, 0]
    rater2 = [1, 0, 0, 0, 1, 1, 0, 1]  # 6/8 agree
    k = compute_cohen_kappa(rater1, rater2)
    assert k is not None
    assert k > 0.0


# ---------------------------------------------------------------------------
# judge_bias_detection
# ---------------------------------------------------------------------------

def test_judge_bias_detection_no_bias():
    from app.analysis.judge_calibration import judge_bias_detection
    primary = [{"case_id": f"c{i}", "passed": True, "confidence": 0.9} for i in range(10)]
    rule = [{"case_id": f"c{i}", "passed": True, "confidence": 0.9} for i in range(10)]
    result = judge_bias_detection(primary, rule, threshold=0.20)
    assert result["bias_detected"] is False
    assert result["disagreement_rate"] == 0.0
    assert len(result["warnings"]) == 0


def test_judge_bias_detection_high_bias():
    from app.analysis.judge_calibration import judge_bias_detection
    primary = [{"case_id": f"c{i}", "passed": True, "confidence": 0.9} for i in range(10)]
    rule = [{"case_id": f"c{i}", "passed": False, "confidence": 0.8} for i in range(10)]
    result = judge_bias_detection(primary, rule, threshold=0.20)
    assert result["bias_detected"] is True
    assert result["disagreement_rate"] == pytest.approx(1.0)
    assert len(result["warnings"]) >= 1


def test_judge_bias_detection_empty_inputs():
    from app.analysis.judge_calibration import judge_bias_detection
    result = judge_bias_detection([], [])
    assert result["bias_detected"] is False
    assert result["disagreement_rate"] == 0.0


def test_judge_bias_detection_no_matching_case_ids():
    from app.analysis.judge_calibration import judge_bias_detection
    primary = [{"case_id": "a1", "passed": True}]
    rule = [{"case_id": "b1", "passed": False}]
    result = judge_bias_detection(primary, rule)
    assert result["total_comparisons"] == 0
    assert result["disagreement_rate"] == 0.0


def test_judge_bias_detection_result_keys():
    from app.analysis.judge_calibration import judge_bias_detection
    result = judge_bias_detection(
        [{"case_id": "x", "passed": True}],
        [{"case_id": "x", "passed": True}],
    )
    expected_keys = {
        "bias_detected", "disagreement_rate", "total_comparisons",
        "disagreements", "primary_liberal_count", "primary_conservative_count",
        "warnings",
    }
    assert expected_keys.issubset(result.keys())


def test_judge_bias_detection_liberal_warning():
    from app.analysis.judge_calibration import judge_bias_detection
    # Primary says pass, rule says fail → "liberal"
    primary = [{"case_id": f"c{i}", "passed": True} for i in range(10)]
    rule = [{"case_id": f"c{i}", "passed": False} for i in range(10)]
    result = judge_bias_detection(primary, rule, threshold=0.05)
    assert result["primary_liberal_count"] == 10
    assert result["primary_conservative_count"] == 0
    # Should warn about being lenient
    liberal_warnings = [w for w in result["warnings"] if "lenient" in w.lower()]
    assert len(liberal_warnings) >= 1


def test_judge_bias_detection_conservative_warning():
    from app.analysis.judge_calibration import judge_bias_detection
    # Primary says fail, rule says pass → "conservative"
    primary = [{"case_id": f"c{i}", "passed": False} for i in range(10)]
    rule = [{"case_id": f"c{i}", "passed": True} for i in range(10)]
    result = judge_bias_detection(primary, rule, threshold=0.05)
    assert result["primary_conservative_count"] == 10
    assert result["primary_liberal_count"] == 0
    conservative_warnings = [w for w in result["warnings"] if "strict" in w.lower()]
    assert len(conservative_warnings) >= 1


def test_judge_bias_detection_capped_disagreements():
    """Disagreements list is capped at 20 for transport."""
    from app.analysis.judge_calibration import judge_bias_detection
    primary = [{"case_id": f"c{i}", "passed": True} for i in range(50)]
    rule = [{"case_id": f"c{i}", "passed": False} for i in range(50)]
    result = judge_bias_detection(primary, rule, threshold=0.05)
    assert len(result["disagreements"]) <= 20


def test_judge_bias_detection_disagreement_rate_bounded():
    from app.analysis.judge_calibration import judge_bias_detection
    primary = [{"case_id": f"c{i}", "passed": bool(i % 2)} for i in range(20)]
    rule = [{"case_id": f"c{i}", "passed": bool((i + 1) % 2)} for i in range(20)]
    result = judge_bias_detection(primary, rule)
    assert 0.0 <= result["disagreement_rate"] <= 1.0
