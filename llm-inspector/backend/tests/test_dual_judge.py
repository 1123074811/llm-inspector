"""
tests/test_dual_judge.py — Tests for dual-blind judging + Cohen's κ

Phase 2.3 verification.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from app.judge.dual_judge import (
    DualJudgeResult,
    KappaAccumulator,
    compute_kappa,
    dual_judge,
)


# ── compute_kappa ─────────────────────────────────────────────────────────────

def test_kappa_perfect_agreement():
    """Both judges agree on 10 items → κ = 1.0"""
    a = [True, False, True, False, True, True, False, True, False, False]
    b = list(a)
    assert compute_kappa(a, b) == pytest.approx(1.0)


def test_kappa_zero_agreement():
    """All disagreements → κ ≤ 0."""
    a = [True, False, True, False, True, False, True, False]
    b = [not x for x in a]
    kappa = compute_kappa(a, b)
    assert kappa <= 0.0


def test_kappa_moderate():
    """7/10 agree → κ in [0.2, 0.7] range."""
    a = [True, True, True, True, True, False, False, False, False, False]
    # Disagree on 3 items (indices 0, 5, 9)
    b = [False, True, True, True, True, True, False, False, False, True]
    assert sum(1 for x, y in zip(a, b) if x == y) == 7
    kappa = compute_kappa(a, b)
    assert 0.2 <= kappa <= 0.7


def test_kappa_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        compute_kappa([True, False], [True])


def test_kappa_single_pair_agree():
    assert compute_kappa([True], [True]) == 1.0


def test_kappa_single_pair_disagree():
    assert compute_kappa([True], [False]) == -1.0


def test_kappa_empty_returns_one():
    assert compute_kappa([], []) == 1.0


# ── KappaAccumulator ──────────────────────────────────────────────────────────

def test_kappa_too_few_samples():
    acc = KappaAccumulator()
    acc.add(True, True)
    acc.add(False, False)
    acc.add(True, False)
    # Only 3 samples < 4 threshold
    assert acc.kappa() is None


def test_kappa_accumulator_basic():
    """Add 10 pairs, compute κ."""
    acc = KappaAccumulator()
    pairs = [
        (True, True), (True, True), (False, False), (True, False),
        (True, True), (False, False), (True, True), (False, False),
        (True, True), (False, True),
    ]
    for r, s in pairs:
        acc.add(r, s)
    assert len(acc) == 10
    kappa = acc.kappa()
    assert kappa is not None
    assert -1.0 <= kappa <= 1.0
    summary = acc.summary()
    assert summary["n"] == 10
    assert summary["rule_pass_rate"] is not None


# ── dual_judge integration ────────────────────────────────────────────────────

def test_dual_judge_rule_only_fingerprint():
    """Protocol / fingerprint methods should skip semantic judge."""
    result = dual_judge(
        judge_method="exact_match",  # not in semantic categories
        response_text="hello",
        params={"target": "hello"},
        run_semantic=True,
    )
    assert isinstance(result, DualJudgeResult)
    assert result.method == "rule_only"
    assert result.semantic_passed is None
    assert result.kappa is None
    assert result.tiebreak_used is False
    assert result.consensus_passed is True


def test_dual_judge_rule_only_when_flag_false():
    """run_semantic=False forces rule-only even on capability methods."""
    with patch("app.judge.dual_judge._run_rule", return_value=(True, {"m": "rule"})):
        result = dual_judge(
            judge_method="constraint_reasoning",
            response_text="ok",
            params={},
            run_semantic=False,
        )
    assert result.method == "rule_only"
    assert result.semantic_passed is None


def test_dual_judge_agree_path():
    """Patch both judges to agree → tiebreak_used=False, method='dual_agree'."""
    with patch("app.judge.dual_judge._run_rule", return_value=(True, {"m": "rule"})), \
         patch("app.judge.dual_judge._run_semantic", return_value=(True, {"m": "sem"})):
        result = dual_judge(
            judge_method="constraint_reasoning",
            response_text="answer",
            params={},
            run_semantic=True,
        )
    assert result.agreement is True
    assert result.tiebreak_used is False
    assert result.method == "dual_agree"
    assert result.consensus_passed is True
    assert result.kappa == 1.0


def test_dual_judge_disagree_triggers_tiebreak():
    """Judges disagree → transparent_judge invoked."""
    tb_called = {"n": 0}

    def _fake_tb(method, text, params):
        tb_called["n"] += 1
        return True, {"m": "tiebreak"}

    with patch("app.judge.dual_judge._run_rule", return_value=(False, {"m": "rule"})), \
         patch("app.judge.dual_judge._run_semantic", return_value=(True, {"m": "sem"})), \
         patch("app.judge.dual_judge._run_tiebreak", side_effect=_fake_tb):
        result = dual_judge(
            judge_method="constraint_reasoning",
            response_text="answer",
            params={},
            run_semantic=True,
        )
    assert result.agreement is False
    assert result.tiebreak_used is True
    assert result.method == "dual_tiebreak"
    assert result.consensus_passed is True  # from tiebreaker
    assert tb_called["n"] == 1


def test_dual_judge_semantic_none_falls_back_to_rule():
    """If semantic judge returns None verdict, use rule judge result."""
    with patch("app.judge.dual_judge._run_rule", return_value=(True, {})), \
         patch("app.judge.dual_judge._run_semantic", return_value=(None, {"err": "x"})):
        result = dual_judge(
            judge_method="constraint_reasoning",
            response_text="x",
            params={},
        )
    assert result.consensus_passed is True
    assert result.method == "rule_only"
