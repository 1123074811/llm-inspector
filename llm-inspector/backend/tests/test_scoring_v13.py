"""
tests/test_scoring_v13.py — v13 scoring: Stanine conversion + ScoreCard fields.
"""
from __future__ import annotations

from app.analysis.stanine import theta_to_stanine
from app.core.schemas import ScoreCard


def test_stanine_boundaries():
    """θ=0 → stanine=5 (mean ability band)."""
    assert theta_to_stanine(0.0) == 5


def test_stanine_low():
    """θ=-2.5 → stanine=1 (well below lowest boundary -1.75)."""
    assert theta_to_stanine(-2.5) == 1


def test_stanine_high():
    """θ=+2.5 → stanine=9 (well above highest boundary +1.75)."""
    assert theta_to_stanine(2.5) == 9


def test_stanine_monotonic():
    """Stanine must be non-decreasing in theta."""
    prev = 0
    for theta in [-3.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 3.0]:
        s = theta_to_stanine(theta)
        assert s >= prev
        assert 1 <= s <= 9
        prev = s


def test_scorecard_has_v13_fields():
    """ScoreCard() has stanine=None, percentile=None by default."""
    card = ScoreCard()
    assert hasattr(card, "stanine")
    assert hasattr(card, "percentile")
    assert card.stanine is None
    assert card.percentile is None


def test_scorecard_to_dict_has_v13_key():
    """to_dict() includes 'v13' key containing stanine + percentile."""
    card = ScoreCard()
    card.stanine = 5
    card.percentile = 50.0
    d = card.to_dict()
    assert "v13" in d
    assert d["v13"]["stanine"] == 5
    assert d["v13"]["percentile"] == 50.0
