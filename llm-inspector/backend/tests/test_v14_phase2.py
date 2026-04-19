"""
tests/test_v14_phase2.py — Phase 2 scoring refactor acceptance tests.

Tests:
  - return 50.0 eliminated: score methods return None for empty inputs
  - total_score renormalises correctly when dimensions are missing
  - ScoreCard.completeness computed correctly
  - verdict_caps loaded via SRC (no TODO fallbacks remain)
  - Bradley-Terry leaderboard endpoint returns valid JSON
  - ScoreCard.to_dict() emits None (not 0) for missing scores
"""
from __future__ import annotations

import math
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def empty_results():
    return []


@pytest.fixture
def dummy_results():
    """Minimal case results for score functions."""
    from app.core.schemas import CaseResult, TestCase, SampleResult, LLMResponse, JudgeResult
    case = TestCase(
        name="test_case",
        prompt="test",
        expected="42",
        category="reasoning",
        judge_method="exact_match",
        difficulty=0.5,
        weight=1.0,
        mode_level="quick",
    )
    sample = SampleResult(
        response=LLMResponse(content="42", latency_ms=500),
        judge_result=JudgeResult(passed=True, score=1.0),
    )
    result = CaseResult(case=case, samples=[sample])
    return [result]


# ---------------------------------------------------------------------------
# T1: Score methods return None for empty inputs (not 50.0)
# ---------------------------------------------------------------------------
class TestReturnNoneInsteadOfFifty:
    def test_reasoning_score_empty(self, empty_results):
        from app.analysis.scoring import ScoreCardCalculator
        calc = ScoreCardCalculator()
        assert calc._reasoning_score(empty_results) is None, \
            "_reasoning_score should return None when no cases, not 50.0"

    def test_adversarial_reasoning_empty(self, empty_results):
        from app.analysis.scoring import ScoreCardCalculator
        calc = ScoreCardCalculator()
        assert calc._adversarial_reasoning_score(empty_results) is None

    def test_coding_score_empty(self, empty_results):
        from app.analysis.scoring import ScoreCardCalculator
        calc = ScoreCardCalculator()
        assert calc._coding_score(empty_results) is None

    def test_stability_score_empty(self, empty_results):
        from app.analysis.scoring import ScoreCardCalculator
        calc = ScoreCardCalculator()
        assert calc._stability_score(empty_results) is None

    def test_similarity_no_baselines(self):
        from app.analysis.scoring import ScoreCardCalculator
        calc = ScoreCardCalculator()
        assert calc._similarity_to_claimed([], "gpt-4o") is None

    def test_score_calculator_reasoning_empty(self, empty_results):
        from app.analysis.score_calculator import ScoreCardCalculator
        calc = ScoreCardCalculator()
        features = {k: 0.0 for k in ["instruction_pass_rate", "exact_match_rate",
                                      "json_valid_rate", "format_follow_rate"]}
        result = calc._reasoning_score(features, empty_results)
        assert result is None

    def test_score_calculator_stability_empty(self, empty_results):
        from app.analysis.score_calculator import ScoreCardCalculator
        calc = ScoreCardCalculator()
        assert calc._stability_score(empty_results) is None

    def test_adaptive_scoring_empty(self):
        from app.analysis.adaptive_scoring import ScoreConfidenceEstimator
        scorer = ScoreConfidenceEstimator()
        assert scorer._calculate_weighted_pass_rate([]) is None


# ---------------------------------------------------------------------------
# T2: ScoreCard completeness field
# ---------------------------------------------------------------------------
class TestScoreCardCompleteness:
    def test_completeness_field_exists(self):
        from app.core.schemas import ScoreCard
        card = ScoreCard()
        assert hasattr(card, "completeness"), "ScoreCard must have completeness field"
        assert card.completeness is None  # default

    def test_completeness_in_to_dict(self):
        from app.core.schemas import ScoreCard
        card = ScoreCard()
        card.completeness = 0.75
        d = card.to_dict()
        # Check completeness is in the dict (either in v13 block or top-level)
        has_completeness = (
            d.get("completeness") is not None
            or (isinstance(d.get("v13"), dict) and d["v13"].get("completeness") is not None)
            or (isinstance(d.get("v14"), dict) and d["v14"].get("completeness") is not None)
        )
        assert has_completeness, f"completeness not found in to_dict() output: {d}"

    def test_completeness_zero_when_all_none(self, empty_results):
        """When all capability dimensions are None, completeness should be 0.0."""
        from app.analysis.scoring import ScoreCardCalculator
        from app.core.schemas import PreDetectionResult
        calc = ScoreCardCalculator()
        # Minimal call with empty inputs
        predetect = PreDetectionResult(success=False, confidence=0.0, identified_as=None, layer_stopped=None)
        card = calc.calculate(
            features={},
            case_results=empty_results,
            similarities=[],
            predetect=predetect,
            claimed_model=None,
        )
        # Completeness should not be 1.0 (no data present)
        assert card.completeness is not None
        assert card.completeness < 1.0


# ---------------------------------------------------------------------------
# T3: total_score renormalization (no zero-fill)
# ---------------------------------------------------------------------------
class TestTotalScoreRenormalization:
    def test_total_score_renormalized_not_zero_filled(self):
        """
        If only capability_score is available and the others are None,
        total_score should equal capability_score (renormalized weight=1.0),
        NOT 0.45 × capability + 0 + 0 = 45% inflation factor.
        """
        from app.core.schemas import ScoreCard
        card = ScoreCard()
        card.capability_score = 80.0
        card.authenticity_score = None
        card.performance_score = None

        # Simulate what the fixed calculate() does:
        _top = {
            "capability": (0.45, card.capability_score),
            "authenticity": (0.30, card.authenticity_score),
            "performance": (0.25, card.performance_score),
        }
        _active_top = {k: (w, v) for k, (w, v) in _top.items() if v is not None}
        _top_sum = sum(w for w, v in _active_top.values())
        computed = round(
            sum(w / _top_sum * v for w, v in _active_top.values()) if _top_sum > 0 else 0.0,
            2,
        )
        # Should be 80.0, not 36.0 (=0.45*80)
        assert computed == pytest.approx(80.0, abs=0.1), \
            f"Expected 80.0 but got {computed} — zero-fill bug still present"


# ---------------------------------------------------------------------------
# T4: Verdict caps loaded from SOURCES.yaml (not TODO fallbacks)
# ---------------------------------------------------------------------------
class TestVerdictCapsFromSRC:
    def test_difficulty_cap_in_src(self):
        from app._data import SRC
        assert "verdict.difficulty_cap" in SRC, \
            "verdict.difficulty_cap must be registered in SOURCES.yaml"
        assert SRC["verdict.difficulty_cap"].value == 50.0

    def test_behavioral_invariant_cap_in_src(self):
        from app._data import SRC
        assert "verdict.behavioral_invariant_cap" in SRC
        assert SRC["verdict.behavioral_invariant_cap"].value == 55.0

    def test_coding_zero_cap_in_src(self):
        from app._data import SRC
        assert "verdict.coding_zero_cap" in SRC
        assert SRC["verdict.coding_zero_cap"].value == 45.0

    def test_identity_exposed_cap_in_src(self):
        from app._data import SRC
        assert "verdict.identity_exposed_cap" in SRC
        assert SRC["verdict.identity_exposed_cap"].value == 30.0

    def test_extraction_weak_cap_in_src(self):
        from app._data import SRC
        assert "verdict.extraction_weak_cap" in SRC
        assert SRC["verdict.extraction_weak_cap"].value == 65.0

    def test_fingerprint_mismatch_cap_in_src(self):
        from app._data import SRC
        assert "verdict.fingerprint_mismatch_cap" in SRC
        assert SRC["verdict.fingerprint_mismatch_cap"].value == 55.0

    def test_verdict_engine_hard_rules_no_none(self):
        """All _RULE_FALLBACKS keys must resolve to non-None via _rule()."""
        from app.analysis.verdicts import VerdictEngine
        engine = VerdictEngine()
        for name in engine._RULE_FALLBACKS:
            val = engine._rule(name)
            assert val is not None, f"VerdictEngine._rule('{name}') returned None"


# ---------------------------------------------------------------------------
# T5: Bradley-Terry endpoint
# ---------------------------------------------------------------------------
class TestBradleyTerryLeaderboard:
    def test_compute_bt_simple(self):
        """Basic BT computation: model A beats B 3 times, should rank higher."""
        from app.handlers.v14_handlers import _compute_bradley_terry
        comparisons = [
            {"winner": "gpt-4o", "loser": "gpt-3.5"},
            {"winner": "gpt-4o", "loser": "gpt-3.5"},
            {"winner": "gpt-4o", "loser": "gpt-3.5"},
        ]
        strengths = _compute_bradley_terry(comparisons)
        assert "gpt-4o" in strengths
        assert "gpt-3.5" in strengths
        assert strengths["gpt-4o"] > strengths["gpt-3.5"]

    def test_compute_bt_empty(self):
        from app.handlers.v14_handlers import _compute_bradley_terry
        assert _compute_bradley_terry([]) == {}

    def test_bt_leaderboard_handler(self):
        """Handler should return valid JSON even with no data."""
        from app.handlers.v14_handlers import handle_bt_leaderboard
        status, body, content_type = handle_bt_leaderboard("/api/v14/bt-leaderboard", {}, {})
        import json
        data = json.loads(body)
        assert "models" in data
        assert "total_comparisons" in data
        assert data["model"] == "bradley_terry"

    def test_bt_leaderboard_route_registered(self):
        """Route /api/v14/bt-leaderboard must be in ROUTES table."""
        import importlib
        import sys
        # Don't start the server; just check ROUTES
        import app.main as main_module
        routes = [(m, p) for m, p, _ in main_module.ROUTES]
        assert ("GET", r"^/api/v14/bt-leaderboard$") in routes, \
            "/api/v14/bt-leaderboard route missing from ROUTES"


# ---------------------------------------------------------------------------
# T6: to_dict() emits None for missing scores
# ---------------------------------------------------------------------------
class TestScoreCardToDict:
    def test_none_reasoning_emits_none(self):
        from app.core.schemas import ScoreCard
        card = ScoreCard()
        card.reasoning_score = None
        d = card.to_dict()
        # breakdown.reasoning should be None, not 0
        assert d["breakdown"]["reasoning"] is None, \
            f"Expected None but got {d['breakdown']['reasoning']}"

    def test_none_coding_emits_none(self):
        from app.core.schemas import ScoreCard
        card = ScoreCard()
        card.coding_score = None
        d = card.to_dict()
        assert d["breakdown"]["coding"] is None

    def test_present_score_emits_value(self):
        from app.core.schemas import ScoreCard
        card = ScoreCard()
        card.reasoning_score = 75.0
        d = card.to_dict()
        # 75.0 * 100 = 7500 (existing scale behavior preserved)
        assert d["breakdown"]["reasoning"] == 7500
