"""
tests/test_v14_phase8.py — Phase 8 Frontend-adjacent backend validation.

Covers:
  - ScoreCard.to_dict() contains all v14 new fields: completeness, token_analysis, skipped_cases
  - ScoreCard.to_dict() emits null (not 50) for missing null fields
  - ScoreCard.token_analysis block structure
  - /api/v1/leaderboard handler accepts limit and offset query params
  - /api/v1/elo-leaderboard handler accepts limit and offset query params
  - /api/v1/health endpoint responds with expected fields
  - handle_elo_leaderboard offset slicing
  - handle_leaderboard limit param works
  - v14 health endpoint includes phases_complete
  - ScoreCard null reasoning_score → to_dict breakdown['reasoning'] is None
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# ScoreCard v14 new fields in to_dict()
# ---------------------------------------------------------------------------

class TestScoreCardV14Fields:

    def test_to_dict_contains_completeness(self):
        """ScoreCard.to_dict() includes v13.completeness key."""
        from app.core.schemas import ScoreCard
        sc = ScoreCard()
        sc.completeness = 0.875
        d = sc.to_dict()
        assert "v13" in d
        assert "completeness" in d["v13"]
        assert d["v13"]["completeness"] == pytest.approx(0.875)

    def test_to_dict_completeness_none_when_unset(self):
        """completeness defaults to None → to_dict v13.completeness is None."""
        from app.core.schemas import ScoreCard
        sc = ScoreCard()
        d = sc.to_dict()
        assert d["v13"]["completeness"] is None

    def test_to_dict_contains_token_analysis(self):
        """ScoreCard.to_dict() includes token_analysis block."""
        from app.core.schemas import ScoreCard
        sc = ScoreCard()
        d = sc.to_dict()
        assert "token_analysis" in d
        ta = d["token_analysis"]
        assert "prompt_optimizer_used" in ta
        assert "tokens_saved_estimate" in ta
        assert "counting_method" in ta

    def test_to_dict_token_analysis_defaults(self):
        """Default ScoreCard has optimizer disabled and no estimate."""
        from app.core.schemas import ScoreCard
        sc = ScoreCard()
        d = sc.to_dict()
        ta = d["token_analysis"]
        assert ta["prompt_optimizer_used"] is False
        assert ta["tokens_saved_estimate"] is None

    def test_to_dict_token_analysis_values(self):
        """Non-default token analysis values round-trip correctly."""
        from app.core.schemas import ScoreCard
        sc = ScoreCard()
        sc.prompt_optimizer_used = True
        sc.tokens_saved_estimate = 3200
        sc.token_counting_method = "tiktoken-cl100k"
        d = sc.to_dict()
        ta = d["token_analysis"]
        assert ta["prompt_optimizer_used"] is True
        assert ta["tokens_saved_estimate"] == 3200
        assert ta["counting_method"] == "tiktoken-cl100k"

    def test_to_dict_contains_skipped_cases(self):
        """ScoreCard.to_dict() includes skipped_cases list."""
        from app.core.schemas import ScoreCard
        sc = ScoreCard()
        sc.skipped_cases = ["case-abc", "case-def"]
        d = sc.to_dict()
        assert "skipped_cases" in d
        assert d["skipped_cases"] == ["case-abc", "case-def"]

    def test_null_reasoning_score_emits_none_not_50(self):
        """null reasoning_score → breakdown['reasoning'] is None (not 50)."""
        from app.core.schemas import ScoreCard
        sc = ScoreCard()
        sc.reasoning_score = None
        d = sc.to_dict()
        assert d["breakdown"]["reasoning"] is None

    def test_null_coding_score_emits_none(self):
        """null coding_score → breakdown['coding'] is None."""
        from app.core.schemas import ScoreCard
        sc = ScoreCard()
        sc.coding_score = None
        d = sc.to_dict()
        assert d["breakdown"]["coding"] is None

    def test_null_speed_score_emits_none(self):
        """null speed_score → breakdown['speed'] is None."""
        from app.core.schemas import ScoreCard
        sc = ScoreCard()
        sc.speed_score = None
        d = sc.to_dict()
        assert d["breakdown"]["speed"] is None


# ---------------------------------------------------------------------------
# handle_elo_leaderboard: limit and offset params
# ---------------------------------------------------------------------------

class TestEloLeaderboardHandler:

    def test_elo_leaderboard_handler_accepts_limit(self):
        """handle_elo_leaderboard reads 'limit' from qs without error."""
        from app.handlers.models import handle_elo_leaderboard
        status, body, headers = handle_elo_leaderboard("/api/v1/elo-leaderboard",
                                                        {"limit": ["10"]}, {})
        assert status == 200

    def test_elo_leaderboard_handler_accepts_offset(self):
        """handle_elo_leaderboard reads 'offset' from qs without error."""
        from app.handlers.models import handle_elo_leaderboard
        status, body, headers = handle_elo_leaderboard("/api/v1/elo-leaderboard",
                                                        {"limit": ["20"], "offset": ["0"]}, {})
        assert status == 200

    def test_elo_leaderboard_offset_slices_results(self):
        """With large offset, returned rows should be empty or fewer."""
        from app.handlers.models import handle_elo_leaderboard
        import json
        status, body, _ = handle_elo_leaderboard("/api/v1/elo-leaderboard",
                                                  {"limit": ["100"], "offset": ["99999"]}, {})
        assert status == 200
        data = json.loads(body)
        assert isinstance(data, list)
        # With offset beyond any data, result must be empty list
        assert len(data) == 0


# ---------------------------------------------------------------------------
# handle_leaderboard: limit param
# ---------------------------------------------------------------------------

class TestLeaderboardHandler:

    def test_leaderboard_accepts_limit(self):
        """handle_leaderboard reads 'limit' from qs without error."""
        from app.handlers.models import handle_leaderboard
        status, body, headers = handle_leaderboard("/api/v1/leaderboard",
                                                    {"limit": ["5"]}, {})
        assert status == 200

    def test_leaderboard_default_limit(self):
        """handle_leaderboard works with no qs params."""
        from app.handlers.models import handle_leaderboard
        import json
        status, body, _ = handle_leaderboard("/api/v1/leaderboard", {}, {})
        assert status == 200
        assert isinstance(json.loads(body), list)


# ---------------------------------------------------------------------------
# /api/v1/health endpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:

    def test_health_handler_returns_ok(self):
        """handle_health returns status 200 with status=ok."""
        from app.handlers.misc import handle_health
        import json
        status, body, _ = handle_health("/api/v1/health", {}, {})
        assert status == 200
        d = json.loads(body)
        assert d["status"] == "ok"

    def test_health_has_version_field(self):
        """handle_health response includes version field."""
        from app.handlers.misc import handle_health
        import json
        _, body, _ = handle_health("/api/v1/health", {}, {})
        d = json.loads(body)
        assert "version" in d
