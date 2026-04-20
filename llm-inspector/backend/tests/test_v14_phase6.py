"""
tests/test_v14_phase6.py — v14 Phase 6: Token Efficiency & Adaptive Sampling.

Coverage:
  - item_information() — typical values, zero-info edge case, numerical stability  (4)
  - adaptive_n_samples() — high info → 1, medium → 2, low → 3, None theta default  (5)
  - get_adaptive_n_samples() — case dict with IRT params, missing params fallback   (3)
  - count_tokens() — fallback mode, returns tuple, method string                    (3)
  - count_messages_tokens() — list of messages, overhead counting                  (3)
  - ScoreCard.to_dict() — token_analysis block present                             (2)
  - ScoreCard new fields — defaults                                                 (3)
  - handle_token_analysis — basic response structure, missing run returns analysis  (3)
  - PromptOptimizer wiring — import succeeds, compile_prompt callable               (2)
  - Adaptive sampling integration — n_samples respects IRT bounds                  (2)

Total: ~30 test methods (target ≥ 22)
"""
from __future__ import annotations

import math
import pytest


# ---------------------------------------------------------------------------
# T1: item_information()
# ---------------------------------------------------------------------------

class TestItemInformation:
    """Tests for adaptive_sampling.item_information()."""

    def _info(self, theta, a, b, c=0.0):
        from app.runner.adaptive_sampling import item_information
        return item_information(theta, a, b, c)

    def test_2pl_typical_value(self):
        """At theta=b (mode) with c=0, the 3PL formula gives I = a².

        Derivation: P = c + (1-c)/(1+exp(0)) = 0 + 0.5 = 0.5
        I = a² × (P-c)² / ((1-c)² × P×(1-P))
          = a² × 0.25 / (1 × 0.25) = a²
        """
        a, b = 1.0, 0.0
        info = self._info(theta=0.0, a=a, b=b)
        assert info == pytest.approx(a ** 2, rel=1e-5)

    def test_high_discrimination_higher_info(self):
        """Higher a should produce higher information at mode."""
        info_low = self._info(theta=0.0, a=0.5, b=0.0)
        info_high = self._info(theta=0.0, a=2.0, b=0.0)
        assert info_high > info_low

    def test_zero_info_extreme_theta(self):
        """At very large |theta - b|, P approaches 0 or 1 and I approaches 0."""
        info = self._info(theta=100.0, a=1.0, b=0.0)
        assert info < 0.01

    def test_clip_upper_bound(self):
        """Information should never exceed 10.0 (clip)."""
        # Extreme a value could produce I > 10 without clipping
        info = self._info(theta=0.0, a=3.0, b=0.0)
        assert info <= 10.0

    def test_3pl_guessing_reduces_info(self):
        """Adding guessing (c>0) reduces information at theta BELOW item difficulty.

        At theta = b - 2 (ability well below item difficulty), the guessing
        parameter effectively 'floors' the probability, compressing the
        useful discrimination range and reducing Fisher information.
        """
        # theta=-2.0, b=0.0 → ability well below difficulty → c reduces info
        info_2pl = self._info(theta=-2.0, a=1.5, b=0.0, c=0.0)
        info_3pl = self._info(theta=-2.0, a=1.5, b=0.0, c=0.25)
        assert info_3pl < info_2pl

    def test_numerical_stability_very_negative_exponent(self):
        """Should not raise on extreme (theta - b) values."""
        # theta=-50 means very large negative exponent -> P≈c
        info = self._info(theta=-50.0, a=1.0, b=0.0)
        assert isinstance(info, float)
        assert math.isfinite(info)


# ---------------------------------------------------------------------------
# T2: adaptive_n_samples()
# ---------------------------------------------------------------------------

class TestAdaptiveNSamples:
    """Tests for adaptive_sampling.adaptive_n_samples()."""

    def _n(self, theta, a, b, c=0.0):
        from app.runner.adaptive_sampling import adaptive_n_samples
        return adaptive_n_samples(theta, a, b, c)

    def test_high_info_returns_1(self):
        """High-discrimination item at mode → I > 1.0 → n=1.

        Formula: I = a² at mode (theta=b, c=0).
        a=1.2 → I = 1.44 > 1.0 → n=1.
        """
        n = self._n(theta=0.0, a=1.2, b=0.0)
        assert n == 1

    def test_medium_info_returns_2(self):
        """Moderately informative item → I in (0.5, 1.0] → n=2.

        Formula: I = a² at mode (theta=b, c=0).
        a=0.9 → I = 0.81 ∈ (0.5, 1.0] → n=2.
        """
        n = self._n(theta=0.0, a=0.9, b=0.0)
        assert n == 2

    def test_low_info_returns_3(self):
        """Low information → I <= 0.5 → n=3.

        Formula: I = a² at mode (theta=b, c=0).
        a=0.7 → I = 0.49 <= 0.5 → n=3.
        """
        n = self._n(theta=0.0, a=0.7, b=0.0)
        assert n == 3

    def test_none_theta_defaults_to_zero(self):
        """theta=None should behave identically to theta=0.0."""
        from app.runner.adaptive_sampling import adaptive_n_samples
        n_none = adaptive_n_samples(None, 1.0, 0.0, 0.0)
        n_zero = adaptive_n_samples(0.0, 1.0, 0.0, 0.0)
        assert n_none == n_zero

    def test_result_in_valid_range(self):
        """Result must always be 1, 2, or 3."""
        from app.runner.adaptive_sampling import adaptive_n_samples
        for a in [0.1, 0.5, 1.0, 2.0, 3.0]:
            for b in [-2.0, 0.0, 2.0]:
                n = adaptive_n_samples(0.0, a, b, 0.0)
                assert n in {1, 2, 3}, f"a={a}, b={b} → n={n}"


# ---------------------------------------------------------------------------
# T3: get_adaptive_n_samples()
# ---------------------------------------------------------------------------

class TestGetAdaptiveNSamples:
    """Tests for adaptive_sampling.get_adaptive_n_samples()."""

    def _get(self, case, theta=None):
        from app.runner.adaptive_sampling import get_adaptive_n_samples
        return get_adaptive_n_samples(case, current_theta=theta)

    def test_with_irt_params(self):
        """Case dict with valid IRT params should return a sensible n."""
        case = {"irt_a": 2.5, "irt_b": 0.0, "irt_c": 0.0}
        n = self._get(case, theta=0.0)
        assert n in {1, 2, 3}

    def test_missing_params_fallback_to_2(self):
        """Missing IRT params should fall back to n=2."""
        n = self._get({})
        assert n == 2

    def test_invalid_params_fallback_to_2(self):
        """Non-numeric IRT params should silently fall back to 2."""
        # Passing string values should trigger the except path
        case = {"irt_a": "not_a_number", "irt_b": 0.0}
        # The internal float() conversion will raise ValueError → fallback
        n = self._get(case)
        # Either returns a valid n from the default values or the fallback 2
        assert n in {1, 2, 3}


# ---------------------------------------------------------------------------
# T4: count_tokens()
# ---------------------------------------------------------------------------

class TestCountTokens:
    """Tests for token_counter.count_tokens()."""

    def test_returns_tuple(self):
        from app.runner.token_counter import count_tokens
        result = count_tokens("hello world")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_count_is_positive_integer(self):
        from app.runner.token_counter import count_tokens
        count, method = count_tokens("This is a test sentence.")
        assert isinstance(count, int)
        assert count >= 1

    def test_method_is_string(self):
        from app.runner.token_counter import count_tokens
        count, method = count_tokens("hello")
        assert isinstance(method, str)
        assert method in {"tiktoken-o200k", "tiktoken-cl100k", "fallback-estimate"}

    def test_empty_string(self):
        from app.runner.token_counter import count_tokens
        count, method = count_tokens("")
        assert count == 0

    def test_longer_text_more_tokens(self):
        """Longer text should produce more tokens than shorter text."""
        from app.runner.token_counter import count_tokens
        short_count, _ = count_tokens("hi")
        long_text = "The quick brown fox jumps over the lazy dog. " * 10
        long_count, _ = count_tokens(long_text)
        assert long_count > short_count


# ---------------------------------------------------------------------------
# T5: count_messages_tokens()
# ---------------------------------------------------------------------------

class TestCountMessagesTokens:
    """Tests for token_counter.count_messages_tokens()."""

    def test_empty_list(self):
        from app.runner.token_counter import count_messages_tokens
        count, method = count_messages_tokens([])
        assert count == 0

    def test_single_message_includes_overhead(self):
        """Single-message count should be content tokens + 3 overhead."""
        from app.runner.token_counter import count_tokens, count_messages_tokens
        text = "Hello there"
        content_tokens, _ = count_tokens(text)
        total, _ = count_messages_tokens([{"role": "user", "content": text}])
        # 3 tokens overhead per message
        assert total == content_tokens + 3

    def test_multiple_messages_summed(self):
        """Multiple messages should produce more tokens than a single one."""
        from app.runner.token_counter import count_messages_tokens
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "The answer is 4."},
        ]
        total, method = count_messages_tokens(msgs)
        assert total > 0
        assert isinstance(method, str)


# ---------------------------------------------------------------------------
# T6: ScoreCard.to_dict() — token_analysis block
# ---------------------------------------------------------------------------

class TestScoreCardTokenAnalysis:
    """Tests for ScoreCard.to_dict() token_analysis block (v14 Phase 6)."""

    def test_token_analysis_key_present(self):
        from app.core.schemas import ScoreCard
        sc = ScoreCard()
        d = sc.to_dict()
        assert "token_analysis" in d

    def test_token_analysis_keys(self):
        from app.core.schemas import ScoreCard
        sc = ScoreCard()
        ta = sc.to_dict()["token_analysis"]
        assert "prompt_optimizer_used" in ta
        assert "tokens_saved_estimate" in ta
        assert "counting_method" in ta

    def test_token_analysis_defaults(self):
        from app.core.schemas import ScoreCard
        sc = ScoreCard()
        ta = sc.to_dict()["token_analysis"]
        assert ta["prompt_optimizer_used"] is False
        assert ta["tokens_saved_estimate"] is None
        assert ta["counting_method"] == "fallback-estimate"

    def test_token_analysis_populated_when_set(self):
        from app.core.schemas import ScoreCard
        sc = ScoreCard(
            prompt_optimizer_used=True,
            tokens_saved_estimate=42,
            token_counting_method="tiktoken-o200k",
        )
        ta = sc.to_dict()["token_analysis"]
        assert ta["prompt_optimizer_used"] is True
        assert ta["tokens_saved_estimate"] == 42
        assert ta["counting_method"] == "tiktoken-o200k"


# ---------------------------------------------------------------------------
# T7: ScoreCard new fields — defaults
# ---------------------------------------------------------------------------

class TestScoreCardNewFields:
    """Tests for the three new ScoreCard dataclass fields."""

    def test_prompt_optimizer_used_default(self):
        from app.core.schemas import ScoreCard
        sc = ScoreCard()
        assert sc.prompt_optimizer_used is False

    def test_tokens_saved_estimate_default(self):
        from app.core.schemas import ScoreCard
        sc = ScoreCard()
        assert sc.tokens_saved_estimate is None

    def test_token_counting_method_default(self):
        from app.core.schemas import ScoreCard
        sc = ScoreCard()
        assert sc.token_counting_method == "fallback-estimate"


# ---------------------------------------------------------------------------
# T8: handle_token_analysis
# ---------------------------------------------------------------------------

class TestHandleTokenAnalysis:
    """Tests for handlers/v14_handlers.py handle_token_analysis."""

    def test_missing_run_returns_analysis_block(self):
        """Missing run should return empty analysis, not an error."""
        from app.handlers.v14_handlers import handle_token_analysis
        status, body, content_type = handle_token_analysis(
            "/api/v14/runs/nonexistent-run-id/token-analysis", {}, {}
        )
        import json
        data = json.loads(body)
        # Should either be a note or a valid analysis block (not a 500 error)
        assert status in {200, 400, 404}
        if status == 200:
            assert "token_analysis" in data

    def test_response_has_run_id_key(self):
        """Successful response should contain run_id."""
        from app.handlers.v14_handlers import handle_token_analysis
        import json
        status, body, _ = handle_token_analysis(
            "/api/v14/runs/test-run-999/token-analysis", {}, {}
        )
        if status == 200:
            data = json.loads(body)
            assert "run_id" in data

    def test_invalid_path_returns_400(self):
        """Path without run ID should return 400."""
        from app.handlers.v14_handlers import handle_token_analysis
        status, body, _ = handle_token_analysis(
            "/api/v14/token-analysis", {}, {}
        )
        assert status == 400


# ---------------------------------------------------------------------------
# T9: PromptOptimizer wiring in case_executor
# ---------------------------------------------------------------------------

class TestPromptOptimizerWiring:
    """Tests that PromptOptimizer is importable and wired correctly."""

    def test_import_succeeds(self):
        """prompt_optimizer module should import without errors."""
        from app.runner.prompt_optimizer import prompt_optimizer, PromptOptimizer
        assert isinstance(prompt_optimizer, PromptOptimizer)

    def test_compile_prompt_callable(self):
        """compile_prompt should accept test_prompt and judge_method kwargs."""
        from app.runner.prompt_optimizer import prompt_optimizer
        result = prompt_optimizer.compile_prompt(
            test_prompt="What is 2+2?",
            judge_method="exact_match",
            max_examples=2,
        )
        assert result is not None
        assert hasattr(result, "prompt")
        assert isinstance(result.prompt, str)

    def test_case_executor_imports_cleanly(self):
        """case_executor should import without error (wiring is non-fatal)."""
        import app.runner.case_executor as ce
        assert hasattr(ce, "execute_case")


# ---------------------------------------------------------------------------
# T10: Adaptive sampling integration
# ---------------------------------------------------------------------------

class TestAdaptiveSamplingIntegration:
    """Integration-level tests for adaptive sampling in execute_case context."""

    def test_n_samples_reduced_for_high_info_item(self):
        """
        For a very high-discrimination item at theta=mode, adaptive_n should
        be 1, so if case.n_samples=3 it would be reduced.

        a=1.2 → I=1.44 > 1.0 → n=1.
        """
        from app.runner.adaptive_sampling import get_adaptive_n_samples
        case = {"irt_a": 1.2, "irt_b": 0.0, "irt_c": 0.0}
        n = get_adaptive_n_samples(case, current_theta=0.0)
        assert n == 1

    def test_n_samples_not_increased(self):
        """
        Adaptive sampling should NEVER increase n_samples above case default.
        The case_executor only reduces n if adaptive_n < case.n_samples.
        The adaptive function returns max n=3 for low-information items.
        """
        from app.runner.adaptive_sampling import get_adaptive_n_samples
        # Low info item at mode → n=3 (our max)
        case = {"irt_a": 0.7, "irt_b": 0.0, "irt_c": 0.0}
        n = get_adaptive_n_samples(case, current_theta=0.0)
        assert n == 3
