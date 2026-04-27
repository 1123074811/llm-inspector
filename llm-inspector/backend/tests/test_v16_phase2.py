"""
test_v16_phase2.py — v16 Phase 2 regression tests.

Validates:
  - RetryConfig has 5xx/truncation/decode/jitter fields
  - Error classifiers work (_is_5xx, _is_truncation, _is_json_decode)
  - _next_max_tokens doubles correctly
  - _jitter adds randomization
  - SampleResult has excluded_from_scoring/retry_type
  - CaseResult.pass_rate excludes excluded_from_scoring
  - CaseResult has rescued field
  - _rescue_round function exists
"""
import pytest


class TestRetryConfigV16:
    def test_extended_fields(self):
        from app.runner.retry_policy import RetryConfig
        cfg = RetryConfig()
        assert cfg.max_retries_5xx == 3
        assert cfg.max_retries_truncation == 2
        assert cfg.max_retries_decode == 1
        assert cfg.jitter_ratio == 0.2

    def test_custom_config(self):
        from app.runner.retry_policy import RetryConfig
        cfg = RetryConfig(max_retries_5xx=5, max_retries_truncation=3, jitter_ratio=0.3)
        assert cfg.max_retries_5xx == 5
        assert cfg.max_retries_truncation == 3
        assert cfg.jitter_ratio == 0.3


class TestErrorClassifiersV16:
    def test_is_5xx_http_error(self):
        from app.runner.retry_policy import _is_5xx_error
        import urllib.error
        try:
            raise urllib.error.HTTPError("url", 500, "Internal Server Error", {}, None)
        except urllib.error.HTTPError as e:
            assert _is_5xx_error(e)

    def test_is_5xx_503(self):
        from app.runner.retry_policy import _is_5xx_error
        import urllib.error
        try:
            raise urllib.error.HTTPError("url", 503, "Service Unavailable", {}, None)
        except urllib.error.HTTPError as e:
            assert _is_5xx_error(e)

    def test_is_not_5xx_400(self):
        from app.runner.retry_policy import _is_5xx_error
        import urllib.error
        try:
            raise urllib.error.HTTPError("url", 400, "Bad Request", {}, None)
        except urllib.error.HTTPError as e:
            assert not _is_5xx_error(e)

    def test_is_truncation_error(self):
        from app.runner.retry_policy import _is_truncation_error

        class MockResp:
            finish_reason = "length"
        assert _is_truncation_error(MockResp())

    def test_is_not_truncation_error(self):
        from app.runner.retry_policy import _is_truncation_error

        class MockResp:
            finish_reason = "stop"
        assert not _is_truncation_error(MockResp())

    def test_is_json_decode_error(self):
        from app.runner.retry_policy import _is_json_decode_error
        try:
            import json
            json.loads("{invalid}")
        except json.JSONDecodeError as e:
            assert _is_json_decode_error(e)

    def test_is_not_json_decode_error(self):
        from app.runner.retry_policy import _is_json_decode_error
        assert not _is_json_decode_error(ValueError("some error"))


class TestNextMaxTokens:
    def test_doubles_on_first_attempt(self):
        from app.runner.retry_policy import _next_max_tokens
        assert _next_max_tokens(100, 1) == 200

    def test_doubles_on_second_attempt(self):
        from app.runner.retry_policy import _next_max_tokens
        assert _next_max_tokens(100, 2) == 400

    def test_capped_at_4096(self):
        from app.runner.retry_policy import _next_max_tokens
        assert _next_max_tokens(4096, 1) == 4096
        assert _next_max_tokens(3000, 2) == 4096


class TestJitter:
    def test_jitter_adds_randomization(self):
        from app.runner.retry_policy import _jitter
        base = 1.0
        results = [_jitter(base, 0.2) for _ in range(100)]
        # Should be in range [0.9, 1.1] with ratio=0.2
        assert all(0.85 <= r <= 1.15 for r in results)
        # Should not all be identical
        assert len(set(round(r, 6) for r in results)) > 1

    def test_jitter_zero_ratio(self):
        from app.runner.retry_policy import _jitter
        assert _jitter(1.0, 0.0) == 1.0


class TestSampleResultV16:
    def test_excluded_from_scoring_field(self):
        from app.core.schemas import SampleResult, LLMResponse
        sr = SampleResult(
            sample_index=0,
            response=LLMResponse(content="test"),
            excluded_from_scoring=True,
            retry_type="truncation",
        )
        assert sr.excluded_from_scoring is True
        assert sr.retry_type == "truncation"

    def test_default_values(self):
        from app.core.schemas import SampleResult, LLMResponse
        sr = SampleResult(sample_index=0, response=LLMResponse(content="test"))
        assert sr.excluded_from_scoring is False
        assert sr.retry_type is None


class TestCaseResultV16:
    def test_pass_rate_excludes_excluded_samples(self):
        from app.core.schemas import TestCase, CaseResult, SampleResult, LLMResponse

        case = TestCase(id="t1", category="test", name="test", user_prompt="hi",
                        expected_type="text", judge_method="exact_match")
        cr = CaseResult(case=case)

        # One passing, one excluded
        cr.samples.append(SampleResult(
            sample_index=0, response=LLMResponse(content="ok"),
            judge_passed=True,
        ))
        cr.samples.append(SampleResult(
            sample_index=1, response=LLMResponse(content="bad"),
            judge_passed=False, excluded_from_scoring=True,
        ))
        # pass_rate should only count the non-excluded sample
        assert cr.pass_rate == 1.0

    def test_rescued_field(self):
        from app.core.schemas import TestCase, CaseResult
        case = TestCase(id="t1", category="test", name="test", user_prompt="hi",
                        expected_type="text", judge_method="exact_match")
        cr = CaseResult(case=case, rescued=True)
        assert cr.rescued is True

    def test_default_rescued_false(self):
        from app.core.schemas import TestCase, CaseResult
        case = TestCase(id="t1", category="test", name="test", user_prompt="hi",
                        expected_type="text", judge_method="exact_match")
        cr = CaseResult(case=case)
        assert cr.rescued is False


class TestRescueRound:
    def test_rescue_round_function_exists(self):
        from app.runner.case_executor import _rescue_round
        assert callable(_rescue_round)

    def test_rescue_round_no_rescue_needed(self):
        from app.runner.case_executor import _rescue_round
        from app.core.schemas import TestCase, CaseResult, SampleResult, LLMResponse

        case = TestCase(id="t1", category="test", name="test", user_prompt="hi",
                        expected_type="text", judge_method="exact_match")
        cr = CaseResult(case=case)
        cr.samples.append(SampleResult(
            sample_index=0, response=LLMResponse(content="ok"),
            judge_passed=True,
        ))
        # Should return unchanged — no rescue needed
        result = _rescue_round(None, "test-model", case, cr)
        assert result is cr  # Same object, no rescue attempted
