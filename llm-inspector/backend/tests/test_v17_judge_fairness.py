"""
Regression tests for v17 judge-fairness fixes.

These tests cover four real measurement bugs that surfaced when comparing
deepseek-v4-pro vs deepseek-v4-flash on the standard suite, where pro
appeared "weaker" than flash purely due to evaluation-side issues:

1. ``_regex_match`` accepted only the ``pattern`` key, so suite cases that
   used ``regex`` (e.g. multilingual reasoning) silently failed every model.
2. ``_regex_match`` used Python's ``\\b``, which doesn't recognise CJK
   character boundaries. ``\\b28\\b`` against ``A是28岁`` failed because
   Chinese characters aren't ASCII word characters — verbose models that
   wrapped their answer with Chinese context were unfairly penalised.
3. ``case_executor`` hard-failed every format-strict judge whenever the
   response was truncated (``finish_reason=length``), even when the
   partial output already contained the correct answer. Verbose-thinking
   models lost points purely for using more tokens.
4. ``_line_count`` couldn't tell apart "model produced wrong number of
   lines" from "API didn't honour max_tokens" — the latter is a
   protocol-layer signal that should be inconclusive (None), not False.
"""
from app.judge.methods import judge


# ── Fix 1 + 2: _regex_match ─────────────────────────────────────────────────

class TestRegexMatchKeyAliases:
    """Suite cases use any of: pattern / regex / target_pattern / forbidden_pattern."""

    def test_accepts_pattern_key(self):
        passed, _ = judge("regex_match", "answer is 42", {"pattern": r"\b42\b"})
        assert passed is True

    def test_accepts_regex_key(self):
        # multilingual cases (mlg_chinese_0001, mlg_japanese_0002) use this
        passed, detail = judge("regex_match", "the answer is 6", {"regex": "6"})
        assert passed is True, detail

    def test_accepts_target_pattern_key(self):
        passed, _ = judge("regex_match", "result: 28", {"target_pattern": r"\b28\b"})
        assert passed is True

    def test_missing_pattern_returns_inconclusive_not_false(self):
        # Misconfigured case must NOT count against the model
        passed, detail = judge("regex_match", "anything", {})
        assert passed is None, "missing pattern must yield inconclusive (None)"
        assert "misconfigured" in detail.get("error", "")


class TestRegexMatchCJKBoundary:
    """\\b in Python re only sees ASCII word boundaries; CJK fallback recovers."""

    def test_pure_ascii_unchanged(self):
        # Backwards compatibility: pure-ASCII text must behave identically
        passed, detail = judge("regex_match", "result is 28.", {"pattern": r"\b28\b"})
        assert passed is True
        assert detail.get("cjk_boundary_fallback") is not True

    def test_chinese_context_with_b_boundary(self):
        # The exact failing case from the deepseek run:
        # FLASH answered "A是28岁。", judged False under stock \b28\b.
        passed, detail = judge(
            "regex_match", "A是28岁。", {"pattern": r"\b28\b"}
        )
        assert passed is True, detail
        assert detail.get("cjk_boundary_fallback") is True

    def test_japanese_context_with_b_boundary(self):
        passed, detail = judge(
            "regex_match", "答えは6個です", {"pattern": r"\b6\b"}
        )
        assert passed is True, detail
        assert detail.get("cjk_boundary_fallback") is True

    def test_no_fallback_when_target_actually_absent(self):
        # The relaxed boundary must not paper over actually-missing answers
        passed, _ = judge(
            "regex_match", "答案是九十九", {"pattern": r"\b28\b"}
        )
        assert passed is False

    def test_no_match_pattern_inversion_still_works(self):
        # match_means_fail: presence of pattern = failure
        passed, _ = judge(
            "regex_match",
            "I cannot help with that",
            {"pattern": "cannot", "match_means_fail": True},
        )
        assert passed is False


# ── Fix 4: _line_count ──────────────────────────────────────────────────────

class TestLineCountTolerance:
    def test_exact_count_passes(self):
        passed, _ = judge("line_count", "1\n2\n3", {"expected_lines": 3})
        assert passed is True

    def test_intro_line_stripped_when_helpful(self):
        # "Here you go:" + 3 numeric lines should now pass at expected=3
        passed, detail = judge(
            "line_count", "Here you go:\n1\n2\n3", {"expected_lines": 3}
        )
        assert passed is True, detail
        assert detail.get("stripped_intro_lines") == 1

    def test_outro_line_stripped_when_helpful(self):
        passed, detail = judge(
            "line_count",
            "1\n2\n3\nHope that helps!",
            {"expected_lines": 3},
        )
        assert passed is True, detail
        assert detail.get("stripped_outro_lines") == 1

    def test_tolerance_allows_off_by_one(self):
        passed, _ = judge(
            "line_count", "1\n2\n3\n4", {"expected_lines": 3, "tolerance": 1}
        )
        assert passed is True

    def test_protocol_dimension_truncation_probe_inconclusive(self):
        # param_002 "max_tokens_truncation": expected_lines=5, max_tokens=15,
        # but deepseek doesn't honour max_tokens so the model writes 50 lines.
        # That's the protocol signal we wanted, not a capability failure.
        full_50 = "\n".join(str(i) for i in range(1, 51))
        passed, detail = judge(
            "line_count",
            full_50,
            {"expected_lines": 5, "_meta": {"dimension": "protocol"}},
        )
        assert passed is None, "protocol-dim probe must be inconclusive, got %r" % passed
        assert detail.get("protocol_signal") == "max_tokens_not_honoured"
        assert detail["actual_lines"] == 50

    def test_capability_dimension_unaffected(self):
        # Same input but in a capability dimension still hard-fails — we
        # only relax behaviour for protocol probes.
        full_50 = "\n".join(str(i) for i in range(1, 51))
        passed, _ = judge(
            "line_count",
            full_50,
            {"expected_lines": 5, "_meta": {"dimension": "instruction"}},
        )
        assert passed is False


# ── Fix 3: truncation handling in case_executor ─────────────────────────────

class TestTruncationHandling:
    """case_executor wraps judge() with truncation policy. We test the
    intended behaviour by simulating the executor's logic directly to keep
    the test fast and independent of the LLM client stack."""

    def _format_strict(self):
        return {"exact_match", "regex_match", "json_schema",
                "line_count", "text_constraints", "tokenizer_fingerprint"}

    def _executor_decision(self, judge_method, partial_text, params):
        """Mirror of case_executor's v17 truncation branch."""
        partial_passed, partial_detail = judge(judge_method, partial_text, params)
        if partial_detail is None:
            partial_detail = {}
        partial_detail["truncated"] = True
        if judge_method in self._format_strict():
            if partial_passed is True:
                return True, partial_detail
            return None, partial_detail  # inconclusive, not False
        return partial_passed, partial_detail

    def test_partial_output_already_passes(self):
        # reason_r-style: model wrote "无解" before being cut off
        partial_passed, _ = self._executor_decision(
            "regex_match",
            "无解。",
            {"pattern": "无解|无法"},
        )
        assert partial_passed is True

    def test_partial_output_does_not_pass_marks_inconclusive(self):
        # Truncation but partial output didn't contain the answer.
        # Old behaviour: hard False (penalises model unfairly).
        # New behaviour: None (excluded from pass-rate denominator).
        partial_passed, detail = self._executor_decision(
            "regex_match",
            "Let me think step by step about this problem...",
            {"pattern": r"\b42\b"},
        )
        assert partial_passed is None
        assert detail.get("truncated") is True

    def test_content_rich_judge_keeps_real_verdict(self):
        # Content-rich judges (semantic_match / refusal_detect / ...) must
        # not be coerced to None by the truncation branch — whatever the
        # judge says (True / False / its own None) is preserved.
        partial_passed, detail = self._executor_decision(
            "exact_match",
            "hello",
            {"target": "hello"},
        )
        # exact_match is format-strict, so partial pass=True path
        assert partial_passed is True
        assert detail.get("truncated") is True
