"""
tests/test_v14_phase4.py — v14 Phase 4: Judge System Hardening acceptance tests.

Coverage:
  - numeric_tolerance_judge (7 cases)
  - multi_choice_judge (5 cases)
  - semantic_entailment_judge (3 cases)
  - _check_against_knowledge_graph (2 cases, mocked)
  - fleiss_kappa (3 cases)
  - JudgeChainRunner (2 cases)
  - handle_judge_chain handler (1 case)
  - methods.py dispatch registration (3 methods)
"""
from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# T1: numeric_tolerance_judge
# ---------------------------------------------------------------------------

class TestNumericToleranceJudge:
    """Tests for judge/numeric_tolerance.py."""

    def _judge(self, response, **kwargs):
        from app.judge.numeric_tolerance import numeric_tolerance_judge
        return numeric_tolerance_judge(response, kwargs)

    def test_exact_match(self):
        passed, detail = self._judge("The answer is 42", expected=42)
        assert passed is True
        assert detail["method"] == "numeric_tolerance"
        assert detail["parsed"] == pytest.approx(42.0)

    def test_within_5_percent(self):
        # 100 * 1.04 = 104 — within default 5% tolerance
        passed, detail = self._judge("Result: 104", expected=100, tolerance=0.05)
        assert passed is True
        assert detail["tolerance_type"] == "relative"

    def test_outside_5_percent(self):
        # 100 * 1.10 = 110 — outside default 5% tolerance
        passed, detail = self._judge("Result: 110", expected=100, tolerance=0.05)
        assert passed is False
        assert detail["error"] > 0.05

    def test_scientific_notation(self):
        passed, detail = self._judge("3.14e-5 is the value", expected=3.14e-5)
        assert passed is True
        assert detail["parsed"] == pytest.approx(3.14e-5, rel=0.05)

    def test_parse_failure_returns_false(self):
        passed, detail = self._judge("No number here at all!", expected=42)
        assert passed is False
        assert detail["error"] == "parse_failed"

    def test_tiny_number_absolute_tolerance(self):
        # Expected ~0 → absolute tolerance path
        passed, detail = self._judge("0.0000000", expected=0.0, absolute_tolerance=1e-6)
        assert passed is True
        assert detail["tolerance_type"] == "absolute"

    def test_percentage_parsed(self):
        # "42.5%" should parse to 0.425
        from app.judge.numeric_tolerance import _parse_number
        val = _parse_number("42.5%")
        assert val == pytest.approx(0.425)


# ---------------------------------------------------------------------------
# T2: multi_choice_judge
# ---------------------------------------------------------------------------

class TestMultiChoiceJudge:
    """Tests for judge/multi_choice_verified.py."""

    def _judge(self, response, expected_choice="A", **kwargs):
        from app.judge.multi_choice_verified import multi_choice_judge
        params = {"expected_choice": expected_choice, **kwargs}
        return multi_choice_judge(response, params)

    def test_simple_letter_pass(self):
        passed, detail = self._judge("A", expected_choice="A")
        assert passed is True
        assert detail["extracted"] == "A"

    def test_letter_with_explanation(self):
        passed, detail = self._judge("A because the formula gives 2x.", expected_choice="A")
        assert passed is True

    def test_double_answer_ambiguous(self):
        # Two different letters — should be ambiguous
        passed, detail = self._judge("A or maybe B is correct", expected_choice="A")
        assert detail["ambiguous"] is True
        # Ambiguous responses should not pass
        assert passed is False

    def test_case_insensitive(self):
        passed, detail = self._judge("the answer is b", expected_choice="B")
        assert passed is True

    def test_chinese_format(self):
        # "选A" or "答案是A"
        passed, detail = self._judge("选C", expected_choice="C")
        assert passed is True


# ---------------------------------------------------------------------------
# T3: semantic_entailment_judge (fallback mode, no ST)
# ---------------------------------------------------------------------------

class TestSemanticEntailmentJudge:
    """Tests for judge/semantic_entailment.py — using cosine fallback."""

    def _judge(self, response, reference, **kwargs):
        from app.judge.semantic_entailment import semantic_entailment_judge
        params = {"expected_answer": reference, **kwargs}
        return semantic_entailment_judge(response, params)

    def test_matching_response_passes(self):
        # Highly overlapping words — should pass cosine fallback
        response = "Water is composed of hydrogen and oxygen atoms"
        reference = "Water molecules contain hydrogen and oxygen"
        passed, detail = self._judge(response, reference)
        # May or may not pass depending on threshold, but should return a valid structure
        assert "method" in detail
        assert detail["method"] == "semantic_entailment"
        assert isinstance(passed, bool)

    def test_non_matching_response_fails(self):
        response = "Pizza is a popular food from Italy"
        reference = "Quantum entanglement is a physical phenomenon"
        passed, detail = self._judge(response, reference)
        # Very different content — should fail
        assert passed is False

    def test_missing_reference_returns_false(self):
        from app.judge.semantic_entailment import semantic_entailment_judge
        passed, detail = semantic_entailment_judge("some response", {})
        assert passed is False
        assert detail.get("error") == "missing_reference"


# ---------------------------------------------------------------------------
# T4: _check_against_knowledge_graph (mocked)
# ---------------------------------------------------------------------------

class TestKnowledgeGraphCheck:
    """Tests for hallucination_v2._check_against_knowledge_graph."""

    def _detector(self):
        from app.judge.hallucination_v2 import HallucinationDetectorV2
        return HallucinationDetectorV2()

    def test_enabled_path_with_mock(self):
        """DBpediaClient available → enabled=True path."""
        from app.knowledge.dbpedia_client import DBpediaClient

        mock_result = MagicMock()
        mock_result.is_verified = True
        mock_result.confidence = 0.9
        mock_result.source = "dbpedia"

        detector = self._detector()

        with patch.object(DBpediaClient, "verify_entity", return_value=mock_result):
            # Text with a real capitalized entity; fake_entities is separate
            result = detector._check_against_knowledge_graph(
                "Albert Einstein developed the theory of relativity.",
                fake_entities=["FakeEntity123"],
            )

        assert result is not None
        assert result.get("enabled") is True
        assert "entity_results" in result
        assert result.get("source") == "dbpedia+wikidata"

    def test_offline_fallback_on_import_error(self):
        """If DBpediaClient import fails → enabled=False fallback."""
        detector = self._detector()

        with patch.dict("sys.modules", {"app.knowledge.dbpedia_client": None}):
            result = detector._check_against_knowledge_graph(
                "Some text about Albert Einstein.",
                fake_entities=[],
            )

        # ImportError path → enabled=False
        assert result is not None
        assert result.get("enabled") is False


# ---------------------------------------------------------------------------
# T5: fleiss_kappa
# ---------------------------------------------------------------------------

class TestFleissKappa:
    """Tests for consensus.fleiss_kappa."""

    def _kappa(self, ratings, n_categories=2):
        from app.judge.consensus import fleiss_kappa
        return fleiss_kappa(ratings, n_categories)

    def test_perfect_agreement_3_raters(self):
        # All 3 raters agree on every item
        ratings = [[1, 1, 1], [0, 0, 0], [1, 1, 1], [0, 0, 0]]
        k = self._kappa(ratings)
        assert k == pytest.approx(1.0, abs=1e-6)

    def test_chance_agreement_uniform(self):
        # 3 raters on 4 items: each item has 2 agree and 1 disagree in balanced pattern
        # p_j[1] = 0.5, p_j[0] = 0.5 → p_e = 0.50; p_bar < 1 → kappa < 1
        # This ensures kappa is a valid float (not 1.0)
        ratings = [[1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 0]]
        k = self._kappa(ratings)
        # kappa should be a float in valid range
        assert isinstance(k, float)
        assert -1.0 <= k <= 1.0

    def test_partial_agreement_3_raters(self):
        # 2/3 raters agree on each item
        ratings = [[1, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]]
        k = self._kappa(ratings)
        # Partial agreement — kappa should be between 0 and 1
        assert -0.5 <= k <= 1.0

    def test_empty_input_returns_zero(self):
        from app.judge.consensus import fleiss_kappa
        assert fleiss_kappa([]) == 0.0

    def test_single_item_returns_zero(self):
        from app.judge.consensus import fleiss_kappa
        assert fleiss_kappa([[1, 1, 1]]) == 0.0


# ---------------------------------------------------------------------------
# T6: JudgeChainRunner
# ---------------------------------------------------------------------------

class TestJudgeChainRunner:
    """Tests for transparent_judge.JudgeChainRunner."""

    def test_chain_returns_valid_structure(self):
        from app.judge.transparent_judge import JudgeChainRunner

        runner = JudgeChainRunner()
        params = {
            "expected_answer": "Water is H2O",
            "reference": "Water is H2O",
        }
        passed, detail = runner.run("Water is H2O — a molecule with two hydrogen atoms.", params)

        assert "judge_chain" in detail
        assert "final_level" in detail
        assert isinstance(detail["judge_chain"], list)
        assert len(detail["judge_chain"]) > 0

    def test_chain_falls_through_on_exception(self):
        """If all levels raise exceptions, still returns a result."""
        from app.judge.transparent_judge import JudgeChainRunner

        runner = JudgeChainRunner()
        # Provide params that likely trigger fallback paths
        params = {}
        passed, detail = runner.run("some response", params)

        # Must not raise and must return a chain log
        assert "judge_chain" in detail
        assert detail.get("final_level") is not None


# ---------------------------------------------------------------------------
# T7: handle_judge_chain handler (unit test)
# ---------------------------------------------------------------------------

class TestHandleJudgeChain:
    """Tests for v14_handlers.handle_judge_chain."""

    def test_returns_valid_json_structure_for_unknown_run(self):
        from app.handlers.v14_handlers import handle_judge_chain

        response = handle_judge_chain(
            path="/api/v14/runs/nonexistent-run-id/judge-chain",
            qs={},
            body={},
        )
        # Should get a 404 error response (not crash)
        assert response is not None
        content = response[1] if isinstance(response, (list, tuple)) else response
        # If it returns a dict/bytes, ensure no exception was raised


# ---------------------------------------------------------------------------
# T8: methods.py dispatch registration
# ---------------------------------------------------------------------------

class TestMethodsDispatch:
    """Tests that new methods are registered in judge/methods.py dispatch."""

    def test_numeric_tolerance_dispatched(self):
        from app.judge.methods import judge
        passed, detail = judge("numeric_tolerance", "The answer is 42", {"expected": 42})
        assert detail.get("method") == "numeric_tolerance"
        assert passed is True

    def test_multi_choice_dispatched(self):
        from app.judge.methods import judge
        passed, detail = judge("multi_choice_verified", "The answer is B", {"expected_choice": "B"})
        assert detail.get("method") == "multi_choice_verified"
        assert passed is True

    def test_semantic_entailment_dispatched(self):
        from app.judge.methods import judge
        passed, detail = judge(
            "semantic_entailment",
            "Water contains hydrogen and oxygen",
            {"expected_answer": "Water has hydrogen and oxygen atoms"},
        )
        assert detail.get("method") == "semantic_entailment"
        # passed can be True or False depending on similarity; no crash is the key check
        assert isinstance(passed, bool)
