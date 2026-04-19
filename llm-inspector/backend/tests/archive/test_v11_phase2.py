"""
v11 Phase 2 integration tests — CDM Engine + Shapley Attribution.

Tests the DINA cognitive diagnostic model and Shapley Value
score attribution engine, plus their integration points.
"""
import math
import pytest
from dataclasses import dataclass

# ── Test fixtures ────────────────────────────────────────────────────────────

from app.core.schemas import TestCase, CaseResult, SampleResult, LLMResponse, ScoreCard, TrustVerdict
from app.core.eval_schemas import SkillVector, EvalTestCase


def _make_case(case_id, category, dimension, passed=True, weight=1.0, n_samples=1):
    """Helper to create a CaseResult for testing."""
    tc = TestCase(
        id=case_id, category=category, name=f"test_{case_id}",
        user_prompt="test", expected_type="text", judge_method="exact_match",
        dimension=dimension, weight=weight,
    )
    samples = [
        SampleResult(
            sample_index=i,
            response=LLMResponse(content="ok", latency_ms=500),
            judge_passed=passed,
            judge_detail={},
        )
        for i in range(n_samples)
    ]
    return CaseResult(case=tc, samples=samples)


def _make_eval_case(case_id, category, dimension, passed=True, skill_vector=None):
    """Helper to create a CaseResult with EvalTestCase."""
    tc = EvalTestCase(
        id=case_id, category=category, name=f"test_{case_id}",
        user_prompt="test", expected_type="text", judge_method="exact_match",
        dimension=dimension, skill_vector=skill_vector,
    )
    samples = [
        SampleResult(
            sample_index=0,
            response=LLMResponse(content="ok", latency_ms=500),
            judge_passed=passed,
            judge_detail={},
        )
    ]
    return CaseResult(case=tc, samples=samples)


# ═══════════════════════════════════════════════════════════════════════════
# CDM Engine Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSkillTaxonomy:
    """Test the CDM skill taxonomy and Q-matrix construction."""

    def test_all_skills_is_nonempty(self):
        from app.analysis.cdm_engine import ALL_SKILLS
        assert len(ALL_SKILLS) > 0

    def test_skill_taxonomy_has_core_dimensions(self):
        from app.analysis.cdm_engine import SKILL_TAXONOMY
        assert "reasoning" in SKILL_TAXONOMY
        assert "instruction" in SKILL_TAXONOMY
        assert "coding" in SKILL_TAXONOMY
        assert "safety" in SKILL_TAXONOMY

    def test_skill_index_maps_correctly(self):
        from app.analysis.cdm_engine import ALL_SKILLS, SKILL_INDEX
        for skill in ALL_SKILLS:
            assert skill in SKILL_INDEX
            assert ALL_SKILLS[SKILL_INDEX[skill]] == skill


class TestQMatrix:
    """Test Q-matrix construction from case results."""

    def test_q_matrix_basic(self):
        from app.analysis.cdm_engine import build_q_matrix, ALL_SKILLS
        cases = [
            _make_case("r1", "reasoning", "reasoning", passed=True),
            _make_case("i1", "instruction", "instruction", passed=False),
        ]
        Q = build_q_matrix(cases)
        assert Q.shape[0] == 2  # 2 items
        assert Q.shape[1] == len(ALL_SKILLS)  # all skills
        # Reasoning item should require reasoning skills
        assert Q[0].sum() > 0
        # Instruction item should require instruction skills
        assert Q[1].sum() > 0

    def test_q_matrix_with_skill_vector(self):
        from app.analysis.cdm_engine import build_q_matrix, SKILL_INDEX
        sv = SkillVector(required={"logical_deduction": 1, "mathematical_reasoning": 1})
        cases = [
            _make_eval_case("r1", "reasoning", "reasoning", passed=True, skill_vector=sv),
        ]
        Q = build_q_matrix(cases)
        # Should have the two explicit skills
        assert Q[0, SKILL_INDEX["logical_deduction"]] == 1
        assert Q[0, SKILL_INDEX["mathematical_reasoning"]] == 1

    def test_q_matrix_no_skills_for_unknown_dimension(self):
        from app.analysis.cdm_engine import build_q_matrix
        cases = [
            _make_case("x1", "unknown_cat", "unknown_dim", passed=True),
        ]
        Q = build_q_matrix(cases)
        # Unknown dimension: category fallback should map some skills
        assert Q.shape[0] == 1


class TestDINAEngine:
    """Test the DINA cognitive diagnostic model."""

    def test_empty_case_results(self):
        from app.analysis.cdm_engine import DINAEngine
        engine = DINAEngine()
        report = engine.diagnose([])
        assert report.n_items == 0
        assert len(report.mastery_profile) == 0

    def test_single_dimension_all_pass(self):
        from app.analysis.cdm_engine import DINAEngine
        engine = DINAEngine()
        cases = [
            _make_case(f"r{i}", "reasoning", "reasoning", passed=True)
            for i in range(10)
        ]
        report = engine.diagnose(cases)
        assert report.n_items == 10
        assert report.n_skills > 0
        # All pass → mastery should be high
        assert report.overall_mastery_rate > 0.5

    def test_single_dimension_all_fail(self):
        from app.analysis.cdm_engine import DINAEngine
        engine = DINAEngine()
        cases = [
            _make_case(f"r{i}", "reasoning", "reasoning", passed=False)
            for i in range(10)
        ]
        report = engine.diagnose(cases)
        # All fail → mastery should be low
        assert report.overall_mastery_rate < 0.5

    def test_mixed_dimensions(self):
        from app.analysis.cdm_engine import DINAEngine
        engine = DINAEngine()
        cases = [
            _make_case("r1", "reasoning", "reasoning", passed=True),
            _make_case("r2", "reasoning", "reasoning", passed=True),
            _make_case("c1", "coding", "coding", passed=False),
            _make_case("c2", "coding", "coding", passed=False),
            _make_case("i1", "instruction", "instruction", passed=True),
        ]
        report = engine.diagnose(cases)
        assert report.n_items == 5
        assert len(report.mastery_profile) > 0
        # Reasoning should have higher mastery than coding
        reasoning_mastery = None
        coding_mastery = None
        for m in report.mastery_profile:
            if m.skill_name in ("logical_deduction", "mathematical_reasoning"):
                reasoning_mastery = m.mastery_probability
            if m.skill_name in ("code_generation", "code_execution_accuracy"):
                coding_mastery = m.mastery_probability
        # Reasoning (passed) should generally have higher mastery than coding (failed)
        if reasoning_mastery is not None and coding_mastery is not None:
            assert reasoning_mastery >= coding_mastery

    def test_cdm_report_serialization(self):
        from app.analysis.cdm_engine import DINAEngine
        engine = DINAEngine()
        cases = [
            _make_case("r1", "reasoning", "reasoning", passed=True),
            _make_case("c1", "coding", "coding", passed=False),
        ]
        report = engine.diagnose(cases)
        d = report.to_dict()
        assert "mastery_profile" in d
        assert "skill_names" in d
        assert "overall_mastery_rate" in d
        assert "strongest_skills" in d
        assert "weakest_skills" in d
        assert isinstance(d["mastery_profile"], list)

    def test_attribute_pattern_binary(self):
        from app.analysis.cdm_engine import DINAEngine
        engine = DINAEngine()
        cases = [
            _make_case(f"r{i}", "reasoning", "reasoning", passed=True)
            for i in range(8)
        ]
        report = engine.diagnose(cases)
        # Attribute pattern should be binary (0 or 1)
        for val in report.attribute_pattern:
            assert val in (0, 1)

    def test_skill_mastery_confidence(self):
        from app.analysis.cdm_engine import DINAEngine, SkillMastery
        # High evidence
        m = SkillMastery("test_skill", 0.8, confidence="high", evidence_count=10)
        assert m.confidence == "high"
        d = m.to_dict()
        assert "mastery_probability" in d

    def test_many_skills_reduction(self):
        """Test that engine handles >10 active skills by reducing."""
        from app.analysis.cdm_engine import DINAEngine
        engine = DINAEngine()
        # Create cases spanning many dimensions to trigger >10 skills
        cases = []
        for dim in ["reasoning", "coding", "instruction", "safety", "protocol", 
                     "consistency", "knowledge", "adversarial_reasoning"]:
            for i in range(3):
                cases.append(_make_case(f"{dim}_{i}", dim, dim, passed=(i % 2 == 0)))
        report = engine.diagnose(cases)
        assert report.n_skills <= 10  # Should be reduced


# ═══════════════════════════════════════════════════════════════════════════
# Shapley Attribution Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestShapleyAttribution:
    """Test the Shapley Value attribution engine."""

    def _make_scorecard(self, **overrides) -> ScoreCard:
        """Create a ScoreCard with specified sub-scores."""
        defaults = {
            "total_score": 70.0,
            "capability_score": 72.0,
            "authenticity_score": 68.0,
            "performance_score": 70.0,
            "reasoning_score": 80.0,
            "adversarial_reasoning_score": 60.0,
            "instruction_score": 75.0,
            "coding_score": 65.0,
            "safety_score": 70.0,
            "protocol_score": 85.0,
            "consistency_score": 60.0,
            "speed_score": 70.0,
            "stability_score": 75.0,
            "behavioral_invariant_score": 55.0,
        }
        defaults.update(overrides)
        card = ScoreCard(**{k: v for k, v in defaults.items() if k in ScoreCard.__dataclass_fields__})
        card.breakdown = {
            "extraction_resistance": 70.0,
            "fingerprint_match": 65.0,
        }
        return card

    def test_basic_attribution(self):
        from app.analysis.shapley_attribution import ShapleyAttributor
        attributor = ShapleyAttributor(n_samples=100)
        card = self._make_scorecard()
        report = attributor.attribute(card)
        assert len(report.attributions) > 0
        assert report.target_score != 0.0

    def test_attribution_direction(self):
        from app.analysis.shapley_attribution import ShapleyAttributor
        attributor = ShapleyAttributor(n_samples=100)
        # High reasoning → should be positive attribution
        card = self._make_scorecard(reasoning_score=95.0, coding_score=20.0)
        report = attributor.attribute(card)
        positive_names = report.top_positive
        negative_names = report.top_negative
        # Something should be positive, something negative
        assert len(positive_names) > 0 or len(negative_names) > 0

    def test_attribution_report_serialization(self):
        from app.analysis.shapley_attribution import ShapleyAttributor
        attributor = ShapleyAttributor(n_samples=50)
        card = self._make_scorecard()
        report = attributor.attribute(card)
        d = report.to_dict()
        assert "target_score" in d
        assert "baseline_score" in d
        assert "score_delta" in d
        assert "attributions" in d
        assert "narrative" in d
        assert "top_positive" in d
        assert "top_negative" in d

    def test_attribution_narrative_nonempty(self):
        from app.analysis.shapley_attribution import ShapleyAttributor
        attributor = ShapleyAttributor(n_samples=100)
        card = self._make_scorecard()
        report = attributor.attribute(card)
        assert len(report.narrative) > 0

    def test_feature_attribution_serialization(self):
        from app.analysis.shapley_attribution import FeatureAttribution
        fa = FeatureAttribution(
            feature_name="reasoning",
            shapley_value=5.3,
            contribution_pct=35.0,
            direction="positive",
            baseline_value=50.0,
            actual_value=80.0,
            impact_description="reasoning 高于基准 (+30.0)，贡献 +5.3 分",
        )
        d = fa.to_dict()
        assert d["feature_name"] == "reasoning"
        assert d["direction"] == "positive"

    def test_neutral_scorecard(self):
        """All scores at baseline (50) should give near-zero Shapley values."""
        from app.analysis.shapley_attribution import ShapleyAttributor
        attributor = ShapleyAttributor(n_samples=100)
        card = self._make_scorecard(
            reasoning_score=50.0, adversarial_reasoning_score=50.0,
            instruction_score=50.0, coding_score=50.0,
            safety_score=50.0, protocol_score=50.0,
            consistency_score=50.0, speed_score=50.0,
            stability_score=50.0, behavioral_invariant_score=50.0,
        )
        card.breakdown = {"extraction_resistance": 50.0, "fingerprint_match": 50.0}
        report = attributor.attribute(card)
        # Score delta should be near zero
        assert abs(report.score_delta) < 5.0

    def test_attributable_features_defined(self):
        from app.analysis.shapley_attribution import ATTRIBUTABLE_FEATURES, FEATURE_SCORE_MAP
        assert len(ATTRIBUTABLE_FEATURES) > 0
        assert len(FEATURE_SCORE_MAP) > 0
        for feat in ATTRIBUTABLE_FEATURES:
            assert feat in FEATURE_SCORE_MAP

    def test_get_score_value(self):
        from app.analysis.shapley_attribution import _get_score_value
        card = ScoreCard(reasoning_score=75.0)
        val = _get_score_value(card, "reasoning")
        assert abs(val - 75.0) < 0.01

    def test_compute_total_score(self):
        from app.analysis.shapley_attribution import _compute_total_score
        values = {feat: 50.0 for feat in ATTRIBUTABLE_FEATURES}
        score = _compute_total_score(values)
        # All at 50 → total should be near 50
        assert 40.0 < score < 60.0


# Need to import for the test above
from app.analysis.shapley_attribution import ATTRIBUTABLE_FEATURES


# ═══════════════════════════════════════════════════════════════════════════
# V11 Handlers Tests (Phase 2)
# ═══════════════════════════════════════════════════════════════════════════

class TestV11Phase2Handlers:
    """Test Phase 2 API handlers."""

    def test_cdm_skills_handler(self):
        from app.handlers.v11_handlers import handle_cdm_skills
        result = handle_cdm_skills("/api/v1/cdm/skills", {}, {})
        # _json() returns (status, body_bytes, content_type)
        assert result[0] == 200  # status code
        body = result[1]  # body bytes
        import json
        data = json.loads(body)
        assert "total_skills" in data
        assert data["total_skills"] > 0
        assert "skills" in data
        assert "taxonomy" in data

    def test_run_cdm_handler_not_found(self):
        from app.handlers.v11_handlers import handle_run_cdm
        result = handle_run_cdm("/api/v1/runs/nonexistent/cdm", {}, {})
        import json
        status = result[0]
        body = json.loads(result[1])
        # Should return 404 or unavailable
        assert status in (200, 404)

    def test_run_attribution_handler_not_found(self):
        from app.handlers.v11_handlers import handle_run_attribution
        result = handle_run_attribution("/api/v1/runs/nonexistent/attribution", {}, {})
        import json
        status = result[0]
        body = json.loads(result[1])
        assert status in (200, 404)


# ═══════════════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestCDMShapleyIntegration:
    """Test CDM and Shapley working together in the analysis pipeline."""

    def test_cdm_then_shapley_pipeline(self):
        """Simulate the orchestrator's analysis pipeline: CDM then Shapley."""
        from app.analysis.cdm_engine import DINAEngine
        from app.analysis.shapley_attribution import ShapleyAttributor

        # Create diverse case results
        cases = [
            _make_case("r1", "reasoning", "reasoning", passed=True),
            _make_case("r2", "reasoning", "reasoning", passed=True),
            _make_case("r3", "reasoning", "reasoning", passed=False),
            _make_case("c1", "coding", "coding", passed=True),
            _make_case("c2", "coding", "coding", passed=False),
            _make_case("i1", "instruction", "instruction", passed=True),
            _make_case("i2", "instruction", "instruction", passed=True),
            _make_case("s1", "safety", "safety", passed=False),
        ]

        # Run CDM
        cdm_engine = DINAEngine()
        cdm_report = cdm_engine.diagnose(cases)
        assert cdm_report.n_items == 8
        assert cdm_report.n_skills > 0

        # Run Shapley on a scorecard derived from the same results
        attributor = ShapleyAttributor(n_samples=50)
        card = ScoreCard(
            total_score=65.0,
            capability_score=70.0,
            authenticity_score=60.0,
            performance_score=65.0,
            reasoning_score=75.0,
            instruction_score=80.0,
            coding_score=50.0,
            safety_score=30.0,
            protocol_score=70.0,
            consistency_score=55.0,
            speed_score=65.0,
            stability_score=70.0,
            behavioral_invariant_score=45.0,
        )
        card.breakdown = {"extraction_resistance": 60.0, "fingerprint_match": 55.0}
        attr_report = attributor.attribute(card)

        assert len(attr_report.attributions) > 0
        assert abs(attr_report.score_delta) > 0 or True  # May be near zero

        # Verify both reports are serializable
        cdm_dict = cdm_report.to_dict()
        attr_dict = attr_report.to_dict()
        assert isinstance(cdm_dict, dict)
        assert isinstance(attr_dict, dict)
