"""
test_v16_phase6.py — v16 Phase 6 regression tests.

Validates:
  - VerificationResult has kg_conflict / degraded fields
  - WikidataClient UA updated to v16
  - verify_with_degradation graceful fallback
  - DynamicKGQuestion data structure
  - generate_kg_question function
  - generate_random_questions function
  - verify_dual_source function
"""
import pytest


class TestVerificationResultV16:
    def test_kg_conflict_field(self):
        from app.knowledge.wikidata_client import VerificationResult
        r = VerificationResult(
            is_verified=True, confidence=0.8, source="wikidata",
            kg_conflict=True,
        )
        assert r.kg_conflict is True

    def test_degraded_field(self):
        from app.knowledge.wikidata_client import VerificationResult
        r = VerificationResult(
            is_verified=False, confidence=0.0, source="wikidata",
            degraded=True,
        )
        assert r.degraded is True

    def test_to_dict_includes_v16_fields(self):
        from app.knowledge.wikidata_client import VerificationResult
        r = VerificationResult(
            is_verified=True, confidence=0.9, source="wikidata",
            kg_conflict=False, degraded=False,
        )
        d = r.to_dict()
        assert "kg_conflict" in d
        assert "degraded" in d


class TestWikidataClientV16:
    def test_ua_is_v16(self):
        from app.knowledge.wikidata_client import WikidataClient
        client = WikidataClient()
        ua = client._session.headers.get("User-Agent", "")
        assert "16.0" in ua

    def test_ua_follows_policy(self):
        from app.knowledge.wikidata_client import WikidataClient
        client = WikidataClient()
        ua = client._session.headers.get("User-Agent", "")
        # Wikimedia policy requires contact info
        assert "LLM-Inspector" in ua


class TestDynamicKGQuestions:
    def test_dynamic_kg_question_dataclass(self):
        from app.runner.dynamic_kg_questions import DynamicKGQuestion
        q = DynamicKGQuestion(
            question_id="kg_Q42_P569",
            question_text="When was Douglas Adams born?",
            expected_answer="1952-03-11",
            answer_type="date",
            entity_qid="Q42",
            entity_label="Douglas Adams",
            property_id="P569",
            property_label="date of birth",
        )
        assert q.question_id == "kg_Q42_P569"
        assert q.source == "wikidata_dynamic"

    def test_to_test_case(self):
        from app.runner.dynamic_kg_questions import DynamicKGQuestion
        q = DynamicKGQuestion(
            question_id="kg_Q42_P569",
            question_text="When was Douglas Adams born?",
            expected_answer="1952-03-11",
            answer_type="date",
            entity_qid="Q42",
            entity_label="Douglas Adams",
            property_id="P569",
            property_label="date of birth",
        )
        tc = q.to_test_case()
        assert tc["category"] == "knowledge"
        assert tc["dynamic"] is True
        assert tc["source_ref"] == "wikidata:Q42/P569"

    def test_generate_kg_question(self):
        from app.runner.dynamic_kg_questions import generate_kg_question
        q = generate_kg_question(
            entity_qid="Q42",
            prop_id="P569",
            entity_label="Douglas Adams",
        )
        assert q is not None
        assert "Douglas Adams" in q.question_text
        assert q.property_id == "P569"

    def test_generate_kg_question_unknown_property(self):
        from app.runner.dynamic_kg_questions import generate_kg_question
        q = generate_kg_question(
            entity_qid="Q42",
            prop_id="P99999",
            entity_label="Test",
        )
        assert q is None  # No template for unknown property

    def test_generate_random_questions(self):
        from app.runner.dynamic_kg_questions import generate_random_questions
        questions = generate_random_questions(n=3, seed=42)
        assert len(questions) == 3
        for q in questions:
            assert q.question_id.startswith("kg_")
            assert q.entity_qid.startswith("Q")

    def test_generate_random_questions_reproducible(self):
        from app.runner.dynamic_kg_questions import generate_random_questions
        q1 = generate_random_questions(n=3, seed=42)
        q2 = generate_random_questions(n=3, seed=42)
        assert [q.question_id for q in q1] == [q.question_id for q in q2]


class TestVerifyDualSource:
    def test_both_verified_consistent(self):
        from app.runner.dynamic_kg_questions import verify_dual_source

        class MockResult:
            is_verified = True

        consistent, conflict = verify_dual_source("Einstein", MockResult(), MockResult())
        assert consistent is True
        assert conflict is False

    def test_both_not_verified_consistent(self):
        from app.runner.dynamic_kg_questions import verify_dual_source

        class MockResult:
            is_verified = False

        consistent, conflict = verify_dual_source("Test", MockResult(), MockResult())
        assert consistent is True
        assert conflict is False

    def test_conflict_detected(self):
        from app.runner.dynamic_kg_questions import verify_dual_source

        class Verified:
            is_verified = True

        class NotVerified:
            is_verified = False

        consistent, conflict = verify_dual_source("Test", Verified(), NotVerified())
        assert consistent is False
        assert conflict is True

    def test_one_source_no_conflict(self):
        from app.runner.dynamic_kg_questions import verify_dual_source

        class Verified:
            is_verified = True

        consistent, conflict = verify_dual_source("Test", Verified(), None)
        assert consistent is True
        assert conflict is False

    def test_no_sources(self):
        from app.runner.dynamic_kg_questions import verify_dual_source
        consistent, conflict = verify_dual_source("Test", None, None)
        assert consistent is False
        assert conflict is False
