"""
tests/test_v15_phase2.py — v15 Phase 2: Authenticity Evidence Ledger & Model Card Diff

Covers:
- AuthenticityEvidence and EvidenceLedger dataclass creation
- EvidenceLedger.wrapper_probability() with zero evidence = 0.5
- EvidenceLedger.wrapper_probability() with one contradicting strength=0.9 → ~0.655
- EvidenceLedger.risk_level() for each tier
- EvidenceLedger.suspected_actual_model() picks top suspected target
- EvidenceLedger.to_dict() structure
- extract_evidence_from_predetect() with empty inputs returns ledger with run_id set
- extract_evidence_from_predetect() with identity_collision adds evidence
- build_model_card_diff() returns ModelCardDiff with correct fields
- ModelCardDiff.to_dict() structure
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# AuthenticityEvidence
# ---------------------------------------------------------------------------

class TestAuthenticityEvidence:
    def test_defaults(self):
        from app.authenticity.evidence_ledger import AuthenticityEvidence
        ev = AuthenticityEvidence()
        assert ev.type == ""
        assert ev.source_layer == ""
        assert ev.direction == "contradicts_claim"
        assert ev.strength == 0.5
        assert ev.reproducible is False
        assert ev.provenance == {}
        assert ev.suspected_target is None

    def test_evidence_id_auto_generated(self):
        from app.authenticity.evidence_ledger import AuthenticityEvidence
        ev1 = AuthenticityEvidence()
        ev2 = AuthenticityEvidence()
        assert len(ev1.evidence_id) == 8
        # IDs should differ
        assert ev1.evidence_id != ev2.evidence_id

    def test_to_dict_keys(self):
        from app.authenticity.evidence_ledger import AuthenticityEvidence
        ev = AuthenticityEvidence(
            type="identity_collision",
            source_layer="L17",
            direction="contradicts_claim",
            strength=0.9,
            strength_rationale="test",
            raw_snippet="I am Kiro",
            claim_target="gpt-4",
            suspected_target="kiro",
            timestamp="2026-01-01T00:00:00Z",
            reproducible=True,
        )
        d = ev.to_dict()
        expected_keys = {
            "evidence_id", "type", "source_layer", "direction", "strength",
            "strength_rationale", "raw_snippet", "claim_target", "suspected_target",
            "timestamp", "reproducible", "provenance",
        }
        assert expected_keys == set(d.keys())
        assert d["type"] == "identity_collision"
        assert d["strength"] == 0.9
        assert d["raw_snippet"] == "I am Kiro"

    def test_strength_rounded_in_to_dict(self):
        from app.authenticity.evidence_ledger import AuthenticityEvidence
        ev = AuthenticityEvidence(strength=0.123456789)
        d = ev.to_dict()
        # to_dict rounds to 3 decimal places
        assert d["strength"] == round(0.123456789, 3)


# ---------------------------------------------------------------------------
# EvidenceLedger — basic construction
# ---------------------------------------------------------------------------

class TestEvidenceLedgerBasic:
    def test_defaults(self):
        from app.authenticity.evidence_ledger import EvidenceLedger
        ledger = EvidenceLedger()
        assert ledger.run_id == ""
        assert ledger.claimed_model == ""
        assert ledger.evidence == []

    def test_add_evidence(self):
        from app.authenticity.evidence_ledger import EvidenceLedger, AuthenticityEvidence
        ledger = EvidenceLedger(run_id="r1", claimed_model="gpt-4")
        ev = AuthenticityEvidence(type="identity_collision", direction="contradicts_claim")
        ledger.add(ev)
        assert len(ledger.evidence) == 1

    def test_contradicting_filter(self):
        from app.authenticity.evidence_ledger import EvidenceLedger, AuthenticityEvidence
        ledger = EvidenceLedger()
        ledger.add(AuthenticityEvidence(direction="contradicts_claim"))
        ledger.add(AuthenticityEvidence(direction="supports_claim"))
        ledger.add(AuthenticityEvidence(direction="neutral"))
        assert len(ledger.contradicting()) == 1
        assert len(ledger.supporting()) == 1

    def test_identity_collisions_filter(self):
        from app.authenticity.evidence_ledger import EvidenceLedger, AuthenticityEvidence
        ledger = EvidenceLedger()
        ledger.add(AuthenticityEvidence(type="identity_collision", direction="contradicts_claim"))
        ledger.add(AuthenticityEvidence(type="tokenizer_mismatch", direction="contradicts_claim"))
        assert len(ledger.identity_collisions()) == 1


# ---------------------------------------------------------------------------
# EvidenceLedger.wrapper_probability()
# ---------------------------------------------------------------------------

class TestWrapperProbability:
    def test_zero_evidence_returns_05(self):
        from app.authenticity.evidence_ledger import EvidenceLedger
        ledger = EvidenceLedger()
        # Prior 0.5/0.5 → odds=1.0 → p=0.5
        assert ledger.wrapper_probability() == 0.5

    def test_one_contradicting_strength_09(self):
        """
        Math: prior_odds=1.0, contradicting strength=0.9
        LR = 1 + 0.9 = 1.9
        odds = 1.0 * 1.9 = 1.9
        p = 1.9 / (1 + 1.9) = 1.9 / 2.9 ≈ 0.6552
        """
        from app.authenticity.evidence_ledger import EvidenceLedger, AuthenticityEvidence
        ledger = EvidenceLedger()
        ledger.add(AuthenticityEvidence(direction="contradicts_claim", strength=0.9))
        p = ledger.wrapper_probability()
        assert abs(p - (1.9 / 2.9)) < 0.001, f"Expected ~0.6552, got {p}"

    def test_supporting_evidence_lowers_probability(self):
        from app.authenticity.evidence_ledger import EvidenceLedger, AuthenticityEvidence
        ledger = EvidenceLedger()
        ledger.add(AuthenticityEvidence(direction="supports_claim", strength=0.8))
        p = ledger.wrapper_probability()
        assert p < 0.5

    def test_neutral_evidence_no_change(self):
        from app.authenticity.evidence_ledger import EvidenceLedger, AuthenticityEvidence
        ledger = EvidenceLedger()
        ledger.add(AuthenticityEvidence(direction="neutral", strength=0.9))
        assert ledger.wrapper_probability() == 0.5

    def test_multiple_contradicting_increases_probability(self):
        from app.authenticity.evidence_ledger import EvidenceLedger, AuthenticityEvidence
        ledger = EvidenceLedger()
        ledger.add(AuthenticityEvidence(direction="contradicts_claim", strength=0.7))
        ledger.add(AuthenticityEvidence(direction="contradicts_claim", strength=0.7))
        p = ledger.wrapper_probability()
        assert p > 0.5

    def test_result_clamped_between_0_and_1(self):
        from app.authenticity.evidence_ledger import EvidenceLedger, AuthenticityEvidence
        ledger = EvidenceLedger()
        for _ in range(20):
            ledger.add(AuthenticityEvidence(direction="contradicts_claim", strength=1.0))
        p = ledger.wrapper_probability()
        assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# EvidenceLedger.risk_level()
# ---------------------------------------------------------------------------

class TestRiskLevel:
    def _ledger_with_probability(self, target_p: float):
        """
        Build a ledger with wrapper_probability close to target_p.
        We'll just test the boundaries with known inputs.
        """
        from app.authenticity.evidence_ledger import EvidenceLedger
        # Mock the probability by patching wrapper_probability
        ledger = EvidenceLedger()
        ledger.wrapper_probability = lambda: target_p  # type: ignore[method-assign]
        return ledger

    def test_trusted_below_020(self):
        ledger = self._ledger_with_probability(0.10)
        assert ledger.risk_level() == "trusted"

    def test_suspicious_between_020_and_055(self):
        ledger = self._ledger_with_probability(0.40)
        assert ledger.risk_level() == "suspicious"

    def test_high_risk_between_055_and_085(self):
        ledger = self._ledger_with_probability(0.70)
        assert ledger.risk_level() == "high_risk"

    def test_fake_above_085(self):
        ledger = self._ledger_with_probability(0.90)
        assert ledger.risk_level() == "fake"

    def test_boundary_020_is_suspicious(self):
        ledger = self._ledger_with_probability(0.20)
        assert ledger.risk_level() == "suspicious"

    def test_boundary_055_is_high_risk(self):
        ledger = self._ledger_with_probability(0.55)
        assert ledger.risk_level() == "high_risk"

    def test_boundary_085_is_fake(self):
        ledger = self._ledger_with_probability(0.85)
        assert ledger.risk_level() == "fake"

    def test_zero_evidence_is_suspicious(self):
        from app.authenticity.evidence_ledger import EvidenceLedger
        ledger = EvidenceLedger()
        # p=0.5 → "suspicious"
        assert ledger.risk_level() == "suspicious"


# ---------------------------------------------------------------------------
# EvidenceLedger.suspected_actual_model()
# ---------------------------------------------------------------------------

class TestSuspectedActualModel:
    def test_no_evidence_returns_none(self):
        from app.authenticity.evidence_ledger import EvidenceLedger
        ledger = EvidenceLedger()
        assert ledger.suspected_actual_model() is None

    def test_picks_strongest_suspected_target(self):
        from app.authenticity.evidence_ledger import EvidenceLedger, AuthenticityEvidence
        ledger = EvidenceLedger()
        ledger.add(AuthenticityEvidence(
            direction="contradicts_claim", strength=0.9,
            suspected_target="claude"
        ))
        ledger.add(AuthenticityEvidence(
            direction="contradicts_claim", strength=0.4,
            suspected_target="gpt-4"
        ))
        assert ledger.suspected_actual_model() == "claude"

    def test_accumulates_strength_for_same_target(self):
        from app.authenticity.evidence_ledger import EvidenceLedger, AuthenticityEvidence
        ledger = EvidenceLedger()
        ledger.add(AuthenticityEvidence(
            direction="contradicts_claim", strength=0.5,
            suspected_target="claude"
        ))
        ledger.add(AuthenticityEvidence(
            direction="contradicts_claim", strength=0.5,
            suspected_target="claude"
        ))
        ledger.add(AuthenticityEvidence(
            direction="contradicts_claim", strength=0.9,
            suspected_target="gpt-4"
        ))
        # claude total = 1.0 > gpt-4 total = 0.9
        assert ledger.suspected_actual_model() == "claude"

    def test_ignores_supporting_evidence(self):
        from app.authenticity.evidence_ledger import EvidenceLedger, AuthenticityEvidence
        ledger = EvidenceLedger()
        ledger.add(AuthenticityEvidence(
            direction="supports_claim", strength=0.9,
            suspected_target="some-model"
        ))
        assert ledger.suspected_actual_model() is None

    def test_ignores_none_suspected_target(self):
        from app.authenticity.evidence_ledger import EvidenceLedger, AuthenticityEvidence
        ledger = EvidenceLedger()
        ledger.add(AuthenticityEvidence(
            direction="contradicts_claim", strength=0.9,
            suspected_target=None
        ))
        assert ledger.suspected_actual_model() is None


# ---------------------------------------------------------------------------
# EvidenceLedger.to_dict()
# ---------------------------------------------------------------------------

class TestEvidenceLedgerToDict:
    def test_to_dict_structure(self):
        from app.authenticity.evidence_ledger import EvidenceLedger, AuthenticityEvidence
        ledger = EvidenceLedger(run_id="run-abc", claimed_model="gpt-4")
        ledger.add(AuthenticityEvidence(
            type="identity_collision",
            direction="contradicts_claim",
            strength=0.9,
            suspected_target="claude",
        ))
        d = ledger.to_dict()
        assert d["run_id"] == "run-abc"
        assert d["claimed_model"] == "gpt-4"
        assert "wrapper_probability" in d
        assert "risk_level" in d
        assert "evidence_count" in d
        assert "suspected_actual_model" in d
        assert "evidence" in d

    def test_to_dict_evidence_count(self):
        from app.authenticity.evidence_ledger import EvidenceLedger, AuthenticityEvidence
        ledger = EvidenceLedger(run_id="r1", claimed_model="m1")
        ledger.add(AuthenticityEvidence(direction="contradicts_claim", strength=0.8))
        ledger.add(AuthenticityEvidence(direction="supports_claim", strength=0.6))
        ledger.add(AuthenticityEvidence(direction="neutral", strength=0.3))
        d = ledger.to_dict()
        counts = d["evidence_count"]
        assert counts["total"] == 3
        assert counts["contradicting"] == 1
        assert counts["supporting"] == 1
        assert counts["neutral"] == 1

    def test_to_dict_evidence_list_items_are_dicts(self):
        from app.authenticity.evidence_ledger import EvidenceLedger, AuthenticityEvidence
        ledger = EvidenceLedger(run_id="r1", claimed_model="m1")
        ledger.add(AuthenticityEvidence(type="tokenizer_mismatch", direction="contradicts_claim"))
        d = ledger.to_dict()
        assert len(d["evidence"]) == 1
        assert isinstance(d["evidence"][0], dict)
        assert d["evidence"][0]["type"] == "tokenizer_mismatch"


# ---------------------------------------------------------------------------
# extract_evidence_from_predetect()
# ---------------------------------------------------------------------------

class TestExtractEvidenceFromPredetect:
    def test_empty_inputs_returns_ledger_with_run_id(self):
        from app.authenticity.evidence_ledger import extract_evidence_from_predetect
        ledger = extract_evidence_from_predetect(
            run_id="run-42",
            claimed_model="gpt-4",
            predetect_result={},
        )
        assert ledger.run_id == "run-42"
        assert ledger.claimed_model == "gpt-4"
        assert isinstance(ledger.evidence, list)

    def test_empty_inputs_no_evidence_added(self):
        from app.authenticity.evidence_ledger import extract_evidence_from_predetect
        ledger = extract_evidence_from_predetect(
            run_id="run-0",
            claimed_model="gpt-4",
            predetect_result={},
        )
        # No identity_exposure, no routing_info, no identified_as
        assert len(ledger.evidence) == 0

    def test_identity_collision_adds_evidence(self):
        from app.authenticity.evidence_ledger import extract_evidence_from_predetect
        identity_exposure = {
            "top_family": "claude",
            "identity_collision": True,
            "posterior": {"claude": 0.85},
            "evidence_snippets": ["I am Claude by Anthropic"],
        }
        ledger = extract_evidence_from_predetect(
            run_id="run-1",
            claimed_model="gpt-4",
            predetect_result={},
            identity_exposure=identity_exposure,
        )
        collisions = ledger.identity_collisions()
        assert len(collisions) == 1
        ev = collisions[0]
        assert ev.type == "identity_collision"
        assert ev.direction == "contradicts_claim"
        assert ev.strength == 0.9
        assert ev.suspected_target == "claude"
        assert ev.claim_target == "gpt-4"
        assert ev.source_layer == "L17"
        assert ev.raw_snippet == "I am Claude by Anthropic"

    def test_identity_collision_same_family_not_added(self):
        """If claimed model matches the top_family, no collision evidence is added."""
        from app.authenticity.evidence_ledger import extract_evidence_from_predetect
        identity_exposure = {
            "top_family": "gpt",
            "identity_collision": True,
            "posterior": {"gpt": 0.9},
        }
        ledger = extract_evidence_from_predetect(
            run_id="run-2",
            claimed_model="gpt-4",
            predetect_result={},
            identity_exposure=identity_exposure,
        )
        collisions = ledger.identity_collisions()
        assert len(collisions) == 0

    def test_tokenizer_match_adds_supporting_evidence(self):
        from app.authenticity.evidence_ledger import extract_evidence_from_predetect
        predetect_result = {"routing_info": {"tokenizer_match": True}}
        ledger = extract_evidence_from_predetect(
            run_id="run-3",
            claimed_model="gpt-4",
            predetect_result=predetect_result,
        )
        supporting = ledger.supporting()
        assert len(supporting) == 1
        assert supporting[0].type == "tokenizer_fingerprint"
        assert supporting[0].direction == "supports_claim"

    def test_tokenizer_mismatch_adds_contradicting_evidence(self):
        from app.authenticity.evidence_ledger import extract_evidence_from_predetect
        predetect_result = {"routing_info": {"tokenizer_match": False}}
        ledger = extract_evidence_from_predetect(
            run_id="run-4",
            claimed_model="gpt-4",
            predetect_result=predetect_result,
        )
        contradicting = ledger.contradicting()
        assert len(contradicting) == 1
        assert contradicting[0].type == "tokenizer_fingerprint"
        assert contradicting[0].direction == "contradicts_claim"

    def test_identified_as_consistent_adds_supporting(self):
        from app.authenticity.evidence_ledger import extract_evidence_from_predetect
        predetect_result = {"identified_as": "gpt-4-turbo"}
        ledger = extract_evidence_from_predetect(
            run_id="run-5",
            claimed_model="gpt-4",
            predetect_result=predetect_result,
        )
        supporting = ledger.supporting()
        assert len(supporting) >= 1
        self_report_evs = [e for e in supporting if e.type == "self_report"]
        assert len(self_report_evs) == 1

    def test_identified_as_inconsistent_adds_contradicting(self):
        from app.authenticity.evidence_ledger import extract_evidence_from_predetect
        predetect_result = {"identified_as": "claude-3-sonnet"}
        ledger = extract_evidence_from_predetect(
            run_id="run-6",
            claimed_model="gpt-4",
            predetect_result=predetect_result,
        )
        contradicting = [e for e in ledger.contradicting() if e.type == "self_report"]
        assert len(contradicting) == 1
        assert contradicting[0].suspected_target == "claude-3-sonnet"

    def test_returns_evidence_ledger_type(self):
        from app.authenticity.evidence_ledger import extract_evidence_from_predetect, EvidenceLedger
        ledger = extract_evidence_from_predetect("r", "m", {})
        assert isinstance(ledger, EvidenceLedger)


# ---------------------------------------------------------------------------
# build_model_card_diff()
# ---------------------------------------------------------------------------

class TestBuildModelCardDiff:
    def test_returns_model_card_diff(self):
        from app.authenticity.model_card_diff import build_model_card_diff, ModelCardDiff
        diff = build_model_card_diff(
            claimed_model="gpt-4",
            suspected_model="claude",
            wrapper_probability=0.72,
            risk_level="high_risk",
            evidence_list=[],
        )
        assert isinstance(diff, ModelCardDiff)

    def test_basic_fields_set(self):
        from app.authenticity.model_card_diff import build_model_card_diff
        diff = build_model_card_diff(
            claimed_model="gpt-4",
            suspected_model="claude-3",
            wrapper_probability=0.80,
            risk_level="fake",
            evidence_list=[],
        )
        assert diff.claimed_model == "gpt-4"
        assert diff.suspected_model == "claude-3"
        assert diff.wrapper_probability == 0.80
        assert diff.risk_level == "fake"

    def test_none_suspected_model(self):
        from app.authenticity.model_card_diff import build_model_card_diff
        diff = build_model_card_diff(
            claimed_model="gpt-4",
            suspected_model=None,
            wrapper_probability=0.30,
            risk_level="suspicious",
            evidence_list=[],
        )
        assert diff.suspected_model is None

    def test_evidence_summary_contains_counts(self):
        from app.authenticity.model_card_diff import build_model_card_diff
        evidence_list = [
            {"direction": "contradicts_claim", "type": "identity_collision"},
            {"direction": "contradicts_claim", "type": "tokenizer_mismatch"},
            {"direction": "supports_claim", "type": "self_report"},
        ]
        diff = build_model_card_diff(
            claimed_model="gpt-4",
            suspected_model=None,
            wrapper_probability=0.60,
            risk_level="high_risk",
            evidence_list=evidence_list,
        )
        assert "2" in diff.evidence_summary  # 2 contradicting
        assert "1" in diff.evidence_summary  # 1 supporting

    def test_collision_snippets_populated(self):
        from app.authenticity.model_card_diff import build_model_card_diff
        evidence_list = [
            {
                "type": "identity_collision",
                "direction": "contradicts_claim",
                "raw_snippet": "I am Claude",
            },
            {
                "type": "identity_collision",
                "direction": "contradicts_claim",
                "raw_snippet": "My name is Claude Sonnet",
            },
        ]
        diff = build_model_card_diff(
            claimed_model="gpt-4",
            suspected_model="claude",
            wrapper_probability=0.90,
            risk_level="fake",
            evidence_list=evidence_list,
        )
        assert len(diff.collision_snippets) == 2
        assert "I am Claude" in diff.collision_snippets

    def test_collision_snippets_limited_to_5(self):
        from app.authenticity.model_card_diff import build_model_card_diff
        evidence_list = [
            {
                "type": "identity_collision",
                "direction": "contradicts_claim",
                "raw_snippet": f"snippet {i}",
            }
            for i in range(10)
        ]
        diff = build_model_card_diff(
            claimed_model="gpt-4",
            suspected_model="claude",
            wrapper_probability=0.90,
            risk_level="fake",
            evidence_list=evidence_list,
        )
        assert len(diff.collision_snippets) <= 5

    def test_fields_is_list(self):
        from app.authenticity.model_card_diff import build_model_card_diff
        diff = build_model_card_diff(
            claimed_model="gpt-4",
            suspected_model=None,
            wrapper_probability=0.5,
            risk_level="suspicious",
            evidence_list=[],
        )
        assert isinstance(diff.fields, list)


# ---------------------------------------------------------------------------
# ModelCardDiff.to_dict()
# ---------------------------------------------------------------------------

class TestModelCardDiffToDict:
    def test_to_dict_structure(self):
        from app.authenticity.model_card_diff import build_model_card_diff
        diff = build_model_card_diff(
            claimed_model="gpt-4",
            suspected_model="claude",
            wrapper_probability=0.72,
            risk_level="high_risk",
            evidence_list=[],
        )
        d = diff.to_dict()
        assert "claimed_model" in d
        assert "suspected_model" in d
        assert "wrapper_probability" in d
        assert "risk_level" in d
        assert "fields" in d
        assert "collision_snippets" in d
        assert "evidence_summary" in d

    def test_to_dict_values_match(self):
        from app.authenticity.model_card_diff import build_model_card_diff
        diff = build_model_card_diff(
            claimed_model="gpt-4",
            suspected_model="claude",
            wrapper_probability=0.88,
            risk_level="fake",
            evidence_list=[],
        )
        d = diff.to_dict()
        assert d["claimed_model"] == "gpt-4"
        assert d["suspected_model"] == "claude"
        assert d["wrapper_probability"] == 0.88
        assert d["risk_level"] == "fake"

    def test_fields_are_list_of_dicts(self):
        from app.authenticity.model_card_diff import build_model_card_diff
        diff = build_model_card_diff(
            claimed_model="gpt-4",
            suspected_model="claude",
            wrapper_probability=0.70,
            risk_level="high_risk",
            evidence_list=[],
        )
        d = diff.to_dict()
        assert isinstance(d["fields"], list)
        for field_item in d["fields"]:
            assert "field" in field_item
            assert "claimed" in field_item
            assert "suspected" in field_item
            assert "match" in field_item

    def test_collision_snippets_in_to_dict(self):
        from app.authenticity.model_card_diff import build_model_card_diff
        evidence_list = [
            {"type": "identity_collision", "direction": "contradicts_claim", "raw_snippet": "I am Claude"}
        ]
        diff = build_model_card_diff(
            claimed_model="gpt-4",
            suspected_model="claude",
            wrapper_probability=0.90,
            risk_level="fake",
            evidence_list=evidence_list,
        )
        d = diff.to_dict()
        assert d["collision_snippets"] == ["I am Claude"]
