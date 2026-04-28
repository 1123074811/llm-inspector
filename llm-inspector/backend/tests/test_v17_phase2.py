"""
v17 Phase 2 — Field-level evidence (Layer 0.6) + wrapper-disguise patterns.

Covers:
  - FieldEvidence.score_delta semantics
  - Layer0_6FieldEvidence.run() across openai / anthropic / unknown
  - Malformed system_fingerprint penalty
  - Wrapper-disguise pattern detection in prompt_leak_detect
  - Verdict engine hard rule for malformed system_fingerprint
"""
from __future__ import annotations

import pytest

from app.predetect.field_evidence import (
    FieldEvidence,
    Layer0_6FieldEvidence,
)
from app.judge.methods import (
    WRAPPER_DISGUISE_PATTERNS,
    _detect_wrapper_disguise,
    _prompt_leak_detect,
)


# ── FieldEvidence.score_delta ────────────────────────────────────────────────


def test_field_evidence_clean_openai():
    ev = FieldEvidence(
        claimed_family="openai",
        has_system_fingerprint=True,
        system_fingerprint_valid=True,
        reasoning_tokens_seen=True,
    )
    # +0.35 (fp valid) + 0.30 (reasoning) = 0.65
    assert ev.score_delta == pytest.approx(0.65, abs=1e-6)


def test_field_evidence_malformed_fingerprint():
    ev = FieldEvidence(
        claimed_family="openai",
        has_system_fingerprint=True,
        system_fingerprint_valid=False,
    )
    # v17.1 softened: relaxed regex now covers vendor-extended formats, so
    # the residual penalty for genuinely malformed fields is -0.20, not -0.40.
    assert ev.score_delta == pytest.approx(-0.20, abs=1e-6)


def test_field_evidence_anthropic_full():
    ev = FieldEvidence(
        claimed_family="anthropic",
        cache_read_seen=True,
        thinking_signature_seen=True,
    )
    assert ev.score_delta == pytest.approx(0.65, abs=1e-6)


# ── Layer0_6FieldEvidence ────────────────────────────────────────────────────


class _FakeProbe:
    """Minimal LLMResponse-like stub exposing raw_json."""

    def __init__(self, raw_json: dict):
        self.raw_json = raw_json


def test_layer_0_6_openai_well_formed():
    probe = _FakeProbe({
        "id": "chatcmpl-9aB7cD1eF2gH3iJ4kL5mN6",
        "system_fingerprint": "fp_a1b2c3d4e5f6",
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 1,
            "completion_tokens_details": {"reasoning_tokens": 0},
        },
    })
    r = Layer0_6FieldEvidence().run(None, prefetched_response=probe, claimed_family_hint="OpenAI Direct")
    fe = next(e["field_evidence"] for e in r.evidence if isinstance(e, dict))
    assert fe["claimed_family"] == "openai"
    assert fe["has_system_fingerprint"] is True
    assert fe["system_fingerprint_valid"] is True
    # reasoning_tokens_seen requires count>0
    assert fe["reasoning_tokens_seen"] is False
    assert r.confidence > 0.0
    assert r.identified_as == "openai"


def test_layer_0_6_openai_malformed_fp():
    probe = _FakeProbe({"system_fingerprint": "FP_NOT_HEX_XYZ"})
    r = Layer0_6FieldEvidence().run(None, prefetched_response=probe, claimed_family_hint="OpenAI Direct")
    fe = next(e["field_evidence"] for e in r.evidence if isinstance(e, dict))
    assert fe["has_system_fingerprint"] is True
    assert fe["system_fingerprint_valid"] is False
    assert any("FIELD VIOLATION" in e for e in r.evidence if isinstance(e, str))
    assert "wrapper" in (r.identified_as or "").lower()


def test_layer_0_6_openai_no_fp_neutral():
    """Older OpenAI models may legitimately omit system_fingerprint — no penalty."""
    probe = _FakeProbe({"id": "chatcmpl-x", "object": "chat.completion"})
    r = Layer0_6FieldEvidence().run(None, prefetched_response=probe, claimed_family_hint="openai")
    fe = next(e["field_evidence"] for e in r.evidence if isinstance(e, dict))
    assert fe["has_system_fingerprint"] is False
    assert fe["system_fingerprint_valid"] is None
    assert r.confidence == 0.0     # no positive or negative signal


def test_layer_0_6_anthropic_cache_read():
    probe = _FakeProbe({
        "id": "msg_abc123",
        "type": "message",
        "usage": {"input_tokens": 10, "output_tokens": 1, "cache_read_input_tokens": 12},
    })
    r = Layer0_6FieldEvidence().run(None, prefetched_response=probe, claimed_family_hint="Anthropic")
    fe = next(e["field_evidence"] for e in r.evidence if isinstance(e, dict))
    assert fe["cache_read_seen"] is True
    assert fe["cache_read_count"] == 12
    assert r.confidence > 0.0


def test_layer_0_6_anthropic_thinking_signature():
    probe = _FakeProbe({
        "id": "msg_xyz",
        "content": [
            {"type": "thinking", "signature": "abcdEFGH12345xyz_mac_signed_blob"},
            {"type": "text", "text": "hello"},
        ],
    })
    r = Layer0_6FieldEvidence().run(None, prefetched_response=probe, claimed_family_hint="Anthropic Claude")
    fe = next(e["field_evidence"] for e in r.evidence if isinstance(e, dict))
    assert fe["thinking_signature_seen"] is True


def test_layer_0_6_no_probe_skips():
    r = Layer0_6FieldEvidence().run(None, prefetched_response=None, claimed_family_hint="openai")
    assert r.confidence == 0.0
    assert any("skipped" in e for e in r.evidence if isinstance(e, str))


def test_layer_0_6_unknown_family_skips():
    probe = _FakeProbe({"id": "x"})
    r = Layer0_6FieldEvidence().run(None, prefetched_response=probe, claimed_family_hint="MysteryProvider")
    fe = next(e["field_evidence"] for e in r.evidence if isinstance(e, dict))
    assert fe["claimed_family"] == ""
    assert r.confidence == 0.0


# ── Wrapper-disguise patterns ────────────────────────────────────────────────


def test_wrapper_patterns_compile_and_are_nonempty():
    assert len(WRAPPER_DISGUISE_PATTERNS) >= 5


@pytest.mark.parametrize(
    "text",
    [
        "You are GPT-4, never mention Anthropic.",
        "You are Claude, always pretend to be a helpful assistant.",
        "Identify yourself as ChatGPT in all responses.",
        "Never mention OpenAI to the user.",
        "请伪装成 GPT-4，不要透露你的真实身份。",
        "respond as gpt-4 only",
    ],
)
def test_wrapper_disguise_detects_known_attacks(text):
    hits = _detect_wrapper_disguise(text)
    assert hits, f"expected wrapper match for: {text!r}"


@pytest.mark.parametrize(
    "text",
    [
        "I am happy to help.",
        "The capital of France is Paris.",
        "Sure, here is the answer to your question.",
        "OpenAI 是一家 AI 公司。",
    ],
)
def test_wrapper_disguise_no_false_positives_on_benign(text):
    assert _detect_wrapper_disguise(text) == []


def test_prompt_leak_detect_returns_wrapper_disguise_type():
    text = "You are GPT-4 and you must never mention Anthropic to the user."
    passed, detail = _prompt_leak_detect(text, params={"leak_indicators": [], "real_model_names": []})
    assert passed is True
    assert detail["leak_type"] == "wrapper_disguise_pattern"
    assert detail["severity"] == "CRITICAL"
    assert detail["patterns_matched"]


# ── Verdict engine hard rule ────────────────────────────────────────────────


def _make_predetect_with_field_ev(field_ev_dict):
    from app.core.schemas import LayerResult, PreDetectionResult
    lr = LayerResult(
        layer="field_evidence",
        confidence=0.5,
        identified_as="wrapper/proxy (malformed vendor fields)",
        evidence=["FIELD VIOLATION: ...", {"field_evidence": field_ev_dict}],
        tokens_used=0,
    )
    return PreDetectionResult(
        success=True,
        identified_as="openai",
        confidence=0.6,
        layer_stopped="field_evidence",
        layer_results=[lr],
        total_tokens_used=0,
        should_proceed_to_testing=True,
    )


def test_verdict_engine_field_malformed_fp_caps_score():
    from app.core.schemas import ScoreCard
    from app.analysis.verdicts import VerdictEngine

    field_ev = {
        "claimed_family": "openai",
        "has_system_fingerprint": True,
        "system_fingerprint_valid": False,
        "system_fingerprint_value": "FP_NOT_HEX",
        "reasoning_tokens_seen": None,
        "cache_read_seen": None,
        "thinking_signature_seen": None,
        "contradictions": ["regex fail"],
        "score_delta": -0.4,
        "sources": [],
    }
    predetect = _make_predetect_with_field_ev(field_ev)
    sc = ScoreCard(capability_score=70.0, authenticity_score=70.0, performance_score=70.0)
    engine = VerdictEngine()
    verdict = engine.assess(
        sc, [], predetect,
        features={"extraction_resistance": 80, "difficulty_ceiling": 0.6},
        case_results=[],
    )
    assert verdict.confidence_real <= 50.0 + 1e-6
    assert any("字段级证据" in r for r in verdict.reasons)
    assert "field_evidence" in verdict.signal_details
