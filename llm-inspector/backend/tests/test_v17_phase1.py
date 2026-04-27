"""
v17 Phase 1 — Protocol-level hard evidence (Layer 0.5) tests.

Covers:
  - ProtocolEvidence.score_delta sign and magnitude
  - Error schema check across OpenAI / Anthropic / Google
  - ID prefix check
  - Layer0_5ProtocolValidator.run() with mocked adapter
  - VerdictEngine hard rules (protocol_error_schema_cap / protocol_id_prefix_cap
    / protocol_auth_pollution_cap)
"""
from __future__ import annotations

import pytest

from app.predetect.protocol_validator import (
    Layer0_5ProtocolValidator,
    ProtocolEvidence,
    _check_error_schema,
    _check_id_prefix,
    _normalize_family,
)


class _FakeAdapter:
    """Minimal adapter stub: only ``bad_request`` is exercised by L0.5."""

    def __init__(self, bad_body: dict | None, base_url: str = "https://example.test"):
        self._bad_body = bad_body or {}
        self.base_url = base_url

    def bad_request(self) -> dict:
        return {"status_code": 400, "body": self._bad_body, "latency_ms": 7}


# ── ProtocolEvidence.score_delta ────────────────────────────────────────────


def test_protocol_evidence_score_delta_clean():
    ev = ProtocolEvidence(
        claimed_family="openai",
        error_schema_match=True,
        response_id_prefix_match=True,
    )
    # Affirmative evidence on both = +0.30
    assert ev.score_delta == pytest.approx(0.30, abs=1e-6)


def test_protocol_evidence_score_delta_violation():
    ev = ProtocolEvidence(
        claimed_family="anthropic",
        error_schema_match=False,
        response_id_prefix_match=False,
        cross_family_auth_pollution=True,
    )
    # -0.5 (auth pollution) + -0.4 (schema) + -0.3 (id prefix) = -1.2
    assert ev.score_delta == pytest.approx(-1.2, abs=1e-6)


# ── _check_error_schema ─────────────────────────────────────────────────────


def test_error_schema_openai_valid():
    body = {"error": {"type": "invalid_request_error", "message": "missing field"}}
    matches, _ = _check_error_schema("openai", body)
    assert matches is True


def test_error_schema_openai_missing_keys():
    body = {"error": {"message": "boom"}}  # missing 'type'
    matches, reason = _check_error_schema("openai", body)
    assert matches is False
    assert "type" in reason


def test_error_schema_anthropic_valid():
    body = {
        "type": "error",
        "error": {"type": "authentication_error", "message": "invalid key"},
    }
    matches, _ = _check_error_schema("anthropic", body)
    assert matches is True


def test_error_schema_anthropic_missing_top_type():
    # Body looks like OpenAI but claimed family is anthropic → contradiction
    body = {"error": {"type": "auth", "message": "x"}}
    matches, reason = _check_error_schema("anthropic", body)
    assert matches is False
    assert "anthropic" in reason.lower() or "type='error'" in reason


def test_error_schema_google_valid():
    body = {
        "error": {
            "code": 401,
            "message": "API key not valid",
            "status": "UNAUTHENTICATED",
        }
    }
    matches, _ = _check_error_schema("google", body)
    assert matches is True


def test_error_schema_no_error_returns_none():
    matches, reason = _check_error_schema("openai", {"id": "chatcmpl-x", "object": "chat.completion"})
    assert matches is None
    assert reason == ""


# ── _check_id_prefix ────────────────────────────────────────────────────────


def test_id_prefix_openai_match():
    body = {"id": "chatcmpl-9aB7cD1eF2gH3iJ4kL5mN6oP", "object": "chat.completion"}
    matches, _ = _check_id_prefix("openai", body)
    assert matches is True


def test_id_prefix_openai_mismatch():
    body = {"id": "msg_01ABCDEFGH123456", "object": "chat.completion"}
    matches, reason = _check_id_prefix("openai", body)
    assert matches is False
    assert "chatcmpl" in reason


def test_id_prefix_anthropic_match():
    body = {"id": "msg_01XYZ123abc456DEF789", "type": "message"}
    matches, _ = _check_id_prefix("anthropic", body)
    assert matches is True


def test_id_prefix_no_id_returns_none():
    matches, _ = _check_id_prefix("openai", {})
    assert matches is None


# ── _normalize_family ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("OpenAI Direct", "openai"),
        ("Azure OpenAI", "openai"),
        ("Anthropic Claude", "anthropic"),
        ("Google Gemini", "google"),
        ("Vertex AI", "google"),
        ("DeepSeek", ""),
        ("", ""),
        (None, ""),
    ],
)
def test_normalize_family(raw, expected):
    assert _normalize_family(raw) == expected


# ── Layer0_5ProtocolValidator integration ───────────────────────────────────


def test_layer0_5_clean_openai():
    adapter = _FakeAdapter({"error": {"type": "invalid_request_error", "message": "x"}})
    r = Layer0_5ProtocolValidator().run(adapter, claimed_family_hint="OpenAI Direct")
    assert r.layer == "protocol"
    assert r.tokens_used == 0
    # Find structured evidence
    proto = next(e["protocol_evidence"] for e in r.evidence if isinstance(e, dict))
    assert proto["claimed_family"] == "openai"
    assert proto["error_schema_match"] is True
    assert proto["cross_family_auth_pollution"] is False


def test_layer0_5_schema_violation_anthropic():
    # Claimed family is anthropic but body is OpenAI-shaped
    adapter = _FakeAdapter({"error": {"type": "auth", "message": "boom"}})
    r = Layer0_5ProtocolValidator().run(adapter, claimed_family_hint="Anthropic Claude")
    proto = next(e["protocol_evidence"] for e in r.evidence if isinstance(e, dict))
    assert proto["error_schema_match"] is False
    assert any("PROTOCOL VIOLATION" in e for e in r.evidence if isinstance(e, str))
    assert r.confidence > 0.0
    # Identified as wrapper since strong negative score_delta
    assert "wrapper" in (r.identified_as or "").lower()


def test_layer0_5_handles_adapter_failure():
    class _BrokenAdapter:
        base_url = "https://broken.test"

        def bad_request(self):
            raise RuntimeError("network down")

    r = Layer0_5ProtocolValidator().run(_BrokenAdapter(), claimed_family_hint="openai")
    assert r.layer == "protocol"
    # No evidence either way: schema check is None, no contradictions
    proto = next(e["protocol_evidence"] for e in r.evidence if isinstance(e, dict))
    assert proto["error_schema_match"] is None


def test_layer0_5_unknown_family_no_assertions():
    adapter = _FakeAdapter({"error": {"type": "x", "message": "y"}})
    r = Layer0_5ProtocolValidator().run(adapter, claimed_family_hint="MysteryProvider")
    proto = next(e["protocol_evidence"] for e in r.evidence if isinstance(e, dict))
    assert proto["claimed_family"] == ""
    assert proto["error_schema_match"] is None
    assert r.confidence == 0.0


# ── VerdictEngine integration smoke test ────────────────────────────────────


def _make_minimal_predetect(layer_results):
    from app.core.schemas import PreDetectionResult
    return PreDetectionResult(
        success=True,
        identified_as="openai",
        confidence=0.6,
        layer_stopped="protocol",
        layer_results=layer_results,
        total_tokens_used=0,
        should_proceed_to_testing=True,
    )


def test_verdict_engine_protocol_hard_rule_fires():
    from app.core.schemas import LayerResult, ScoreCard
    from app.analysis.verdicts import VerdictEngine

    proto_ev = {
        "claimed_family": "openai",
        "error_schema_match": False,
        "response_id_prefix_match": None,
        "cross_family_auth_pollution": False,
        "contradictions": ["error envelope wrong"],
        "score_delta": -0.4,
        "sources": [],
    }
    lr = LayerResult(
        layer="protocol",
        confidence=0.5,
        identified_as="wrapper/proxy (protocol contradictions)",
        evidence=["PROTOCOL VIOLATION: ...", {"protocol_evidence": proto_ev}],
        tokens_used=0,
    )
    predetect = _make_minimal_predetect([lr])

    sc = ScoreCard(
        capability_score=70.0,
        authenticity_score=70.0,
        performance_score=70.0,
    )
    engine = VerdictEngine()
    verdict = engine.assess(
        sc, [], predetect,
        features={"extraction_resistance": 80, "difficulty_ceiling": 0.6},
        case_results=[],
    )
    # confidence_real should be capped at <=50 due to protocol_error_schema_cap
    assert verdict.confidence_real <= 50.0 + 1e-6
    # Reason should mention protocol layer evidence
    assert any("协议层证据" in r for r in verdict.reasons)
    # signal_details should carry the structured evidence
    assert "protocol_evidence" in verdict.signal_details
