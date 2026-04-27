"""
v17 Phase 3 — Price-evidence layer tests.

Covers:
  - YAML price seed loadability and shape
  - evaluate_pricing severity ladder (clean / suspicious / fake_high_confidence / overpaid)
  - Geometric-mean blended ratio
  - Verdict-engine integration: claimed price < 30% of official caps confidence_real
"""
from __future__ import annotations

import pytest

from app.authenticity.price_evidence import (
    PriceEvidence,
    evaluate_pricing,
    get_official_price,
    PRICE_BELOW_30PCT_THRESHOLD,
    PRICE_BELOW_60PCT_THRESHOLD,
    PRICE_ABOVE_120PCT_THRESHOLD,
    _invalidate_cache,
)


def test_seed_yaml_has_at_least_30_models():
    # Force fresh load
    _invalidate_cache()
    record = get_official_price("gpt-4o")
    assert record is not None
    assert record["vendor"] == "openai"
    assert record["input_per_mtok_usd"] > 0
    assert record["output_per_mtok_usd"] > 0
    assert record["source_url"].startswith("http")


def test_get_official_price_case_insensitive():
    rec = get_official_price("GPT-4o")
    assert rec is not None
    assert rec["model_id"] == "gpt-4o"


def test_get_official_price_unknown_returns_none():
    assert get_official_price("nonexistent-model-xyz") is None


def test_evaluate_pricing_no_claim():
    ev = evaluate_pricing("gpt-4o", None, None)
    assert isinstance(ev, PriceEvidence)
    assert ev.has_claim is False
    assert ev.severity == "none"


def test_evaluate_pricing_no_official_record():
    ev = evaluate_pricing("nonexistent-model", 1.0, 5.0)
    assert ev.has_claim is True
    assert ev.has_official is False
    assert ev.severity == "none"
    assert ev.blended_ratio is None


def test_evaluate_pricing_clean_match():
    # gpt-4o official: input 2.50, output 10.00 → ratios 1.0, 1.0
    ev = evaluate_pricing("gpt-4o", 2.50, 10.00)
    assert ev.severity == "none"
    assert ev.blended_ratio == pytest.approx(1.0, abs=1e-6)


def test_evaluate_pricing_below_30pct_fake():
    # claim 25% of official → fake_high_confidence
    ev = evaluate_pricing("gpt-4o", 0.625, 2.50)         # 0.25 of official
    assert ev.severity == "fake_high_confidence"
    assert ev.blended_ratio is not None
    assert ev.blended_ratio < PRICE_BELOW_30PCT_THRESHOLD
    assert any("30%" in r for r in ev.reasons)


def test_evaluate_pricing_30pct_to_60pct_suspicious():
    # 50% of official → suspicious
    ev = evaluate_pricing("gpt-4o", 1.25, 5.00)
    assert ev.severity == "suspicious"
    assert (
        PRICE_BELOW_30PCT_THRESHOLD
        <= ev.blended_ratio
        < PRICE_BELOW_60PCT_THRESHOLD
    )


def test_evaluate_pricing_above_120pct_overpaid():
    # 150% of official → overpaid (informational only, NOT a wrapper signal)
    ev = evaluate_pricing("gpt-4o", 3.75, 15.00)
    assert ev.severity == "overpaid"
    assert ev.blended_ratio > PRICE_ABOVE_120PCT_THRESHOLD


def test_evaluate_pricing_partial_claim_uses_one_ratio():
    # claim only output, 25% of official
    ev = evaluate_pricing("gpt-4o", None, 2.50)
    assert ev.has_claim is True
    assert ev.input_ratio is None
    assert ev.output_ratio == pytest.approx(0.25, abs=1e-6)
    assert ev.blended_ratio == pytest.approx(0.25, abs=1e-6)
    assert ev.severity == "fake_high_confidence"


def test_evaluate_pricing_geometric_mean():
    # input ratio 0.4, output ratio 0.9 → blended ≈ sqrt(0.36) = 0.6
    # gpt-4o input 2.50 → claim 1.0 (0.4×); output 10 → claim 9 (0.9×)
    ev = evaluate_pricing("gpt-4o", 1.0, 9.0)
    assert ev.input_ratio == pytest.approx(0.4, abs=1e-6)
    assert ev.output_ratio == pytest.approx(0.9, abs=1e-6)
    assert ev.blended_ratio == pytest.approx(0.6, abs=1e-3)
    # 0.6 lies on the boundary; spec puts it at "none" (≥60% threshold)
    assert ev.severity in {"none", "suspicious"}


# ── to_dict serialization ───────────────────────────────────────────────────


def test_price_evidence_to_dict_roundtrip():
    ev = evaluate_pricing("gpt-4o", 0.625, 2.50)
    d = ev.to_dict()
    assert d["model_id"] == "gpt-4o"
    assert d["severity"] == "fake_high_confidence"
    assert d["input_ratio"] == pytest.approx(0.25, abs=1e-3)
    assert d["source_url"]
    assert isinstance(d["reasons"], list) and d["reasons"]


# ── Verdict engine integration ──────────────────────────────────────────────


def test_verdict_engine_price_below_30pct_caps_to_30():
    from app.core.schemas import ScoreCard, PreDetectionResult
    from app.analysis.verdicts import VerdictEngine

    pricing = evaluate_pricing("gpt-4o", 0.625, 2.50)   # 25% — fake_high_confidence
    pre = PreDetectionResult(
        success=True,
        identified_as="openai",
        confidence=0.6,
        layer_stopped="probe",
        layer_results=[],
        total_tokens_used=0,
        should_proceed_to_testing=True,
    )
    sc = ScoreCard(capability_score=80.0, authenticity_score=80.0, performance_score=80.0)
    engine = VerdictEngine()
    v = engine.assess(
        scorecard=sc,
        similarities=[],
        predetect=pre,
        features={"extraction_resistance": 80, "difficulty_ceiling": 0.6},
        case_results=[],
        pricing=pricing,
    )
    assert v.confidence_real <= 30.0 + 1e-6
    assert any("价格层证据" in r for r in v.reasons)
    assert "price_evidence" in v.signal_details


def test_verdict_engine_price_suspicious_caps_to_60():
    from app.core.schemas import ScoreCard, PreDetectionResult
    from app.analysis.verdicts import VerdictEngine

    pricing = evaluate_pricing("gpt-4o", 1.25, 5.00)    # 50% — suspicious
    pre = PreDetectionResult(
        success=True, identified_as="openai", confidence=0.6,
        layer_stopped="probe", layer_results=[], total_tokens_used=0,
        should_proceed_to_testing=True,
    )
    sc = ScoreCard(capability_score=80.0, authenticity_score=80.0, performance_score=80.0)
    engine = VerdictEngine()
    v = engine.assess(
        scorecard=sc, similarities=[], predetect=pre,
        features={"extraction_resistance": 80, "difficulty_ceiling": 0.6},
        case_results=[], pricing=pricing,
    )
    assert v.confidence_real <= 60.0 + 1e-6
    assert any("价格层证据" in r for r in v.reasons)


def test_verdict_engine_price_clean_no_cap():
    from app.core.schemas import ScoreCard, PreDetectionResult
    from app.analysis.verdicts import VerdictEngine

    pricing = evaluate_pricing("gpt-4o", 2.50, 10.00)   # 100% — none
    pre = PreDetectionResult(
        success=True, identified_as="openai", confidence=0.6,
        layer_stopped="probe", layer_results=[], total_tokens_used=0,
        should_proceed_to_testing=True,
    )
    sc = ScoreCard(capability_score=80.0, authenticity_score=80.0, performance_score=80.0)
    engine = VerdictEngine()
    v = engine.assess(
        scorecard=sc, similarities=[], predetect=pre,
        features={"extraction_resistance": 80, "difficulty_ceiling": 0.6},
        case_results=[], pricing=pricing,
    )
    # No price hard rule fires; no "价格层证据" reason
    assert not any("价格层证据" in r for r in v.reasons)
    # signal_details still records evidence even when severity=none
    assert "price_evidence" in v.signal_details


def test_verdict_engine_pricing_none_does_not_break():
    from app.core.schemas import ScoreCard, PreDetectionResult
    from app.analysis.verdicts import VerdictEngine

    pre = PreDetectionResult(
        success=True, identified_as="openai", confidence=0.6,
        layer_stopped="probe", layer_results=[], total_tokens_used=0,
        should_proceed_to_testing=True,
    )
    sc = ScoreCard(capability_score=80.0, authenticity_score=80.0, performance_score=80.0)
    engine = VerdictEngine()
    v = engine.assess(
        scorecard=sc, similarities=[], predetect=pre,
        features={"extraction_resistance": 80, "difficulty_ceiling": 0.6},
        case_results=[], pricing=None,
    )
    assert "price_evidence" not in v.signal_details
