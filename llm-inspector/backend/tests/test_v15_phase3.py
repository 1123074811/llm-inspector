"""Tests for v15 Phase 3: None≠0 score semantics."""
from __future__ import annotations
import pytest


def test_scorecard_none_values_stay_null():
    from app.core.schemas import ScoreCard
    sc = ScoreCard()
    # Fields with None default should output null
    d = sc.to_dict()
    assert d["breakdown"]["reasoning"] is None
    assert d["breakdown"]["coding"] is None
    assert d["breakdown"]["speed"] is None
    assert d["breakdown"]["stability"] is None


def test_scorecard_zero_is_zero():
    from app.core.schemas import ScoreCard
    sc = ScoreCard()
    sc.reasoning_score = 0.0
    d = sc.to_dict()
    assert d["breakdown"]["reasoning"] == 0


def test_scorecard_measurement_block():
    from app.core.schemas import ScoreCard
    sc = ScoreCard()
    sc.reasoning_score = 0.8
    sc.coding_score = 0.6
    d = sc.to_dict()
    assert "measurement" in d
    assert d["measurement"]["measured_dims"] >= 2


def test_scorecard_total_score_null_safe():
    from app.core.schemas import ScoreCard
    sc = ScoreCard()
    sc.total_score = None  # test None total score
    d = sc.to_dict()
    assert d["total_score"] is None


def test_scorecard_pct_helper_precision():
    from app.core.schemas import ScoreCard
    sc = ScoreCard()
    sc.reasoning_score = 0.756
    d = sc.to_dict()
    # round(0.756 * 100) == 76
    assert d["breakdown"]["reasoning"] == 76


def test_timing_refs_loadable():
    from app.predetect.layers_l18_l19 import _load_timing_refs
    refs = _load_timing_refs()
    assert isinstance(refs, dict)
    # Should have at least some families
    assert len(refs) >= 1


def test_dist_refs_loadable():
    from app.predetect.layers_l18_l19 import _load_dist_refs
    refs = _load_dist_refs()
    assert isinstance(refs, dict)


def test_timing_refs_json_valid():
    import json, os
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "app", "_data", "timing_refs.json"
    )
    assert os.path.exists(path), f"timing_refs.json not found at {path}"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert "families" in data
    assert "_provenance" in data


def test_token_dist_refs_json_valid():
    import json, os
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "app", "_data", "token_dist_refs.json"
    )
    assert os.path.exists(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert "families" in data


def test_timing_refs_has_expected_families():
    from app.predetect.layers_l18_l19 import _load_timing_refs
    refs = _load_timing_refs()
    for family in ("claude", "gpt", "gemini"):
        assert family in refs, f"Expected family '{family}' in timing_refs"
        assert "ttft_ms_mean" in refs[family]
        assert "ttft_ms_std" in refs[family]


def test_dist_refs_has_expected_families():
    from app.predetect.layers_l18_l19 import _load_dist_refs
    refs = _load_dist_refs()
    for family in ("claude", "gpt"):
        assert family in refs, f"Expected family '{family}' in dist_refs"
        assert "avg_len" in refs[family]


def test_timing_refs_sampled_false_by_default():
    from app.predetect.layers_l18_l19 import _load_timing_refs
    refs = _load_timing_refs()
    for family, data in refs.items():
        assert data.get("sampled") is False, (
            f"Family '{family}' should have sampled=false (placeholder data)"
        )


def test_scorecard_breakdown_extra_fields_null_when_missing():
    from app.core.schemas import ScoreCard
    sc = ScoreCard()
    d = sc.to_dict()
    # breakdown extras that come from card.breakdown dict — should be None when not set
    assert d["breakdown"]["knowledge_score"] is None
    assert d["breakdown"]["tool_use_score"] is None
    assert d["breakdown"]["extraction_resistance"] is None
