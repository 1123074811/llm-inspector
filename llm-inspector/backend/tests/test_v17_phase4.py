"""
v17 Phase 4 — Timing baseline truthification tests.

Real API sampling is NOT exercised here (it requires user keys and incurs
token cost).  Instead this module validates:

  * the offline helpers (``_quantiles``, ``_sha256_of_records``)
  * the persistence/merge logic (``_save_results``)
  * the L18/L19 placeholder gate that returns ``confidence=0`` and skip
    reason ``all_baselines_placeholder`` when no family has ``sampled=True``
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make the script importable as a module (it lives outside the app package)
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import sample_timing_references as srt  # noqa: E402


# ── Helper functions ────────────────────────────────────────────────────────


def test_quantiles_empty_returns_zeros():
    q = srt._quantiles([])
    assert q == {"p10": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0}


def test_quantiles_single_value_constant():
    q = srt._quantiles([42.0])
    assert q == {"p10": 42.0, "p25": 42.0, "p50": 42.0, "p75": 42.0, "p90": 42.0}


def test_quantiles_known_distribution():
    # Symmetric around 50; p50 should be 50.0
    values = list(range(1, 101))
    q = srt._quantiles(values)
    assert q["p50"] == pytest.approx(50.5, abs=0.5)
    assert q["p10"] < q["p25"] < q["p50"] < q["p75"] < q["p90"]


def test_sha256_deterministic():
    rec = [{"ttft": 100.0, "tps": 50.0, "len": 20}]
    h1 = srt._sha256_of_records(rec)
    h2 = srt._sha256_of_records(rec)
    assert h1 == h2 and len(h1) == 64


def test_sha256_changes_with_payload():
    h1 = srt._sha256_of_records([{"x": 1}])
    h2 = srt._sha256_of_records([{"x": 2}])
    assert h1 != h2


# ── _save_results merge + provenance refresh ────────────────────────────────


def test_save_results_creates_file_and_clears_placeholder(tmp_path):
    output = tmp_path / "timing_refs.json"
    output.write_text(json.dumps({
        "_provenance": {
            "sampling_required": True,
            "note": "PLACEHOLDER",
            "version": "v15.0-placeholder",
        },
        "families": {
            "gpt": {"sampled": False, "ttft_ms_mean": 600},
        },
    }), encoding="utf-8")

    fake_data = {
        "sampled": True,
        "sample_size": 100,
        "sampled_at": "2026-04-27T00:00:00Z",
        "model_version": "gpt-4o-mini",
        "ttft_ms_mean": 350.0,
        "ttft_ms_std": 80.0,
        "ttft_ms_quantiles": {"p10": 250, "p25": 300, "p50": 340, "p75": 400, "p90": 480},
        "tps_mean": 120.0,
        "tps_std": 15.0,
        "tps_quantiles": {"p10": 100, "p25": 110, "p50": 120, "p75": 130, "p90": 145},
        "avg_response_len_words": 42.0,
        "repetition_rate_4gram": 0.001,
        "raw_data_sha256": "deadbeef" * 8,
    }
    srt._save_results(str(output), {"gpt": fake_data})

    saved = json.loads(output.read_text(encoding="utf-8"))
    # Placeholder marker cleared, real provenance recorded
    assert saved["_provenance"]["sampling_required"] is False
    assert saved["_provenance"]["version"] == "v17.0-self-measurement"
    assert "placeholder_replaced_at" in saved["_provenance"]
    # Family data updated and quantiles preserved
    assert saved["families"]["gpt"]["sampled"] is True
    assert saved["families"]["gpt"]["ttft_ms_quantiles"]["p90"] == 480


# ── L18 / L19 placeholder gate ──────────────────────────────────────────────


def _make_layer_dicts_with_timings(ttft_ms_seq, tps_seq):
    """Layer 18 reads ttft_ms / tps from prior dict-shaped layer results."""
    return [
        {"layer": "probe", "ttft_ms": ttft, "tps": tps}
        for ttft, tps in zip(ttft_ms_seq, tps_seq)
    ]


def test_layer18_returns_zero_confidence_on_placeholder_baselines(monkeypatch):
    from app.predetect import layers_l18_l19 as ll

    placeholder_refs = {
        "gpt": {"sampled": False, "ttft_ms_mean": 600, "ttft_ms_std": 150},
        "claude": {"sampled": False, "ttft_ms_mean": 800, "ttft_ms_std": 200},
    }
    monkeypatch.setattr(ll, "_load_timing_refs", lambda: placeholder_refs)

    layer = ll.Layer18TimingSideChannel()
    prior = _make_layer_dicts_with_timings([100.0] * 8, [50.0] * 8)
    out = layer.run(adapter=None, model_name="gpt-4o", layer_results_so_far=prior)

    assert out["confidence"] == 0.0
    assert out["skipped"] is True
    assert out["reason"] == "all_baselines_placeholder"
    assert any("placeholder" in e.lower() for e in out["evidence"])


def test_layer18_uses_real_baseline_when_any_sampled(monkeypatch):
    from app.predetect import layers_l18_l19 as ll

    refs = {
        "gpt": {"sampled": True, "ttft_ms_mean": 600, "ttft_ms_std": 150},
        "claude": {"sampled": False, "ttft_ms_mean": 800, "ttft_ms_std": 200},
    }
    monkeypatch.setattr(ll, "_load_timing_refs", lambda: refs)

    layer = ll.Layer18TimingSideChannel()
    prior = _make_layer_dicts_with_timings([600.0] * 8, [50.0] * 8)
    out = layer.run(adapter=None, model_name="gpt-4o", layer_results_so_far=prior)

    assert out["skipped"] is False
    assert out["confidence"] > 0.0
    assert out["closest_family"] == "gpt"
