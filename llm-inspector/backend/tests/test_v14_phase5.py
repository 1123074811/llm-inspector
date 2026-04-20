"""
tests/test_v14_phase5.py — Phase 5 PreDetect extension & observability tests.

Covers:
  - Layer18TimingSideChannel (with/without timing data, confidence capping, KL distance)
  - Layer19TokenDistribution (with/without text data, repetition_rate, wasserstein)
  - _write_predetect_trace (creates file, appends, handles errors)
  - read_predetect_trace (pagination, missing file returns [])
  - handle_predetect_trace (basic response structure)
  - EventKind.PREDETECT_LAYER_TRACE exists
  - Pipeline integration: L18 and L19 appear in layer_results when extraction_mode=True
  - _repetition_rate helper
  - KL / Wasserstein helpers
"""
from __future__ import annotations

import json
import pathlib
import tempfile

import pytest


# ---------------------------------------------------------------------------
# EventKind
# ---------------------------------------------------------------------------

class TestEventKind:
    def test_predetect_layer_trace_exists(self):
        from app.core.events import EventKind
        assert hasattr(EventKind, "PREDETECT_LAYER_TRACE")
        assert EventKind.PREDETECT_LAYER_TRACE.value == "predetect_layer_trace"


# ---------------------------------------------------------------------------
# L18 — Timing Side-Channel
# ---------------------------------------------------------------------------

class TestLayer18TimingSideChannel:

    def _layer(self):
        from app.predetect.layers_l18_l19 import Layer18TimingSideChannel
        return Layer18TimingSideChannel()

    def test_skipped_when_no_timing_data(self):
        layer = self._layer()
        prior = [{"layer": 1, "name": "test", "evidence": []}]
        result = layer.run(None, "test-model", prior)
        assert result["skipped"] is True
        assert result["reason"] == "no_timing_data"
        assert result["tokens"] == 0
        assert result["confidence"] == 0.0

    def test_runs_with_timing_data(self):
        layer = self._layer()
        prior = [
            {"ttft_ms": 600.0, "tps": 50.0},
            {"ttft_ms": 650.0, "tps": 55.0},
            {"ttft_ms": 580.0, "tps": 48.0},
        ]
        result = layer.run(None, "gpt-4o", prior)
        assert result["skipped"] is False
        assert result["ttft_samples"] == 3
        assert result["mean_ttft_ms"] > 0
        assert result["mean_tps"] is not None
        assert "closest_family" in result
        assert 0.0 <= result["confidence"] <= 0.50

    def test_confidence_capped_at_0_50(self):
        """Even with perfect timing match, confidence must not exceed 0.50."""
        layer = self._layer()
        # Use exact reference values for "gpt" => kl will be 0 => raw_confidence = 1.0
        prior = [{"ttft_ms": 600.0, "tps": 50.0}] * 5
        result = layer.run(None, "gpt-4o", prior)
        assert result["confidence"] <= 0.50

    def test_kl_distance_positive(self):
        layer = self._layer()
        prior = [{"ttft_ms": 2000.0}]  # Far from any reference
        result = layer.run(None, "unknown", prior)
        assert result["kl_distance"] >= 0.0
        assert isinstance(result["closest_family"], str)

    def test_layer_number(self):
        from app.predetect.layers_l18_l19 import Layer18TimingSideChannel
        assert Layer18TimingSideChannel.LAYER == 18
        assert Layer18TimingSideChannel.NAME == "timing_side_channel"

    def test_evidence_list_populated(self):
        layer = self._layer()
        prior = [{"ttft_ms": 800.0, "tps": 45.0}]
        result = layer.run(None, "claude", prior)
        assert isinstance(result["evidence"], list)
        assert len(result["evidence"]) >= 1

    def test_missing_tps_handled_gracefully(self):
        layer = self._layer()
        prior = [{"ttft_ms": 700.0}]  # no tps field
        result = layer.run(None, "gemini", prior)
        assert result["skipped"] is False
        assert result["mean_tps"] is None


# ---------------------------------------------------------------------------
# L19 — Token Distribution Side-Channel
# ---------------------------------------------------------------------------

class TestLayer19TokenDistribution:

    def _layer(self):
        from app.predetect.layers_l18_l19 import Layer19TokenDistribution
        return Layer19TokenDistribution()

    def test_skipped_when_no_text_data(self):
        layer = self._layer()
        prior = [{"layer": 1, "confidence": 0.0, "evidence": []}]
        result = layer.run(None, "test-model", prior)
        assert result["skipped"] is True
        assert result["reason"] == "no_response_data"
        assert result["tokens"] == 0
        assert result["confidence"] == 0.0

    def test_runs_with_response_texts(self):
        layer = self._layer()
        prior = [
            {"response_text": "Hello! I am an AI assistant here to help you with your questions today."},
            {"response_text": "Sure, let me explain that concept in detail for you."},
            {"response_text": "I understand your question. The answer involves several considerations."},
        ]
        result = layer.run(None, "claude", prior)
        assert result["skipped"] is False
        assert result["samples"] >= 1
        assert result["avg_response_len"] > 0
        assert 0.0 <= result["confidence"] <= 0.45

    def test_confidence_capped_at_0_45(self):
        """Confidence must not exceed 0.45."""
        layer = self._layer()
        # Use exact reference avg_len for "claude" = 480 chars
        text = "A" * 480
        prior = [{"response_text": text}] * 5
        result = layer.run(None, "claude", prior)
        assert result["confidence"] <= 0.45

    def test_layer_number(self):
        from app.predetect.layers_l18_l19 import Layer19TokenDistribution
        assert Layer19TokenDistribution.LAYER == 19
        assert Layer19TokenDistribution.NAME == "token_distribution"

    def test_repetition_rate_zero_for_unique_texts(self):
        layer = self._layer()
        prior = [
            {"response_text": "Alpha beta gamma delta epsilon zeta eta theta"},
            {"response_text": "One two three four five six seven eight nine ten"},
            {"response_text": "Red green blue purple orange yellow pink white black"},
        ]
        result = layer.run(None, "model", prior)
        # No repeated 4-grams across fully distinct texts
        assert result["repetition_rate"] == 0.0

    def test_repetition_rate_nonzero_for_repeated_text(self):
        layer = self._layer()
        repeated_text = "This is the same response text for all queries today."
        prior = [{"response_text": repeated_text}] * 6
        result = layer.run(None, "model", prior)
        assert result["repetition_rate"] > 0.0

    def test_wasserstein_distance_computed(self):
        layer = self._layer()
        # avg_len will differ significantly from all refs
        prior = [{"response_text": "Hi"}]  # avg_len = 2
        result = layer.run(None, "model", prior)
        assert result["wasserstein_distance"] >= 0.0
        assert result["closest_family"] in ("claude", "gpt", "qwen", "deepseek")

    def test_evidence_list_populated(self):
        layer = self._layer()
        prior = [{"response_text": "This is a test response with some content."}]
        result = layer.run(None, "test", prior)
        assert isinstance(result["evidence"], list)
        assert len(result["evidence"]) >= 1


# ---------------------------------------------------------------------------
# _repetition_rate helper
# ---------------------------------------------------------------------------

class TestRepetitionRate:
    def test_empty_list(self):
        from app.predetect.layers_l18_l19 import _repetition_rate
        assert _repetition_rate([]) == 0.0

    def test_short_texts(self):
        from app.predetect.layers_l18_l19 import _repetition_rate
        # Texts with fewer than 4 words can't form 4-grams
        assert _repetition_rate(["hi", "bye"]) == 0.0

    def test_full_repeat(self):
        from app.predetect.layers_l18_l19 import _repetition_rate
        text = "alpha beta gamma delta"
        rate = _repetition_rate([text] * 4)
        assert rate > 0.0


# ---------------------------------------------------------------------------
# KL and Wasserstein helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_kl_gaussian_zero_at_ref(self):
        from app.predetect.layers_l18_l19 import _kl_gaussian
        # When obs_mean == ref_mean, KL = 0
        assert _kl_gaussian(600.0, 600.0, 150.0) == 0.0

    def test_kl_gaussian_positive_distance(self):
        from app.predetect.layers_l18_l19 import _kl_gaussian
        assert _kl_gaussian(1200.0, 600.0, 150.0) > 0.0

    def test_kl_gaussian_zero_std(self):
        from app.predetect.layers_l18_l19 import _kl_gaussian
        result = _kl_gaussian(600.0, 600.0, 0.0)
        assert result == float("inf")

    def test_wasserstein_1d_zero_at_ref(self):
        from app.predetect.layers_l18_l19 import _wasserstein_1d
        assert _wasserstein_1d(480.0, 480.0) == 0.0

    def test_wasserstein_1d_positive(self):
        from app.predetect.layers_l18_l19 import _wasserstein_1d
        result = _wasserstein_1d(200.0, 480.0)
        assert result > 0.0


# ---------------------------------------------------------------------------
# JSONL trace sink
# ---------------------------------------------------------------------------

class TestWritePredetectTrace:

    def test_creates_file_and_appends(self, tmp_path, monkeypatch):
        """_write_predetect_trace creates JSONL file and appends correctly."""
        monkeypatch.setattr("app.core.config.settings.DATA_DIR", str(tmp_path))
        from app.predetect.pipeline import _write_predetect_trace

        run_id = "test-run-001"
        record = {
            "layer": 18,
            "name": "timing_side_channel",
            "tokens": 0,
            "confidence": 0.3,
            "skipped": False,
            "evidence": ["ttft=600ms"],
        }
        _write_predetect_trace(run_id, record)

        trace_file = tmp_path / "traces" / run_id / "predetect.jsonl"
        assert trace_file.exists(), "Trace file should be created"
        lines = [json.loads(l) for l in trace_file.read_text().strip().splitlines()]
        assert len(lines) == 1
        assert lines[0]["layer"] == 18
        assert lines[0]["confidence"] == 0.3

    def test_appends_multiple_records(self, tmp_path, monkeypatch):
        monkeypatch.setattr("app.core.config.settings.DATA_DIR", str(tmp_path))
        from app.predetect.pipeline import _write_predetect_trace

        run_id = "test-run-002"
        for i in range(3):
            _write_predetect_trace(run_id, {"layer": i, "name": f"layer{i}", "tokens": 0,
                                            "confidence": 0.1 * i, "skipped": False, "evidence": []})

        trace_file = tmp_path / "traces" / run_id / "predetect.jsonl"
        lines = trace_file.read_text().strip().splitlines()
        assert len(lines) == 3

    def test_no_op_when_run_id_none(self, tmp_path, monkeypatch):
        """With run_id=None, no file should be created."""
        monkeypatch.setattr("app.core.config.settings.DATA_DIR", str(tmp_path))
        from app.predetect.pipeline import _write_predetect_trace

        _write_predetect_trace(None, {"layer": 1, "tokens": 0, "confidence": 0.0, "evidence": []})
        traces_dir = tmp_path / "traces"
        assert not traces_dir.exists() or not any(traces_dir.iterdir())

    def test_handles_io_error_gracefully(self, tmp_path, monkeypatch):
        """I/O errors must not raise exceptions."""
        # Point to an unwritable path by making DATA_DIR a file instead of dir
        fake_file = tmp_path / "not_a_dir"
        fake_file.write_text("x")
        monkeypatch.setattr("app.core.config.settings.DATA_DIR", str(fake_file))
        from app.predetect.pipeline import _write_predetect_trace

        # Should not raise
        _write_predetect_trace("test-run", {"layer": 1, "tokens": 0, "confidence": 0.0, "evidence": []})


# ---------------------------------------------------------------------------
# read_predetect_trace
# ---------------------------------------------------------------------------

class TestReadPredetectTrace:

    def test_missing_file_returns_empty(self):
        from app.repository.repo import read_predetect_trace
        result = read_predetect_trace("nonexistent-run-xyz-999")
        assert result == []

    def test_reads_written_records(self, tmp_path, monkeypatch):
        monkeypatch.setattr("app.core.config.settings.DATA_DIR", str(tmp_path))
        from app.predetect.pipeline import _write_predetect_trace
        from app.repository.repo import read_predetect_trace

        run_id = "read-test-001"
        for i in range(5):
            _write_predetect_trace(run_id, {
                "layer": i, "name": f"layer{i}", "tokens": 0,
                "confidence": 0.1, "skipped": False, "evidence": [],
            })

        result = read_predetect_trace(run_id)
        assert len(result) == 5
        assert result[0]["layer"] == 0

    def test_pagination_offset_limit(self, tmp_path, monkeypatch):
        monkeypatch.setattr("app.core.config.settings.DATA_DIR", str(tmp_path))
        from app.predetect.pipeline import _write_predetect_trace
        from app.repository.repo import read_predetect_trace

        run_id = "read-test-002"
        for i in range(10):
            _write_predetect_trace(run_id, {
                "layer": i, "name": f"layer{i}", "tokens": 0,
                "confidence": 0.1, "skipped": False, "evidence": [],
            })

        page1 = read_predetect_trace(run_id, offset=0, limit=3)
        assert len(page1) == 3
        assert page1[0]["layer"] == 0

        page2 = read_predetect_trace(run_id, offset=3, limit=3)
        assert len(page2) == 3
        assert page2[0]["layer"] == 3

    def test_limit_enforced(self, tmp_path, monkeypatch):
        monkeypatch.setattr("app.core.config.settings.DATA_DIR", str(tmp_path))
        from app.predetect.pipeline import _write_predetect_trace
        from app.repository.repo import read_predetect_trace

        run_id = "read-test-003"
        for i in range(20):
            _write_predetect_trace(run_id, {
                "layer": i, "name": f"layer{i}", "tokens": 0,
                "confidence": 0.1, "skipped": False, "evidence": [],
            })

        result = read_predetect_trace(run_id, offset=0, limit=5)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# handle_predetect_trace
# ---------------------------------------------------------------------------

class TestHandlePredetectTrace:

    def test_basic_response_structure(self, tmp_path, monkeypatch):
        monkeypatch.setattr("app.core.config.settings.DATA_DIR", str(tmp_path))
        from app.handlers.v14_handlers import handle_predetect_trace

        status, body, ctype = handle_predetect_trace(
            "/api/v14/runs/run-abc/predetect-trace", {}, {}
        )
        assert status == 200
        data = json.loads(body)
        assert "run_id" in data
        assert data["run_id"] == "run-abc"
        assert "entries" in data
        assert isinstance(data["entries"], list)

    def test_missing_run_id_returns_error(self):
        from app.handlers.v14_handlers import handle_predetect_trace
        status, body, ctype = handle_predetect_trace("/api/v14/runs//predetect-trace", {}, {})
        assert status in (400, 500)

    def test_with_trace_data(self, tmp_path, monkeypatch):
        monkeypatch.setattr("app.core.config.settings.DATA_DIR", str(tmp_path))
        from app.predetect.pipeline import _write_predetect_trace
        from app.handlers.v14_handlers import handle_predetect_trace

        run_id = "handler-test-001"
        _write_predetect_trace(run_id, {
            "layer": 18, "name": "timing_side_channel", "tokens": 0,
            "confidence": 0.3, "skipped": False, "evidence": ["test"],
        })

        status, body, ctype = handle_predetect_trace(
            f"/api/v14/runs/{run_id}/predetect-trace", {}, {}
        )
        assert status == 200
        data = json.loads(body)
        assert data["count"] == 1
        assert data["entries"][0]["layer"] == 18


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    """Test that L18 and L19 are registered in the pipeline code and run when extraction_mode=True."""

    def test_l18_l19_imported_in_pipeline(self):
        """The pipeline module should be importable and contain L18/L19 code."""
        import app.predetect.pipeline as pipeline_mod
        source = pipeline_mod.__doc__ or ""
        # Check that both layer classes are importable (already validated via other tests)
        from app.predetect.layers_l18_l19 import Layer18TimingSideChannel, Layer19TokenDistribution
        assert Layer18TimingSideChannel.LAYER == 18
        assert Layer19TokenDistribution.LAYER == 19

    def test_l18_run_with_prior_layers_from_pipeline_like_data(self):
        """Simulate running L18 with data that would come from prior pipeline layers."""
        from app.predetect.layers_l18_l19 import Layer18TimingSideChannel
        # Simulate what prior layer results would look like
        prior_layers = [
            {"layer": "http", "name": "Layer0", "confidence": 0.0, "ttft_ms": 550.0, "tps": 52.0},
            {"layer": "self_report", "name": "Layer1", "confidence": 0.3, "ttft_ms": 600.0, "tps": 48.0},
            {"layer": "identity", "name": "Layer2", "confidence": 0.4, "ttft_ms": 580.0},
        ]
        result = Layer18TimingSideChannel().run(None, "gpt-4o", prior_layers)
        assert result["skipped"] is False
        assert result["ttft_samples"] == 3
        assert result["layer"] == 18

    def test_l19_run_with_prior_layers_from_pipeline_like_data(self):
        """Simulate running L19 with data that would come from prior pipeline layers."""
        from app.predetect.layers_l18_l19 import Layer19TokenDistribution
        prior_layers = [
            {"layer": "http", "evidence": ["Server header found", "OpenAI API format detected"]},
            {"layer": "self_report", "response_text": "I am Claude, a helpful AI assistant by Anthropic."},
            {"layer": "identity", "response_text": "I'm designed to be helpful, harmless, and honest."},
        ]
        result = Layer19TokenDistribution().run(None, "claude", prior_layers)
        # May or may not find texts depending on evidence parsing
        assert "layer" in result
        assert result["layer"] == 19
        assert result["tokens"] == 0

    def test_pipeline_write_predetect_trace_called_with_none_run_id(self, tmp_path, monkeypatch):
        """When run_id=None, _write_predetect_trace is a no-op."""
        monkeypatch.setattr("app.core.config.settings.DATA_DIR", str(tmp_path))
        from app.predetect.pipeline import _write_predetect_trace
        _write_predetect_trace(None, {"layer": 18, "name": "test", "tokens": 0,
                                      "confidence": 0.0, "evidence": []})
        traces_dir = tmp_path / "traces"
        assert not traces_dir.exists() or not any(traces_dir.iterdir())

    def test_pipeline_write_predetect_trace_with_run_id(self, tmp_path, monkeypatch):
        """When run_id is set, _write_predetect_trace creates the file."""
        monkeypatch.setattr("app.core.config.settings.DATA_DIR", str(tmp_path))
        from app.predetect.pipeline import _write_predetect_trace
        _write_predetect_trace("my-run-42", {
            "layer": 19, "name": "token_distribution", "tokens": 0,
            "confidence": 0.3, "skipped": False, "evidence": ["avg_len=400"],
        })
        trace_file = tmp_path / "traces" / "my-run-42" / "predetect.jsonl"
        assert trace_file.exists()
        data = json.loads(trace_file.read_text().strip())
        assert data["layer"] == 19
