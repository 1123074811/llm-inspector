"""
test_v16_phase8.py — v16 Phase 8 regression tests.

Validates:
  - EventKind new v16 event types
  - TraceWriter JSONL output
  - TraceWriter schema validation
"""
import pytest
import pathlib as _pl


class TestEventKindV16:
    def test_retry_truncation_event(self):
        from app.core.events import EventKind
        assert EventKind.RETRY_TRUNCATION.value == "retry.truncation"

    def test_retry_5xx_event(self):
        from app.core.events import EventKind
        assert EventKind.RETRY_5XX.value == "retry.5xx"

    def test_retry_decode_event(self):
        from app.core.events import EventKind
        assert EventKind.RETRY_DECODE.value == "retry.decode"

    def test_excluded_sample_event(self):
        from app.core.events import EventKind
        assert EventKind.EXCLUDED_SAMPLE.value == "sample.excluded"

    def test_identity_exposure_event(self):
        from app.core.events import EventKind
        assert EventKind.IDENTITY_EXPOSURE.value == "identity.exposure_detected"

    def test_system_prompt_leaked_event(self):
        from app.core.events import EventKind
        assert EventKind.SYSTEM_PROMPT_LEAKED.value == "system_prompt.leaked"

    def test_model_list_probed_event(self):
        from app.core.events import EventKind
        assert EventKind.MODEL_LIST_PROBED.value == "model_list.probed"

    def test_judge_chain_step_event(self):
        from app.core.events import EventKind
        assert EventKind.JUDGE_CHAIN_STEP.value == "judge_chain.step"

    def test_all_v16_events_are_strings(self):
        from app.core.events import EventKind
        v16_events = [
            EventKind.RETRY_TRUNCATION, EventKind.RETRY_5XX,
            EventKind.RETRY_DECODE, EventKind.EXCLUDED_SAMPLE,
            EventKind.IDENTITY_EXPOSURE, EventKind.SYSTEM_PROMPT_LEAKED,
            EventKind.MODEL_LIST_PROBED, EventKind.JUDGE_CHAIN_STEP,
        ]
        for e in v16_events:
            assert isinstance(e.value, str)


class TestTraceWriter:
    def test_write_single_record(self, tmp_path):
        from app.core.trace_writer import TraceWriter
        # Override data dir for testing
        writer = TraceWriter("test-trace-run")
        writer._run_dir = tmp_path / "test-trace-run"
        writer._run_dir.mkdir(parents=True, exist_ok=True)

        result = writer.write("judge_chain", {
            "case_id": "case_001",
            "judge_chain_step": 1,
            "method": "exact_match",
            "verdict": "pass",
            "confidence": 0.95,
        })
        assert result == 1

        # Verify file exists and has content
        path = writer._run_dir / "judge_chain.jsonl"
        assert path.exists()
        import json
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["case_id"] == "case_001"
        assert record["verdict"] == "pass"
        assert "timestamp" in record

    def test_write_batch(self, tmp_path):
        from app.core.trace_writer import TraceWriter
        writer = TraceWriter("test-batch-run")
        writer._run_dir = tmp_path / "test-batch-run"
        writer._run_dir.mkdir(parents=True, exist_ok=True)

        records = [
            {"step": "dns", "name": "DNS Lookup", "passed": True, "duration_ms": 50},
            {"step": "tls", "name": "TLS Handshake", "passed": True, "duration_ms": 120},
        ]
        count = writer.write_batch("preflight", records)
        assert count == 2
        assert writer.count("preflight") == 2

    def test_unknown_trace_type(self):
        from app.core.trace_writer import TraceWriter
        writer = TraceWriter("test-unknown")
        result = writer.write("nonexistent", {"data": "test"})
        assert result == 0

    def test_close_returns_counts(self, tmp_path):
        from app.core.trace_writer import TraceWriter
        writer = TraceWriter("test-close-run")
        writer._run_dir = tmp_path / "test-close-run"
        writer._run_dir.mkdir(parents=True, exist_ok=True)

        writer.write("errors", {"attempt": 1, "error_type": "5xx", "action": "retry"})
        writer.write("errors", {"attempt": 2, "error_type": "timeout", "action": "abort"})
        counts = writer.close()
        assert counts.get("errors") == 2

    def test_schemas_defined(self):
        from app.core.trace_writer import TraceWriter
        assert "preflight" in TraceWriter.SCHEMAS
        assert "predetect" in TraceWriter.SCHEMAS
        assert "judge_chain" in TraceWriter.SCHEMAS
        assert "errors" in TraceWriter.SCHEMAS
        assert "token_audit" in TraceWriter.SCHEMAS
