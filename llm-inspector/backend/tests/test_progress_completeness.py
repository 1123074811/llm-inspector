"""
test_progress_completeness.py — Phase 4: run progress & observability tests.

Tests:
1. test_watchdog_marks_stale_runs
2. test_b03_unhandled_exception_reaches_terminal_state
3. test_event_emit_returns_event
4. test_event_kind_values
5. test_timeline_svg_returns_200_on_empty_trace
"""
from __future__ import annotations

import time
import threading
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

import pytest


# ── 1. Watchdog ───────────────────────────────────────────────────────────────

class TestWatchdog:
    """Tests for the RunWatchdog that marks stale runs as partial_failed."""

    def test_watchdog_marks_stale_runs(self):
        """
        A run with status 'running' whose updated_at is far in the past
        should be marked as partial_failed by the watchdog.
        """
        from app.tasks.watchdog import RunWatchdog

        old_ts = (datetime.now(timezone.utc) - timedelta(hours=2)).strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
        fake_run = {
            "id": "stale-run-001",
            "status": "running",
            "updated_at": old_ts,
            "created_at": old_ts,
        }

        updated_statuses: dict[str, str] = {}

        def fake_list_runs(limit=50, offset=0):
            return [fake_run]

        def fake_get_run(run_id):
            return fake_run

        def fake_update_run_status(run_id, status, **kwargs):
            updated_statuses[run_id] = status

        with patch("app.tasks.watchdog.RunWatchdog.run_once", wraps=lambda self: None):
            pass  # just checking import works

        # Run directly with mocked repo
        with (
            patch("app.repository.repo.list_runs", side_effect=fake_list_runs),
            patch("app.repository.repo.get_run", side_effect=fake_get_run),
            patch("app.repository.repo.update_run_status", side_effect=fake_update_run_status),
        ):
            watchdog = RunWatchdog(max_duration_sec=10)  # very short threshold
            marked = watchdog.run_once()

        assert marked == 1
        assert updated_statuses.get("stale-run-001") == "partial_failed"

    def test_watchdog_ignores_fresh_runs(self):
        """A run updated just now should NOT be marked stale."""
        from app.tasks.watchdog import RunWatchdog

        fresh_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        fake_run = {
            "id": "fresh-run-001",
            "status": "running",
            "updated_at": fresh_ts,
            "created_at": fresh_ts,
        }
        updated_statuses: dict[str, str] = {}

        with (
            patch("app.repository.repo.list_runs", return_value=[fake_run]),
            patch("app.repository.repo.get_run", return_value=fake_run),
            patch("app.repository.repo.update_run_status",
                  side_effect=lambda run_id, status, **kw: updated_statuses.__setitem__(run_id, status)),
        ):
            watchdog = RunWatchdog(max_duration_sec=3600)
            marked = watchdog.run_once()

        assert marked == 0
        assert "fresh-run-001" not in updated_statuses

    def test_watchdog_ignores_terminal_runs(self):
        """Already-completed runs must never be touched."""
        from app.tasks.watchdog import RunWatchdog

        old_ts = (datetime.now(timezone.utc) - timedelta(hours=3)).strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
        for terminal_status in ("completed", "failed", "partial_failed", "cancelled"):
            fake_run = {
                "id": f"terminal-{terminal_status}",
                "status": terminal_status,
                "updated_at": old_ts,
                "created_at": old_ts,
            }
            updated_statuses: dict[str, str] = {}
            with (
                patch("app.repository.repo.list_runs", return_value=[fake_run]),
                patch("app.repository.repo.get_run", return_value=fake_run),
                patch("app.repository.repo.update_run_status",
                      side_effect=lambda run_id, status, **kw: updated_statuses.__setitem__(run_id, status)),
            ):
                marked = RunWatchdog(max_duration_sec=10).run_once()
            assert marked == 0, f"Should not mark terminal status={terminal_status}"


# ── 2. B-03: Unhandled exception reaches terminal state ──────────────────────

class TestB03UnhandledExceptionTerminalState:
    """
    Verifies that if an unhandled exception is raised inside execute_full()
    or execute_continue() the run status is set to a terminal state (failed).
    """

    def _make_lifecycle(self, run_id: str):
        from app.orchestration.run_lifecycle import RunLifecycleManager
        mgr = RunLifecycleManager.__new__(RunLifecycleManager)
        mgr.run_id = run_id
        mgr.run_metadata = {
            "id": run_id,
            "status": "running",
            "base_url": "http://fake-api/v1",
            "model_name": "fake-model",
            "test_mode": "quick",
            "suite_version": "v1",
            "api_key_encrypted": b"fake",
            "predetect_result": None,
        }
        mgr.adapter = MagicMock()
        mgr.base_url = "http://fake-api/v1"
        mgr.execution = MagicMock()
        mgr.reporting = MagicMock()

        # Tracer mock — supports start/finish/add_event/flush_to_jsonl
        tracer_mock = MagicMock()
        tracer_mock.span = MagicMock(return_value=MagicMock(
            __enter__=MagicMock(return_value=MagicMock()),
            __exit__=MagicMock(return_value=False),
        ))
        mgr.tracer = tracer_mock
        return mgr

    def test_unhandled_exception_in_initialize_marks_failed(self):
        """
        When _initialize() itself raises unexpectedly the run should land
        in 'failed' state.
        """
        run_id = "b03-test-001"
        final_statuses: dict[str, str] = {}
        current_run = {"id": run_id, "status": "running"}

        def fake_update(rid, status, **kw):
            current_run["status"] = status
            final_statuses[rid] = status

        with (
            patch("app.orchestration.run_lifecycle.repo.get_run", return_value=current_run),
            patch("app.orchestration.run_lifecycle.repo.update_run_status",
                  side_effect=fake_update),
            patch("app.orchestration.run_lifecycle.remove_tracer"),
            patch("app.core.events.sse_publisher"),
        ):
            from app.orchestration.run_lifecycle import RunLifecycleManager

            mgr = RunLifecycleManager.__new__(RunLifecycleManager)
            mgr.run_id = run_id

            # Make tracer.start() a no-op
            tracer_mock = MagicMock()
            tracer_mock.span = MagicMock(return_value=MagicMock(
                __enter__=MagicMock(return_value=MagicMock()),
                __exit__=MagicMock(return_value=False),
            ))
            mgr.tracer = tracer_mock
            mgr.execution = MagicMock()
            mgr.reporting = MagicMock()

            # _initialize raises an unhandled error
            def exploding_init():
                raise RuntimeError("Simulated catastrophic init failure")

            mgr._initialize = exploding_init
            mgr.execute_full()

        assert final_statuses.get(run_id) == "failed", (
            f"Expected 'failed', got {final_statuses.get(run_id)!r}"
        )

    def test_unhandled_exception_does_not_leave_run_in_running(self):
        """
        After execute_full() raises internally the status must NOT be 'running'.
        """
        run_id = "b03-test-002"
        run_states: list[str] = []
        db_run = {"id": run_id, "status": "running"}

        def track_update(rid, status, **kw):
            db_run["status"] = status
            run_states.append(status)

        with (
            patch("app.orchestration.run_lifecycle.repo.get_run", return_value=db_run),
            patch("app.orchestration.run_lifecycle.repo.update_run_status",
                  side_effect=track_update),
            patch("app.orchestration.run_lifecycle.remove_tracer"),
            patch("app.core.events.sse_publisher"),
        ):
            from app.orchestration.run_lifecycle import RunLifecycleManager

            mgr = RunLifecycleManager.__new__(RunLifecycleManager)
            mgr.run_id = run_id
            tracer_mock = MagicMock()
            tracer_mock.span = MagicMock(return_value=MagicMock(
                __enter__=MagicMock(return_value=MagicMock()),
                __exit__=MagicMock(return_value=False),
            ))
            mgr.tracer = tracer_mock
            mgr.execution = MagicMock()
            mgr.reporting = MagicMock()

            def boom():
                raise RuntimeError("boom")

            mgr._initialize = boom
            mgr.execute_full()

        assert db_run["status"] != "running", "Run must NOT stay in 'running' after an exception"
        terminal = {"failed", "partial_failed", "completed", "cancelled"}
        assert db_run["status"] in terminal, f"Status {db_run['status']!r} is not terminal"


# ── 3. Event emit returns Event ───────────────────────────────────────────────

class TestEventEmit:
    """Tests for the structured event bus in core/events.py."""

    def test_event_emit_returns_event(self):
        """emit() should return an Event with the correct run_id, kind, and payload."""
        with patch("app.core.events.sse_publisher"):
            from app.core.events import emit, EventKind, Event

            ev = emit("run123", EventKind.CASE_RESULT, case_id="x", passed=True)

        assert isinstance(ev, Event)
        assert ev.run_id == "run123"
        assert ev.kind == EventKind.CASE_RESULT
        assert ev.payload["case_id"] == "x"
        assert ev.payload["passed"] is True
        assert ev.timestamp > 0

    def test_event_emit_populates_timestamp(self):
        """The event timestamp should be close to now."""
        before = time.time()
        with patch("app.core.events.sse_publisher"):
            from app.core.events import emit, EventKind
            ev = emit("ts-test", EventKind.RUN_STARTED)
        after = time.time()

        assert before <= ev.timestamp <= after

    def test_event_to_dict(self):
        """to_dict() should include all required keys."""
        with patch("app.core.events.sse_publisher"):
            from app.core.events import emit, EventKind
            ev = emit("dict-test", EventKind.PHASE_TRANSITION, phase="phase1")

        d = ev.to_dict()
        assert d["run_id"] == "dict-test"
        assert d["kind"] == "phase.transition"
        assert "timestamp" in d
        assert d["payload"]["phase"] == "phase1"

    def test_event_to_jsonl(self):
        """to_jsonl() should return a valid JSON line ending with newline."""
        import json as _json
        with patch("app.core.events.sse_publisher"):
            from app.core.events import emit, EventKind
            ev = emit("jsonl-test", EventKind.CASE_SKIP, reason="budget")

        line = ev.to_jsonl()
        assert line.endswith("\n")
        parsed = _json.loads(line.strip())
        assert parsed["kind"] == "case.skip"


# ── 4. EventKind values are unique ───────────────────────────────────────────

class TestEventKindValues:
    """Verify all EventKind enum values are unique, non-empty strings."""

    def test_event_kind_values_are_unique(self):
        from app.core.events import EventKind

        values = [k.value for k in EventKind]
        assert len(values) == len(set(values)), "Duplicate EventKind values detected"

    def test_event_kind_values_are_strings(self):
        from app.core.events import EventKind

        for k in EventKind:
            assert isinstance(k.value, str), f"{k.name} value is not a string"
            assert len(k.value) > 0, f"{k.name} value is empty"

    def test_event_kind_expected_members(self):
        """Spot-check that key members are present."""
        from app.core.events import EventKind

        expected = {
            "PROBE_REQUEST", "CASE_RESULT", "JUDGE_RULE",
            "CB_OPEN", "RETRY_SCHEDULED", "RUN_STARTED",
            "RUN_COMPLETED", "RUN_FAILED", "PHASE_TRANSITION",
        }
        actual_names = {k.name for k in EventKind}
        missing = expected - actual_names
        assert not missing, f"Missing EventKind members: {missing}"


# ── 5. Timeline SVG returns 200 on empty trace ───────────────────────────────

class TestTimelineSvg:
    """Tests for the GET /api/v1/runs/{id}/timeline.svg endpoint."""

    def test_timeline_svg_returns_200_on_empty_trace(self):
        """
        Calling handle_run_timeline_svg with a run_id that has no trace data
        should still return HTTP 200 and a valid SVG document.
        """
        from app.handlers.v11_handlers import handle_run_timeline_svg

        with (
            patch("app.handlers.v11_handlers._load_trace_events", return_value=[]),
            patch("app.handlers.v11_handlers.get_tracer", return_value=None),
        ):
            result = handle_run_timeline_svg(
                "/api/v1/runs/no-trace-run/timeline.svg", {}, {}
            )

        status, body, content_type = result
        assert status == 200
        assert b"<svg" in body
        assert "svg" in content_type

    def test_timeline_svg_contains_run_id(self):
        """The SVG should mention the run ID in its title."""
        from app.handlers.v11_handlers import handle_run_timeline_svg

        run_id = "my-observable-run-xyz"
        with (
            patch("app.handlers.v11_handlers._load_trace_events", return_value=[]),
            patch("app.handlers.v11_handlers.get_tracer", return_value=None),
        ):
            status, body, _ = handle_run_timeline_svg(
                f"/api/v1/runs/{run_id}/timeline.svg", {}, {}
            )

        assert status == 200
        # run_id may be truncated — check at least first 16 chars
        assert run_id[:16].encode() in body

    def test_timeline_svg_with_trace_events(self):
        """SVG generated from actual trace events should contain span names."""
        from app.handlers.v11_handlers import handle_run_timeline_svg
        import time as _t

        now = _t.time()
        fake_events = [
            {"trace_event": "span_start", "span": "predetect", "timestamp": now - 5.0},
            {"trace_event": "span_end",   "span": "predetect", "timestamp": now - 2.0, "status": "ok"},
            {"trace_event": "span_start", "span": "phase1",    "timestamp": now - 2.0},
            {"trace_event": "span_end",   "span": "phase1",    "timestamp": now,       "status": "ok"},
        ]

        with (
            patch("app.handlers.v11_handlers._load_trace_events", return_value=fake_events),
            patch("app.handlers.v11_handlers.get_tracer", return_value=None),
        ):
            status, body, content_type = handle_run_timeline_svg(
                "/api/v1/runs/trace-run-001/timeline.svg", {}, {}
            )

        assert status == 200
        assert b"predetect" in body
        assert b"phase1" in body
        assert b"<svg" in body

    def test_timeline_svg_invalid_path_returns_400(self):
        """A malformed path should return 400."""
        from app.handlers.v11_handlers import handle_run_timeline_svg

        result = handle_run_timeline_svg("/api/v1/timeline.svg", {}, {})
        status = result[0]
        assert status == 400
