"""
tests/test_v14_phase7.py — Phase 7 Test Reliability & 100% Progress Guarantee.

Covers:
  - Progress formula (B7 fix): skipped cases do not keep progress below 100%
  - RetryConfig defaults
  - with_retry: success on first try
  - with_retry: network error → retries → eventual success
  - with_retry: network error → exhausted → returns failure outcome
  - with_retry: HTTP 429 → respects max_retries_429 limit
  - with_retry: non-retryable error → re-raises immediately
  - _write_error_event: creates file and appends JSON
  - list_stale_runs: supports after_id cursor parameter
  - Watchdog scan: processes more than 500 runs (pagination)
  - handle_circuit_breaker_history: returns list with expected fields
  - ScoreCard.skipped_cases field exists with default
  - ScoreCard.to_dict() includes skipped_cases
"""
from __future__ import annotations

import json
import pathlib
import tempfile
import time
import unittest.mock as mock

import pytest


# ---------------------------------------------------------------------------
# Progress formula (B7 fix)
# ---------------------------------------------------------------------------

class TestProgressFormula:

    def test_progress_with_skipped_reaches_100(self):
        """completed=8, total=10, skipped=2 → progress=100 (not 80)."""
        from app.repository.repo import update_run_progress
        assert update_run_progress(8, 10, skipped=2) == 100

    def test_progress_no_skipped(self):
        """completed=5, total=10, skipped=0 → progress=50."""
        from app.repository.repo import update_run_progress
        assert update_run_progress(5, 10, skipped=0) == 50

    def test_progress_skipped_exceeds_total_no_division_by_zero(self):
        """skipped >= total should not cause division by zero."""
        from app.repository.repo import update_run_progress
        # denominator is max(1, total - skipped) = max(1, 0) = 1
        result = update_run_progress(0, 10, skipped=10)
        assert 0 <= result <= 100

    def test_progress_zero_total_no_division_by_zero(self):
        """total=0 should not cause division by zero."""
        from app.repository.repo import update_run_progress
        result = update_run_progress(0, 0, skipped=0)
        assert 0 <= result <= 100

    def test_progress_clamped_to_100(self):
        """Progress should never exceed 100."""
        from app.repository.repo import update_run_progress
        result = update_run_progress(10, 5, skipped=0)
        assert result <= 100

    def test_progress_default_skipped_zero(self):
        """Default skipped=0, formula works normally."""
        from app.repository.repo import update_run_progress
        assert update_run_progress(3, 10) == 30


# ---------------------------------------------------------------------------
# ScoreCard.skipped_cases
# ---------------------------------------------------------------------------

class TestScoreCardSkippedCases:

    def test_skipped_cases_field_exists_with_default(self):
        """ScoreCard.skipped_cases is a list with empty default."""
        from app.core.schemas import ScoreCard
        sc = ScoreCard()
        assert hasattr(sc, "skipped_cases")
        assert sc.skipped_cases == []

    def test_skipped_cases_to_dict(self):
        """ScoreCard.to_dict() must include skipped_cases key."""
        from app.core.schemas import ScoreCard
        sc = ScoreCard()
        sc.skipped_cases = ["case-1", "case-2"]
        d = sc.to_dict()
        assert "skipped_cases" in d
        assert d["skipped_cases"] == ["case-1", "case-2"]

    def test_skipped_cases_empty_list_in_to_dict(self):
        """Empty skipped_cases still appears in to_dict output."""
        from app.core.schemas import ScoreCard
        sc = ScoreCard()
        d = sc.to_dict()
        assert "skipped_cases" in d
        assert d["skipped_cases"] == []


# ---------------------------------------------------------------------------
# RetryConfig defaults
# ---------------------------------------------------------------------------

class TestRetryConfig:

    def test_defaults(self):
        from app.runner.retry_policy import RetryConfig
        cfg = RetryConfig()
        assert cfg.max_retries_network == 3
        assert cfg.max_retries_429 == 2
        assert cfg.backoff_base_s == 0.5
        assert cfg.backoff_max_s == 8.0

    def test_custom_values(self):
        from app.runner.retry_policy import RetryConfig
        cfg = RetryConfig(max_retries_network=1, max_retries_429=0, backoff_base_s=0.1, backoff_max_s=2.0)
        assert cfg.max_retries_network == 1
        assert cfg.max_retries_429 == 0


# ---------------------------------------------------------------------------
# with_retry — success cases
# ---------------------------------------------------------------------------

class TestWithRetrySuccess:

    def test_success_on_first_try(self):
        from app.runner.retry_policy import with_retry, RetryConfig
        calls = []

        def _fn():
            calls.append(1)
            return "ok"

        cfg = RetryConfig()
        outcome = with_retry(_fn, config=cfg)
        assert outcome.success is True
        assert outcome.attempts == 1
        assert outcome.final_error is None
        assert outcome.result == "ok"
        assert len(calls) == 1

    def test_retry_events_empty_on_success(self):
        from app.runner.retry_policy import with_retry
        outcome = with_retry(lambda: 42)
        assert outcome.success is True
        assert outcome.retry_events == []


# ---------------------------------------------------------------------------
# with_retry — network error retry
# ---------------------------------------------------------------------------

class TestWithRetryNetworkError:

    def test_network_error_retries_then_succeeds(self):
        from app.runner.retry_policy import with_retry, RetryConfig
        calls = []

        def _fn():
            calls.append(1)
            if len(calls) < 3:
                raise ConnectionError("connection refused")
            return "success"

        cfg = RetryConfig(max_retries_network=3, backoff_base_s=0.0)
        with mock.patch("time.sleep"):
            outcome = with_retry(_fn, config=cfg)

        assert outcome.success is True
        assert outcome.attempts == 3
        assert len(calls) == 3

    def test_network_error_exhausted_returns_failure(self):
        from app.runner.retry_policy import with_retry, RetryConfig

        def _fn():
            raise ConnectionError("connection refused")

        cfg = RetryConfig(max_retries_network=2, backoff_base_s=0.0)
        with mock.patch("time.sleep"):
            outcome = with_retry(_fn, config=cfg)

        assert outcome.success is False
        assert outcome.attempts == 3  # 1 initial + 2 retries
        assert outcome.final_error is not None
        # Should have 3 events: 2 "retry" + 1 "exhausted"
        assert len(outcome.retry_events) == 3

    def test_network_error_uses_exponential_backoff(self):
        from app.runner.retry_policy import with_retry, RetryConfig
        sleep_calls = []

        def _fn():
            raise TimeoutError("timed out")

        cfg = RetryConfig(max_retries_network=3, backoff_base_s=1.0, backoff_max_s=10.0)
        with mock.patch("time.sleep", side_effect=lambda s: sleep_calls.append(s)):
            with_retry(_fn, config=cfg)

        # delays should be 1.0, 2.0, 4.0 (exponential) — capped at 10.0
        assert len(sleep_calls) == 3
        assert sleep_calls[0] == 1.0
        assert sleep_calls[1] == 2.0
        assert sleep_calls[2] == 4.0


# ---------------------------------------------------------------------------
# with_retry — HTTP 429
# ---------------------------------------------------------------------------

class TestWithRetry429:

    def test_429_respects_max_retries(self):
        from app.runner.retry_policy import with_retry, RetryConfig

        def _fn():
            raise Exception("HTTP 429 Too Many Requests — rate_limit exceeded")

        cfg = RetryConfig(max_retries_429=1, backoff_base_s=0.0)
        with mock.patch("time.sleep"):
            outcome = with_retry(_fn, config=cfg)

        assert outcome.success is False
        assert outcome.attempts == 2  # 1 initial + 1 retry

    def test_429_uses_retry_after_from_message(self):
        from app.runner.retry_policy import with_retry, RetryConfig
        sleep_calls = []

        def _fn():
            raise Exception("429 rate_limit — Retry-After: 5")

        cfg = RetryConfig(max_retries_429=1, backoff_base_s=0.0, backoff_max_s=60.0)
        with mock.patch("time.sleep", side_effect=lambda s: sleep_calls.append(s)):
            with_retry(_fn, config=cfg)

        assert len(sleep_calls) >= 1
        assert sleep_calls[0] == 5.0


# ---------------------------------------------------------------------------
# with_retry — non-retryable error
# ---------------------------------------------------------------------------

class TestWithRetryNonRetryable:

    def test_non_retryable_reraises_immediately(self):
        from app.runner.retry_policy import with_retry, RetryConfig
        calls = []

        def _fn():
            calls.append(1)
            raise ValueError("bad input")

        cfg = RetryConfig()
        with pytest.raises(ValueError, match="bad input"):
            with_retry(_fn, config=cfg)

        assert len(calls) == 1  # No retry — re-raised immediately


# ---------------------------------------------------------------------------
# _write_error_event
# ---------------------------------------------------------------------------

class TestWriteErrorEvent:

    def test_creates_file_and_appends_json(self, tmp_path):
        from app.runner.retry_policy import _write_error_event
        run_id = "test-run-abcdef"
        event = {"error_type": "network", "attempt": 1}

        with mock.patch("app.core.config.settings") as mock_settings:
            mock_settings.DATA_DIR = str(tmp_path)
            _write_error_event(run_id, event)

        errors_file = tmp_path / "traces" / run_id / "errors.jsonl"
        assert errors_file.exists()
        lines = errors_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["error_type"] == "network"

    def test_appends_multiple_events(self, tmp_path):
        from app.runner.retry_policy import _write_error_event
        run_id = "test-run-multi"

        with mock.patch("app.core.config.settings") as mock_settings:
            mock_settings.DATA_DIR = str(tmp_path)
            _write_error_event(run_id, {"seq": 1})
            _write_error_event(run_id, {"seq": 2})

        errors_file = tmp_path / "traces" / run_id / "errors.jsonl"
        lines = errors_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

    def test_non_fatal_on_bad_path(self):
        """_write_error_event must not raise even if path is unwritable."""
        from app.runner.retry_policy import _write_error_event

        with mock.patch("app.core.config.settings") as mock_settings:
            mock_settings.DATA_DIR = "/nonexistent_path_that_cannot_be_created/abc"
            # Should not raise
            _write_error_event("run-xyz", {"test": True})


# ---------------------------------------------------------------------------
# list_stale_runs — cursor pagination
# ---------------------------------------------------------------------------

class TestListStaleRuns:

    def test_list_stale_runs_exists_and_accepts_after_id(self):
        """list_stale_runs must accept after_id parameter."""
        import inspect
        from app.repository.repo import list_stale_runs
        sig = inspect.signature(list_stale_runs)
        assert "after_id" in sig.parameters
        assert "limit" in sig.parameters

    def test_list_stale_runs_returns_list(self):
        """list_stale_runs() with no stale runs returns empty list (not error)."""
        from app.repository.repo import list_stale_runs
        result = list_stale_runs(limit=10, after_id=None)
        assert isinstance(result, list)

    def test_list_stale_runs_after_id_cursor(self):
        """Calling list_stale_runs with after_id='zzzzz' returns empty (no higher IDs)."""
        from app.repository.repo import list_stale_runs
        result = list_stale_runs(limit=10, after_id="zzzzzzzzzzz-high-sentinel")
        assert result == []


# ---------------------------------------------------------------------------
# Watchdog pagination — processes more than 500 runs
# ---------------------------------------------------------------------------

class TestWatchdogPagination:

    def test_watchdog_uses_list_stale_runs_not_list_runs(self):
        """Watchdog.run_once() must call list_stale_runs (not list_runs)."""
        import app.tasks.watchdog as watchdog_module
        import app.repository.repo as repo_module

        wd = watchdog_module.RunWatchdog(max_duration_sec=0)  # expire everything instantly

        with mock.patch.object(repo_module, "list_stale_runs", return_value=[]) as mock_stale, \
             mock.patch.object(repo_module, "list_runs") as mock_list_runs:
            wd.run_once()

        mock_stale.assert_called()
        mock_list_runs.assert_not_called()

    def test_watchdog_paginates_all_batches(self):
        """Watchdog iterates multiple pages until batch is empty."""
        import app.tasks.watchdog as watchdog_module
        import app.repository.repo as repo_module

        wd = watchdog_module.RunWatchdog(max_duration_sec=9999)  # not expired

        batch1 = [{"id": f"run-{i:04d}", "status": "running",
                   "created_at": "2020-01-01T00:00:00"} for i in range(100)]
        batch2 = [{"id": f"run-{i:04d}", "status": "running",
                   "created_at": "2020-01-01T00:00:00"} for i in range(100, 200)]

        call_count = {"n": 0}

        def fake_list_stale_runs(limit=100, after_id=None):
            call_count["n"] += 1
            if after_id is None:
                return batch1
            elif after_id == batch1[-1]["id"]:
                return batch2
            else:
                return []

        with mock.patch.object(repo_module, "list_stale_runs", side_effect=fake_list_stale_runs):
            wd.run_once()

        # Should have called at least 3 times (page1, page2, empty page)
        assert call_count["n"] >= 3


# ---------------------------------------------------------------------------
# handle_circuit_breaker_history
# ---------------------------------------------------------------------------

class TestCircuitBreakerHistory:

    def _parse_result(self, result):
        """Helper: unpack handler result tuple (status, body, ct) → (status, dict)."""
        if isinstance(result, tuple):
            status, body_bytes, _ct = result
            data = json.loads(body_bytes)
            return status, data
        return 200, result

    def test_handle_circuit_breaker_history_returns_200(self):
        from app.handlers.v14_handlers import handle_circuit_breaker_history
        result = handle_circuit_breaker_history("/api/v14/circuit-breaker/history", {}, {})
        status, data = self._parse_result(result)
        assert status == 200
        assert "events" in data
        assert "count" in data
        assert isinstance(data["events"], list)

    def test_circuit_breaker_history_with_limit_param(self):
        from app.handlers.v14_handlers import handle_circuit_breaker_history
        result = handle_circuit_breaker_history(
            "/api/v14/circuit-breaker/history",
            {"limit": "5"},
            {},
        )
        _, data = self._parse_result(result)
        assert data["limit"] == 5

    def test_circuit_breaker_history_state_filter(self):
        from app.handlers.v14_handlers import handle_circuit_breaker_history
        result = handle_circuit_breaker_history(
            "/api/v14/circuit-breaker/history",
            {"state": "open"},
            {},
        )
        _, data = self._parse_result(result)
        assert data["state_filter"] == "open"
        # All returned events (if any) should have state == "open"
        for evt in data["events"]:
            assert evt.get("state") == "open"

    def test_get_cb_event_history_returns_list(self):
        from app.core.circuit_breaker import get_cb_event_history
        result = get_cb_event_history()
        assert isinstance(result, list)

    def test_circuit_breaker_records_open_event(self):
        """When circuit opens, event_history captures the 'opened' event."""
        from app.core.circuit_breaker import CircuitBreaker, get_cb_event_history, _cb_event_history

        # Use a fresh CB so we can trigger an event
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout_sec=60.0)
        _cb_event_history.clear()

        cb.record_failure("http://test-endpoint/", "error 1")
        cb.record_failure("http://test-endpoint/", "error 2")  # threshold reached

        history = get_cb_event_history()
        opened_events = [e for e in history if e["event_type"] == "opened"]
        assert len(opened_events) >= 1
        assert opened_events[0]["base_url"] == "http://test-endpoint/"
        assert opened_events[0]["state"] == "open"
