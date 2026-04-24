"""Tests for v15 Phase 1: Preflight connection check."""
from __future__ import annotations
import json
import pytest


def test_error_taxonomy_imports():
    from app.preflight.error_taxonomy import ErrorCode, ErrorDetail, make_error
    assert ErrorCode.E_NET_DNS_FAIL.value == "E_NET_DNS_FAIL"
    err = make_error(ErrorCode.E_UPSTREAM_SERVICE_UNAVAILABLE)
    assert err.retryable is True
    assert "503" in err.user_message_zh or "不可用" in err.user_message_zh
    # Key test: 503 message must NOT say URL/key error
    assert "URL" not in err.user_message_zh
    assert "Key" not in err.user_message_zh


def test_error_detail_to_dict():
    from app.preflight.error_taxonomy import ErrorCode, make_error
    err = make_error(ErrorCode.E_AUTH_INVALID_KEY, raw_status=401, raw_body="Unauthorized")
    d = err.to_dict()
    assert d["code"] == "E_AUTH_INVALID_KEY"
    assert d["retryable"] is False
    assert d["raw_status"] == 401


def test_preflight_report_dataclass():
    from app.preflight.connection_check import PreflightReport, PreflightStep
    step = PreflightStep("A1", "input_validation", True, 1.5)
    report = PreflightReport(passed=True, steps=[step])
    d = report.to_dict()
    assert d["passed"] is True
    assert len(d["steps"]) == 1
    assert d["steps"][0]["step"] == "A1"


def test_preflight_fails_empty_url():
    from app.preflight.connection_check import run_preflight
    report = run_preflight("", "sk-test", "gpt-4")
    assert report.passed is False
    assert report.first_error is not None
    assert "URL" in report.first_error.code or "E_URL" in report.first_error.code


def test_preflight_fails_missing_key():
    from app.preflight.connection_check import run_preflight
    report = run_preflight("https://api.openai.com/v1", "", "gpt-4")
    assert report.passed is False
    assert "KEY" in report.first_error.code or "E_API" in report.first_error.code


def test_preflight_fails_invalid_url_format():
    from app.preflight.connection_check import run_preflight
    report = run_preflight("not-a-url", "sk-test", "gpt-4")
    assert report.passed is False
    assert "E_URL" in report.first_error.code


def test_preflight_dns_fail():
    """DNS failure or connection error for non-existent host."""
    from app.preflight.connection_check import run_preflight
    report = run_preflight("http://this-host-definitely-does-not-exist-xyz123.com/v1",
                           "sk-test", "gpt-4", timeout=3.0)
    assert report.passed is False
    err = report.first_error
    assert err is not None
    # Accept any network-level or upstream error — exact code depends on DNS wildcard / firewall
    assert err.code in (
        "E_NET_DNS_FAIL", "E_NET_CONN_REFUSED", "E_NET_TIMEOUT",
        "E_UPSTREAM_BAD_GATEWAY", "E_UPSTREAM_INTERNAL",
        "E_UPSTREAM_SERVICE_UNAVAILABLE", "E_UNKNOWN",
    )


def test_event_kinds_exist():
    from app.core.events import EventKind
    assert hasattr(EventKind, "PREFLIGHT_STARTED")
    assert hasattr(EventKind, "PREFLIGHT_PASSED")
    assert hasattr(EventKind, "PREFLIGHT_FAILED")


def test_preflight_retryable_flags():
    from app.preflight.error_taxonomy import ErrorCode, make_error
    # Auth errors are NOT retryable
    assert make_error(ErrorCode.E_AUTH_INVALID_KEY).retryable is False
    assert make_error(ErrorCode.E_AUTH_FORBIDDEN).retryable is False
    assert make_error(ErrorCode.E_URL_INVALID_FORMAT).retryable is False
    # Service unavailability IS retryable
    assert make_error(ErrorCode.E_UPSTREAM_SERVICE_UNAVAILABLE).retryable is True
    assert make_error(ErrorCode.E_UPSTREAM_RATE_LIMITED).retryable is True
    assert make_error(ErrorCode.E_NET_TIMEOUT).retryable is True


def test_all_error_codes_documented():
    """Every ErrorCode must have an entry in _ERROR_DETAILS."""
    from app.preflight.error_taxonomy import ErrorCode, _ERROR_DETAILS
    for code in ErrorCode:
        assert code in _ERROR_DETAILS, f"ErrorCode {code} missing from _ERROR_DETAILS"


def test_preflight_step_to_dict():
    from app.preflight.connection_check import PreflightStep
    step = PreflightStep("A1", "input_validation", True, 2.5, notes="URL format OK")
    d = step.to_dict()
    assert d["step"] == "A1"
    assert d["passed"] is True
    assert d["error"] is None
    assert "duration_ms" in d


def test_update_run_field_exists():
    """repo.update_run_field must be callable (used by preflight lifecycle)."""
    from app.repository import repo
    assert callable(getattr(repo, "update_run_field", None))


def test_lifecycle_has_preflight_method():
    """RunLifecycleManager must have _step_preflight."""
    from app.orchestration.run_lifecycle import RunLifecycleManager
    assert callable(getattr(RunLifecycleManager, "_step_preflight", None))


def test_db_migration_005_registered():
    """DB migration 005 (preflight_report column) must be registered."""
    from app.core.db_migrations import _migrations
    assert 5 in _migrations


def test_frontend_preflight_status_labels():
    """app.js must include preflight_running / preflight_failed status labels."""
    import pathlib
    js_path = pathlib.Path(__file__).parent.parent.parent / "frontend" / "app.js"
    content = js_path.read_text(encoding="utf-8")
    assert "preflight_running" in content, "app.js missing 'preflight_running' label"
    assert "preflight_failed" in content, "app.js missing 'preflight_failed' label"
