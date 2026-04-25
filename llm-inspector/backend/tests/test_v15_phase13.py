"""
Tests for v15 Phase 13: Preflight Connection Check.

Covers:
  - preflight/error_taxonomy.py: ErrorCode enum, ErrorDetail dataclass, make_error()
  - preflight/connection_check.py: PreflightStep, PreflightReport, _check_inputs(),
    _check_schema(), _note_capabilities(), run_preflight() (with mock/stub)
"""
from __future__ import annotations
import pytest


# ---------------------------------------------------------------------------
# error_taxonomy: ErrorCode
# ---------------------------------------------------------------------------

def test_error_code_is_string_enum():
    from app.preflight.error_taxonomy import ErrorCode
    assert isinstance(ErrorCode.E_URL_EMPTY, str)
    assert ErrorCode.E_URL_EMPTY == "E_URL_EMPTY"


def test_error_code_all_codes_accessible():
    from app.preflight.error_taxonomy import ErrorCode
    expected = [
        "E_URL_EMPTY", "E_URL_INVALID_FORMAT", "E_API_KEY_MISSING",
        "E_NET_DNS_FAIL", "E_NET_CONN_REFUSED", "E_NET_TIMEOUT",
        "E_AUTH_INVALID_KEY", "E_AUTH_FORBIDDEN",
        "E_MODEL_NOT_FOUND", "E_MODEL_NAME_EMPTY",
        "E_UPSTREAM_RATE_LIMITED", "E_UPSTREAM_INTERNAL",
        "E_SCHEMA_INVALID_JSON", "E_SCHEMA_MISSING_CHOICES",
        "E_UNKNOWN",
    ]
    for code in expected:
        assert hasattr(ErrorCode, code), f"ErrorCode.{code} not found"


def test_error_code_retryable_vs_non():
    from app.preflight.error_taxonomy import ErrorCode, _ERROR_DETAILS
    # Network errors should be retryable
    assert _ERROR_DETAILS[ErrorCode.E_NET_TIMEOUT][0] is True
    assert _ERROR_DETAILS[ErrorCode.E_NET_DNS_FAIL][0] is True
    # Auth errors should NOT be retryable
    assert _ERROR_DETAILS[ErrorCode.E_AUTH_INVALID_KEY][0] is False
    assert _ERROR_DETAILS[ErrorCode.E_URL_EMPTY][0] is False


# ---------------------------------------------------------------------------
# error_taxonomy: make_error()
# ---------------------------------------------------------------------------

def test_make_error_returns_error_detail():
    from app.preflight.error_taxonomy import make_error, ErrorCode, ErrorDetail
    err = make_error(ErrorCode.E_URL_EMPTY)
    assert isinstance(err, ErrorDetail)


def test_make_error_code_matches():
    from app.preflight.error_taxonomy import make_error, ErrorCode
    err = make_error(ErrorCode.E_API_KEY_MISSING)
    assert err.code == "E_API_KEY_MISSING"


def test_make_error_retryable_flag():
    from app.preflight.error_taxonomy import make_error, ErrorCode
    retryable = make_error(ErrorCode.E_NET_TIMEOUT)
    not_retryable = make_error(ErrorCode.E_URL_EMPTY)
    assert retryable.retryable is True
    assert not_retryable.retryable is False


def test_make_error_messages_not_empty():
    from app.preflight.error_taxonomy import make_error, ErrorCode
    err = make_error(ErrorCode.E_AUTH_INVALID_KEY)
    assert err.user_message_zh
    assert err.user_message_en


def test_make_error_with_raw_status():
    from app.preflight.error_taxonomy import make_error, ErrorCode
    err = make_error(ErrorCode.E_UPSTREAM_INTERNAL, raw_status=500)
    assert err.raw_status == 500


def test_make_error_with_raw_body_truncated():
    from app.preflight.error_taxonomy import make_error, ErrorCode
    long_body = "x" * 500
    err = make_error(ErrorCode.E_UNKNOWN, raw_body=long_body)
    assert err.raw_body_excerpt is not None
    assert len(err.raw_body_excerpt) <= 200


def test_make_error_to_dict():
    from app.preflight.error_taxonomy import make_error, ErrorCode
    err = make_error(ErrorCode.E_MODEL_NOT_FOUND, raw_status=404)
    d = err.to_dict()
    assert "code" in d
    assert "retryable" in d
    assert "message" in d
    assert "message_en" in d
    assert d["raw_status"] == 404


def test_make_error_source_layer_default():
    from app.preflight.error_taxonomy import make_error, ErrorCode
    err = make_error(ErrorCode.E_UNKNOWN)
    assert err.source_layer == "preflight"


def test_make_error_custom_source_layer():
    from app.preflight.error_taxonomy import make_error, ErrorCode
    err = make_error(ErrorCode.E_UNKNOWN, source_layer="connection_check")
    assert err.source_layer == "connection_check"


# ---------------------------------------------------------------------------
# connection_check: PreflightStep
# ---------------------------------------------------------------------------

def test_preflight_step_to_dict_passed():
    from app.preflight.connection_check import PreflightStep
    step = PreflightStep("A1", "input_validation", True, 5.3)
    d = step.to_dict()
    assert d["step"] == "A1"
    assert d["name"] == "input_validation"
    assert d["passed"] is True
    assert d["duration_ms"] == pytest.approx(5.3, abs=0.1)
    assert d["error"] is None


def test_preflight_step_to_dict_failed():
    from app.preflight.connection_check import PreflightStep
    from app.preflight.error_taxonomy import make_error, ErrorCode
    err = make_error(ErrorCode.E_URL_EMPTY)
    step = PreflightStep("A1", "input_validation", False, 1.0, error=err)
    d = step.to_dict()
    assert d["passed"] is False
    assert d["error"] is not None
    assert "code" in d["error"]


# ---------------------------------------------------------------------------
# connection_check: PreflightReport
# ---------------------------------------------------------------------------

def test_preflight_report_to_dict():
    from app.preflight.connection_check import PreflightReport, PreflightStep
    report = PreflightReport(
        passed=True,
        steps=[PreflightStep("A1", "input_validation", True, 1.0)],
        total_duration_ms=10.0,
        capabilities={"stream": None, "logprobs": False},
    )
    d = report.to_dict()
    assert d["passed"] is True
    assert len(d["steps"]) == 1
    assert d["total_duration_ms"] == pytest.approx(10.0, abs=0.1)
    assert "capabilities" in d
    assert "checked_at" in d


def test_preflight_report_failed():
    from app.preflight.connection_check import PreflightReport
    from app.preflight.error_taxonomy import make_error, ErrorCode
    err = make_error(ErrorCode.E_URL_EMPTY)
    report = PreflightReport(passed=False, first_error=err)
    d = report.to_dict()
    assert d["passed"] is False
    assert d["first_error"] is not None


# ---------------------------------------------------------------------------
# connection_check: _check_inputs() (pure function, no network)
# ---------------------------------------------------------------------------

def test_check_inputs_valid():
    from app.preflight.connection_check import _check_inputs
    err, notes = _check_inputs("https://api.example.com", "sk-test123", "gpt-4")
    assert err is None
    assert notes  # should have some notes


def test_check_inputs_empty_url():
    from app.preflight.connection_check import _check_inputs
    err, _ = _check_inputs("", "sk-test", "gpt-4")
    assert err is not None
    assert err.code == "E_URL_EMPTY"


def test_check_inputs_invalid_url_format():
    from app.preflight.connection_check import _check_inputs
    err, _ = _check_inputs("not-a-url", "sk-test", "gpt-4")
    assert err is not None
    assert err.code == "E_URL_INVALID_FORMAT"


def test_check_inputs_http_url_valid():
    from app.preflight.connection_check import _check_inputs
    err, _ = _check_inputs("http://localhost:8080", "sk-test", "gpt-4")
    assert err is None


def test_check_inputs_missing_model():
    from app.preflight.connection_check import _check_inputs
    err, _ = _check_inputs("https://api.example.com", "sk-test", "")
    assert err is not None
    assert err.code == "E_MODEL_NAME_EMPTY"


def test_check_inputs_missing_api_key():
    from app.preflight.connection_check import _check_inputs
    err, _ = _check_inputs("https://api.example.com", "", "gpt-4")
    assert err is not None
    assert err.code == "E_API_KEY_MISSING"


def test_check_inputs_none_url():
    from app.preflight.connection_check import _check_inputs
    err, _ = _check_inputs(None, "sk-test", "gpt-4")
    assert err is not None


def test_check_inputs_whitespace_url():
    from app.preflight.connection_check import _check_inputs
    err, _ = _check_inputs("   ", "sk-test", "gpt-4")
    assert err is not None
    assert err.code == "E_URL_EMPTY"


# ---------------------------------------------------------------------------
# connection_check: _check_schema() (pure function, no network)
# ---------------------------------------------------------------------------

def test_check_schema_valid():
    from app.preflight.connection_check import _check_schema
    body = {
        "choices": [{"message": {"content": "Hello"}, "finish_reason": "stop"}],
        "usage": {"total_tokens": 5},
    }
    err, notes = _check_schema(body)
    assert err is None
    assert "schema OK" in notes


def test_check_schema_missing_choices():
    from app.preflight.connection_check import _check_schema
    err, notes = _check_schema({"id": "abc", "object": "chat.completion"})
    assert err is not None
    assert err.code == "E_SCHEMA_MISSING_CHOICES"


def test_check_schema_empty_choices():
    from app.preflight.connection_check import _check_schema
    err, _ = _check_schema({"choices": []})
    assert err is not None
    assert err.code == "E_SCHEMA_MISSING_CHOICES"


def test_check_schema_not_a_dict():
    from app.preflight.connection_check import _check_schema
    err, _ = _check_schema("not a dict")
    assert err is not None
    assert err.code == "E_SCHEMA_INVALID_JSON"


def test_check_schema_error_key_passes():
    """If upstream returned an error object, schema check should not block."""
    from app.preflight.connection_check import _check_schema
    # Upstream error body passes through (error handling was in A3)
    err, _ = _check_schema({"error": {"message": "bad request", "code": "invalid_model"}})
    assert err is None


# ---------------------------------------------------------------------------
# connection_check: _note_capabilities() (pure function)
# ---------------------------------------------------------------------------

def test_note_capabilities_basic():
    from app.preflight.connection_check import _note_capabilities
    body = {
        "choices": [{"message": {"content": "OK"}, "logprobs": None}],
    }
    caps = _note_capabilities(body)
    assert isinstance(caps, dict)
    assert "stream" in caps
    assert "logprobs" in caps
    assert "tools" in caps


def test_note_capabilities_empty_body():
    from app.preflight.connection_check import _note_capabilities
    caps = _note_capabilities({})
    assert caps["stream"] is None
    assert caps["logprobs"] is False


def test_note_capabilities_logprobs_present():
    from app.preflight.connection_check import _note_capabilities
    body = {
        "choices": [{"message": {"content": "OK"}, "logprobs": [{"token": "OK"}]}],
    }
    caps = _note_capabilities(body)
    # logprobs key present in choices[0] → logprobs capability is truthy
    assert "logprobs" in caps


# ---------------------------------------------------------------------------
# _http_status_to_error_code mapping
# ---------------------------------------------------------------------------

def test_http_status_401_maps_to_invalid_key():
    from app.preflight.connection_check import _http_status_to_error_code
    code = _http_status_to_error_code(401, "")
    assert code.value == "E_AUTH_INVALID_KEY"


def test_http_status_403_maps_to_forbidden():
    from app.preflight.connection_check import _http_status_to_error_code
    code = _http_status_to_error_code(403, "")
    assert code.value == "E_AUTH_FORBIDDEN"


def test_http_status_404_maps_to_model_not_found():
    from app.preflight.connection_check import _http_status_to_error_code
    code = _http_status_to_error_code(404, "")
    assert code.value == "E_MODEL_NOT_FOUND"


def test_http_status_429_maps_to_rate_limited():
    from app.preflight.connection_check import _http_status_to_error_code
    code = _http_status_to_error_code(429, "")
    assert code.value == "E_UPSTREAM_RATE_LIMITED"


def test_http_status_500_maps_to_internal():
    from app.preflight.connection_check import _http_status_to_error_code
    code = _http_status_to_error_code(500, "")
    assert code.value == "E_UPSTREAM_INTERNAL"


def test_http_status_502_maps_to_bad_gateway():
    from app.preflight.connection_check import _http_status_to_error_code
    code = _http_status_to_error_code(502, "")
    assert code.value == "E_UPSTREAM_BAD_GATEWAY"


def test_http_status_503_maps_to_service_unavailable():
    from app.preflight.connection_check import _http_status_to_error_code
    code = _http_status_to_error_code(503, "")
    assert code.value == "E_UPSTREAM_SERVICE_UNAVAILABLE"


def test_http_status_400_with_auth_body():
    from app.preflight.connection_check import _http_status_to_error_code
    code = _http_status_to_error_code(400, "invalid api_key provided")
    assert code.value == "E_AUTH_INVALID_KEY"


def test_http_status_400_with_model_body():
    from app.preflight.connection_check import _http_status_to_error_code
    code = _http_status_to_error_code(400, "model not found")
    assert code.value == "E_MODEL_NOT_FOUND"


# ---------------------------------------------------------------------------
# run_preflight() with mocked network (A1 fails fast — no network needed)
# ---------------------------------------------------------------------------

def test_run_preflight_invalid_url_fails_at_a1():
    from app.preflight.connection_check import run_preflight
    report = run_preflight("", "sk-test", "gpt-4", timeout=1.0)
    assert report.passed is False
    assert len(report.steps) >= 1
    step_a1 = next(s for s in report.steps if s.step == "A1")
    assert step_a1.passed is False


def test_run_preflight_missing_model_fails_at_a1():
    from app.preflight.connection_check import run_preflight
    report = run_preflight("https://api.example.com", "sk-test", "", timeout=1.0)
    assert report.passed is False
    step_a1 = next(s for s in report.steps if s.step == "A1")
    assert step_a1.passed is False


def test_run_preflight_unreachable_host_fails_at_a2():
    """A valid URL but unreachable host should fail at A2 (TCP connect)."""
    from app.preflight.connection_check import run_preflight
    # Use a valid format URL but definitely unreachable host
    report = run_preflight(
        "http://127.0.0.1:19999",  # nothing listening on this port
        "sk-test",
        "gpt-4",
        timeout=1.0,
    )
    assert report.passed is False
    # A1 should pass (valid format)
    step_a1 = next((s for s in report.steps if s.step == "A1"), None)
    if step_a1:
        assert step_a1.passed is True
    # A2 should fail
    step_a2 = next((s for s in report.steps if s.step == "A2"), None)
    if step_a2:
        assert step_a2.passed is False


def test_run_preflight_report_has_total_duration():
    from app.preflight.connection_check import run_preflight
    report = run_preflight("", "sk-test", "gpt-4", timeout=1.0)
    assert report.total_duration_ms >= 0.0


def test_run_preflight_report_to_dict_is_serializable():
    from app.preflight.connection_check import run_preflight
    import json
    report = run_preflight("", "sk-test", "gpt-4", timeout=1.0)
    d = report.to_dict()
    # Should be JSON serializable
    json.dumps(d)


def test_run_preflight_invalid_url_format_non_retryable():
    from app.preflight.connection_check import run_preflight
    report = run_preflight("ftp://bad-scheme.com", "sk-test", "gpt-4", timeout=1.0)
    assert report.passed is False
    if report.first_error:
        assert report.first_error.retryable is False
