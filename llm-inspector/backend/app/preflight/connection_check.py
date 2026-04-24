"""
preflight/connection_check.py — Preflight connectivity and authentication check.

Runs 5 check stages before any actual LLM testing:
  A1. URL format validation
  A2. TCP/HTTP reachability
  A3. API Key authentication
  A4. Response schema validation
  A5. Capability notes (non-blocking)

References:
  - RFC 9110 HTTP Semantics
  - RFC 9457 Problem Details for HTTP APIs
  - OpenAI API Reference
"""
from __future__ import annotations

import json
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

from app.core.logging import get_logger
from app.preflight.error_taxonomy import ErrorCode, ErrorDetail, make_error

logger = get_logger(__name__)


@dataclass
class PreflightStep:
    """Result of a single preflight check step."""
    step: str                    # "A1" .. "A5"
    name: str
    passed: bool
    duration_ms: float
    error: ErrorDetail | None = None
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "name": self.name,
            "passed": self.passed,
            "duration_ms": round(self.duration_ms, 1),
            "error": self.error.to_dict() if self.error else None,
            "notes": self.notes,
        }


@dataclass
class PreflightReport:
    """Full preflight check report."""
    passed: bool
    steps: list[PreflightStep] = field(default_factory=list)
    first_error: ErrorDetail | None = None
    total_duration_ms: float = 0.0
    capabilities: dict = field(default_factory=dict)  # {stream: bool, logprobs: bool, tools: bool}
    checked_at: str = ""

    def to_dict(self) -> dict:
        import datetime
        return {
            "passed": self.passed,
            "first_error": self.first_error.to_dict() if self.first_error else None,
            "steps": [s.to_dict() for s in self.steps],
            "total_duration_ms": round(self.total_duration_ms, 1),
            "capabilities": self.capabilities,
            "checked_at": self.checked_at or datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00","Z"),
        }


def run_preflight(base_url: str, api_key: str, model_name: str,
                  timeout: float = 10.0) -> PreflightReport:
    """
    Run all preflight checks. Returns a PreflightReport.
    Stops at first hard failure (non-retryable) but continues for retryable ones.
    """
    import datetime
    t_start = time.time()
    steps: list[PreflightStep] = []
    first_error: ErrorDetail | None = None
    capabilities: dict = {}

    def add_step(step: PreflightStep):
        nonlocal first_error
        steps.append(step)
        if not step.passed and first_error is None and step.error:
            first_error = step.error
        logger.info(f"Preflight {step.step} {step.name}: {'PASS' if step.passed else 'FAIL'}",
                    run_preflight=True, step=step.step, passed=step.passed)

    # --- A1: Input validation ---
    t = time.time()
    a1_err, a1_notes = _check_inputs(base_url, api_key, model_name)
    add_step(PreflightStep("A1", "input_validation", a1_err is None,
                           (time.time() - t) * 1000, a1_err, a1_notes))
    if a1_err and not a1_err.retryable:
        return PreflightReport(False, steps, first_error,
                               (time.time() - t_start) * 1000,
                               checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00","Z"))

    # --- A2: TCP / HTTP reachability ---
    t = time.time()
    a2_err, a2_notes = _check_reachability(base_url, timeout)
    add_step(PreflightStep("A2", "reachability", a2_err is None,
                           (time.time() - t) * 1000, a2_err, a2_notes))
    if a2_err:
        return PreflightReport(False, steps, first_error,
                               (time.time() - t_start) * 1000,
                               checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00","Z"))

    # --- A3: Auth + minimal completion probe ---
    t = time.time()
    a3_err, probe_body, a3_notes = _check_auth_and_probe(base_url, api_key, model_name, timeout)
    add_step(PreflightStep("A3", "auth_and_probe", a3_err is None,
                           (time.time() - t) * 1000, a3_err, a3_notes))
    if a3_err:
        # A3 failed: skip A4 (schema check would always fail with empty probe_body,
        # producing a misleading E_SCHEMA_MISSING_CHOICES cascading error).
        add_step(PreflightStep("A4", "response_schema", False, 0.0,
                               None, "skipped — A3 did not return a valid response"))
        add_step(PreflightStep("A5", "capability_notes", True, 0.0,
                               None, "skipped — A3 failed"))
        return PreflightReport(False, steps, first_error,
                               (time.time() - t_start) * 1000,
                               checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00","Z"))

    # --- A4: Response schema ---
    t = time.time()
    a4_err, a4_notes = _check_schema(probe_body)
    add_step(PreflightStep("A4", "response_schema", a4_err is None,
                           (time.time() - t) * 1000, a4_err, a4_notes))

    # --- A5: Capability notes (stream / logprobs — non-blocking) ---
    t = time.time()
    capabilities = _note_capabilities(probe_body)
    add_step(PreflightStep("A5", "capability_notes", True,
                           (time.time() - t) * 1000, None,
                           f"stream={capabilities.get('stream')}, logprobs={capabilities.get('logprobs')}"))

    passed = first_error is None
    return PreflightReport(passed, steps, first_error,
                           (time.time() - t_start) * 1000,
                           capabilities,
                           checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00","Z"))


# ── Private helpers ──────────────────────────────────────────────────────────

def _check_inputs(base_url: str, api_key: str, model_name: str
                  ) -> tuple[ErrorDetail | None, str]:
    if not base_url or not base_url.strip():
        return make_error(ErrorCode.E_URL_EMPTY), ""
    url = base_url.strip()
    if not re.match(r"^https?://", url):
        return make_error(ErrorCode.E_URL_INVALID_FORMAT), f"got: {url[:60]}"
    if not model_name or not model_name.strip():
        return make_error(ErrorCode.E_MODEL_NAME_EMPTY), ""
    if not api_key or not api_key.strip():
        return make_error(ErrorCode.E_API_KEY_MISSING), ""
    return None, "URL/key/model format OK"


def _check_reachability(base_url: str, timeout: float) -> tuple[ErrorDetail | None, str]:
    """Try a TCP connect probe to the base URL host."""
    try:
        parsed = urllib.parse.urlparse(base_url.rstrip("/"))
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        # TCP connect probe
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.close()
        return None, f"TCP OK to {host}:{port}"
    except socket.gaierror as e:
        return make_error(ErrorCode.E_NET_DNS_FAIL, raw_body=str(e)), str(e)
    except ConnectionRefusedError as e:
        return make_error(ErrorCode.E_NET_CONN_REFUSED, raw_body=str(e)), str(e)
    except socket.timeout as e:
        return make_error(ErrorCode.E_NET_TIMEOUT, raw_body=str(e)), str(e)
    except OSError as e:
        msg = str(e).lower()
        if "ssl" in msg or "certificate" in msg:
            return make_error(ErrorCode.E_TLS_INVALID, raw_body=str(e)), str(e)
        return make_error(ErrorCode.E_NET_CONN_REFUSED, raw_body=str(e)), str(e)


def _check_auth_and_probe(base_url: str, api_key: str, model_name: str,
                           timeout: float) -> tuple[ErrorDetail | None, dict, str]:
    """Send a minimal chat/completions request (1 token) to verify auth."""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = json.dumps({
        "model": model_name,
        "messages": [{"role": "user", "content": "Reply with the single word: OK"}],
        "max_tokens": 5,
        "temperature": 0,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body_bytes = resp.read()
            try:
                body = json.loads(body_bytes)
            except json.JSONDecodeError:
                return make_error(ErrorCode.E_SCHEMA_INVALID_JSON,
                                  raw_status=resp.status,
                                  raw_body=body_bytes[:200].decode("utf-8", errors="replace")), {}, ""
            return None, body, f"HTTP {resp.status}"
    except urllib.error.HTTPError as e:
        raw_body = ""
        try:
            raw_body = e.read()[:500].decode("utf-8", errors="replace")
        except Exception:
            pass
        code = _http_status_to_error_code(e.code, raw_body)
        return make_error(code, raw_status=e.code, raw_body=raw_body), {}, f"HTTP {e.code}"
    except socket.timeout as e:
        return make_error(ErrorCode.E_NET_TIMEOUT, raw_body=str(e) or f"timed out after {timeout}s"), {}, f"timeout after {timeout}s"
    except ConnectionRefusedError as e:
        return make_error(ErrorCode.E_NET_CONN_REFUSED, raw_body=str(e)), {}, str(e)
    except socket.gaierror as e:
        return make_error(ErrorCode.E_NET_DNS_FAIL, raw_body=str(e)), {}, str(e)
    except Exception as e:
        return make_error(ErrorCode.E_UNKNOWN, raw_body=str(e)), {}, str(e)


def _http_status_to_error_code(status: int, body: str) -> ErrorCode:
    if status == 401:
        return ErrorCode.E_AUTH_INVALID_KEY
    if status == 403:
        return ErrorCode.E_AUTH_FORBIDDEN
    if status == 404:
        return ErrorCode.E_MODEL_NOT_FOUND
    if status == 429:
        return ErrorCode.E_UPSTREAM_RATE_LIMITED
    if status == 500:
        return ErrorCode.E_UPSTREAM_INTERNAL
    if status in (502, 504):
        return ErrorCode.E_UPSTREAM_BAD_GATEWAY
    if status == 503:
        return ErrorCode.E_UPSTREAM_SERVICE_UNAVAILABLE
    return ErrorCode.E_UNKNOWN


def _check_schema(body: dict) -> tuple[ErrorDetail | None, str]:
    """Validate OpenAI-compatible response schema."""
    if not isinstance(body, dict):
        return make_error(ErrorCode.E_SCHEMA_INVALID_JSON), "not a dict"
    if "error" in body:
        # upstream returned an error object — already caught in probe step
        return None, "upstream error object (handled)"
    if "choices" not in body or not body.get("choices"):
        return make_error(ErrorCode.E_SCHEMA_MISSING_CHOICES), f"keys: {list(body.keys())}"
    return None, "schema OK"


def _note_capabilities(body: dict) -> dict:
    """Non-blocking capability detection from probe response."""
    return {
        "stream": None,    # Not tested in basic preflight (would need stream=True probe)
        "logprobs": "logprobs" in body.get("choices", [{}])[0] if body.get("choices") else False,
        "tools": None,
    }
