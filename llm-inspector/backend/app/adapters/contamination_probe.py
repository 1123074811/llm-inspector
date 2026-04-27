"""
adapters/contamination_probe.py — v17 Phase 1: Cross-family auth pollution probe.

Many wrapper / proxy services advertise themselves as Anthropic, OpenAI, or
Google compatible but accept *any* auth scheme on the same endpoint, which
is a strong signal of being a multi-vendor passthrough rather than the real
upstream.  This module sends ONE bad-auth request per probe (no chat
completion is performed, so token cost is essentially zero) and reports
whether the response shape contradicts the claimed family.

Design constraints
------------------
- Pure stdlib (urllib).  Aligns with adapters/openai_compat.py.
- Single short request per probe; total wall-clock < 5s under good network.
- Never raises — all failures collapse into ProbeOutcome with explanation.
- Does not depend on the caller's adapter object — works on raw base_url.

References
----------
- OpenAI error envelope: ``{"error": {"code", "param", "message", "type"}}``
  https://platform.openai.com/docs/api-reference/authentication
- Anthropic error envelope: ``{"type": "error", "error": {"type", "message"}}``
  https://docs.anthropic.com/en/api/errors
- Google Gemini error envelope: ``{"error": {"code", "message", "status", "details"}}``
  https://ai.google.dev/api/rest
"""
from __future__ import annotations

import json
import ssl
import time
import urllib.error
import urllib.request
import urllib.parse
from dataclasses import dataclass, field
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)


def _build_ssl_ctx() -> ssl.SSLContext:
    try:
        import certifi  # type: ignore[import]
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


_SSL_CTX = _build_ssl_ctx()


@dataclass
class ProbeOutcome:
    """Single contamination probe outcome (pure data, JSON-serializable)."""

    probe_name: str
    target_family: str          # what the caller claims this endpoint serves
    sent_auth_scheme: str       # "bearer" | "x-api-key" | "both"
    status_code: int | None
    error_envelope: dict[str, Any] = field(default_factory=dict)
    contradicts_claim: bool = False
    contradiction_reason: str = ""
    latency_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "probe_name": self.probe_name,
            "target_family": self.target_family,
            "sent_auth_scheme": self.sent_auth_scheme,
            "status_code": self.status_code,
            "error_envelope": self.error_envelope,
            "contradicts_claim": self.contradicts_claim,
            "contradiction_reason": self.contradiction_reason,
            "latency_ms": self.latency_ms,
        }


def _send_bad_auth(
    url: str,
    headers: dict[str, str],
    payload: dict,
    timeout: int = 8,
) -> tuple[int | None, dict[str, Any], int]:
    """Return (status_code, parsed_body, latency_ms).  Never raises."""
    t0 = time.monotonic()
    data = json.dumps(payload).encode("utf-8")
    try:
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, context=_SSL_CTX, timeout=timeout) as resp:
            body_raw = resp.read()
            try:
                body = json.loads(body_raw.decode("utf-8"))
            except Exception:
                body = {"raw": body_raw.decode("utf-8", errors="replace")[:512]}
            return resp.status, body, int((time.monotonic() - t0) * 1000)
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode("utf-8"))
        except Exception:
            body = {}
        return e.code, body, int((time.monotonic() - t0) * 1000)
    except Exception as e:
        return None, {"transport_error": str(e)[:200]}, int((time.monotonic() - t0) * 1000)


def _looks_like_openai_error(body: dict) -> bool:
    """OpenAI error envelope: error.{code, param, message, type}."""
    err = body.get("error") if isinstance(body, dict) else None
    if not isinstance(err, dict):
        return False
    return all(k in err for k in ("type", "message"))


def _looks_like_anthropic_error(body: dict) -> bool:
    """Anthropic error envelope: top-level type='error' + error.{type, message}."""
    if not isinstance(body, dict):
        return False
    if body.get("type") == "error" and isinstance(body.get("error"), dict):
        err = body["error"]
        return "type" in err and "message" in err
    return False


def _looks_like_google_error(body: dict) -> bool:
    """Google Gemini error envelope: error.{code, message, status}."""
    err = body.get("error") if isinstance(body, dict) else None
    if not isinstance(err, dict):
        return False
    return "status" in err and "code" in err


# ── Public probe entry points ────────────────────────────────────────────────


def probe_anthropic_with_bearer(base_url: str) -> ProbeOutcome:
    """Send Authorization: Bearer to an Anthropic /v1/messages endpoint.

    A real Anthropic upstream returns 401 with Anthropic-shaped error envelope
    (``{"type":"error","error":{"type":"authentication_error",...}}``).  A
    proxy that re-routes by Authorization header may return 200 or an
    OpenAI-shaped error envelope, which contradicts the claim.
    """
    url = base_url.rstrip("/") + "/v1/messages"
    headers = {
        "Authorization": "Bearer sk-fake-llm-inspector-probe-v17",
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
        "User-Agent": "LLMInspector/17.0 ContaminationProbe",
    }
    payload = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 1,
        "messages": [{"role": "user", "content": "hi"}],
    }
    status, body, latency = _send_bad_auth(url, headers, payload)
    contradicts = False
    reason = ""
    if status == 200:
        contradicts = True
        reason = "Anthropic endpoint accepted Bearer-only auth and returned 200 (no x-api-key)"
    elif _looks_like_openai_error(body) and not _looks_like_anthropic_error(body):
        contradicts = True
        reason = "Error envelope matches OpenAI shape, not Anthropic"
    return ProbeOutcome(
        probe_name="anthropic_bearer_pollution",
        target_family="anthropic",
        sent_auth_scheme="bearer",
        status_code=status,
        error_envelope=body if isinstance(body, dict) else {},
        contradicts_claim=contradicts,
        contradiction_reason=reason,
        latency_ms=latency,
    )


def probe_openai_with_xapikey(base_url: str) -> ProbeOutcome:
    """Send x-api-key to an OpenAI-style /chat/completions endpoint.

    A real OpenAI upstream ignores ``x-api-key`` and returns 401 because
    Authorization is missing.  A wrapper that re-routes by ``x-api-key`` may
    accept the request as if it were Anthropic-style.
    """
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "x-api-key": "sk-fake-llm-inspector-probe-v17",
        "Content-Type": "application/json",
        "User-Agent": "LLMInspector/17.0 ContaminationProbe",
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
    }
    status, body, latency = _send_bad_auth(url, headers, payload)
    contradicts = False
    reason = ""
    if status == 200:
        contradicts = True
        reason = "OpenAI endpoint accepted x-api-key-only auth and returned 200"
    elif _looks_like_anthropic_error(body) and not _looks_like_openai_error(body):
        contradicts = True
        reason = "Error envelope matches Anthropic shape, not OpenAI"
    return ProbeOutcome(
        probe_name="openai_xapikey_pollution",
        target_family="openai",
        sent_auth_scheme="x-api-key",
        status_code=status,
        error_envelope=body if isinstance(body, dict) else {},
        contradicts_claim=contradicts,
        contradiction_reason=reason,
        latency_ms=latency,
    )


def run_contamination_probes(base_url: str, target_family: str) -> list[ProbeOutcome]:
    """Dispatch the appropriate cross-family probes for the claimed family.

    Returns a list of ProbeOutcome (possibly empty if the family has no
    well-established cross-pollination test).  Network errors do not raise.
    """
    outcomes: list[ProbeOutcome] = []
    fam = (target_family or "").lower()
    try:
        if "anthropic" in fam or "claude" in fam:
            outcomes.append(probe_anthropic_with_bearer(base_url))
        elif "openai" in fam or "gpt" in fam or "azure" in fam:
            outcomes.append(probe_openai_with_xapikey(base_url))
    except Exception as e:
        logger.warning("contamination probe failed", base_url=base_url, error=str(e))
    return outcomes
