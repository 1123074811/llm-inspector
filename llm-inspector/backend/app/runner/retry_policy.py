"""
runner/retry_policy.py — Tiered retry policy for LLM API calls.

Implements exponential backoff for network errors and Retry-After
header handling for HTTP 429 rate-limit responses.

Reference:
    Google SRE Book, Chapter 22: "Handling Overload"
    URL: https://sre.google/sre-book/handling-overload/

    RFC 7231 Section 7.1.3: Retry-After header semantics.
    URL: https://datatracker.ietf.org/doc/html/rfc7231#section-7.1.3
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import time
from dataclasses import dataclass, field

from app.core.logging import get_logger

logger = get_logger(__name__)

# -- Helpers ------------------------------------------------------------------

def _is_network_error(exc: BaseException) -> bool:
    """Return True for transient network/connectivity errors worth retrying."""
    exc_str = str(type(exc).__name__).lower()
    exc_msg = str(exc).lower()
    # ConnectionError, TimeoutError, socket.timeout, urllib.error.URLError, etc.
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    if "connectionerror" in exc_str or "timeout" in exc_str:
        return True
    if "connection" in exc_msg and ("refused" in exc_msg or "reset" in exc_msg):
        return True
    return False


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Return True for HTTP 429 rate-limit responses."""
    exc_str = str(exc).lower()
    exc_type = str(type(exc).__name__).lower()
    if "429" in exc_str or "rate_limit" in exc_str or "rate limit" in exc_str:
        return True
    if "ratelimit" in exc_type or "toomanyrequests" in exc_type:
        return True
    return False


def _parse_retry_after(exc: BaseException) -> float | None:
    """
    Try to extract Retry-After seconds from an exception message.

    Looks for patterns like:
        "Retry-After: 30"
        "retry_after=30"
        "retry after 30 seconds"
    Returns None if not found.
    """
    exc_str = str(exc)
    patterns = [
        r"[Rr]etry-?[Aa]fter[:\s=]+(\d+(?:\.\d+)?)",
        r"[Rr]etry[_\s][Aa]fter[:\s=]+(\d+(?:\.\d+)?)",
    ]
    for pat in patterns:
        m = re.search(pat, exc_str)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    return None


# v16 Phase 2: Additional error classifiers

def _is_5xx_error(exc: BaseException) -> bool:
    """Return True for upstream 5xx HTTP errors (500/502/503/504)."""
    exc_str = str(exc).lower()
    exc_type = str(type(exc).__name__).lower()
    if "httperror" in exc_type:
        # urllib.error.HTTPError — check status code in message
        for code in (500, 502, 503, 504):
            if str(code) in exc_str:
                return True
    # Also detect from LLMResponse-like objects
    if hasattr(exc, "status_code") and exc.status_code in (500, 502, 503, 504):
        return True
    if hasattr(exc, "http_status") and exc.http_status in (500, 502, 503, 504):
        return True
    return False


def _is_truncation_error(exc: BaseException) -> bool:
    """Return True for truncation errors (finish_reason=length)."""
    if hasattr(exc, "finish_reason") and exc.finish_reason == "length":
        return True
    exc_str = str(exc).lower()
    if "finish_reason" in exc_str and "length" in exc_str:
        return True
    return False


def _is_json_decode_error(exc: BaseException) -> bool:
    """Return True for JSON decode / parse errors from upstream."""
    exc_type = str(type(exc).__name__).lower()
    if "json" in exc_type and "decode" in exc_type:
        return True
    exc_str = str(exc).lower()
    if "parse_error" in exc_str or "json" in exc_str and "decode" in exc_str:
        return True
    return False


def _next_max_tokens(current: int, attempt: int) -> int:
    """v16 Phase 2: Calculate next max_tokens for truncation retry.

    Doubles max_tokens on each truncation retry, capped at 4096.
    """
    return min(current * (2 ** attempt), 4096)


def _jitter(delay: float, ratio: float = 0.2) -> float:
    """v16 Phase 2: Add random jitter to delay.

    Reference: AWS Architecture Blog "Exponential Backoff and Jitter"
    delay * (1 - ratio/2) <= actual <= delay * (1 + ratio/2)
    """
    import random
    return delay * (1.0 + (random.random() - 0.5) * ratio)


# -- Data structures ----------------------------------------------------------

@dataclass
class RetryConfig:
    """Configuration for the tiered retry policy."""
    max_retries_network: int = 3
    max_retries_429: int = 2
    # v16 Phase 2: Extended retry tiers
    max_retries_5xx: int = 3           # Upstream 5xx with exponential backoff + jitter
    max_retries_truncation: int = 2    # finish_reason=length — double max_tokens and retry
    max_retries_decode: int = 1        # JSON parse failure — single retry
    backoff_base_s: float = 0.5
    backoff_max_s: float = 8.0
    jitter_ratio: float = 0.2          # AWS Architecture Blog: "Exponential Backoff and Jitter"


@dataclass
class RetryOutcome:
    """Result of a with_retry() call."""
    success: bool
    attempts: int
    final_error: str | None = None
    retry_events: list[dict] = field(default_factory=list)
    result: object = None


# -- Error event persistence --------------------------------------------------

def _write_error_event(run_id: str, event_dict: dict) -> None:
    """
    Append a JSON line to data/traces/{run_id}/errors.jsonl.

    Non-fatal: any I/O error is silently logged.
    """
    try:
        from app.core.config import settings
        base = pathlib.Path(settings.DATA_DIR) / "traces" / run_id
        base.mkdir(parents=True, exist_ok=True)
        target = base / "errors.jsonl"
        with open(target, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event_dict, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.debug("_write_error_event failed (non-fatal)", error=str(exc))


# -- Core retry logic ---------------------------------------------------------

def with_retry(
    func,
    *args,
    config: RetryConfig | None = None,
    run_id: str | None = None,
    **kwargs,
) -> RetryOutcome:
    """
    Call ``func(*args, **kwargs)`` with tiered retry logic.

    Retry tiers:
    - Network errors (ConnectionError / TimeoutError / OSError):
      Exponential backoff up to ``config.max_retries_network`` retries.
      Delay = min(backoff_base_s * 2^attempt, backoff_max_s).

    - HTTP 429 rate-limit errors:
      Read Retry-After from exception message if available, else
      use ``backoff_base_s * 2``. Up to ``config.max_retries_429`` retries.

    - All other exceptions:
      Re-raised immediately (no retry).

    Args:
        func: Callable to invoke.
        *args: Positional arguments forwarded to func.
        config: RetryConfig instance; uses defaults if None.
        run_id: If provided, retry events are appended to
                data/traces/{run_id}/errors.jsonl.
        **kwargs: Keyword arguments forwarded to func.

    Returns:
        RetryOutcome with success flag, attempt count, and event log.
    """
    cfg = config or RetryConfig()
    network_attempts = 0
    rate_limit_attempts = 0
    five_xx_attempts = 0
    truncation_attempts = 0
    decode_attempts = 0
    retry_events: list[dict] = []
    attempt = 0

    while True:
        attempt += 1
        try:
            result = func(*args, **kwargs)
            return RetryOutcome(
                success=True,
                attempts=attempt,
                final_error=None,
                retry_events=retry_events,
                result=result,
            )
        except Exception as exc:
            if _is_rate_limit_error(exc):
                rate_limit_attempts += 1
                if rate_limit_attempts > cfg.max_retries_429:
                    event = {
                        "attempt": attempt,
                        "error_type": "rate_limit",
                        "error": str(exc)[:300],
                        "action": "exhausted",
                    }
                    retry_events.append(event)
                    if run_id:
                        _write_error_event(run_id, event)
                    return RetryOutcome(
                        success=False,
                        attempts=attempt,
                        final_error=str(exc),
                        retry_events=retry_events,
                    )
                # Determine sleep duration
                retry_after = _parse_retry_after(exc)
                delay = retry_after if retry_after is not None else cfg.backoff_base_s * 2
                delay = min(delay, cfg.backoff_max_s)
                delay = _jitter(delay, cfg.jitter_ratio)
                event = {
                    "attempt": attempt,
                    "error_type": "rate_limit_429",
                    "error": str(exc)[:300],
                    "action": "retry",
                    "delay_s": round(delay, 3),
                }
                retry_events.append(event)
                if run_id:
                    _write_error_event(run_id, event)
                logger.debug(
                    "RetryPolicy: 429 rate-limit, retrying",
                    attempt=attempt,
                    delay_s=round(delay, 3),
                    run_id=run_id,
                )
                time.sleep(delay)

            elif _is_5xx_error(exc):
                # v16 Phase 2: 5xx retry with exponential backoff + jitter
                five_xx_attempts += 1
                if five_xx_attempts > cfg.max_retries_5xx:
                    event = {
                        "attempt": attempt,
                        "error_type": "5xx",
                        "error": str(exc)[:300],
                        "action": "exhausted",
                    }
                    retry_events.append(event)
                    if run_id:
                        _write_error_event(run_id, event)
                    return RetryOutcome(
                        success=False,
                        attempts=attempt,
                        final_error=str(exc),
                        retry_events=retry_events,
                    )
                delay = min(cfg.backoff_base_s * (2 ** (five_xx_attempts - 1)), cfg.backoff_max_s)
                delay = _jitter(delay, cfg.jitter_ratio)
                event = {
                    "attempt": attempt,
                    "error_type": "5xx",
                    "error": str(exc)[:300],
                    "action": "retry",
                    "delay_s": round(delay, 3),
                }
                retry_events.append(event)
                if run_id:
                    _write_error_event(run_id, event)
                logger.debug(
                    "RetryPolicy: 5xx upstream error, retrying",
                    attempt=attempt,
                    delay_s=round(delay, 3),
                    run_id=run_id,
                )
                time.sleep(delay)

            elif _is_truncation_error(exc):
                # v16 Phase 2: Truncation retry — double max_tokens
                truncation_attempts += 1
                if truncation_attempts > cfg.max_retries_truncation:
                    event = {
                        "attempt": attempt,
                        "error_type": "truncation",
                        "error": str(exc)[:300],
                        "action": "exhausted",
                    }
                    retry_events.append(event)
                    if run_id:
                        _write_error_event(run_id, event)
                    return RetryOutcome(
                        success=False,
                        attempts=attempt,
                        final_error=str(exc),
                        retry_events=retry_events,
                    )
                # Compute next max_tokens
                current_mt = getattr(exc, "max_tokens", None) or kwargs.get("max_tokens", 256)
                next_mt = _next_max_tokens(current_mt, truncation_attempts)
                event = {
                    "attempt": attempt,
                    "error_type": "truncation",
                    "error": str(exc)[:300],
                    "action": "retry_with_doubled_tokens",
                    "max_tokens": next_mt,
                }
                retry_events.append(event)
                if run_id:
                    _write_error_event(run_id, event)
                logger.debug(
                    "RetryPolicy: truncation, retrying with doubled max_tokens",
                    attempt=attempt,
                    max_tokens=next_mt,
                    run_id=run_id,
                )
                # Inject new max_tokens into kwargs for next attempt
                kwargs["max_tokens"] = next_mt

            elif _is_json_decode_error(exc):
                # v16 Phase 2: JSON decode retry — single retry without delay
                decode_attempts += 1
                if decode_attempts > cfg.max_retries_decode:
                    event = {
                        "attempt": attempt,
                        "error_type": "json_decode",
                        "error": str(exc)[:300],
                        "action": "exhausted",
                    }
                    retry_events.append(event)
                    if run_id:
                        _write_error_event(run_id, event)
                    return RetryOutcome(
                        success=False,
                        attempts=attempt,
                        final_error=str(exc),
                        retry_events=retry_events,
                    )
                event = {
                    "attempt": attempt,
                    "error_type": "json_decode",
                    "error": str(exc)[:300],
                    "action": "retry",
                }
                retry_events.append(event)
                if run_id:
                    _write_error_event(run_id, event)
                logger.debug(
                    "RetryPolicy: JSON decode error, retrying",
                    attempt=attempt,
                    run_id=run_id,
                )

            elif _is_network_error(exc):
                network_attempts += 1
                if network_attempts > cfg.max_retries_network:
                    event = {
                        "attempt": attempt,
                        "error_type": "network",
                        "error": str(exc)[:300],
                        "action": "exhausted",
                    }
                    retry_events.append(event)
                    if run_id:
                        _write_error_event(run_id, event)
                    return RetryOutcome(
                        success=False,
                        attempts=attempt,
                        final_error=str(exc),
                        retry_events=retry_events,
                    )
                # Exponential backoff + jitter
                delay = min(cfg.backoff_base_s * (2 ** (network_attempts - 1)), cfg.backoff_max_s)
                delay = _jitter(delay, cfg.jitter_ratio)
                event = {
                    "attempt": attempt,
                    "error_type": "network",
                    "error": str(exc)[:300],
                    "action": "retry",
                    "delay_s": round(delay, 3),
                }
                retry_events.append(event)
                if run_id:
                    _write_error_event(run_id, event)
                logger.debug(
                    "RetryPolicy: network error, retrying",
                    attempt=attempt,
                    delay_s=round(delay, 3),
                    run_id=run_id,
                )
                time.sleep(delay)

            else:
                # Non-retryable — re-raise immediately
                raise
