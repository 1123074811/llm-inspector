"""
Async OpenAI-compatible LLM adapter.

Phase A of asyncio migration — pure asyncio I/O using asyncio.open_connection()
and http.client in a thread (asyncio.to_thread) for HTTPS.
No third-party dependency required (no aiohttp/httpx).

Design:
- AsyncOpenAICompatibleAdapter mirrors the sync adapter's public API
  but all methods are coroutines (async def).
- Uses asyncio.to_thread() to offload the blocking urllib calls, which
  releases the event loop during I/O wait without changing the wire protocol.
- Stage 2 (future): replace to_thread wrapping with native asyncio streams.
"""
from __future__ import annotations

import asyncio
import json
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import hashlib
from typing import Callable

from app.core.schemas import (
    LLMRequest, LLMResponse, StreamChunk, StreamCaptureResult,
)
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_RETRYABLE_CODES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_RETRY_BASE_DELAY_SEC = 1.0

_SSL_CTX = ssl.create_default_context()


async def _async_with_retry(coro_fn: Callable, max_retries: int = _MAX_RETRIES) -> LLMResponse:
    """Async retry wrapper — exponential back-off on retryable status codes."""
    last_result = None
    for attempt in range(max_retries + 1):
        result: LLMResponse = await coro_fn()
        if result is not None and result.ok:
            return result

        is_retryable = False
        if result is not None:
            if result.status_code in _RETRYABLE_CODES:
                is_retryable = True
            elif result.error_type == "timeout":
                is_retryable = True

        if not is_retryable:
            return result
        if attempt < max_retries:
            delay = _RETRY_BASE_DELAY_SEC * (2 ** attempt)
            await asyncio.sleep(delay)
        last_result = result

    return last_result or LLMResponse(
        error_type="unknown",
        error_message="Async retry loop exhausted without receiving response",
        latency_ms=0,
    )


class AsyncOpenAICompatibleAdapter:
    """Async adapter for OpenAI-compatible /chat/completions endpoints.

    Usage:
        adapter = AsyncOpenAICompatibleAdapter(base_url, api_key)
        resp = await adapter.achat(req)

    Backward compatibility:
        The synchronous OpenAICompatibleAdapter is still available and
        unchanged. This class is a drop-in async counterpart.
    """

    @staticmethod
    def _normalize_api_key(api_key: str) -> str:
        key = (api_key or "").strip()
        if key.lower().startswith("bearer "):
            key = key[7:].strip()
        return key

    @staticmethod
    def _looks_like_qianfan_host(base_url: str) -> bool:
        try:
            host = (urllib.parse.urlparse(base_url).hostname or "").lower()
        except Exception:
            host = ""
        return host.endswith("qianfan.baidubce.com")

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self._api_key = self._normalize_api_key(api_key)
        self._base_headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (compatible; LLMInspector/3.0-async)",
        }
        # Qianfan IAM token cache (thread-safe enough: only written from async context)
        self._qianfan_ak: str | None = None
        self._qianfan_sk: str | None = None
        self._qianfan_iam_token: str | None = None
        self._qianfan_iam_expiry: float = 0.0

        if self._looks_like_qianfan_host(self.base_url):
            if ":" in self._api_key:
                ak, sk = self._api_key.split(":", 1)
                if ak.strip() and sk.strip():
                    self._qianfan_ak = ak.strip()
                    self._qianfan_sk = sk.strip()

    # ── Auth helpers ──────────────────────────────────────────────────────────

    async def _auth_headers(self) -> dict:
        headers = dict(self._base_headers)
        if self._qianfan_ak and self._qianfan_sk:
            token = await self._fetch_qianfan_token()
            if token:
                headers["Authorization"] = f"Bearer {token}"
        return headers

    async def _fetch_qianfan_token(self) -> str | None:
        now = time.monotonic()
        if self._qianfan_iam_token and now < self._qianfan_iam_expiry:
            return self._qianfan_iam_token
        # Token fetch is a short blocking call — wrap in to_thread
        try:
            result = await asyncio.to_thread(self._fetch_qianfan_token_sync)
            return result
        except Exception as e:
            logger.warning("Qianfan async token fetch failed", error=str(e))
            return None

    def _fetch_qianfan_token_sync(self) -> str | None:
        ak = self._qianfan_ak
        sk = self._qianfan_sk
        token_url = (
            "https://aip.baidubce.com/oauth/2.0/token"
            f"?grant_type=client_credentials&client_id={urllib.parse.quote(ak)}"
            f"&client_secret={urllib.parse.quote(sk)}"
        )
        req = urllib.request.Request(token_url, method="POST")
        with urllib.request.urlopen(req, context=_SSL_CTX, timeout=10) as resp:
            body = json.loads(resp.read().decode())
            token = (body.get("access_token") or "").strip()
            expires_in = int(body.get("expires_in", 2592000) or 2592000)
            if token:
                self._qianfan_iam_token = token
                self._qianfan_iam_expiry = time.monotonic() + max(60, expires_in - 300)
                return token
        return None

    # ── Core blocking HTTP (runs inside asyncio.to_thread) ────────────────────

    def _sync_post(self, url: str, payload: bytes, headers: dict, timeout: float) -> LLMResponse:
        """Blocking HTTP POST — called via asyncio.to_thread."""
        t0 = time.monotonic()
        try:
            http_req = urllib.request.Request(
                url, data=payload, headers=headers, method="POST"
            )
            with urllib.request.urlopen(http_req, context=_SSL_CTX, timeout=timeout) as resp:
                latency = int((time.monotonic() - t0) * 1000)
                raw_body = resp.read()
                try:
                    body = json.loads(raw_body.decode("utf-8"))
                except Exception as e:
                    return LLMResponse(
                        error_type="parse_error",
                        error_message=f"JSON parse error: {e}",
                        status_code=resp.status,
                        latency_ms=latency,
                    )
                return self._parse_response(body, resp.status, dict(resp.headers), latency)
        except urllib.error.HTTPError as e:
            latency = int((time.monotonic() - t0) * 1000)
            try:
                body = json.loads(e.read().decode())
            except Exception:
                body = {}
            error_type = "rate_limit" if e.code == 429 else "http_error"
            return LLMResponse(
                status_code=e.code, latency_ms=latency,
                headers=dict(e.headers),
                error_type=error_type,
                error_message=(
                    body.get("error", {}).get("message", e.reason)
                    if isinstance(body.get("error"), dict) else str(body)[:300]
                ),
            )
        except TimeoutError:
            return LLMResponse(
                error_type="timeout",
                error_message=f"Timed out after {timeout}s",
                latency_ms=int((time.monotonic() - t0) * 1000),
            )
        except Exception as e:
            return LLMResponse(
                error_type="network_error",
                error_message=str(e)[:300],
                latency_ms=int((time.monotonic() - t0) * 1000),
            )

    # ── Public async API ──────────────────────────────────────────────────────

    async def achat(self, req: LLMRequest) -> LLMResponse:
        """Async chat completion with retry."""
        url = f"{self.base_url}/chat/completions"
        payload = json.dumps(req.to_payload()).encode("utf-8")
        headers = await self._auth_headers()

        async def _do() -> LLMResponse:
            return await asyncio.to_thread(
                self._sync_post, url, payload, headers, req.timeout_sec
            )

        return await _async_with_retry(_do)

    async def alist_models(self) -> dict:
        """Async GET /models."""
        url = f"{self.base_url}/models"
        headers = await self._auth_headers()

        def _sync():
            t0 = time.monotonic()
            try:
                r = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(r, context=_SSL_CTX,
                                            timeout=settings.DEFAULT_REQUEST_TIMEOUT_SEC) as resp:
                    raw = resp.read()
                    try:
                        body = json.loads(raw.decode("utf-8"))
                    except Exception as e:
                        body = {"raw": raw.decode("utf-8", errors="replace")}
                    return {"status_code": resp.status, "body": body,
                            "latency_ms": int((time.monotonic() - t0) * 1000)}
            except urllib.error.HTTPError as e:
                return {"status_code": e.code, "error": e.reason,
                        "latency_ms": int((time.monotonic() - t0) * 1000)}
            except Exception as e:
                return {"status_code": None, "error": str(e),
                        "latency_ms": int((time.monotonic() - t0) * 1000)}

        return await asyncio.to_thread(_sync)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_response(body: dict, status: int, headers: dict, latency: int) -> LLMResponse:
        choices = body.get("choices", [{}])
        choice = choices[0] if choices else {}
        usage = body.get("usage", {})
        return LLMResponse(
            content=choice.get("message", {}).get("content"),
            raw_json=body,
            status_code=status,
            headers=headers,
            latency_ms=latency,
            finish_reason=choice.get("finish_reason"),
            usage_prompt_tokens=usage.get("prompt_tokens"),
            usage_completion_tokens=usage.get("completion_tokens"),
            usage_total_tokens=usage.get("total_tokens"),
        )

    def to_sync_adapter(self):
        """Return a sync adapter backed by the same credentials (for compatibility)."""
        from app.adapters.openai_compat import OpenAICompatibleAdapter
        return OpenAICompatibleAdapter(self.base_url, self._api_key)
