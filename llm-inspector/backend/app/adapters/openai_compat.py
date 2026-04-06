"""
OpenAI-compatible LLM adapter.
Uses stdlib urllib — no httpx/requests dependency.
"""
from __future__ import annotations

import json
import ssl
import time
import urllib.error
import urllib.request
import urllib.parse
import hashlib
from app.core.schemas import (
    LLMRequest, LLMResponse, StreamChunk, StreamCaptureResult
)
from app.core.db import get_conn, now_iso, json_col, from_json_col

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_RETRYABLE_CODES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_RETRY_BASE_DELAY_SEC = 1.0

def _with_retry(fn, max_retries: int = _MAX_RETRIES):
    last_result = None
    for attempt in range(max_retries + 1):
        result = fn()
        if result.ok:
            return result
        if result.status_code not in _RETRYABLE_CODES:
            return result
        if attempt < max_retries:
            delay = _RETRY_BASE_DELAY_SEC * (2 ** attempt)
            time.sleep(delay)
        last_result = result
    return last_result

# Build a permissive SSL context (some self-hosted APIs use self-signed certs)
_SSL_CTX = ssl.create_default_context()


class OpenAICompatibleAdapter:
    """Adapter for OpenAI-compatible /chat/completions endpoints.

    Also supports Baidu Qianfan AK/SK bootstrap when base_url points to
    qianfan.baidubce.com and api_key is provided as "<AK>:<SK>".
    """

    @staticmethod
    def _normalize_api_key(api_key: str) -> str:
        """Normalize user-supplied key to avoid common auth formatting mistakes."""
        key = (api_key or "").strip()
        # Users often paste "Bearer <token>" from docs/cURL.
        # Adapter always injects Bearer, so strip a leading prefix to avoid
        # sending "Authorization: Bearer Bearer ...".
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

    @staticmethod
    def _split_ak_sk(api_key: str) -> tuple[str, str] | None:
        """Accept AK/SK in the form "AK:SK" for Qianfan token bootstrap."""
        if ":" not in api_key:
            return None
        ak, sk = api_key.split(":", 1)
        ak = ak.strip()
        sk = sk.strip()
        if not ak or not sk:
            return None
        return ak, sk

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self._api_key = self._normalize_api_key(api_key)
        self._headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        # Provider-specific bootstrap: Baidu Qianfan may require IAM token from AK/SK
        self._qianfan_ak: str | None = None
        self._qianfan_sk: str | None = None
        self._qianfan_iam_token: str | None = None
        self._qianfan_iam_token_expiry_monotonic: float = 0.0

        if self._looks_like_qianfan_host(self.base_url):
            pair = self._split_ak_sk(self._api_key)
            if pair:
                self._qianfan_ak, self._qianfan_sk = pair

    # ── public API ────────────────────────────────────────────────────────────

    def list_models(self) -> dict:
        """GET /models — returns raw JSON or error dict."""
        url = f"{self.base_url}/models"
        t0 = time.monotonic()
        try:
            req = urllib.request.Request(url, headers=self._auth_headers())
            with urllib.request.urlopen(req, context=_SSL_CTX,
                                        timeout=settings.DEFAULT_REQUEST_TIMEOUT_SEC) as resp:
                raw_body = resp.read()
                try:
                    body = json.loads(raw_body.decode("utf-8"))
                except Exception as e:
                    return {
                        "status_code": resp.status,
                        "error": f"JSON parse error: {str(e)}",
                        "body": raw_body.decode("utf-8", errors="replace"),
                        "headers": dict(resp.headers),
                        "latency_ms": int((time.monotonic() - t0) * 1000),
                    }
                return {
                    "status_code": resp.status,
                    "body": body,
                    "headers": dict(resp.headers),
                    "latency_ms": int((time.monotonic() - t0) * 1000),
                }
        except urllib.error.HTTPError as e:
            return {"status_code": e.code, "error": e.reason,
                    "headers": dict(e.headers),
                    "latency_ms": int((time.monotonic() - t0) * 1000)}
        except Exception as e:
            return {"status_code": None, "error": str(e),
                    "latency_ms": int((time.monotonic() - t0) * 1000)}

    def head_request(self) -> dict:
        """HEAD / — used by Layer 0 to capture response headers cheaply."""
        url = f"{self.base_url}/"
        t0 = time.monotonic()
        try:
            req = urllib.request.Request(url, method="HEAD", headers=self._auth_headers())
            with urllib.request.urlopen(req, context=_SSL_CTX, timeout=10) as resp:
                return {"status_code": resp.status, "headers": dict(resp.headers),
                        "latency_ms": int((time.monotonic() - t0) * 1000)}
        except urllib.error.HTTPError as e:
            return {"status_code": e.code, "headers": dict(e.headers),
                    "latency_ms": int((time.monotonic() - t0) * 1000)}
        except Exception as e:
            return {"error": str(e), "headers": {},
                    "latency_ms": int((time.monotonic() - t0) * 1000)}

    def bad_request(self) -> dict:
        """POST /chat/completions with missing fields — observe error shape."""
        url = f"{self.base_url}/chat/completions"
        t0 = time.monotonic()
        payload = json.dumps({"model": "test"}).encode()  # missing messages
        try:
            req = urllib.request.Request(url, data=payload,
                                         headers=self._auth_headers(), method="POST")
            with urllib.request.urlopen(req, context=_SSL_CTX, timeout=10) as resp:
                raw_body = resp.read()
                try:
                    body = json.loads(raw_body.decode("utf-8"))
                except Exception as e:
                    body = {"raw": raw_body.decode("utf-8", errors="replace")}
                return {"status_code": resp.status, "body": body,
                        "latency_ms": int((time.monotonic() - t0) * 1000)}
        except urllib.error.HTTPError as e:
            try:
                body = json.loads(e.read().decode())
            except Exception:
                body = {}
            return {"status_code": e.code, "body": body, "headers": dict(e.headers),
                    "latency_ms": int((time.monotonic() - t0) * 1000)}
        except Exception as e:
            return {"error": str(e),
                    "latency_ms": int((time.monotonic() - t0) * 1000)}

    def _get_cache_key(self, req: LLMRequest) -> str:
        payload = req.to_payload()
        # Ensure consistent key by sorting keys in JSON
        payload_str = json.dumps(payload, sort_keys=True)
        key_src = f"{self.base_url}:{payload_str}"
        return hashlib.sha256(key_src.encode()).hexdigest()

    def _get_cache(self, key: str) -> LLMResponse | None:
        try:
            conn = get_conn()
            row = conn.execute(
                "SELECT response_json FROM llm_response_cache WHERE cache_key=? AND expires_at > ?",
                (key, now_iso())
            ).fetchone()
            if row:
                data = from_json_col(row["response_json"])
                return LLMResponse(**data)
        except Exception as e:
            logger.warning("Cache fetch error", error=str(e))
        return None

    def _save_cache(self, key: str, resp: LLMResponse, ttl_hours: int = 24) -> None:
        if not resp.ok or resp.error_type:
            return
        try:
            conn = get_conn()
            # Calculate expiry (naive string comparison OK for ISO)
            from datetime import datetime, timedelta, timezone
            expires_at = (datetime.now(timezone.utc) + timedelta(hours=ttl_hours)).isoformat()
            conn.execute(
                "INSERT OR REPLACE INTO llm_response_cache (cache_key, response_json, created_at, expires_at) VALUES (?,?,?,?)",
                (key, json_col(resp.to_dict()), now_iso(), expires_at)
            )
            conn.commit()
        except Exception as e:
            logger.warning("Cache save error", error=str(e))

    def _qianfan_fetch_iam_token(self) -> str | None:
        """Fetch Baidu IAM access_token using AK/SK pair, cache for reuse."""
        if not self._qianfan_ak or not self._qianfan_sk:
            return None

        now_mono = time.monotonic()
        if self._qianfan_iam_token and now_mono < self._qianfan_iam_token_expiry_monotonic:
            return self._qianfan_iam_token

        token_url = (
            "https://aip.baidubce.com/oauth/2.0/token"
            f"?grant_type=client_credentials&client_id={urllib.parse.quote(self._qianfan_ak)}"
            f"&client_secret={urllib.parse.quote(self._qianfan_sk)}"
        )
        try:
            req = urllib.request.Request(token_url, method="POST")
            with urllib.request.urlopen(
                req, context=_SSL_CTX, timeout=settings.DEFAULT_REQUEST_TIMEOUT_SEC
            ) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                body = json.loads(raw)
                token = (body.get("access_token") or "").strip()
                expires_in = int(body.get("expires_in", 2592000) or 2592000)
                if not token:
                    logger.warning("Qianfan IAM token fetch returned empty token", body=raw[:300])
                    return None
                # Refresh 5 min early to avoid near-expiry races
                self._qianfan_iam_token = token
                self._qianfan_iam_token_expiry_monotonic = now_mono + max(60, expires_in - 300)
                logger.info("Qianfan IAM token bootstrap success")
                return token
        except Exception as e:
            logger.warning("Qianfan IAM token bootstrap failed", error=str(e))
            return None

    def _auth_headers(self) -> dict:
        """Resolve provider-specific auth header while preserving OpenAI-compatible default."""
        # Default: OpenAI-compatible bearer with provided token
        headers = dict(self._headers)

        # Qianfan BCE v3 key mode: api_key starts with "bce-v3/..."
        # The default _headers already sets "Authorization: Bearer {api_key}"
        # which is exactly what Qianfan's OpenAI-compatible endpoint expects.
        # No special handling needed — just use the default Bearer header.
        if self._looks_like_qianfan_host(self.base_url) and self._api_key.lower().startswith("bce-v3/"):
            return headers

        # Qianfan AK/SK mode: api_key is "AK:SK" -> bootstrap IAM access_token.
        if self._qianfan_ak and self._qianfan_sk:
            token = self._qianfan_fetch_iam_token()
            if token:
                headers["Authorization"] = f"Bearer {token}"

        return headers

    def chat(self, req: LLMRequest) -> LLMResponse:
        """Synchronous non-streaming chat completion with internal caching."""
        cache_key = self._get_cache_key(req)
        cached = self._get_cache(cache_key)
        if cached:
            logger.info("Cache hit for LLM request", model=req.model)
            return cached

        def _do():
            url = f"{self.base_url}/chat/completions"
            payload = json.dumps(req.to_payload()).encode("utf-8")
            t0 = time.monotonic()
            try:
                http_req = urllib.request.Request(
                    url, data=payload, headers=self._auth_headers(), method="POST"
                )
                with urllib.request.urlopen(
                    http_req, context=_SSL_CTX, timeout=req.timeout_sec
                ) as resp:
                    latency = int((time.monotonic() - t0) * 1000)
                    raw_body = resp.read()
                    try:
                        body = json.loads(raw_body.decode("utf-8"))
                    except Exception as e:
                        return LLMResponse(
                            error_type="parse_error",
                            error_message=f"JSON parse error: {str(e)[:100]}. Body: {raw_body.decode('utf-8', errors='replace')[:200]}",
                            status_code=resp.status,
                            headers=dict(resp.headers),
                            latency_ms=latency,
                        )
                    parsed_resp = self._parse_response(body, resp.status,
                                                      dict(resp.headers), latency)
                    self._save_cache(cache_key, parsed_resp)
                    return parsed_resp
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
                    error_message=body.get("error", {}).get("message", e.reason)
                    if isinstance(body.get("error"), dict) else str(body)[:300],
                )
            except TimeoutError:
                return LLMResponse(
                    error_type="timeout",
                    error_message=f"Timed out after {req.timeout_sec}s",
                    latency_ms=int((time.monotonic() - t0) * 1000),
                )
            except Exception as e:
                return LLMResponse(
                    error_type="parse_error",
                    error_message=str(e)[:300],
                    latency_ms=int((time.monotonic() - t0) * 1000),
                )

        result = _with_retry(_do)
        if result is not None:
            return result

        t0 = time.monotonic()
        return LLMResponse(
            error_type="unknown",
            error_message="Retry loop returned None",
            latency_ms=int((time.monotonic() - t0) * 1000),
        )

    def chat_stream(self, req: LLMRequest) -> StreamCaptureResult:
        """Streaming chat — captures SSE chunks up to MAX_STREAM_CHUNKS."""
        stream_req = LLMRequest(
            model=req.model,
            messages=req.messages,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            stream=True,
            response_format=req.response_format,
            timeout_sec=req.timeout_sec,
            extra_params=req.extra_params,
        )
        url = f"{self.base_url}/chat/completions"
        payload = json.dumps(stream_req.to_payload()).encode("utf-8")
        chunks: list[StreamChunk] = []
        combined: list[str] = []
        t0 = time.monotonic()
        first_token_ms = None
        truncated = False

        try:
            http_req = urllib.request.Request(
                url, data=payload, headers=self._auth_headers(), method="POST"
            )
            with urllib.request.urlopen(
                http_req, context=_SSL_CTX, timeout=req.timeout_sec
            ) as resp:
                for raw_bytes in resp:
                    line = raw_bytes.decode("utf-8").rstrip("\r\n")
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    if len(chunks) >= settings.MAX_STREAM_CHUNKS:
                        truncated = True
                        break
                    arrived = int((time.monotonic() - t0) * 1000)
                    if first_token_ms is None and data.strip():
                        first_token_ms = arrived
                    delta_text = None
                    finish_reason = None
                    try:
                        obj = json.loads(data)
                        delta_text = (
                            obj.get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content", "")
                        ) or None
                        finish_reason = (
                            obj.get("choices", [{}])[0].get("finish_reason")
                        )
                    except Exception:
                        pass
                    if delta_text:
                        combined.append(delta_text)
                    chunks.append(StreamChunk(
                        index=len(chunks), arrived_at_ms=arrived,
                        raw_line=line, delta_text=delta_text,
                        finish_reason=finish_reason,
                    ))
        except Exception as e:
            return StreamCaptureResult(
                chunks=chunks,
                combined_text="".join(combined),
                latency_ms=int((time.monotonic() - t0) * 1000),
                first_token_ms=first_token_ms,
                error_type="stream_error",
                error_message=str(e)[:300],
                truncated=truncated,
            )

        return StreamCaptureResult(
            chunks=chunks,
            combined_text="".join(combined),
            latency_ms=int((time.monotonic() - t0) * 1000),
            first_token_ms=first_token_ms,
            truncated=truncated,
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_response(body: dict, status: int,
                        headers: dict, latency: int) -> LLMResponse:
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
