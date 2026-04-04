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
from app.core.schemas import (
    LLMRequest, LLMResponse, StreamChunk, StreamCaptureResult
)
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Build a permissive SSL context (some self-hosted APIs use self-signed certs)
_SSL_CTX = ssl.create_default_context()


class OpenAICompatibleAdapter:
    """Adapter for any OpenAI-compatible /chat/completions endpoint."""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    # ── public API ────────────────────────────────────────────────────────────

    def list_models(self) -> dict:
        """GET /models — returns raw JSON or error dict."""
        url = f"{self.base_url}/models"
        t0 = time.monotonic()
        try:
            req = urllib.request.Request(url, headers=self._headers)
            with urllib.request.urlopen(req, context=_SSL_CTX,
                                        timeout=settings.DEFAULT_REQUEST_TIMEOUT_SEC) as resp:
                body = json.loads(resp.read().decode())
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
            req = urllib.request.Request(url, method="HEAD", headers=self._headers)
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
                                         headers=self._headers, method="POST")
            with urllib.request.urlopen(req, context=_SSL_CTX, timeout=10) as resp:
                body = json.loads(resp.read().decode())
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

    def chat(self, req: LLMRequest) -> LLMResponse:
        """Synchronous non-streaming chat completion."""
        url = f"{self.base_url}/chat/completions"
        payload = json.dumps(req.to_payload()).encode("utf-8")
        t0 = time.monotonic()
        try:
            http_req = urllib.request.Request(
                url, data=payload, headers=self._headers, method="POST"
            )
            with urllib.request.urlopen(
                http_req, context=_SSL_CTX, timeout=req.timeout_sec
            ) as resp:
                latency = int((time.monotonic() - t0) * 1000)
                body = json.loads(resp.read().decode("utf-8"))
                return self._parse_response(body, resp.status,
                                            dict(resp.headers), latency)
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
                url, data=payload, headers=self._headers, method="POST"
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
