"""
predetect/model_discovery.py — v16 Phase 4: /v1/models Probe

Probes the OpenAI-compatible /v1/models endpoint to discover
available models and detect shell/relay services.

References:
  - OpenAI API: https://platform.openai.com/docs/api-reference/models/list
  - Azure OpenAI: https://learn.microsoft.com/azure/ai-services/openai/reference#list-models
  - Berger (1985) "Statistical Decision Theory and Bayesian Analysis" — Jeffreys prior
"""
from __future__ import annotations

import json
import urllib.request
import urllib.error
import ssl
from dataclasses import dataclass, field
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModelListReport:
    """Result of /v1/models probe."""
    available: list[str] = field(default_factory=list)
    claimed_present: bool = False
    suspicious_neighbors: list[str] = field(default_factory=list)
    cross_family_models: list[str] = field(default_factory=list)
    raw_payload: dict | None = None
    source_url: str = "https://platform.openai.com/docs/api-reference/models/list"
    http_status: int | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "claimed_present": self.claimed_present,
            "suspicious_neighbors": self.suspicious_neighbors,
            "cross_family_models": self.cross_family_models,
            "http_status": self.http_status,
            "error": self.error,
        }


# Model family prefixes for cross-family detection
_MODEL_FAMILIES: dict[str, list[str]] = {
    "openai": ["gpt-", "o1-", "o3-", "dall-e", "whisper", "tts-", "text-embedding"],
    "anthropic": ["claude-"],
    "google": ["gemini-", "gemma-"],
    "deepseek": ["deepseek-"],
    "qwen": ["qwen"],
    "zhipu": ["glm-", "chatglm"],
    "meta": ["llama-"],
    "mistral": ["mistral-", "codestral-"],
    "minimax": ["abab", "minimax-"],
    "moonshot": ["moonshot-"],
    "yi": ["yi-"],
    "baichuan": ["baichuan-"],
}


def _infer_family(model_name: str) -> str | None:
    """Infer model family from model name prefix."""
    name_lower = model_name.lower()
    for family, prefixes in _MODEL_FAMILIES.items():
        for prefix in prefixes:
            if name_lower.startswith(prefix):
                return family
    return None


def _find_cross_family(claimed_model: str, available: list[str]) -> list[str]:
    """Find models from a different family than the claimed model."""
    claimed_family = _infer_family(claimed_model)
    if not claimed_family:
        return []
    cross = []
    for m in available:
        m_family = _infer_family(m)
        if m_family and m_family != claimed_family:
            cross.append(m)
    return cross


def _find_suspicious_neighbors(claimed_model: str, available: list[str]) -> list[str]:
    """Find models with similar but different names (typical relay gateway)."""
    claimed_lower = claimed_model.lower()
    suspicious = []
    for m in available:
        m_lower = m.lower()
        if claimed_lower != m_lower:
            # Check if they share a common family prefix
            claimed_family = _infer_family(claimed_model)
            m_family = _infer_family(m)
            if claimed_family and m_family == claimed_family:
                suspicious.append(m)
    return suspicious


def list_models(
    base_url: str,
    api_key: str,
    claimed_model: str = "",
    timeout: float = 8.0,
    verify_ssl: bool = True,
) -> ModelListReport:
    """
    Probe the /v1/models endpoint.

    Args:
        base_url: API base URL (e.g., https://api.openai.com)
        api_key: Bearer token for authentication.
        claimed_model: The model the user claims to be testing.
        timeout: Request timeout in seconds.
        verify_ssl: Whether to verify TLS certificates.

    Returns:
        ModelListReport with available models and analysis.
    """
    # Normalize URL: ensure /v1/models path
    url = base_url.rstrip("/")
    if not url.endswith("/v1/models"):
        if url.endswith("/v1"):
            url += "/models"
        else:
            url += "/v1/models"

    req = urllib.request.Request(
        url,
        method="GET",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
    )

    ssl_context = None
    if verify_ssl:
        ssl_context = ssl.create_default_context()
    else:
        ssl_context = ssl._create_unverified_context()

    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ssl_context) as resp:
            body_bytes = resp.read()
            http_status = resp.status
            try:
                body = json.loads(body_bytes)
            except json.JSONDecodeError:
                return ModelListReport(
                    http_status=http_status,
                    error="Non-JSON response from /v1/models",
                )

        # Extract model IDs
        model_ids = []
        data = body.get("data", [])
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "id" in item:
                    model_ids.append(item["id"])

        model_ids.sort()

        # Check if claimed model is present
        claimed_present = False
        if claimed_model:
            claimed_present = any(
                m.lower() == claimed_model.lower() for m in model_ids
            )

        # Find cross-family and suspicious neighbors
        cross_family = _find_cross_family(claimed_model, model_ids) if claimed_model else []
        suspicious = _find_suspicious_neighbors(claimed_model, model_ids) if claimed_model else []

        report = ModelListReport(
            available=model_ids,
            claimed_present=claimed_present,
            suspicious_neighbors=suspicious,
            cross_family_models=cross_family,
            raw_payload=body,
            http_status=http_status,
        )

        logger.info(
            "Model discovery complete",
            available_count=len(model_ids),
            claimed_present=claimed_present,
            cross_family_count=len(cross_family),
            suspicious_count=len(suspicious),
        )

        return report

    except urllib.error.HTTPError as e:
        raw_body = ""
        try:
            raw_body = e.read()[:500].decode("utf-8", errors="replace")
        except Exception:
            pass
        return ModelListReport(
            http_status=e.code,
            error=f"HTTP {e.code}: {raw_body[:200]}",
        )
    except urllib.error.URLError as e:
        return ModelListReport(error=f"URL error: {e.reason}")
    except Exception as e:
        return ModelListReport(error=f"Unexpected: {e}")


def compute_shell_posterior(
    report: ModelListReport,
    claimed_model: str,
    prior: float = 0.5,
) -> float:
    """
    Compute posterior probability of shell/relay service using Bayesian update.

    Uses Jeffreys prior (Berger 1985) by default (0.5 = uniform on log-odds).

    Rules (registered in SOURCES.yaml identity.model_list_rules):
      - claimed_present == False → +0.4 posterior
      - cross_family_models present → +0.5 posterior
      - only 1 model with near-miss name → +0.2 posterior

    Args:
        report: ModelListReport from list_models().
        claimed_model: The model the user claims to be testing.
        prior: Prior probability of shell service (default 0.5 = Jeffreys).

    Returns:
        Posterior probability [0, 1] that this is a shell/relay service.
    """
    posterior = prior

    # Rule 1: Claimed model not present in /v1/models
    if claimed_model and not report.claimed_present:
        posterior = min(1.0, posterior + 0.4)

    # Rule 2: Cross-family models detected
    if report.cross_family_models:
        posterior = min(1.0, posterior + 0.5)

    # Rule 3: Only 1 model with near-miss name (typical relay gateway)
    if len(report.available) == 1 and report.suspicious_neighbors:
        posterior = min(1.0, posterior + 0.2)

    return round(posterior, 4)
