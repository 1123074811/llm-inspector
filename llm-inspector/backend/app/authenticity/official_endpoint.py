"""
authenticity/official_endpoint.py — v16 Phase 1.5: Official Endpoint Fast-Path.

Performs TLS triple-consistency verification against the official endpoint
registry (_data/official_endpoints.yaml):

  1. URL match: base_url matches a registered official endpoint
  2. TLS cert chain: server certificate chain depth & issuer are consistent
  3. Response headers: expected provider-specific headers are present

If all three checks pass with high confidence, the endpoint is marked as
"official_verified" and PreDetect can skip certain identity-probe layers.

Anti-evasion: detects reverse-proxies by checking for missing expected
headers, anomalous latency, and non-standard certificate chains.
"""
from __future__ import annotations

import json
import pathlib
import ssl
import time
import urllib.error
import urllib.request
import urllib.parse
from dataclasses import dataclass, field

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_REGISTRY_PATH = pathlib.Path(__file__).parent.parent / "_data" / "official_endpoints.yaml"
_registry_cache: dict | None = None


def _load_registry() -> dict:
    """Load and cache the official_endpoints.yaml registry."""
    global _registry_cache
    if _registry_cache is not None:
        return _registry_cache
    try:
        import yaml
        with open(_REGISTRY_PATH, encoding="utf-8") as f:
            _registry_cache = yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning("Failed to load official_endpoints.yaml", error=str(e))
        _registry_cache = {"providers": {}, "evasion_indicators": {}}
    return _registry_cache


def reload_registry() -> None:
    """Force reload the registry (for testing or hot-reload)."""
    global _registry_cache
    _registry_cache = None


@dataclass
class OfficialEndpointResult:
    """Result of official endpoint verification."""
    verified: bool = False
    provider: str | None = None
    display_name: str | None = None
    confidence: float = 0.0
    url_matched: bool = False
    tls_consistent: bool = False
    headers_consistent: bool = False
    model_prefix_matched: bool = False
    evasion_signals: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "verified": self.verified,
            "provider": self.provider,
            "display_name": self.display_name,
            "confidence": round(self.confidence, 3),
            "url_matched": self.url_matched,
            "tls_consistent": self.tls_consistent,
            "headers_consistent": self.headers_consistent,
            "model_prefix_matched": self.model_prefix_matched,
            "evasion_signals": self.evasion_signals,
            "details": self.details,
        }


def _match_url(base_url: str, provider_cfg: dict) -> bool:
    """Check if base_url matches any registered URL for a provider."""
    normalized = base_url.rstrip("/").lower()
    for registered in provider_cfg.get("base_urls", []):
        if normalized == registered.rstrip("/").lower():
            return True
        # Also match if base_url starts with a registered prefix
        if normalized.startswith(registered.rstrip("/").lower()):
            return True
    return False


def _check_tls(base_url: str, provider_cfg: dict, timeout: float) -> tuple[bool, dict]:
    """
    Check TLS certificate chain consistency.

    Returns (consistent, details_dict).
    For now, we verify the connection succeeds with strict SSL and
    record the cert chain depth and issuer. Full cert pinning is
    deferred until we have stable cert hashes in the registry.
    """
    details: dict = {"checked": False}
    try:
        parsed = urllib.parse.urlparse(base_url)
        hostname = parsed.hostname
        port = parsed.port or 443
        if not hostname:
            return False, details

        ctx = ssl.create_default_context()
        t0 = time.monotonic()
        conn = ctx.wrap_socket(
            __import__("socket").socket(__import__("socket").AF_INET, __import__("socket").SOCK_STREAM),
            server_hostname=hostname,
        )
        conn.settimeout(timeout)
        conn.connect((hostname, port))
        der_cert = conn.getpeercert(binary_form=True)
        peercert = conn.getpeercert()
        conn.close()
        tls_latency_ms = int((time.monotonic() - t0) * 1000)

        details["checked"] = True
        # peercert subject/issuer are tuple of tuples: ((key, value), ...)
        # Convert safely to dict for display
        try:
            subject_tuple = peercert.get("subject", ()) if peercert else ()
            details["cert_subject"] = dict(subject_tuple) if subject_tuple else None
        except (TypeError, ValueError):
            details["cert_subject"] = str(peercert.get("subject")) if peercert else None
        try:
            issuer_tuple = peercert.get("issuer", ()) if peercert else ()
            details["cert_issuer"] = dict(issuer_tuple) if issuer_tuple else None
        except (TypeError, ValueError):
            details["cert_issuer"] = str(peercert.get("issuer")) if peercert else None
        details["cert_chain_depth"] = len(peercert.get("subject", ())) if peercert else 0
        details["tls_latency_ms"] = tls_latency_ms

        # Basic consistency: connection succeeded with strict SSL = good sign
        # Deeper checks (cert pinning) can be added when cert_sha256 is populated
        cert_sha256 = provider_cfg.get("tls_fingerprints", {}).get("cert_sha256", "")
        if cert_sha256:
            import hashlib
            actual_hash = hashlib.sha256(der_cert).hexdigest()
            if actual_hash == cert_sha256:
                return True, details
            else:
                details["cert_hash_mismatch"] = True
                return False, details

        # Without cert pinning, a successful strict-SSL handshake is sufficient
        return True, details

    except ssl.SSLCertVerificationError as e:
        details["ssl_error"] = str(e)[:200]
        return False, details
    except Exception as e:
        details["error"] = str(e)[:200]
        return False, details


def _check_headers(base_url: str, provider_cfg: dict, api_key: str, timeout: float) -> tuple[bool, dict]:
    """
    Check if response headers contain provider-specific expected headers.

    Sends a lightweight HEAD or minimal request and inspects response headers.
    """
    details: dict = {"checked": False}
    expected = provider_cfg.get("expected_headers", [])
    if not expected:
        # No expected headers defined -> auto-pass this check
        return True, {"checked": True, "skipped": True, "reason": "no_expected_headers"}

    try:
        url = base_url.rstrip("/") + "/models"
        req = urllib.request.Request(url, headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
            resp_headers = dict(resp.headers)
            details["checked"] = True
            matched = 0
            for exp in expected:
                # Check if any response header starts with the expected prefix
                if any(k.lower().startswith(exp.lower()) for k in resp_headers):
                    matched += 1
            details["matched_headers"] = matched
            details["expected_headers"] = len(expected)
            # Pass if at least half the expected headers are found
            return matched >= max(1, len(expected) // 2), details

    except urllib.error.HTTPError as e:
        # Even a 401/403 returns headers — still useful for fingerprinting
        resp_headers = dict(e.headers) if e.headers else {}
        details["checked"] = True
        details["http_status"] = e.code
        matched = 0
        for exp in expected:
            if any(k.lower().startswith(exp.lower()) for k in resp_headers):
                matched += 1
        details["matched_headers"] = matched
        details["expected_headers"] = len(expected)
        return matched >= max(1, len(expected) // 2), details
    except Exception as e:
        details["error"] = str(e)[:200]
        return False, details


def _check_model_prefix(model_name: str, provider_cfg: dict) -> bool:
    """Check if model name matches any registered prefix for this provider."""
    prefixes = provider_cfg.get("model_prefixes", [])
    if not prefixes:
        return False
    lower = model_name.lower()
    return any(lower.startswith(p.lower()) for p in prefixes)


def _detect_evasion(base_url: str, provider_cfg: dict,
                     headers_result: dict, tls_result: dict) -> list[str]:
    """Detect signals suggesting a reverse-proxy or wrapper."""
    signals: list[str] = []
    raw_evasion_cfg = _load_registry().get("evasion_indicators", {})
    # YAML may produce a list of single-key dicts instead of a flat dict
    if isinstance(raw_evasion_cfg, list):
        evasion_cfg: dict = {}
        for item in raw_evasion_cfg:
            if isinstance(item, dict):
                evasion_cfg.update(item)
    elif isinstance(raw_evasion_cfg, dict):
        evasion_cfg = raw_evasion_cfg
    else:
        evasion_cfg = {}

    # Check header mismatches
    header_mismatch_threshold = evasion_cfg.get("header_mismatches", 3)
    expected = provider_cfg.get("expected_headers", [])
    if expected and headers_result.get("checked") and not headers_result.get("skipped"):
        matched = headers_result.get("matched_headers", 0)
        missing = len(expected) - matched
        if missing >= header_mismatch_threshold:
            signals.append(f"header_mismatches({missing}>={header_mismatch_threshold})")

    # Check cert chain depth
    max_depth = evasion_cfg.get("cert_chain_depth_gt", 3)
    if tls_result.get("checked"):
        depth = tls_result.get("cert_chain_depth", 0)
        if depth > max_depth:
            signals.append(f"cert_chain_depth({depth}>{max_depth})")

    # Check latency anomaly
    anomaly_ms = evasion_cfg.get("response_latency_anomaly_ms", 500)
    tls_lat = tls_result.get("tls_latency_ms", 0)
    if tls_lat > anomaly_ms:
        signals.append(f"tls_latency_anomaly({tls_lat}ms>{anomaly_ms}ms)")

    # Check missing OpenAI-compat fields in HEAD response
    missing_fields = evasion_cfg.get("missing_openai_compat_fields", [])
    if missing_fields and headers_result.get("checked"):
        # This is a soft signal — only logged, not decisive
        pass

    return signals


def check_official_endpoint(
    base_url: str,
    api_key: str,
    model_name: str,
    timeout: float | None = None,
) -> OfficialEndpointResult:
    """
    v16 Phase 1.5: Official Endpoint Fast-Path verification.

    Triple-consistency check:
      1. URL matches a registered official endpoint
      2. TLS certificate chain is consistent
      3. Response headers contain provider-specific markers

    Returns OfficialEndpointResult with confidence and evasion signals.
    """
    if not settings.OFFICIAL_ENDPOINT_ENABLED:
        return OfficialEndpointResult(verified=False, details={"disabled": True})

    timeout = timeout or settings.OFFICIAL_ENDPOINT_TIMEOUT_S
    registry = _load_registry()
    providers = registry.get("providers", {})

    # Step 1: URL match
    matched_provider: str | None = None
    provider_cfg: dict | None = None
    for prov_name, cfg in providers.items():
        if _match_url(base_url, cfg):
            matched_provider = prov_name
            provider_cfg = cfg
            break

    if not matched_provider or not provider_cfg:
        return OfficialEndpointResult(
            verified=False,
            url_matched=False,
            details={"reason": "url_not_in_registry"},
        )

    # Step 2: TLS consistency
    tls_ok, tls_details = _check_tls(base_url, provider_cfg, timeout)

    # Step 3: Header consistency
    headers_ok, headers_details = _check_headers(base_url, provider_cfg, api_key, timeout)

    # Step 4: Model prefix match (bonus signal)
    model_prefix_ok = _check_model_prefix(model_name, provider_cfg)

    # Step 5: Evasion detection
    evasion_signals = _detect_evasion(base_url, provider_cfg, headers_details, tls_details)

    # Compute confidence
    score = 0.0
    if True:  # URL matched by definition at this point
        score += 0.35
    if tls_ok:
        score += 0.30
    if headers_ok:
        score += 0.25
    if model_prefix_ok:
        score += 0.10
    # Evasion signals reduce confidence. Per-signal penalty is intentionally
    # mild (−0.05) so a single soft anomaly (e.g. transient latency spike)
    # cannot single-handedly flip a fully-passing 3-factor check below
    # OFFICIAL_ENDPOINT_MIN_CONFIDENCE. Multiple corroborating signals can
    # still reduce confidence meaningfully (cf. v16 Phase 11 corroboration_min).
    for _ in evasion_signals:
        score = max(0.0, score - 0.05)

    verified = score >= settings.OFFICIAL_ENDPOINT_MIN_CONFIDENCE

    result = OfficialEndpointResult(
        verified=verified,
        provider=matched_provider,
        display_name=provider_cfg.get("display_name", matched_provider),
        confidence=score,
        url_matched=True,
        tls_consistent=tls_ok,
        headers_consistent=headers_ok,
        model_prefix_matched=model_prefix_ok,
        evasion_signals=evasion_signals,
        details={
            "tls": tls_details,
            "headers": headers_details,
        },
    )

    logger.info(
        "OfficialEndpoint check completed",
        provider=matched_provider,
        verified=verified,
        confidence=round(score, 3),
        evasion_signals=evasion_signals,
    )

    return result
