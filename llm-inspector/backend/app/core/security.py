"""
Security utilities:
  - AES-256-GCM encryption for API keys
  - SSRF-safe URL validation
"""
import base64
import hashlib
import ipaddress
import os
import socket
import urllib.parse
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


# ── SSRF guard ────────────────────────────────────────────────────────────────

_BLOCKED_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]


def validate_and_sanitize_url(url: str) -> str:
    """
    Validate a user-supplied base URL:
    1. scheme must be http or https
    2. hostname must resolve
    3. all resolved IPs must be public (no SSRF)
    4. strips query-string and fragment
    Returns sanitised URL or raises ValueError.
    """
    parsed = urllib.parse.urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported scheme '{parsed.scheme}'. Only http/https allowed.")

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL is missing a hostname.")

    # Reject obvious private hostnames before DNS resolution
    _reject_private_hostname(hostname)

    # DNS resolution + IP check
    try:
        addr_infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise ValueError(f"DNS resolution failed for '{hostname}': {exc}") from exc

    for info in addr_infos:
        ip_str = info[4][0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        for net in _BLOCKED_NETWORKS:
            if ip in net:
                raise ValueError(
                    f"Hostname '{hostname}' resolves to a private/blocked address ({ip_str})."
                )

    # Rebuild clean URL: scheme + netloc + path only
    clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")
    return normalize_openai_compatible_base_url(clean)


_CHAT_COMPLETIONS_SUFFIX = "/chat/completions"


def normalize_openai_compatible_base_url(url: str) -> str:
    """
    OpenAI-compatible adapters POST to ``{base_url}/chat/completions``.
    Users sometimes paste the full endpoint; strip trailing ``/chat/completions``.
    """
    u = url.rstrip("/")
    while u.endswith(_CHAT_COMPLETIONS_SUFFIX):
        u = u[: -len(_CHAT_COMPLETIONS_SUFFIX)].rstrip("/")
    return u


def _reject_private_hostname(hostname: str) -> None:
    lower = hostname.lower()
    blocked_names = (
        "localhost", "ip6-localhost", "ip6-loopback",
        "broadcasthost", "local",
    )
    if lower in blocked_names:
        raise ValueError(f"Hostname '{hostname}' is not allowed.")
    # Block *.local mDNS
    if lower.endswith(".local") or lower.endswith(".internal"):
        raise ValueError(f"Hostname '{hostname}' is not allowed (mDNS/internal).")


# ── API Key encryption ─────────────────────────────────────────────────────────

class ApiKeyManager:
    """AES-256-GCM encrypt/decrypt for stored API keys."""

    def __init__(self, key_bytes: bytes):
        assert len(key_bytes) == 32, "AES key must be exactly 32 bytes"
        self._cipher = AESGCM(key_bytes)

    def encrypt(self, api_key: str) -> tuple[str, str]:
        """
        Returns (encrypted_b64, hash_prefix).
        encrypted_b64 = base64(nonce[12] + ciphertext)
        hash_prefix   = first 16 hex chars of SHA-256(api_key)  — for deduplication only
        """
        nonce = os.urandom(12)
        ct = self._cipher.encrypt(nonce, api_key.encode("utf-8"), None)
        encrypted_b64 = base64.b64encode(nonce + ct).decode("ascii")
        hash_prefix = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]
        return encrypted_b64, hash_prefix

    def decrypt(self, encrypted_b64: str) -> str:
        raw = base64.b64decode(encrypted_b64)
        nonce, ct = raw[:12], raw[12:]
        return self._cipher.decrypt(nonce, ct, None).decode("utf-8")


# Singleton — imported everywhere that needs encryption
def get_key_manager() -> ApiKeyManager:
    from app.core.config import settings
    return ApiKeyManager(settings.aes_key)
