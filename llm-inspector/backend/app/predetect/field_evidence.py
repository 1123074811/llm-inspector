"""
predetect/field_evidence.py — v17 Phase 2: field-level hard evidence layer.

Layer 0.6 (after L0.5 protocol validation, before L1 self-report).

Many vendors include high-entropy, hard-to-fake fields in their chat
completion responses that wrappers usually drop, mangle, or fake poorly:

  * OpenAI         ``system_fingerprint`` matching ``^fp_[a-f0-9]{10,}$``
  * OpenAI o1/o3   ``usage.completion_tokens_details.reasoning_tokens > 0``
  * Anthropic      ``usage.cache_read_input_tokens`` round-trippable
  * Anthropic      ``thinking.signature`` cryptographic blob

This layer is **passive**: it extracts these signals from a response that
was *already* fetched (the orchestrator's quick probe).  It never sends
new requests, so token cost is exactly zero.  Cache-read and thinking
signature ideally need a 2nd call — those are deferred to v17.1; the
current layer reports presence/absence of the *single* known field per
family from the prefetched probe.

A field being **present** with a valid format pushes authenticity *up*.
A field being **absent or malformed** does *not* push authenticity down
on its own (many legitimate older models lack ``system_fingerprint``);
absence is recorded as evidence but does not contribute negative score.

The only **negative** signal in this layer is "fake-looking field
content" — e.g. a value claiming ``fp_…`` shape but failing the regex,
which is symptomatic of cargo-culted wrappers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field as dc_field
from typing import Any

from app.core.schemas import LayerResult
from app.core.logging import get_logger

logger = get_logger(__name__)


# ── Field detectors (per family) ────────────────────────────────────────────

# OpenAI ``system_fingerprint`` shape: lower-case hex, length ≥ 10.
# https://platform.openai.com/docs/api-reference/chat/object#chat/object-system_fingerprint
_OPENAI_FP_RE = re.compile(r"^fp_[a-f0-9]{10,}$")


@dataclass
class FieldEvidence:
    """Structured field-level evidence for the verdict engine."""

    claimed_family: str = ""

    # OpenAI-class: system_fingerprint
    has_system_fingerprint: bool | None = None        # None=field missing
    system_fingerprint_valid: bool | None = None      # None=N/A, True=regex ok
    system_fingerprint_value: str | None = None

    # OpenAI o1/o3 thinking models: reasoning_tokens
    reasoning_tokens_seen: bool | None = None
    reasoning_tokens_count: int = 0

    # Anthropic prompt cache: cache_read_input_tokens
    cache_read_seen: bool | None = None
    cache_read_count: int = 0

    # Anthropic thinking: signature blob
    thinking_signature_seen: bool | None = None

    contradictions: list[str] = dc_field(default_factory=list)
    sources: list[str] = dc_field(default_factory=list)

    @property
    def score_delta(self) -> float:
        """Bayesian-fusion score contribution.

        Positive evidence (present + well-formed) is bounded.  Negative
        evidence (malformed value claiming the right shape) is penalised
        more heavily because it indicates active fakery.
        """
        s = 0.0
        if self.system_fingerprint_valid is True:
            s += 0.35
        elif self.has_system_fingerprint is True and self.system_fingerprint_valid is False:
            s -= 0.40                                              # malformed = active fakery
        if self.reasoning_tokens_seen is True:
            s += 0.30
        if self.cache_read_seen is True:
            s += 0.25
        if self.thinking_signature_seen is True:
            s += 0.40
        return s

    def to_dict(self) -> dict[str, Any]:
        return {
            "claimed_family": self.claimed_family,
            "has_system_fingerprint": self.has_system_fingerprint,
            "system_fingerprint_valid": self.system_fingerprint_valid,
            "system_fingerprint_value": self.system_fingerprint_value,
            "reasoning_tokens_seen": self.reasoning_tokens_seen,
            "reasoning_tokens_count": self.reasoning_tokens_count,
            "cache_read_seen": self.cache_read_seen,
            "cache_read_count": self.cache_read_count,
            "thinking_signature_seen": self.thinking_signature_seen,
            "contradictions": self.contradictions,
            "score_delta": round(self.score_delta, 4),
            "sources": self.sources,
        }


def _normalize_family(raw: str | None) -> str:
    if not raw:
        return ""
    s = raw.lower()
    if "anthropic" in s or "claude" in s:
        return "anthropic"
    if "google" in s or "gemini" in s or "vertex" in s:
        return "google"
    if "openai" in s or "gpt" in s or "azure" in s:
        return "openai"
    return ""


def _extract_openai_fields(raw: dict, ev: FieldEvidence) -> None:
    """Populate OpenAI-family fields on the evidence object."""
    fp = raw.get("system_fingerprint")
    if fp is not None:
        ev.has_system_fingerprint = True
        if isinstance(fp, str):
            ev.system_fingerprint_value = fp[:64]
            ev.system_fingerprint_valid = bool(_OPENAI_FP_RE.match(fp))
            if not ev.system_fingerprint_valid:
                ev.contradictions.append(
                    f"system_fingerprint value {fp!r} fails ^fp_[a-f0-9]{{10,}}$"
                )
        else:
            ev.system_fingerprint_valid = False
            ev.contradictions.append(
                f"system_fingerprint is non-string ({type(fp).__name__})"
            )
    else:
        ev.has_system_fingerprint = False

    # reasoning_tokens — under usage.completion_tokens_details.reasoning_tokens
    usage = raw.get("usage") or {}
    details = usage.get("completion_tokens_details") if isinstance(usage, dict) else None
    if isinstance(details, dict) and "reasoning_tokens" in details:
        rt = details.get("reasoning_tokens")
        try:
            ev.reasoning_tokens_count = int(rt or 0)
            ev.reasoning_tokens_seen = ev.reasoning_tokens_count > 0
        except (TypeError, ValueError):
            ev.reasoning_tokens_seen = False
    else:
        ev.reasoning_tokens_seen = None


def _extract_anthropic_fields(raw: dict, ev: FieldEvidence) -> None:
    """Populate Anthropic-family fields on the evidence object."""
    usage = raw.get("usage") or {}
    if isinstance(usage, dict) and "cache_read_input_tokens" in usage:
        try:
            ev.cache_read_count = int(usage.get("cache_read_input_tokens") or 0)
            # We can only assert "seen" when the value is non-zero; a fresh
            # call with no cache hit returns 0 legitimately.
            ev.cache_read_seen = ev.cache_read_count > 0
        except (TypeError, ValueError):
            ev.cache_read_seen = False
    else:
        ev.cache_read_seen = None

    # thinking signature: response['content'] is a list of blocks; check for
    # a block of type 'thinking' carrying a 'signature' string.
    content = raw.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "thinking":
                sig = block.get("signature")
                if isinstance(sig, str) and len(sig) >= 8:
                    ev.thinking_signature_seen = True
                    return
        ev.thinking_signature_seen = False
    else:
        ev.thinking_signature_seen = None


# ── Layer entry point ────────────────────────────────────────────────────────


class Layer0_6FieldEvidence:
    """Layer 0.6 — passive field-level evidence extraction (zero tokens)."""

    def run(
        self,
        adapter,
        prefetched_response: Any | None = None,
        claimed_family_hint: str | None = None,
    ) -> LayerResult:
        """Extract structured field evidence from the orchestrator's quick probe.

        Parameters
        ----------
        adapter : unused (kept for layer signature symmetry)
        prefetched_response : LLMResponse-like object exposing ``raw_json``,
            or None if no quick probe was done.
        claimed_family_hint : free-form family string from L0 (header match).
        """
        family = _normalize_family(claimed_family_hint)
        ev = FieldEvidence(
            claimed_family=family,
            sources=[
                "https://platform.openai.com/docs/api-reference/chat/object",
                "https://docs.anthropic.com/en/api/messages",
            ],
        )
        evidence: list[Any] = []

        raw: dict | None = None
        if prefetched_response is not None:
            raw = getattr(prefetched_response, "raw_json", None)
            if raw is None and isinstance(prefetched_response, dict):
                raw = prefetched_response

        if not isinstance(raw, dict) or not raw:
            evidence.append("Field evidence layer skipped: no prefetched response body")
            evidence.append({"field_evidence": ev.to_dict()})
            return LayerResult(
                layer="field_evidence",
                confidence=0.0,
                identified_as=None,
                evidence=evidence,
                tokens_used=0,
            )

        if family == "openai":
            _extract_openai_fields(raw, ev)
        elif family == "anthropic":
            _extract_anthropic_fields(raw, ev)
        else:
            evidence.append(f"Field evidence layer skipped: unknown family {family!r}")
            evidence.append({"field_evidence": ev.to_dict()})
            return LayerResult(
                layer="field_evidence",
                confidence=0.0,
                identified_as=None,
                evidence=evidence,
                tokens_used=0,
            )

        # Build human-readable evidence strings
        if ev.system_fingerprint_valid is True:
            evidence.append(
                f"system_fingerprint present and well-formed ({ev.system_fingerprint_value})"
            )
        elif ev.system_fingerprint_valid is False:
            evidence.append(
                f"FIELD VIOLATION: system_fingerprint malformed ({ev.system_fingerprint_value!r})"
            )
        if ev.reasoning_tokens_seen is True:
            evidence.append(f"reasoning_tokens={ev.reasoning_tokens_count} (o1/o3-class)")
        if ev.cache_read_seen is True:
            evidence.append(f"cache_read_input_tokens={ev.cache_read_count}")
        if ev.thinking_signature_seen is True:
            evidence.append("thinking.signature present (Anthropic Sonnet 4+/Opus 4+)")

        # Map score_delta to confidence:
        delta = ev.score_delta
        if delta <= -0.3:
            confidence = min(0.9, 0.4 - delta)
            identified = "wrapper/proxy (malformed vendor fields)"
        elif delta >= 0.25:
            confidence = min(0.7, 0.3 + delta)
            identified = family or None
        else:
            confidence = 0.0
            identified = None

        evidence.append({"field_evidence": ev.to_dict()})

        return LayerResult(
            layer="field_evidence",
            confidence=confidence,
            identified_as=identified,
            evidence=evidence,
            tokens_used=0,
        )


__all__ = ["Layer0_6FieldEvidence", "FieldEvidence"]
