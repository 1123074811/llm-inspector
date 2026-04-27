"""
predetect/protocol_validator.py — v17 Phase 1: protocol-level hard evidence.

Layer 0.5 (between L0/HTTP-headers and L1/SelfReport).

The original L0 layer in ``layers_l0_l2.py`` only does a fuzzy header
``startswith()`` match.  This validator goes one level deeper and checks
*structural* contracts that wrappers find hard to fake:

  1. ID prefix on the response object (``chatcmpl-`` for OpenAI, ``msg_`` for
     Anthropic).
  2. Error envelope schema (OpenAI requires ``error.{code,param,message,type}``;
     Anthropic top-level ``type:"error"`` + ``error.{type,message}``).
  3. Cross-family auth pollution (Bearer accepted on Anthropic endpoints, or
     ``x-api-key`` accepted on OpenAI endpoints) — see
     ``adapters/contamination_probe.py``.

Output is a ``LayerResult`` whose ``evidence`` list also carries a
JSON-serialized ``ProtocolEvidence`` dict (key ``protocol_evidence``) that
``analysis/verdicts.py`` consumes for hard-rule decisions.

This layer reuses the cheap ``adapter.bad_request()`` payload that L0 already
calls (no extra tokens for the schema check).  Only the cross-family
contamination probe sends a *new* request, and only when ``deep_probe=True``
(by default that is gated to Standard+ modes via the pipeline).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.core.schemas import LayerResult
from app.core.logging import get_logger
from app.adapters.contamination_probe import (
    ProbeOutcome,
    run_contamination_probes,
)

logger = get_logger(__name__)


# ── Per-family protocol contracts (sources in _data/SOURCES.yaml) ────────────

# Required keys in the *error* sub-object for each upstream family.
# Source URLs:
#   OpenAI    — https://platform.openai.com/docs/api-reference/errors
#   Anthropic — https://docs.anthropic.com/en/api/errors
#   Google    — https://ai.google.dev/api/rest
ERROR_SCHEMA_REQUIRED = {
    "openai":    {"type", "message"},          # "code","param" optional but common
    "anthropic": {"type", "message"},          # within the inner error object
    "google":    {"code", "message", "status"},
}

# Regex for the ``id`` field on a *successful* completion.  We can only
# evaluate this when an upstream actually returns a chat completion (Layer 1
# self-report does that), so this validator records the *expected* pattern
# and lets Layer 1's prefetched response feed back into the rule.
ID_PREFIX_PATTERN = {
    "openai":    re.compile(r"^chatcmpl-[A-Za-z0-9]{8,}$"),
    "anthropic": re.compile(r"^msg_[A-Za-z0-9]{12,}$"),
}


@dataclass
class ProtocolEvidence:
    """Structured protocol-level evidence (consumed by verdict engine)."""

    claimed_family: str = ""                                # "openai" | "anthropic" | ...
    sse_frames_match: bool | None = None                    # None = stream not exercised yet
    response_id_prefix_match: bool | None = None            # None = no completion seen yet
    error_schema_match: bool | None = None                  # None = no error body seen
    cross_family_auth_pollution: bool = False               # True = wrapper signal
    contradictions: list[str] = field(default_factory=list)
    contamination_probes: list[dict] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)

    @property
    def score_delta(self) -> float:
        """Net logit added to BayesianFusion: positive = real, negative = fake."""
        s = 0.0
        # Hard violations push toward fake
        if self.cross_family_auth_pollution:
            s -= 0.5
        if self.error_schema_match is False:
            s -= 0.4
        if self.response_id_prefix_match is False:
            s -= 0.3
        # Affirmative matches give modest authenticity boost
        if self.error_schema_match is True:
            s += 0.15
        if self.response_id_prefix_match is True:
            s += 0.15
        return s

    def to_dict(self) -> dict[str, Any]:
        return {
            "claimed_family": self.claimed_family,
            "sse_frames_match": self.sse_frames_match,
            "response_id_prefix_match": self.response_id_prefix_match,
            "error_schema_match": self.error_schema_match,
            "cross_family_auth_pollution": self.cross_family_auth_pollution,
            "contradictions": self.contradictions,
            "contamination_probes": self.contamination_probes,
            "score_delta": round(self.score_delta, 4),
            "sources": self.sources,
        }


def _check_error_schema(family: str, body: Any) -> tuple[bool | None, str]:
    """Return (matches_or_none, contradiction_reason).

    ``None`` means the body did not contain an error object so the schema
    contract could not be evaluated (no evidence either way).
    """
    if not isinstance(body, dict):
        return None, ""
    if family == "anthropic":
        if body.get("type") != "error":
            # If response was a 4xx/5xx for an Anthropic endpoint, the wrapper
            # *must* return ``type:"error"`` at the top level.  Absence is a
            # contradiction.
            if "error" in body or "message" in body:
                return False, "Anthropic error envelope missing top-level type='error'"
            return None, ""
        err = body.get("error")
        if not isinstance(err, dict):
            return False, "Anthropic error has no nested error object"
        missing = ERROR_SCHEMA_REQUIRED["anthropic"] - err.keys()
        if missing:
            return False, f"Anthropic error missing required keys: {sorted(missing)}"
        return True, ""

    # OpenAI / Google share the ``error`` top-level key
    err = body.get("error")
    if not isinstance(err, dict):
        return None, ""
    required = ERROR_SCHEMA_REQUIRED.get(family, set())
    missing = required - err.keys()
    if missing:
        return False, f"{family} error missing required keys: {sorted(missing)}"
    return True, ""


def _check_id_prefix(family: str, completion_body: Any) -> tuple[bool | None, str]:
    """Return (matches_or_none, contradiction_reason)."""
    if not isinstance(completion_body, dict):
        return None, ""
    cid = completion_body.get("id")
    if not isinstance(cid, str) or not cid:
        return None, ""
    pat = ID_PREFIX_PATTERN.get(family)
    if pat is None:
        return None, ""
    if pat.match(cid):
        return True, ""
    return False, f"{family} completion id {cid!r} does not match {pat.pattern}"


def _normalize_family(raw: str | None) -> str:
    """Map a free-form family string to canonical form: openai|anthropic|google."""
    if not raw:
        return ""
    s = raw.lower()
    if "anthropic" in s or "claude" in s:
        return "anthropic"
    if "google" in s or "gemini" in s or "vertex" in s:
        return "google"
    if "openai" in s or "gpt" in s or "azure openai" in s:
        return "openai"
    return ""


# ── Layer entry point ────────────────────────────────────────────────────────


class Layer0_5ProtocolValidator:
    """Layer 0.5 — structural protocol contract verification.

    Cost: 0 tokens normally, +1 throwaway request when ``deep_probe=True``.
    """

    def run(
        self,
        adapter,
        claimed_family_hint: str | None = None,
        prefetched_bad_request: dict | None = None,
        prefetched_completion: dict | None = None,
        deep_probe: bool = False,
    ) -> LayerResult:
        evidence: list[str] = []
        family = _normalize_family(claimed_family_hint)

        # If the caller did not hand us a bad_request payload, fetch one cheaply.
        bad = prefetched_bad_request
        if bad is None:
            try:
                bad = adapter.bad_request()
            except Exception as e:
                logger.warning("Layer0_5: bad_request failed", error=str(e))
                bad = {}
        body = bad.get("body") if isinstance(bad, dict) else None

        proto = ProtocolEvidence(
            claimed_family=family,
            sources=[
                "https://platform.openai.com/docs/api-reference/errors",
                "https://docs.anthropic.com/en/api/errors",
                "https://ai.google.dev/api/rest",
            ],
        )

        # 1. Error envelope schema check (uses already-fetched bad_request body)
        if family:
            matches, reason = _check_error_schema(family, body)
            proto.error_schema_match = matches
            if matches is True:
                evidence.append(f"Error envelope conforms to {family} schema")
            elif matches is False:
                evidence.append(f"PROTOCOL VIOLATION: {reason}")
                proto.contradictions.append(reason)

        # 2. Response id prefix check (if a completion is available)
        if family and prefetched_completion is not None:
            matches, reason = _check_id_prefix(family, prefetched_completion)
            proto.response_id_prefix_match = matches
            if matches is True:
                evidence.append(f"Completion id matches {family} prefix pattern")
            elif matches is False:
                evidence.append(f"PROTOCOL VIOLATION: {reason}")
                proto.contradictions.append(reason)

        # 3. Cross-family auth pollution probe (only when deep_probe=True)
        if deep_probe and family:
            try:
                base_url = getattr(adapter, "base_url", "")
                outcomes: list[ProbeOutcome] = run_contamination_probes(base_url, family)
                proto.contamination_probes = [o.to_dict() for o in outcomes]
                if any(o.contradicts_claim for o in outcomes):
                    proto.cross_family_auth_pollution = True
                    for o in outcomes:
                        if o.contradicts_claim:
                            msg = f"AUTH POLLUTION: {o.contradiction_reason}"
                            evidence.append(msg)
                            proto.contradictions.append(o.contradiction_reason)
                else:
                    if outcomes:
                        evidence.append(
                            f"Cross-family auth contamination probe clean ({len(outcomes)} probe(s))"
                        )
            except Exception as e:
                logger.warning("contamination probe wrapper failed", error=str(e))

        # Convert ProtocolEvidence -> confidence delta -> LayerResult.confidence
        # We map score_delta in [-1.2, +0.3] to confidence in [0, 0.9].
        delta = proto.score_delta
        # Hard violation present ⇒ high confidence we're seeing a wrapper.
        if delta <= -0.4:
            confidence = min(0.9, 0.5 - delta)        # -0.5 → 1.0 → cap 0.9
            identified = "wrapper/proxy (protocol contradictions)"
        elif delta >= 0.2:
            confidence = min(0.7, 0.4 + delta)
            identified = family or None
        else:
            confidence = 0.0
            identified = None

        # Attach structured evidence dict (consumed by verdict engine)
        evidence.append({"protocol_evidence": proto.to_dict()})  # type: ignore[arg-type]

        return LayerResult(
            layer="protocol",
            confidence=confidence,
            identified_as=identified,
            evidence=evidence,
            tokens_used=0,
        )


__all__ = [
    "Layer0_5ProtocolValidator",
    "ProtocolEvidence",
    "ERROR_SCHEMA_REQUIRED",
    "ID_PREFIX_PATTERN",
]
