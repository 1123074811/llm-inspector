"""
authenticity/evidence_ledger.py — Authenticity evidence registration and aggregation.

Evidence types:
  - identity_collision: Model response contains a different model's name
  - tokenizer_mismatch: Tokenizer fingerprint doesn't match claimed model
  - refusal_style_mismatch: Refusal pattern matches a different model family
  - timing_anomaly: TTFT/TPS doesn't match claimed model family
  - token_distribution_anomaly: Response length distribution doesn't match
  - knowledge_cutoff_contradiction: Stated cutoff conflicts with claimed model
  - tool_capability_mismatch: Tool behavior inconsistent with claimed model
  - self_report_contradiction: Direct identity question got contradictory answer
  - system_prompt_leak: Extracted system prompt reveals actual identity

Evidence strength guidelines (must be documented):
  - 1.0: Direct explicit statement of different identity (e.g., "I am Kiro")
  - 0.9: Strong pattern match to different family (refusal template exact match)
  - 0.7: Moderate statistical signal (timing 2σ from claimed, 1σ from alternate)
  - 0.5: Weak signal (single data point, noisy measure)
  - 0.3: Circumstantial (consistent with but not exclusive to one family)
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AuthenticityEvidence:
    """A single piece of authenticity evidence."""
    evidence_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: str = ""                          # See type list above
    source_layer: str = ""                  # L1, L6, L17, judge, preflight, ...
    direction: str = "contradicts_claim"    # supports_claim | contradicts_claim | neutral
    strength: float = 0.5                   # 0-1 (see guidelines above)
    strength_rationale: str = ""
    raw_snippet: Optional[str] = None       # Actual text from model response
    claim_target: str = ""                  # Claimed model name
    suspected_target: Optional[str] = None  # Other family/model detected
    timestamp: str = ""
    reproducible: bool = False              # Can a human verify this?
    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "evidence_id": self.evidence_id,
            "type": self.type,
            "source_layer": self.source_layer,
            "direction": self.direction,
            "strength": round(self.strength, 3),
            "strength_rationale": self.strength_rationale,
            "raw_snippet": self.raw_snippet,
            "claim_target": self.claim_target,
            "suspected_target": self.suspected_target,
            "timestamp": self.timestamp,
            "reproducible": self.reproducible,
            "provenance": self.provenance,
        }


@dataclass
class EvidenceLedger:
    """
    Aggregates all authenticity evidence for a run.

    Fusion method: Bayesian odds update (Jeffreys 1961 reference):
        posterior_odds = prior_odds × Π LR_i
        where LR_i = P(evidence_i | wrapper) / P(evidence_i | genuine)

    Simplified: strength of contradicting evidence reduces P(genuine).
    """
    run_id: str = ""
    claimed_model: str = ""
    evidence: list[AuthenticityEvidence] = field(default_factory=list)

    def add(self, ev: AuthenticityEvidence) -> None:
        self.evidence.append(ev)

    def contradicting(self) -> list[AuthenticityEvidence]:
        return [e for e in self.evidence if e.direction == "contradicts_claim"]

    def supporting(self) -> list[AuthenticityEvidence]:
        return [e for e in self.evidence if e.direction == "supports_claim"]

    def identity_collisions(self) -> list[AuthenticityEvidence]:
        return [e for e in self.evidence if e.type == "identity_collision"]

    def wrapper_probability(self) -> float:
        """
        Compute P(wrapper_or_mismatch) using simplified Bayesian odds update.

        Prior: P(genuine) = 0.5 (uniform, no prior preference)
        Each contradicting evidence with strength s contributes LR = (1 + s)
        as a multiplier for the wrapper hypothesis.
        Each supporting evidence with strength s contributes LR = 1 / (1 + s).

        Returns P(wrapper) in [0, 1].
        Reference: Jeffreys H. (1961) Theory of Probability, 3rd ed. Oxford.
        """
        prior_genuine = 0.5
        prior_wrapper = 0.5
        odds = prior_wrapper / prior_genuine  # start at 1.0

        for ev in self.evidence:
            if ev.direction == "contradicts_claim":
                # Evidence against genuine: increases wrapper odds
                lr = 1.0 + ev.strength  # LR > 1 favors wrapper
            elif ev.direction == "supports_claim":
                # Evidence for genuine: decreases wrapper odds
                lr = 1.0 / (1.0 + ev.strength)
            else:
                lr = 1.0  # neutral
            odds *= lr

        p_wrapper = odds / (1.0 + odds)
        return round(min(max(p_wrapper, 0.0), 1.0), 4)

    def risk_level(self) -> str:
        """Map wrapper_probability to risk level string."""
        p = self.wrapper_probability()
        if p < 0.20:
            return "trusted"
        elif p < 0.55:
            return "suspicious"
        elif p < 0.85:
            return "high_risk"
        else:
            return "fake"

    def suspected_actual_model(self) -> Optional[str]:
        """Return the most-named suspected target family across contradicting evidence."""
        counts: dict[str, float] = {}
        for ev in self.contradicting():
            if ev.suspected_target:
                counts[ev.suspected_target] = counts.get(ev.suspected_target, 0.0) + ev.strength
        if not counts:
            return None
        return max(counts, key=lambda k: counts[k])

    def to_dict(self) -> dict:
        p = self.wrapper_probability()
        return {
            "run_id": self.run_id,
            "claimed_model": self.claimed_model,
            "wrapper_probability": p,
            "risk_level": self.risk_level(),
            "evidence_count": {
                "total": len(self.evidence),
                "contradicting": len(self.contradicting()),
                "supporting": len(self.supporting()),
                "neutral": len([e for e in self.evidence if e.direction == "neutral"]),
                "identity_collisions": len(self.identity_collisions()),
            },
            "suspected_actual_model": self.suspected_actual_model(),
            "evidence": [e.to_dict() for e in self.evidence],
        }


def extract_evidence_from_predetect(
    run_id: str,
    claimed_model: str,
    predetect_result: dict,
    identity_exposure: dict | None = None,
) -> EvidenceLedger:
    """
    Build an EvidenceLedger from predetect pipeline results.
    This is the main integration point called after predetect completes.
    """
    import time
    ledger = EvidenceLedger(run_id=run_id, claimed_model=claimed_model)
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # 1. Identity exposure result (L17)
    if identity_exposure:
        top_family = identity_exposure.get("top_family")
        collision = identity_exposure.get("identity_collision", False)
        posterior = identity_exposure.get("posterior", {})

        if collision and top_family:
            # A collision means model admitted to being something else
            claimed_lower = claimed_model.lower()
            if top_family.lower() not in claimed_lower:
                ev = AuthenticityEvidence(
                    type="identity_collision",
                    source_layer="L17",
                    direction="contradicts_claim",
                    strength=0.9,
                    strength_rationale="Model identity exposure found high-posterior non-claimed family",
                    raw_snippet=identity_exposure.get("evidence_snippets", [None])[0],
                    claim_target=claimed_model,
                    suspected_target=top_family,
                    timestamp=ts,
                    reproducible=True,
                    provenance={"layer": "L17", "method": "identity_exposure", "run_id": run_id},
                )
                ledger.add(ev)

    # 2. Tokenizer fingerprint (from predetect routing_info if present)
    routing = predetect_result.get("routing_info", {})
    tokenizer_match = routing.get("tokenizer_match")
    if tokenizer_match is not None:
        direction = "supports_claim" if tokenizer_match else "contradicts_claim"
        strength = 0.7 if tokenizer_match else 0.65
        ev = AuthenticityEvidence(
            type="tokenizer_fingerprint",
            source_layer="L3",
            direction=direction,
            strength=strength,
            strength_rationale="Tokenizer encoding pattern match/mismatch with claimed model family",
            claim_target=claimed_model,
            timestamp=ts,
            reproducible=True,
            provenance={"layer": "L3", "method": "tokenizer_fingerprint"},
        )
        ledger.add(ev)

    # 3. Self-report consistency (L1/L2)
    identified_as = predetect_result.get("identified_as")
    if identified_as:
        claimed_lower = claimed_model.lower()
        identified_lower = identified_as.lower()
        # Check if identified family is consistent with claimed
        is_consistent = (
            identified_lower in claimed_lower
            or claimed_lower in identified_lower
            or any(part in claimed_lower for part in identified_lower.split("-"))
        )
        direction = "supports_claim" if is_consistent else "contradicts_claim"
        strength = 0.6 if is_consistent else 0.7
        ev = AuthenticityEvidence(
            type="self_report",
            source_layer="L1",
            direction=direction,
            strength=strength,
            strength_rationale=f"Model self-identified as '{identified_as}' vs claimed '{claimed_model}'",
            claim_target=claimed_model,
            suspected_target=None if is_consistent else identified_as,
            timestamp=ts,
            reproducible=True,
            provenance={"layer": "L1", "method": "self_report", "identified_as": identified_as},
        )
        ledger.add(ev)

    return ledger
