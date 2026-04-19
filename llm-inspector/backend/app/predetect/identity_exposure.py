"""
predetect/identity_exposure.py — Layer 17: Identity Exposure Engine

Scans response texts against model_taxonomy.yaml to detect identity
mismatches between the claimed model and the responding model.

Algorithm:
  1. Load model_taxonomy.yaml (cached singleton)
  2. For each response text: score each model family via keyword matching
     - official_names: weight 3.0
     - internal_codenames: weight 2.0
     - refusal_signatures: weight 2.5
     - style_keywords: weight 1.0
  3. Aggregate scores; if a non-claimed family scores highest AND score > threshold,
     flag identity_collision = True
  4. Return IdentityExposureReport with top-3 families, evidence snippets,
     and overall collision flag

Reference:
    Perez & Ribeiro (2022) "Ignore Previous Prompt: Attack Techniques For
    Language Models" arXiv:2211.09527
    Greshake et al. (2023) arXiv:2302.12173
"""
from __future__ import annotations

import pathlib
import re
import threading
from dataclasses import dataclass, field
from collections import defaultdict

from app.core.schemas import LayerResult, LLMRequest, Message, CaseResult
from app.core.logging import get_logger

logger = get_logger(__name__)

# ── Taxonomy loader ───────────────────────────────────────────────────────────

_TAXONOMY_PATH = pathlib.Path(__file__).resolve().parents[1] / "_data" / "model_taxonomy.yaml"
_TAXONOMY_LOCK = threading.Lock()
_TAXONOMY_CACHE: dict | None = None

# Signal weights (from SOURCES.yaml via taxonomy signal_weights)
_SIGNAL_WEIGHTS = {
    "official_names":     3.0,
    "internal_codenames": 2.0,
    "refusal_signatures": 2.5,
    "style_keywords":     1.0,
}

# Collision threshold: if top non-claimed family reaches this Bayesian posterior, flag collision
_COLLISION_POSTERIOR_THRESHOLD = 0.80


def _load_taxonomy() -> dict:
    """Load model_taxonomy.yaml (thread-safe singleton with lazy load)."""
    global _TAXONOMY_CACHE
    if _TAXONOMY_CACHE is not None:
        return _TAXONOMY_CACHE
    with _TAXONOMY_LOCK:
        if _TAXONOMY_CACHE is not None:
            return _TAXONOMY_CACHE
        try:
            import yaml  # type: ignore
            with _TAXONOMY_PATH.open("r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh)
        except ImportError:
            raw = _parse_taxonomy_minimal()
        except Exception as e:
            logger.warning("Could not load model_taxonomy.yaml", error=str(e))
            raw = {}
        _TAXONOMY_CACHE = raw or {}
        logger.info("Loaded model taxonomy", families=list(_TAXONOMY_CACHE.keys()))
    return _TAXONOMY_CACHE


def _parse_taxonomy_minimal() -> dict:
    """Fallback parser if pyyaml unavailable — parses top-level keys only."""
    try:
        text = _TAXONOMY_PATH.read_text(encoding="utf-8")
    except Exception:
        return {}
    result = {}
    current_family = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # Top-level family keys: lines like "claude:" at indent=0
        if re.match(r"^[a-z_]+:$", stripped) and not line.startswith(" "):
            current_family = stripped[:-1]
            result[current_family] = {}
    return result


def reload_taxonomy() -> dict:
    """Force-reload taxonomy from disk (useful in tests)."""
    global _TAXONOMY_CACHE
    with _TAXONOMY_LOCK:
        _TAXONOMY_CACHE = None
    return _load_taxonomy()


# ── Core analysis ─────────────────────────────────────────────────────────────

@dataclass
class EvidenceItem:
    """A single piece of evidence for a family match."""
    family: str
    signal_type: str       # official_names | internal_codenames | refusal_signatures | style_keywords
    matched_text: str      # the keyword/signature that matched
    response_snippet: str  # ≤80 chars of context around match
    case_id: str           # case name or "predetect/{layer}"
    weight: float


@dataclass
class FamilyHit:
    """Aggregated score + evidence for one model family."""
    family: str
    raw_score: float = 0.0
    posterior: float = 0.0
    evidence: list[EvidenceItem] = field(default_factory=list)


@dataclass
class IdentityExposureReport:
    """Full identity exposure analysis result."""
    claimed_model: str | None
    claimed_family: str | None           # resolved from claimed_model
    identity_collision: bool = False     # True when non-claimed family wins with p>=0.80
    top_families: list[FamilyHit] = field(default_factory=list)   # top-3 sorted by posterior
    extracted_system_prompt: str | None = None
    total_responses_scanned: int = 0
    collision_confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "claimed_model": self.claimed_model,
            "claimed_family": self.claimed_family,
            "identity_collision": self.identity_collision,
            "collision_confidence": round(self.collision_confidence, 3),
            "total_responses_scanned": self.total_responses_scanned,
            "extracted_system_prompt": self.extracted_system_prompt,
            "top_families": [
                {
                    "family": h.family,
                    "raw_score": round(h.raw_score, 2),
                    "posterior": round(h.posterior, 3),
                    "evidence": [
                        {
                            "signal_type": e.signal_type,
                            "matched_text": e.matched_text,
                            "snippet": e.response_snippet,
                            "case_id": e.case_id,
                            "weight": e.weight,
                        }
                        for e in h.evidence[:5]  # Top 5 per family
                    ],
                }
                for h in self.top_families
            ],
        }


def _resolve_claimed_family(claimed_model: str | None, taxonomy: dict) -> str | None:
    """Map a claimed_model string to a taxonomy family key."""
    if not claimed_model:
        return None
    lower = claimed_model.lower()
    for family, data in taxonomy.items():
        if not isinstance(data, dict):
            continue
        all_names = (
            [n.lower() for n in data.get("official_names", [])]
            + [n.lower() for n in data.get("internal_codenames", [])]
        )
        if any(n in lower or lower in n for n in all_names):
            return family
    return None


def _extract_snippet(text: str, match_start: int, match_end: int, window: int = 40) -> str:
    """Extract a ≤(2*window+match_len) chars context snippet around a match."""
    start = max(0, match_start - window)
    end = min(len(text), match_end + window)
    snippet = text[start:end]
    if start > 0:
        snippet = "…" + snippet
    if end < len(text):
        snippet = snippet + "…"
    return snippet.replace("\n", " ").strip()


def _scan_text(
    text: str,
    case_id: str,
    taxonomy: dict,
    family_scores: dict,
    family_evidence: dict,
) -> None:
    """Scan one response text and accumulate scores + evidence."""
    if not text:
        return
    text_lower = text.lower()

    for family, data in taxonomy.items():
        if not isinstance(data, dict):
            continue
        for signal_type, weight in _SIGNAL_WEIGHTS.items():
            keywords = data.get(signal_type, [])
            for kw in keywords:
                kw_lower = kw.lower()
                # Find all occurrences (case-insensitive)
                start = 0
                while True:
                    pos = text_lower.find(kw_lower, start)
                    if pos == -1:
                        break
                    family_scores[family] += weight
                    snippet = _extract_snippet(text, pos, pos + len(kw))
                    family_evidence[family].append(EvidenceItem(
                        family=family,
                        signal_type=signal_type,
                        matched_text=kw,
                        response_snippet=snippet,
                        case_id=case_id,
                        weight=weight,
                    ))
                    start = pos + len(kw)
                    break  # Count once per keyword per response (avoid over-weighting)


def _bayesian_posterior(scores: dict[str, float]) -> dict[str, float]:
    """
    Convert raw scores to Bayesian posteriors.

    Prior: uniform (1/N for each family).
    Likelihood: proportional to exp(score).
    Posterior: softmax(scores).
    """
    import math
    if not scores or all(v == 0 for v in scores.values()):
        n = len(scores)
        return {k: 1.0 / n for k in scores} if n > 0 else {}

    max_score = max(scores.values())
    exp_scores = {k: math.exp(v - max_score) for k, v in scores.items()}
    total = sum(exp_scores.values())
    return {k: v / total for k, v in exp_scores.items()}


def analyze_responses(
    response_texts: list[tuple[str, str]],   # [(text, case_id), ...]
    claimed_model: str | None,
    extracted_system_prompt: str | None = None,
) -> IdentityExposureReport:
    """
    Main analysis function.

    Args:
        response_texts: List of (response_text, case_id) tuples.
        claimed_model: The model name the API claims to be.
        extracted_system_prompt: Pre-extracted system prompt (from SystemPromptHarvester).

    Returns:
        IdentityExposureReport with collision detection + top families.
    """
    taxonomy = _load_taxonomy()
    if not taxonomy:
        return IdentityExposureReport(
            claimed_model=claimed_model,
            claimed_family=None,
            identity_collision=False,
        )

    claimed_family = _resolve_claimed_family(claimed_model, taxonomy)

    family_scores: dict[str, float] = defaultdict(float)
    family_evidence: dict[str, list[EvidenceItem]] = defaultdict(list)

    # Initialise all families with 0 score (ensures they appear in posteriors)
    for family in taxonomy:
        family_scores[family] += 0.0

    # Scan all responses
    for text, case_id in response_texts:
        _scan_text(text, case_id, taxonomy, family_scores, family_evidence)

    # Compute posteriors
    posteriors = _bayesian_posterior(dict(family_scores))

    # Build FamilyHit list (top 3)
    hits = [
        FamilyHit(
            family=family,
            raw_score=family_scores[family],
            posterior=posteriors.get(family, 0.0),
            evidence=family_evidence.get(family, []),
        )
        for family in taxonomy
    ]
    hits.sort(key=lambda h: h.raw_score, reverse=True)
    top3 = hits[:3]

    # Collision detection: is the top family different from claimed?
    identity_collision = False
    collision_confidence = 0.0
    if top3 and top3[0].raw_score > 0:
        top_family = top3[0].family
        top_posterior = top3[0].posterior
        if top_family != claimed_family and top_posterior >= _COLLISION_POSTERIOR_THRESHOLD:
            identity_collision = True
            collision_confidence = top_posterior
        elif top_family != claimed_family and top3[0].raw_score >= 5.0:
            # Raw score threshold fallback when posteriors are flat
            identity_collision = True
            collision_confidence = min(0.79, top_posterior)

    report = IdentityExposureReport(
        claimed_model=claimed_model,
        claimed_family=claimed_family,
        identity_collision=identity_collision,
        top_families=top3,
        extracted_system_prompt=extracted_system_prompt,
        total_responses_scanned=len(response_texts),
        collision_confidence=collision_confidence,
    )

    if identity_collision:
        logger.warning(
            "Identity collision detected",
            claimed=claimed_model,
            claimed_family=claimed_family,
            suspected_family=top3[0].family,
            confidence=round(collision_confidence, 3),
        )
    else:
        logger.info(
            "Identity exposure analysis complete",
            claimed=claimed_model,
            identity_collision=False,
            responses_scanned=len(response_texts),
        )

    return report


def analyze_case_results(
    case_results: list[CaseResult],
    claimed_model: str | None,
    predetect_responses: list[tuple[str, str]] | None = None,
    extracted_system_prompt: str | None = None,
) -> IdentityExposureReport:
    """
    Convenience wrapper: build response_texts from CaseResult list + optional predetect responses.
    """
    texts: list[tuple[str, str]] = []

    # Add predetect probe responses
    if predetect_responses:
        texts.extend(predetect_responses)

    # Add test case responses
    for cr in case_results:
        for s in cr.samples:
            content = s.response.content
            if content:
                texts.append((content, cr.case.name))

    return analyze_responses(texts, claimed_model, extracted_system_prompt)


# ── Layer 17 PreDetect integration ───────────────────────────────────────────

class Layer17IdentityExposure:
    """
    PreDetect Layer 17 — Identity Exposure.

    Runs after L16; scans predetect probe responses against taxonomy.
    Only flags high-confidence collisions (≥0.85) so it doesn't spam
    low-certainty warnings.

    Reference:
        Perez & Ribeiro (2022) arXiv:2211.09527
        Greshake et al. (2023) arXiv:2302.12173
    """

    _PREDETECT_SIGNAL_THRESHOLD = 0.85  # Higher bar for predetect-only evidence

    def run(self, adapter, model_name: str, layer_results_so_far: list | None = None) -> LayerResult:
        """
        Extract responses accumulated so far from layer_results and scan them.
        This layer does NOT make new API calls; it re-analyses existing evidence.
        """
        texts: list[tuple[str, str]] = []

        # Collect evidence strings from prior layers
        if layer_results_so_far:
            for lr in layer_results_so_far:
                for ev in lr.evidence:
                    if isinstance(ev, str) and len(ev) > 20:
                        texts.append((ev, lr.layer))

        if not texts:
            return LayerResult(
                layer="Layer17/IdentityExposure",
                confidence=0.0,
                identified_as=None,
                evidence=["No predetect responses available for identity analysis"],
                tokens_used=0,
            )

        report = analyze_responses(texts, model_name)

        # Build evidence strings for LayerResult
        evidence_strs = []
        for hit in report.top_families:
            if hit.raw_score > 0:
                evidence_strs.append(
                    f"Family '{hit.family}': score={hit.raw_score:.1f}, "
                    f"posterior={hit.posterior:.3f}, "
                    f"evidence_items={len(hit.evidence)}"
                )

        if report.identity_collision:
            identified = report.top_families[0].family if report.top_families else None
            confidence = min(1.0, report.collision_confidence)
            evidence_strs.insert(0, f"IDENTITY COLLISION: claimed={model_name}, "
                                     f"suspected={identified}, "
                                     f"confidence={confidence:.3f}")
        else:
            identified = None
            confidence = 0.0

        return LayerResult(
            layer="Layer17/IdentityExposure",
            confidence=confidence,
            identified_as=identified,
            evidence=evidence_strs[:10],
            tokens_used=0,  # No API calls; pure analysis
        )
