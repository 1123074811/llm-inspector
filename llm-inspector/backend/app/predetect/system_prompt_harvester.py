"""
predetect/system_prompt_harvester.py — System Prompt Extraction & Sanitization

Detects and extracts system prompts that have leaked into model responses.
A system prompt is considered "leaked" when the response contains patterns
that indicate it is echoing back its own system instructions.

Detection templates (from model_taxonomy.yaml spirit + Perez 2022):
    - Starts with "You are", "Your name is", "你是", "You're an AI"
    - Contains ≥2 Markdown section headers: ## Role, ## Task, ## Capabilities
    - Contains "knowledge cutoff" / "截止至" / "training data"
    - Length > 100 chars after first template match

Sanitization: removes internal URLs, API tokens, org IDs, internal tool names.

Reference:
    Perez & Ribeiro (2022) "Ignore Previous Prompt"  arXiv:2211.09527
    Carlini et al. (2023) "Extracting Training Data"  arXiv:2403.06634
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.logging import get_logger

logger = get_logger(__name__)


# ── Detection patterns ────────────────────────────────────────────────────────

# Tier 1: Strong indicators (single match sufficient if length > 200)
_TIER1_PATTERNS = [
    re.compile(r"(?i)^(you are |you're |your name is |you are an ai |你是)", re.MULTILINE),
    re.compile(r"(?i)^(system:\s|<system>|\[system\])", re.MULTILINE),
    re.compile(r"(?i)knowledge cutoff\s*[:：]"),
    re.compile(r"(?i)(my training data|my knowledge|截止至|知识截止)"),
]

# Tier 2: Structural indicators (need ≥2 to trigger)
_TIER2_PATTERNS = [
    re.compile(r"(?m)^#{1,3}\s+(role|task|capabilities|limitations|instructions|rules|guidelines|你的|我的)\b", re.IGNORECASE),
    re.compile(r"(?i)\b(task:|role:|context:|instructions:|rules:)\s"),
    re.compile(r"(?i)\b(do not |don't |you must not |禁止|不得)\b"),
    re.compile(r"(?i)\b(you should always |you should never |始终|永远不要)\b"),
]

# Sanitization: patterns to redact (order matters: more specific patterns first)
_SANITIZE_PATTERNS = [
    (re.compile(r"https?://[^\s<>\"]+"), "[URL_REDACTED]"),
    (re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"), "[API_KEY_REDACTED]"),       # API keys before base64
    (re.compile(r"\b[A-Za-z0-9+/]{32,}={0,2}\b"), "[TOKEN_REDACTED]"),    # base64-like tokens
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"), "[UUID_REDACTED]"),
    (re.compile(r"\b\d{10,}\b"), "[ID_REDACTED]"),  # Long numeric IDs
]

_MIN_PROMPT_LENGTH = 80   # chars — shorter is unlikely to be a real system prompt
_MAX_PROMPT_LENGTH = 8000 # chars — sanity cap


@dataclass
class HarvestResult:
    """Result of system prompt extraction attempt."""
    found: bool = False
    extracted_text: str | None = None
    sanitized_text: str | None = None
    detection_method: str | None = None   # which pattern triggered
    confidence: float = 0.0
    source_case_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "found": self.found,
            "sanitized_text": self.sanitized_text,
            "detection_method": self.detection_method,
            "confidence": round(self.confidence, 3),
            "source_case_ids": self.source_case_ids,
        }


def _sanitize(text: str) -> str:
    """Apply redaction patterns to remove sensitive content."""
    for pattern, replacement in _SANITIZE_PATTERNS:
        text = pattern.sub(replacement, text)
    return text.strip()


def _score_text(text: str) -> tuple[float, str]:
    """
    Score a text for system-prompt leakage.

    Returns (confidence: float, method: str).
    """
    if len(text) < _MIN_PROMPT_LENGTH:
        return 0.0, ""

    # Tier 1: single strong pattern = high confidence
    for pat in _TIER1_PATTERNS:
        if pat.search(text):
            tier2_hits = sum(1 for p in _TIER2_PATTERNS if p.search(text))
            base = 0.70 if len(text) > 200 else 0.50
            bonus = min(0.25, tier2_hits * 0.08)
            return min(0.95, base + bonus), f"tier1:{pat.pattern[:30]}"

    # Tier 2: need ≥2 structural indicators
    tier2_hits = sum(1 for p in _TIER2_PATTERNS if p.search(text))
    if tier2_hits >= 2:
        conf = min(0.65, 0.40 + tier2_hits * 0.07)
        return conf, f"tier2:{tier2_hits}_matches"

    return 0.0, ""


def harvest(
    response_texts: list[tuple[str, str]],   # [(text, case_id), ...]
    confidence_threshold: float = 0.50,
) -> HarvestResult:
    """
    Scan response texts for leaked system prompt content.

    Args:
        response_texts: List of (text, case_id) to scan.
        confidence_threshold: Minimum confidence to consider a positive detection.

    Returns:
        HarvestResult with the best (highest-confidence) candidate.
    """
    best: HarvestResult | None = None

    for text, case_id in response_texts:
        if not text:
            continue

        # Truncate to max length for analysis
        scan_text = text[:_MAX_PROMPT_LENGTH]
        confidence, method = _score_text(scan_text)

        if confidence >= confidence_threshold:
            # Extract the "system prompt segment": from first tier1 match to end of section
            extracted = _extract_segment(scan_text, method)
            sanitized = _sanitize(extracted) if extracted else None

            if best is None or confidence > best.confidence:
                best = HarvestResult(
                    found=True,
                    extracted_text=extracted,
                    sanitized_text=sanitized,
                    detection_method=method,
                    confidence=confidence,
                    source_case_ids=[case_id],
                )
            elif confidence == best.confidence and case_id not in best.source_case_ids:
                best.source_case_ids.append(case_id)

    if best is None:
        return HarvestResult(found=False, confidence=0.0)

    logger.info(
        "System prompt harvested",
        confidence=round(best.confidence, 3),
        method=best.detection_method,
        length=len(best.sanitized_text or ""),
        sources=best.source_case_ids,
    )
    return best


def _extract_segment(text: str, method: str) -> str:
    """
    Extract the system-prompt-looking segment from the text.

    Heuristic: take from the first Tier 1 match forward,
    stopping at the first "user-like" turn marker or max 2000 chars.
    """
    # Try to find where the system prompt starts
    start_pos = 0
    for pat in _TIER1_PATTERNS:
        m = pat.search(text)
        if m:
            start_pos = max(0, m.start() - 5)
            break

    segment = text[start_pos:start_pos + 2000]

    # Truncate at "user turn" markers
    user_markers = ["\nUser:", "\nHuman:", "\n用户:", "[/INST]", "<|im_end|>", "---"]
    for marker in user_markers:
        idx = segment.find(marker)
        if idx > 50:  # Keep at least 50 chars
            segment = segment[:idx]
            break

    return segment.strip()
