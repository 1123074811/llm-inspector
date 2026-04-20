"""
judge/multi_choice_verified.py — Strict multiple-choice answer verification.

Implements the "strict letter match" evaluation protocol from MMLU:
    Hendrycks et al. (2021) "Measuring Massive Multitask Language Understanding"
    arXiv: https://arxiv.org/abs/2009.03300

Also supports GPQA Diamond (Rein et al. 2023) and CMMLU formats.
"""
from __future__ import annotations

import re

from app.core.logging import get_logger

logger = get_logger(__name__)

# -- Extraction patterns (MMLU strict letter match protocol) ------------------

# Pattern 1: Letter at sentence start followed by punctuation/space
_LEADING = re.compile(r"^([A-Da-d])\s*[.):\s]")

# Pattern 2: Explicit answer declaration in English
_ANSWER_EN = re.compile(r"\bAnswer\s*[:\-]\s*([A-Da-d])\b", re.IGNORECASE)

# Pattern 3: "The answer is X" / "answer: X" pattern
_THE_ANSWER = re.compile(
    r"\b(?:the\s+)?answer\s+(?:is\s+|would\s+be\s+|=\s*)?([A-Da-d])\b",
    re.IGNORECASE,
)

# Pattern 4: Chinese MCQ formats (CMMLU: "选A", "答案是A", "正确答案为A")
_ANSWER_ZH = re.compile(r"[选答案是为]+\s*([A-Da-d])", re.IGNORECASE)

# Pattern 5: "option X" / "choice X"
_OPTION = re.compile(r"\b(?:option|choice)\s+([A-Da-d])\b", re.IGNORECASE)

# Pattern 6: Bare letter assertion "A because..." / "A, because..."
_BARE_LETTER = re.compile(r"\b([A-Da-d])\s*[,\s](?:because|since|as\b|is\b)", re.IGNORECASE)

# Pattern 7: Bare single-letter response (entire response is just a letter, possibly with whitespace/punctuation)
_SOLE_LETTER = re.compile(r"^\s*([A-Da-d])\s*[.):\s]*$")

_ALL_PATTERNS = [
    _SOLE_LETTER,
    _LEADING,
    _ANSWER_EN,
    _THE_ANSWER,
    _ANSWER_ZH,
    _OPTION,
    _BARE_LETTER,
]


def _extract_choice_letter(text: str) -> str | None:
    """
    Extract the single answer letter from a response string.

    Tries multiple patterns in priority order.  If two or more *different*
    letters are found across all matches the response is considered ambiguous
    and None is returned.

    Returns:
        Uppercase letter ('A'–'D') or None if extraction fails or is ambiguous.
    """
    if not text:
        return None

    candidates: list[str] = []

    for pattern in _ALL_PATTERNS:
        for m in pattern.finditer(text):
            letter = m.group(1).upper()
            candidates.append(letter)

    if not candidates:
        return None

    unique = set(candidates)
    if len(unique) > 1:
        # Ambiguous: multiple different letters found (double-answer)
        return None

    return unique.pop()


def multi_choice_judge(
    response: str,
    params: dict,
) -> tuple[bool, dict]:
    """
    Strict multiple-choice answer verification.

    Args:
        response: Model response text.
        params:
            expected_choice  — correct answer letter, e.g. ``"A"`` (required)
            choices          — optional list of choice texts for logging

    Returns:
        (passed, detail_dict)

    Reference:
        Hendrycks et al. (2021) MMLU, arXiv:2009.03300
    """
    expected_raw = params.get("expected_choice")
    if not expected_raw:
        return False, {
            "method": "multi_choice_verified",
            "error": "missing_expected_choice",
        }

    expected = str(expected_raw).strip().upper()
    if expected not in ("A", "B", "C", "D"):
        return False, {
            "method": "multi_choice_verified",
            "error": "invalid_expected_choice",
            "expected_raw": expected_raw,
            "note": "expected_choice must be one of A, B, C, D",
        }

    extracted = _extract_choice_letter(response)
    ambiguous = extracted is None and bool(response)

    # Re-check whether None was truly ambiguous or just not found
    # (ambiguous only if multiple different letters were detected)
    _all_found: set[str] = set()
    for pattern in _ALL_PATTERNS:
        for m in pattern.finditer(response):
            _all_found.add(m.group(1).upper())
    truly_ambiguous = len(_all_found) > 1

    passed = extracted is not None and extracted == expected

    return passed, {
        "method": "multi_choice_verified",
        "extracted": extracted,
        "expected": expected,
        "ambiguous": truly_ambiguous,
        "passed": passed,
        "choices": params.get("choices"),
    }
