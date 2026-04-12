"""
SemanticJudge — LLM-as-Judge for open-ended evaluation.
Falls back to local rule-based evaluation when judge API is not configured.
"""
from __future__ import annotations

import json
import re
import urllib.request
import urllib.error
from dataclasses import dataclass

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SemanticResult:
    passed: bool
    score: float  # 0-100
    reasoning: str
    confidence: float  # 0-1
    method: str  # "llm" or "local"


_JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator. Score the following AI response.

## Original Prompt
{prompt}

## AI Response
{response}

## Evaluation Criteria
{criteria}

## Instructions
1. Evaluate the response against the criteria.
2. Output a JSON object with exactly these fields:
   - "verdict": "pass", "partial", or "fail"
   - "score": integer 0-100
   - "reasoning": brief explanation (1-2 sentences)
   - "confidence": float 0.0-1.0

Output ONLY the JSON, no other text."""


def llm_judge_available() -> bool:
    """Check if the LLM judge API is configured."""
    return bool(settings.JUDGE_API_URL and settings.JUDGE_API_KEY)


def llm_judge(
    prompt: str,
    response: str,
    criteria: str,
) -> SemanticResult:
    """
    Call the configured LLM judge API for semantic evaluation.
    Returns SemanticResult with method="llm".
    Raises on network/parse errors — caller should handle gracefully.
    """
    judge_prompt = _JUDGE_PROMPT_TEMPLATE.format(
        prompt=prompt[:500],
        response=response[:2000],
        criteria=criteria,
    )

    payload = {
        "model": settings.JUDGE_MODEL,
        "messages": [{"role": "user", "content": judge_prompt}],
        "max_tokens": 300,
        "temperature": 0.0,
    }

    url = settings.JUDGE_API_URL.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.JUDGE_API_KEY}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=settings.JUDGE_TIMEOUT) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        logger.warning("LLM judge API call failed", error=str(e))
        raise

    # Parse response
    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

    # Try to extract JSON from the response
    try:
        # Strip markdown fences if present
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
            cleaned = cleaned.rsplit("```", 1)[0]

        parsed = json.loads(cleaned.strip())
        verdict = parsed.get("verdict", "fail")
        score = int(parsed.get("score", 0))
        reasoning = str(parsed.get("reasoning", ""))
        confidence = float(parsed.get("confidence", 0.5))

        return SemanticResult(
            passed=verdict in ("pass", "partial"),
            score=max(0, min(100, score)),
            reasoning=reasoning,
            confidence=max(0.0, min(1.0, confidence)),
            method="llm",
        )
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.warning("Failed to parse LLM judge response", error=str(e), content=content[:200])
        # Fallback: try to detect pass/fail from text
        lower_content = content.lower()
        if "pass" in lower_content and "fail" not in lower_content:
            return SemanticResult(passed=True, score=70, reasoning=content[:200], confidence=0.3, method="llm")
        return SemanticResult(passed=False, score=30, reasoning=content[:200], confidence=0.3, method="llm")


# ---------------------------------------------------------------------------
# Helpers for local_semantic_judge
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    "的 了 在 是 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好 "
    "自己 这 他 她 它 们 那 些 什么 没 吧 吗 呢 啊 哦 嗯 "
    "the a an is are was were be been being have has had do does did will would shall should "
    "may might can could of in to for on with at by from as into through during before after "
    "above below between out off over under again further then once here there when where why "
    "how all each every both few more most other some such no nor not only own same so than "
    "too very and but or if while that this it its i me my we our you your he him his she her they them their".split()
)

_UNCERTAINTY_PHRASES_EN = [
    "i'm not sure", "i don't know", "i am not certain", "it's unclear",
    "i cannot determine", "uncertain", "hard to say", "not enough information",
    "it depends", "possibly", "i'm unsure",
]
_UNCERTAINTY_PHRASES_ZH = [
    "不确定", "无法确认", "不清楚", "很难说", "不太确定", "无法判断",
    "信息不足", "不好说", "可能是",
]

_COPY_PASTE_PATTERNS = [
    r"^>{3,}",                          # markdown nested quotes
    r"\[object\s+\w+\]",               # JS object stringification leak
    r"undefined{2,}",                   # JS undefined leak
    r"(.{30,})\1{2,}",                 # long repeated substring
    r"lorem ipsum",                     # placeholder text
]


def _extract_keywords(text: str) -> list[str]:
    """Extract meaningful tokens from text, filtering stopwords and short tokens."""
    tokens = re.findall(r"[\w\u4e00-\u9fff]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def _ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    """Generate n-gram set from token list."""
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def _parse_length_constraint(criteria: str) -> tuple[int | None, int | None]:
    """Try to extract min/max sentence or word count hints from criteria text.

    Supports patterns like:
      - "用2-3句话回答" → (2, 3) sentences
      - "in 50 words or fewer" → (None, 50) words
      - "至少100字" → (100, None) chars
    Returns (min_val, max_val) as rough targets, or (None, None).
    """
    # Chinese sentence count: "用N-M句话"
    m = re.search(r"用\s*(\d+)\s*[-~到]\s*(\d+)\s*句", criteria)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"(\d+)\s*句", criteria)
    if m:
        n = int(m.group(1))
        return max(1, n - 1), n + 1
    # English word count
    m = re.search(r"(?:in|under|within|at most|no more than)\s+(\d+)\s+words?", criteria, re.IGNORECASE)
    if m:
        return None, int(m.group(1))
    m = re.search(r"(?:at least|minimum)\s+(\d+)\s+words?", criteria, re.IGNORECASE)
    if m:
        return int(m.group(1)), None
    # Chinese char count: "至少N字" / "不超过N字"
    m = re.search(r"至少\s*(\d+)\s*字", criteria)
    if m:
        return int(m.group(1)), None
    m = re.search(r"不超过\s*(\d+)\s*字", criteria)
    if m:
        return None, int(m.group(1))
    return None, None


# ---------------------------------------------------------------------------
# Main local_semantic_judge
# ---------------------------------------------------------------------------

def local_semantic_judge(
    prompt: str,
    response: str,
    criteria: str,
    reference_answer: str = "",
    required_keywords: list[str] | None = None,
    forbidden_patterns: list[str] | None = None,
    accept_uncertainty: bool = False,
    pass_threshold: float = 45.0,
) -> SemanticResult:
    """Enhanced local semantic evaluation using multi-signal scoring.

    Provides a much richer evaluation than simple keyword matching by combining
    five independent scoring dimensions: relevance, completeness, structure,
    constraint compliance, and confidence calibration.

    Total score range: 0-100.
    """
    if required_keywords is None:
        required_keywords = []
    if forbidden_patterns is None:
        forbidden_patterns = []

    reasoning_parts: list[str] = []
    signals_with_data = 0  # how many dimensions actually had data to evaluate

    # -- 1. Relevance score (0-30) -------------------------------------------
    relevance_score = 0.0
    prompt_kw = _extract_keywords(prompt)
    response_kw = _extract_keywords(response)

    if prompt_kw:
        signals_with_data += 1
        # Unigram overlap: how many prompt keywords appear in response
        prompt_kw_set = set(prompt_kw)
        response_kw_set = set(response_kw)
        hit_count = len(prompt_kw_set & response_kw_set)
        coverage = hit_count / len(prompt_kw_set) if prompt_kw_set else 0.0

        # Bigram overlap adds credit for topical coherence
        prompt_bigrams = _ngrams(prompt_kw, 2)
        response_bigrams = _ngrams(response_kw, 2)
        bigram_overlap = _jaccard(prompt_bigrams, response_bigrams) if prompt_bigrams else 0.0

        # Weighted: 70% unigram coverage, 30% bigram coherence
        relevance_raw = 0.7 * coverage + 0.3 * bigram_overlap
        relevance_score = relevance_raw * 30.0

        reasoning_parts.append(
            f"Relevance: keyword coverage={coverage:.2f}, bigram overlap={bigram_overlap:.2f} -> {relevance_score:.1f}/30"
        )
    else:
        # No meaningful keywords in prompt — give a baseline if response is non-empty
        if len(response.strip()) > 10:
            relevance_score = 15.0
            reasoning_parts.append("Relevance: no prompt keywords extracted, baseline 15/30 for non-empty response")
        else:
            reasoning_parts.append("Relevance: empty/trivial response -> 0/30")

    # -- 2. Completeness score (0-25) ----------------------------------------
    completeness_score = 0.0
    completeness_signals = 0

    # 2a. Required keyword coverage
    if required_keywords:
        signals_with_data += 1
        completeness_signals += 1
        lower_response = response.lower()
        kw_hits = [kw for kw in required_keywords if kw.lower() in lower_response]
        kw_coverage = len(kw_hits) / len(required_keywords)
        kw_part = kw_coverage * 25.0
        completeness_score += kw_part * 0.5  # half weight from keywords
        reasoning_parts.append(
            f"Completeness/keywords: {len(kw_hits)}/{len(required_keywords)} matched -> kw_coverage={kw_coverage:.2f}"
        )

    # 2b. Reference answer similarity (word-level Jaccard + bigram Jaccard)
    if reference_answer:
        signals_with_data += 1
        completeness_signals += 1
        ref_tokens = _extract_keywords(reference_answer)
        resp_tokens = _extract_keywords(response)

        ref_word_set = set(ref_tokens)
        resp_word_set = set(resp_tokens)
        word_jaccard = _jaccard(ref_word_set, resp_word_set)

        ref_bigrams = _ngrams(ref_tokens, 2)
        resp_bigrams = _ngrams(resp_tokens, 2)
        bigram_jaccard = _jaccard(ref_bigrams, resp_bigrams)

        # Combined: 55% word-level, 45% bigram-level (bigram captures phrase accuracy)
        ref_similarity = 0.55 * word_jaccard + 0.45 * bigram_jaccard
        ref_part = ref_similarity * 25.0

        if required_keywords:
            completeness_score += ref_part * 0.5  # other half from reference
        else:
            completeness_score += ref_part  # full weight from reference

        reasoning_parts.append(
            f"Completeness/reference: word_jaccard={word_jaccard:.2f}, bigram_jaccard={bigram_jaccard:.2f}, combined={ref_similarity:.2f}"
        )

    # If neither keywords nor reference provided, give partial credit for non-trivial response
    if not required_keywords and not reference_answer:
        word_count = len(response.split())
        if word_count >= 5:
            completeness_score = 12.5  # half credit
            reasoning_parts.append("Completeness: no reference data, baseline 12.5/25 for non-trivial response")
        else:
            reasoning_parts.append("Completeness: no reference data and short response -> 0/25")

    # -- 3. Structure score (0-20) -------------------------------------------
    structure_score = 0.0
    signals_with_data += 1

    resp_stripped = response.strip()
    sentences = [s.strip() for s in re.split(r'[.!?。！？\n]+', resp_stripped) if s.strip()]
    word_count = len(response.split())

    # 3a. Has clear sentences (not just fragments): 0-8 points
    if len(sentences) >= 1 and any(len(s.split()) >= 3 or len(s) >= 6 for s in sentences):
        # At least one sentence with 3+ words (EN) or 6+ chars (ZH)
        structure_score += 8.0
    elif resp_stripped:
        structure_score += 3.0  # some text but fragmented

    # 3b. Reasonable length for question type: 0-7 points
    # Heuristic: questions typically expect answers between 10 and 2000 words
    if 10 <= word_count <= 2000:
        structure_score += 7.0
    elif 3 <= word_count < 10:
        structure_score += 4.0
    elif word_count > 2000:
        structure_score += 5.0  # verbose but not penalized heavily

    # 3c. No copy-paste artifacts: 0-5 points
    artifact_found = False
    for pat in _COPY_PASTE_PATTERNS:
        try:
            if re.search(pat, response, re.IGNORECASE | re.MULTILINE):
                artifact_found = True
                break
        except re.error:
            pass
    if not artifact_found:
        structure_score += 5.0
    else:
        reasoning_parts.append("Structure: copy-paste artifact detected (-5)")

    reasoning_parts.append(f"Structure: sentences={len(sentences)}, words={word_count} -> {structure_score:.1f}/20")

    # -- 4. Constraint compliance score (0-15) --------------------------------
    constraint_score = 15.0  # start at max, deduct for violations
    constraint_deductions: list[str] = []

    # 4a. Forbidden patterns
    for pat in forbidden_patterns:
        try:
            if re.search(pat, response, re.IGNORECASE | re.MULTILINE):
                constraint_score -= 5.0
                constraint_deductions.append(f"forbidden pattern hit: {pat[:30]}")
        except re.error:
            pass  # invalid regex, skip

    # 4b. Length constraints from criteria
    min_val, max_val = _parse_length_constraint(criteria)
    if min_val is not None or max_val is not None:
        signals_with_data += 1
        # Decide unit: if criteria mentions "句" count sentences, "字" count chars, else words
        if "句" in criteria:
            measured = len(sentences)
            unit = "sentences"
        elif "字" in criteria:
            # Chinese character count (exclude spaces/punctuation)
            measured = len(re.findall(r'[\u4e00-\u9fff]', response))
            unit = "chars(zh)"
        else:
            measured = word_count
            unit = "words"

        if min_val is not None and measured < min_val:
            penalty = min(5.0, (min_val - measured) * 2.0)
            constraint_score -= penalty
            constraint_deductions.append(f"too short: {measured} {unit} < min {min_val}")
        if max_val is not None and measured > max_val:
            penalty = min(5.0, (measured - max_val) * 1.5)
            constraint_score -= penalty
            constraint_deductions.append(f"too long: {measured} {unit} > max {max_val}")

    # 4c. Format expectations from criteria (e.g., "JSON", "列表", "list")
    criteria_lower = criteria.lower() if criteria else ""
    if "json" in criteria_lower:
        try:
            import json as _json
            _json.loads(response.strip())
        except (ValueError, TypeError):
            # Check if response contains a JSON block
            json_match = re.search(r'\{[^{}]*\}|\[[^\[\]]*\]', response)
            if not json_match:
                constraint_score -= 4.0
                constraint_deductions.append("expected JSON format not found")

    constraint_score = max(0.0, constraint_score)
    if constraint_deductions:
        reasoning_parts.append(f"Constraints: deductions=[{'; '.join(constraint_deductions)}] -> {constraint_score:.1f}/15")
    else:
        reasoning_parts.append(f"Constraints: no violations -> {constraint_score:.1f}/15")

    # -- 5. Confidence calibration score (0-10) ------------------------------
    calibration_score = 5.0  # default neutral
    lower_response = response.lower()

    is_uncertain = any(p in lower_response for p in _UNCERTAINTY_PHRASES_EN) or \
                   any(p in response for p in _UNCERTAINTY_PHRASES_ZH)

    if accept_uncertainty and is_uncertain:
        # Expressing uncertainty when it's acceptable is good
        calibration_score = 9.0
        reasoning_parts.append("Calibration: appropriate uncertainty expressed -> 9/10")
    elif not accept_uncertainty and is_uncertain:
        # Uncertainty when a definite answer was expected — mild penalty
        calibration_score = 3.0
        reasoning_parts.append("Calibration: uncertain response when definite answer expected -> 3/10")
    elif accept_uncertainty and not is_uncertain:
        # Gave a definite answer to a potentially uncertain question — okay if correct
        calibration_score = 5.0
        reasoning_parts.append("Calibration: definite answer to uncertain question, neutral -> 5/10")
    else:
        # Normal case: definite answer expected, definite answer given
        # Check for overconfidence markers when response is very short
        overconfident_markers = [
            "obviously", "clearly", "without doubt", "definitely", "absolutely",
            "毫无疑问", "显然", "肯定是", "必然",
        ]
        has_overconfidence = any(m in lower_response for m in overconfident_markers)
        if has_overconfidence and word_count < 15:
            calibration_score = 3.0
            reasoning_parts.append("Calibration: overconfident short answer -> 3/10")
        else:
            calibration_score = 6.0
            reasoning_parts.append(f"Calibration: normal -> {calibration_score:.1f}/10")

    # -- Final aggregation ---------------------------------------------------
    total_score = relevance_score + completeness_score + structure_score + constraint_score + calibration_score
    total_score = max(0.0, min(100.0, total_score))
    passed = total_score >= pass_threshold

    # Confidence is based on how many signal dimensions had actual data
    # (5 possible signal sources: prompt keywords, required keywords,
    #  reference answer, structure, length constraints)
    confidence = min(1.0, 0.3 + 0.14 * signals_with_data)  # 0.3 base, up to ~1.0

    reasoning_summary = (
        f"Total={total_score:.1f}/100 (relevance={relevance_score:.1f}, "
        f"completeness={completeness_score:.1f}, structure={structure_score:.1f}, "
        f"constraints={constraint_score:.1f}, calibration={calibration_score:.1f}). "
        + " | ".join(reasoning_parts)
    )

    logger.debug(
        "local_semantic_judge completed",
        score=round(total_score, 1),
        passed=passed,
        confidence=round(confidence, 2),
        signals_with_data=signals_with_data,
    )

    return SemanticResult(
        passed=passed,
        score=round(total_score, 1),
        reasoning=reasoning_summary,
        confidence=round(confidence, 2),
        method="local_enhanced",
    )
