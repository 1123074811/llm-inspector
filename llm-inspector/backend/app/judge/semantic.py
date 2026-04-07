"""
SemanticJudge — LLM-as-Judge for open-ended evaluation.
Falls back to local rule-based evaluation when judge API is not configured.
"""
from __future__ import annotations

import json
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
