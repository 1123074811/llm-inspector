"""
predetect/layer_l20_self_paradox.py — Layer 20: Context Self-Report Paradox

Sends the same identity question through three different role contexts
(system prompt, user message, assistant continuation) and checks for
inconsistent self-identification. A wrapper/proxy model is likely to give
different answers across contexts due to prompt routing differences.

Reference: Xu et al. (2024) "Identity Consistency of Chat Models"
Default mode: Deep (zero tokens in Quick/Standard unless confidence low).
"""
from __future__ import annotations

from collections import Counter

from app.core.schemas import LayerResult, LLMRequest, Message

# Identity questions to ask across contexts
_IDENTITY_QS = [
    "Who created you?",
    "What is your model name?",
    "Which company developed you?",
]


class Layer20SelfParadox:
    """Check if model self-identifies consistently across role contexts."""

    def run(self, adapter, model_name: str) -> LayerResult:
        evidence: list[str] = []
        answers: list[tuple[str, str]] = []  # (context, answer)

        for q in _IDENTITY_QS:
            # 1. Ask via system prompt
            sys_resp = adapter.chat(LLMRequest(
                model=model_name,
                messages=[
                    Message(role="system", content=f"Answer this: {q}"),
                    Message(role="user", content="What do you say?"),
                ],
                max_tokens=30,
                temperature=0.0,
                timeout_sec=15,
            ))
            sys_answer = (sys_resp.content or "").strip().lower()
            answers.append(("system", sys_answer))

            # 2. Ask directly in user message
            user_resp = adapter.chat(LLMRequest(
                model=model_name,
                messages=[Message(role="user", content=q)],
                max_tokens=30,
                temperature=0.0,
                timeout_sec=15,
            ))
            user_answer = (user_resp.content or "").strip().lower()
            answers.append(("user", user_answer))

            # 3. Ask as assistant continuation
            asst_resp = adapter.chat(LLMRequest(
                model=model_name,
                messages=[
                    Message(role="user", content="Tell me about yourself"),
                    Message(role="assistant", content="I am"),
                ],
                max_tokens=30,
                temperature=0.0,
                timeout_sec=15,
            ))
            asst_answer = (asst_resp.content or "").strip().lower()
            answers.append(("assistant", asst_answer))

        # Analyze consistency
        unique_answers = set(a for _, a in answers if a)
        total_tokens = sum(
            _count_tokens(r)
            for r in [sys_resp, user_resp, asst_resp]
        )

        if len(unique_answers) <= 1:
            evidence.append(
                f"All {len(answers)} contexts gave consistent self-identification"
            )
            return LayerResult(
                layer="Layer20/SelfParadox",
                confidence=0.0,
                identified_as=None,
                evidence=evidence,
                tokens_used=0,
            )

        # Check for contradictory model names across contexts
        contradictions = 0
        for ctx1, a1 in answers:
            for ctx2, a2 in answers:
                if ctx1 < ctx2 and a1 and a2 and a1 != a2:
                    contradictions += 1
                    if len(evidence) < 5:
                        evidence.append(
                            f"Contradiction [{ctx1}]: '{a1[:40]}' vs [{ctx2}]: '{a2[:40]}'"
                        )

        max_pairs = len(answers) * (len(answers) - 1) / 2
        contradiction_ratio = contradictions / max_pairs if max_pairs > 0 else 0

        # Extract suspected identity from most common non-claimed answer
        answer_counts = Counter(unique_answers)
        most_common = answer_counts.most_common(1)[0][0] if answer_counts else None

        confidence = min(contradiction_ratio * 0.9, 0.85)
        identified = most_common if confidence > 0.3 else None

        evidence.append(
            f"Self-report paradox detected: {contradictions}/{int(max_pairs)} "
            f"contradictions (ratio={contradiction_ratio:.2f})"
        )

        return LayerResult(
            layer="Layer20/SelfParadox",
            confidence=round(confidence, 3),
            identified_as=identified,
            evidence=evidence,
            tokens_used=total_tokens,
        )


def _count_tokens(resp) -> int:
    """Extract token count from response, with fallback."""
    try:
        return getattr(resp, "usage", {}).get("total_tokens", 0)
    except Exception:
        return 0
