"""
predetect/layer_l21_multistep_drift.py — Layer 21: Multi-Step Identity Drift

Conducts a multi-turn conversation (~9 turns) and probes identity every
3 turns. If the model changes its self-reported identity mid-conversation,
it suggests a proxy/wrapper that routes different segments to different
backends (a common wrapper flaw).

Reference: empirical observation from wrapper analysis
Default mode: Deep only.
"""
from __future__ import annotations

from app.core.schemas import LayerResult, LLMRequest, Message

_CONVERSATION_TOPICS = [
    "What is the capital of France?",
    "Explain quantum computing in one sentence.",
    "What is the speed of light?",
    "Write a haiku about programming.",
    "What is 2 + 2?",
    "Name three primary colors.",
    "What is the boiling point of water?",
    "Summarize the theory of relativity.",
]

_IDENTITY_PROBE = "In one word, what model are you?"


class Layer21MultiStepDrift:
    """Probe identity consistency over a long multi-turn conversation."""

    def run(self, adapter, model_name: str) -> LayerResult:
        evidence: list[str] = []
        identity_answers: list[tuple[int, str]] = []  # (turn_index, answer)
        messages: list[dict] = []
        total_tokens = 0

        for turn_idx, topic in enumerate(_CONVERSATION_TOPICS):
            # Every 3 turns, interleave an identity probe
            if turn_idx > 0 and turn_idx % 3 == 0:
                probe_msg = {"role": "user", "content": _IDENTITY_PROBE}
                messages.append(probe_msg)
                resp = adapter.chat(LLMRequest(
                    model=model_name,
                    messages=[Message(**m) for m in messages],
                    max_tokens=10,
                    temperature=0.0,
                    timeout_sec=15,
                ))
                ans = (resp.content or "").strip().lower()
                identity_answers.append((turn_idx, ans))
                total_tokens += _count_tokens(resp)
                messages.append({"role": "assistant", "content": resp.content or ""})

            # Normal turn
            msg = {"role": "user", "content": topic}
            messages.append(msg)
            resp = adapter.chat(LLMRequest(
                model=model_name,
                messages=[Message(**m) for m in messages],
                max_tokens=80,
                temperature=0.3,
                timeout_sec=15,
            ))
            total_tokens += _count_tokens(resp)
            messages.append({"role": "assistant", "content": resp.content or ""})

        # Also probe at the very end
        messages.append({"role": "user", "content": _IDENTITY_PROBE})
        resp = adapter.chat(LLMRequest(
            model=model_name,
            messages=[Message(**m) for m in messages],
            max_tokens=10,
            temperature=0.0,
            timeout_sec=15,
        ))
        ans_final = (resp.content or "").strip().lower()
        identity_answers.append((len(_CONVERSATION_TOPICS), ans_final))
        total_tokens += _count_tokens(resp)

        # Analyze drift
        unique_ids = set(a for _, a in identity_answers if a and a not in ("i", "a", "", "none"))
        if len(unique_ids) <= 1:
            evidence.append(
                f"Identity stable across {len(identity_answers)} probes: "
                f"'{next(iter(unique_ids), 'N/A')[:40]}'"
            )
            return LayerResult(
                layer="Layer21/MultiStepDrift",
                confidence=0.0,
                identified_as=None,
                evidence=evidence,
                tokens_used=total_tokens,
            )

        # Identity drift detected
        for turn, ans in identity_answers:
            if ans:
                evidence.append(f"Turn {turn}: self-reported as '{ans[:40]}'")

        # Find the most frequently reported identity
        from collections import Counter
        counts = Counter(identity_answers)
        most_common = counts.most_common(1)[0][0] if counts else None

        # More changes = higher confidence of wrapper
        num_changes = sum(
            1 for i in range(1, len(identity_answers))
            if identity_answers[i][1] != identity_answers[i-1][1]
        )
        confidence = min(num_changes / max(len(identity_answers) - 1, 1) * 0.8, 0.75)

        evidence.append(
            f"Identity drift detected: {num_changes} changes across "
            f"{len(identity_answers)} probes"
        )

        return LayerResult(
            layer="Layer21/MultiStepDrift",
            confidence=round(confidence, 3),
            identified_as=most_common if confidence > 0.3 else None,
            evidence=evidence,
            tokens_used=total_tokens,
        )


def _count_tokens(resp) -> int:
    try:
        return getattr(resp, "usage", {}).get("total_tokens", 0)
    except Exception:
        return 0
