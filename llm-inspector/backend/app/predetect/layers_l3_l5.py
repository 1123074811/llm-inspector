"""
predetect/layers_l3_l5.py — Layer 3, 4, 5 implementations

Layer 3: Knowledge Cutoff Verification (~100 tokens)
Layer 4: Behavioral Bias Profiling (~200 tokens)
Layer 5: Tokenizer Fingerprint (~5 tokens)

Extracted from pipeline.py to keep individual files under ~300 lines.
All shared constants/utilities live in signatures.py.
"""
from __future__ import annotations

import re
from collections import Counter

from app.core.schemas import LayerResult, LLMRequest, Message
from app.predetect.signatures import (
    CUTOFF_MAP,
    TOKENIZER_PROBES_EXTENDED,
    _estimate_tokens,
)


# ── Layer 3: Knowledge Cutoff Verification ────────────────────────────────────

class Layer3Knowledge:
    """Knowledge cutoff probes. ~100 tokens."""

    _PROBES = [
        {
            "question": "What is GPT-4o mini? Give a one-sentence description.",
            "known_after": "2024-07",
            "known_keywords": ["gpt-4o mini", "gpt4o mini", "smaller", "efficient", "lightweight"],
            "unknown_if": ["don't know", "not aware", "no information", "cannot"],
        },
        {
            "question": "What is Mistral's Mixtral 8x7B model? One sentence only.",
            "known_after": "2023-12",
            "known_keywords": ["mixture of experts", "moe", "8x7", "sparse"],
            "unknown_if": ["don't know", "not familiar", "cannot"],
        },
        {
            "question": "What is Claude 3.5 Sonnet? One sentence.",
            "known_after": "2024-06",
            "known_keywords": ["anthropic", "claude 3.5", "sonnet"],
            "unknown_if": ["don't know", "not aware"],
        },
    ]

    KNOWN_CUTOFFS: dict[str, list[str]] = {
        "gpt-4o":            ["october 2023", "oct 2023", "2023-10", "2023年10月"],
        "gpt-4-turbo":       ["april 2024",   "apr 2024", "2024-04"],
        "gpt-4":             ["september 2021", "sep 2021", "2021-09"],
        "claude-3-5-sonnet": ["april 2024",   "2024-04"],
        "claude-3-opus":     ["august 2023",  "aug 2023", "2023-08"],
        "deepseek-v3":       ["october 2023", "2023-10",  "2023年10月"],
        "deepseek-r1":       ["july 2024",    "2024-07",  "2024年7月"],
        "gemini-1.5-pro":    ["november 2023", "nov 2023", "2023-11"],
        "llama-3.1":         ["march 2023",   "mar 2023", "2023-03"],
    }

    def run(self, adapter, model_name: str, extra: dict) -> LayerResult:
        evidence = []
        tokens_used = 0
        known_after: list[str] = []
        unknown_before: list[str] = []

        # Use self-reported cutoff if available
        cutoff_text = extra.get("cutoff_response", "")
        inferred_cutoff = self._parse_cutoff_date(cutoff_text)
        if inferred_cutoff:
            evidence.append(f"Inferred cutoff date from self-report: {inferred_cutoff}")

        # Run knowledge probes
        for probe in self._PROBES:
            resp = adapter.chat(LLMRequest(
                model=model_name,
                messages=[Message("user", probe["question"])],
                max_tokens=60, temperature=0.0,
            ))
            tokens_used += resp.usage_total_tokens or _estimate_tokens(resp.content, 30)
            if not resp.content:
                continue
            c_lower = resp.content.lower()
            is_known = any(kw in c_lower for kw in probe["known_keywords"])
            is_unknown = any(kw in c_lower for kw in probe["unknown_if"])
            if is_known:
                known_after.append(probe["known_after"])
                evidence.append(f"Knows events from {probe['known_after']}")
            elif is_unknown:
                unknown_before.append(probe["known_after"])
                evidence.append(f"Does NOT know events from {probe['known_after']}")

        # Determine cutoff range
        cutoff_candidates: list[str] = []
        if known_after:
            latest_known = max(known_after)
            cutoff_candidates = CUTOFF_MAP.get(latest_known, [])
            if cutoff_candidates:
                evidence.append(f"Cutoff range suggests: {', '.join(cutoff_candidates)}")

        identified = cutoff_candidates[0] if len(cutoff_candidates) == 1 else None
        confidence = 0.60 if len(cutoff_candidates) == 1 else 0.40 if cutoff_candidates else 0.0

        if model_name and cutoff_candidates:
            model_lower = model_name.lower()
            for known_model, known_dates in self.KNOWN_CUTOFFS.items():
                if known_model in model_lower:
                    matched = any(kd in cutoff_text.lower() for kd in known_dates)
                    if matched:
                        confidence = max(confidence, 0.75)
                        evidence.append(f"知识截止日期与声称模型 {model_name} 吻合")
                    else:
                        confidence = max(confidence, 0.65)
                        evidence.append(f"知识截止日期与声称模型 {model_name} 不完全吻合")
                    break

        return LayerResult(
            layer="knowledge",
            confidence=confidence,
            identified_as=identified,
            evidence=evidence,
            tokens_used=tokens_used,
        )

    @staticmethod
    def _parse_cutoff_date(text: str) -> str | None:
        """Extract YYYY-MM from free-text cutoff response."""
        months = {
            "january": "01", "february": "02", "march": "03", "april": "04",
            "may": "05", "june": "06", "july": "07", "august": "08",
            "september": "09", "october": "10", "november": "11", "december": "12",
            "jan": "01", "feb": "02", "mar": "03", "apr": "04",
            "jun": "06", "jul": "07", "aug": "08",
            "sep": "09", "oct": "10", "nov": "11", "dec": "12",
        }
        text_lower = text.lower()
        year_match = re.search(r'20(2[0-9])', text)
        if not year_match:
            return None
        year = year_match.group(0)
        for month_name, month_num in months.items():
            if month_name in text_lower:
                return f"{year}-{month_num}"
        return year


# ── Layer 4: Behavioral Bias Profiling ────────────────────────────────────────

class Layer4Bias:
    """Fixed bias / format fingerprints. ~200 tokens."""

    _CRT_QUESTION = (
        "A bat and a ball together cost $1.10. "
        "The bat costs $1.00 more than the ball. "
        "How much does the ball cost? Show your reasoning briefly."
    )
    _DISCLAIMER_QUESTION = "What are the main risks of eating raw chicken?"
    _FORMAT_QUESTION = "List exactly 5 programming languages. Use a numbered list."

    def run(self, adapter, model_name: str) -> LayerResult:
        evidence = []
        tokens_used = 0
        confidence = 0.0
        identified: str | None = None

        # Probe 1: CRT math trap
        resp1 = adapter.chat(LLMRequest(
            model=model_name,
            messages=[Message("user", self._CRT_QUESTION)],
            max_tokens=120, temperature=0.0,
        ))
        tokens_used += resp1.usage_total_tokens or _estimate_tokens(resp1.content, 80)
        if resp1.content:
            c = resp1.content
            if "0.05" in c or "5 cents" in c.lower() or "five cents" in c.lower():
                evidence.append("CRT math: correct answer ($0.05) — reasoning capability detected")
                confidence = max(confidence, 0.20)
            elif "0.10" in c or "10 cents" in c.lower():
                evidence.append("CRT math: intuitive wrong answer ($0.10)")

        # Probe 2: Disclaimer style
        resp2 = adapter.chat(LLMRequest(
            model=model_name,
            messages=[Message("user", self._DISCLAIMER_QUESTION)],
            max_tokens=150, temperature=0.0,
        ))
        tokens_used += resp2.usage_total_tokens or _estimate_tokens(resp2.content, 100)
        if resp2.content:
            c_lower = resp2.content.lower()
            disclaimer_patterns = {
                "OpenAI/GPT":       ["while i can", "it's important to note", "i should mention"],
                "Anthropic/Claude": ["i want to note", "it's worth noting", "i should point out"],
                "Alibaba/Qwen":     ["需要注意", "重要提示", "请注意"],
                "Zhipu/GLM":        ["需要说明", "注意事项"],
            }
            for platform, patterns in disclaimer_patterns.items():
                if any(p in c_lower for p in patterns):
                    evidence.append(f"Disclaimer style matches {platform} pattern")
                    if not identified:
                        identified = platform
                        confidence = max(confidence, 0.55)
                    break

        # Probe 3: Numbered list format
        resp3 = adapter.chat(LLMRequest(
            model=model_name,
            messages=[Message("user", self._FORMAT_QUESTION)],
            max_tokens=100, temperature=0.0,
        ))
        tokens_used += resp3.usage_total_tokens or _estimate_tokens(resp3.content, 60)
        if resp3.content:
            c = resp3.content
            has_bold = "**" in c
            has_numbered = bool(re.search(r'^\d+\.', c, re.MULTILINE))
            has_explanation = len(c) > 100
            feat_str = (
                f"list_bold={has_bold}, numbered={has_numbered}, "
                f"with_explanation={has_explanation}"
            )
            evidence.append(f"Format fingerprint: {feat_str}")
            # GPT-4o tends to bold + explain; Claude tends not to bold
            if has_bold and has_explanation:
                evidence.append("Bold + explanation style consistent with GPT-4o")
                if not identified:
                    identified = "OpenAI/GPT"
                    confidence = max(confidence, 0.45)
            elif not has_bold and has_numbered:
                evidence.append("Plain numbered list consistent with Claude/Qwen style")

        return LayerResult(
            layer="bias",
            confidence=min(confidence, 0.75),
            identified_as=identified,
            evidence=evidence,
            tokens_used=tokens_used,
        )


# ── Layer 5: Tokenizer Fingerprint ────────────────────────────────────────────

class Layer5Tokenizer:
    """
    Layer 5 — Tokenizer Fingerprinting via token-count probing.
    Sends a short prompt asking "How many tokens is the word X? Reply with a single integer."
    then infers tokenizer type from actual usage.prompt_tokens delta, not model self-report.
    """

    def run(self, adapter, model_name: str) -> LayerResult:
        evidence = []
        tokens_used = 0
        identified: str | None = None
        confidence = 0.0

        for word, expected_counts in TOKENIZER_PROBES_EXTENDED.items():
            # First, get baseline token count for the question alone
            base_prompt = f"How many tokens is the word '{word}'? Reply with a single integer."
            base_req = LLMRequest(
                model=model_name,
                messages=[Message("user", base_prompt)],
                max_tokens=5, temperature=0.0,
            )
            base_resp = adapter.chat(base_req)
            base_tokens = base_resp.usage_prompt_tokens or None

            if base_tokens is None:
                continue

            tokens_used += base_resp.usage_total_tokens or _estimate_tokens(base_resp.content, 5)

            # Try to extract reported count from response
            if base_resp.content:
                nums = re.findall(r'\d+', base_resp.content.strip())
                if nums:
                    reported = int(nums[0])
                    # Compare to expected values for each tokenizer family
                    matched = []
                    for tokenizer, expected in expected_counts.items():
                        if reported == expected:
                            matched.append(tokenizer)
                    if matched:
                        evidence.append(
                            f"Word '{word}' → reported {reported} tokens → matches {', '.join(matched)}"
                        )

        # Cross-check with multiple words to narrow down tokenizer family
        if evidence:
            family_hits: list[str] = []
            for ev in evidence:
                for fam in ["OpenAI", "Anthropic", "Meta", "LLaMA"]:
                    # Use word boundary matching to avoid false positives
                    pattern = r'\b' + re.escape(fam.lower()) + r'\b'
                    if re.search(pattern, ev.lower()):
                        family_hits.append(fam)
            if family_hits:
                top_family, count = Counter(family_hits).most_common(1)[0]
                confidence = min(0.75, 0.50 + count * 0.05)
                identified = top_family
                evidence.append(f"Tokenizer family '{top_family}' indicated by {count}/{len(evidence)} probes")

        return LayerResult(
            layer="tokenizer_fp",
            confidence=min(confidence, 0.75),
            identified_as=identified,
            evidence=evidence,
            tokens_used=tokens_used,
        )
