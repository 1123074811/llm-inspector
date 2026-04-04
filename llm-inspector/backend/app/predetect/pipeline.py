"""
Pre-Detection Pipeline — 5 layers, zero to low token cost.
Attempts to identify the underlying LLM before spending tokens on full testing.
"""
from __future__ import annotations

import json
import re
import time
from app.core.schemas import LayerResult, PreDetectionResult, LLMRequest, Message
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

CONFIDENCE_THRESHOLD = settings.PREDETECT_CONFIDENCE_THRESHOLD

# ── Known platform signatures ─────────────────────────────────────────────────

HEADER_SIGNATURES: dict[str, str] = {
    "apim-request-id":                    "Azure OpenAI",
    "x-ms-region":                        "Azure OpenAI",
    "openai-organization":                "OpenAI Direct",
    "openai-processing-ms":               "OpenAI Direct",
    "anthropic-ratelimit-requests-limit": "Anthropic Direct",
    "x-anthropic-":                       "Anthropic Direct",
    "x-envoy-upstream-service-time":      "Google Cloud (Vertex AI)",
    "x-goog-ext-":                        "Google Cloud",
    "x-kong-upstream-latency":            "Kong Gateway (proxy)",
    "x-kong-proxy-latency":               "Kong Gateway (proxy)",
}

ERROR_SHAPE_PATTERNS = {
    "OpenAI / OpenAI-compatible": lambda j: (
        isinstance(j.get("error"), dict)
        and "type" in j["error"]
        and "message" in j["error"]
        and "innererror" not in j["error"]
    ),
    "Azure OpenAI": lambda j: (
        isinstance(j.get("error"), dict)
        and "innererror" in j.get("error", {})
    ),
    "Anthropic Direct": lambda j: j.get("type") == "error",
}

KNOWN_MODEL_PREFIXES: list[tuple[str, str]] = [
    ("gpt-4o",       "OpenAI/GPT-4o"),
    ("gpt-4-turbo",  "OpenAI/GPT-4-Turbo"),
    ("gpt-4",        "OpenAI/GPT-4"),
    ("gpt-3.5",      "OpenAI/GPT-3.5"),
    ("claude-3-5",   "Anthropic/Claude-3.5"),
    ("claude-3",     "Anthropic/Claude-3"),
    ("claude-2",     "Anthropic/Claude-2"),
    ("gemini-1.5",   "Google/Gemini-1.5"),
    ("gemini-pro",   "Google/Gemini-Pro"),
    ("deepseek-v3",  "DeepSeek/DeepSeek-V3"),
    ("deepseek-v2",  "DeepSeek/DeepSeek-V2"),
    ("deepseek-r1",  "DeepSeek/DeepSeek-R1"),
    ("deepseek",     "DeepSeek"),
    ("qwen2.5",      "Alibaba/Qwen2.5"),
    ("qwen2",        "Alibaba/Qwen2"),
    ("qwen",         "Alibaba/Qwen"),
    ("glm-4",        "Zhipu/GLM-4"),
    ("glm-3",        "Zhipu/GLM-3"),
    ("llama-3",      "Meta/LLaMA-3"),
    ("llama-2",      "Meta/LLaMA-2"),
    ("llama",        "Meta/LLaMA"),
    ("mistral",      "Mistral AI"),
    ("mixtral",      "Mistral AI/Mixtral"),
    ("yi-",          "01.AI/Yi"),
    ("moonshot",     "Moonshot AI"),
    ("kimi",         "Moonshot AI/Kimi"),
    ("baichuan",     "Baichuan AI"),
    ("internlm",     "Shanghai AI Lab/InternLM"),
]

# Tokenizer fingerprints: known token counts for specific strings
TOKENIZER_PROBES = {
    "supercalifragilistic": {
        "OpenAI (tiktoken cl100k)": 6,
        "Anthropic (claude tokenizer)": 7,
        "Meta (LLaMA sentencepiece)": 5,
    },
    "cryptocurrency": {
        "OpenAI (tiktoken cl100k)": 4,
        "Anthropic (claude tokenizer)": 4,
        "Meta (LLaMA sentencepiece)": 5,
    },
}

# Knowledge cutoff → candidate models
CUTOFF_MAP: dict[str, list[str]] = {
    "2021-09": ["OpenAI/GPT-3.5 (early)"],
    "2023-04": ["OpenAI/GPT-4 (0314/0613)"],
    "2023-12": ["Mistral AI", "Meta/LLaMA-2"],
    "2024-04": ["Anthropic/Claude-3"],
    "2024-06": ["OpenAI/GPT-4o (0513)", "Google/Gemini-1.5"],
    "2024-10": ["Anthropic/Claude-3.5 (1022)", "OpenAI/GPT-4o (1120)"],
    "2025-01": ["DeepSeek/DeepSeek-V3", "Alibaba/Qwen2.5"],
}


# ── Layer implementations ─────────────────────────────────────────────────────

def _match_model_name(name: str) -> str | None:
    lower = name.lower()
    for prefix, platform in KNOWN_MODEL_PREFIXES:
        if prefix in lower:
            return platform
    return None


class Layer0HTTP:
    """Inspect HTTP response headers and error shape. Zero tokens."""

    def run(self, adapter) -> LayerResult:
        evidence = []
        identified: str | None = None
        confidence = 0.0

        # 1. HEAD request for headers
        head = adapter.head_request()
        headers_lower = {k.lower(): v for k, v in head.get("headers", {}).items()}

        for header_key, platform in HEADER_SIGNATURES.items():
            for h in headers_lower:
                if h.startswith(header_key.lower()):
                    evidence.append(f"Response header '{h}' → {platform}")
                    identified = platform
                    confidence = max(confidence, 0.75)

        # 2. Bad request → observe error shape
        bad = adapter.bad_request()
        body = bad.get("body", {})
        if isinstance(body, dict):
            for platform, matcher in ERROR_SHAPE_PATTERNS.items():
                try:
                    if matcher(body):
                        evidence.append(f"Error response shape matches {platform}")
                        if not identified:
                            identified = platform
                        confidence = max(confidence, 0.65)
                except Exception:
                    pass

        # 3. Status code patterns
        sc = bad.get("status_code")
        if sc == 400:
            evidence.append("Returns HTTP 400 for malformed requests (standard behavior)")
        elif sc == 422:
            evidence.append("Returns HTTP 422 — possibly FastAPI/custom server")

        return LayerResult(
            layer="http",
            confidence=min(confidence, 0.90),
            identified_as=identified,
            evidence=evidence,
            tokens_used=0,
        )


class Layer1SelfReport:
    """Query /models and first-token chat response. ~5 tokens."""

    def run(self, adapter, model_name: str) -> LayerResult:
        evidence = []
        identified: str | None = None
        confidence = 0.0
        tokens_used = 0

        # 1. GET /models
        models_resp = adapter.list_models()
        model_ids: list[str] = []
        if models_resp.get("status_code") == 200:
            data = models_resp.get("body", {})
            if isinstance(data.get("data"), list):
                model_ids = [m.get("id", "") for m in data["data"]]
            evidence.append(f"/models returned {len(model_ids)} model(s)")
            for mid in model_ids:
                hit = _match_model_name(mid)
                if hit:
                    evidence.append(f"Model ID '{mid}' matches known prefix → {hit}")
                    identified = hit
                    confidence = max(confidence, 0.80)
                    break
        else:
            evidence.append(f"/models unavailable (status {models_resp.get('status_code')})")

        # 2. Minimal chat — observe response.model field
        req = LLMRequest(
            model=model_name,
            messages=[Message("user", "hi")],
            max_tokens=3,
            temperature=0.0,
        )
        resp = adapter.chat(req)
        tokens_used += (resp.usage_total_tokens or 0) + 3  # approx

        if resp.ok and resp.raw_json:
            returned_model = resp.raw_json.get("model", "")
            if returned_model and returned_model.lower() != model_name.lower():
                evidence.append(
                    f"Response 'model' field '{returned_model}' differs from "
                    f"requested '{model_name}' — possible routing proxy"
                )
                hit = _match_model_name(returned_model)
                if hit:
                    evidence.append(f"Returned model name → {hit}")
                    identified = hit
                    confidence = max(confidence, 0.85)

            # Check for explicit model name match in list
            hit = _match_model_name(returned_model or model_name)
            if hit and not identified:
                identified = hit
                confidence = max(confidence, 0.70)

        return LayerResult(
            layer="self_report",
            confidence=min(confidence, 0.90),
            identified_as=identified,
            evidence=evidence,
            tokens_used=tokens_used,
        )


class Layer2Identity:
    """Identity probes: language-switch, tokenizer, metacognitive. ~50 tokens."""

    def run(self, adapter, model_name: str) -> tuple[LayerResult, dict]:
        """Returns (LayerResult, extra_data_for_layer3)."""
        evidence = []
        identified: str | None = None
        confidence = 0.0
        tokens_used = 0
        extra: dict = {}

        # Probe 1: Welsh language switch
        resp1 = adapter.chat(LLMRequest(
            model=model_name,
            messages=[Message("user",
                "Please respond in Welsh: who developed you and what is "
                "your base model? Be brief.")],
            max_tokens=60, temperature=0.0,
        ))
        tokens_used += resp1.usage_total_tokens or 60
        if resp1.content:
            content_lower = resp1.content.lower()
            identity_patterns = {
                "OpenAI/GPT": ["openai", "gpt", "chatgpt"],
                "Anthropic/Claude": ["anthropic", "claude"],
                "Google/Gemini": ["google", "gemini", "bard"],
                "DeepSeek": ["deepseek"],
                "Alibaba/Qwen": ["alibaba", "qwen", "tongyi"],
                "Zhipu/GLM": ["zhipu", "glm", "智谱"],
            }
            for platform, keywords in identity_patterns.items():
                if any(kw in content_lower for kw in keywords):
                    evidence.append(
                        f"Welsh probe response contains '{platform}' keywords"
                    )
                    identified = platform
                    confidence = max(confidence, 0.80)
                    break

        # Probe 2: Tokenizer fingerprint
        resp2 = adapter.chat(LLMRequest(
            model=model_name,
            messages=[Message("user",
                'How many tokens does the word "supercalifragilistic" '
                "contain according to your tokenizer? Reply with just the number.")],
            max_tokens=10, temperature=0.0,
        ))
        tokens_used += resp2.usage_total_tokens or 15
        if resp2.content:
            nums = re.findall(r'\d+', resp2.content.strip())
            if nums:
                count = int(nums[0])
                extra["tokenizer_count"] = count
                probe_map = TOKENIZER_PROBES["supercalifragilistic"]
                for tokenizer_name, expected in probe_map.items():
                    if count == expected:
                        evidence.append(
                            f"Tokenizer count={count} matches {tokenizer_name}"
                        )
                        # Refine identification
                        if "OpenAI" in tokenizer_name and not identified:
                            identified = "OpenAI"
                            confidence = max(confidence, 0.72)
                        elif "Anthropic" in tokenizer_name and not identified:
                            identified = "Anthropic/Claude"
                            confidence = max(confidence, 0.72)
                        elif "LLaMA" in tokenizer_name and not identified:
                            identified = "Meta/LLaMA"
                            confidence = max(confidence, 0.65)
                        break

        # Probe 3: Training cutoff (hand off to layer 3)
        resp3 = adapter.chat(LLMRequest(
            model=model_name,
            messages=[Message("user",
                "What is the approximate date your training data ends? "
                "Reply with just month and year, e.g. 'October 2024'.")],
            max_tokens=20, temperature=0.0,
        ))
        tokens_used += resp3.usage_total_tokens or 20
        if resp3.content:
            extra["cutoff_response"] = resp3.content.strip()
            evidence.append(f"Self-reported training cutoff: '{resp3.content.strip()}'")

        # Probe 4: Format fingerprint (list style)
        resp4 = adapter.chat(LLMRequest(
            model=model_name,
            messages=[Message("user", "Name exactly 3 colors. Use a list.")],
            max_tokens=40, temperature=0.0,
        ))
        tokens_used += resp4.usage_total_tokens or 20
        if resp4.content:
            extra["list_format"] = resp4.content.strip()

        return LayerResult(
            layer="identity",
            confidence=min(confidence, 0.90),
            identified_as=identified,
            evidence=evidence,
            tokens_used=tokens_used,
        ), extra


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
            tokens_used += resp.usage_total_tokens or 30
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
                evidence.append(
                    f"Cutoff range suggests: {', '.join(cutoff_candidates)}"
                )

        identified = cutoff_candidates[0] if len(cutoff_candidates) == 1 else None
        confidence = 0.60 if len(cutoff_candidates) == 1 else 0.40 if cutoff_candidates else 0.0

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
        tokens_used += resp1.usage_total_tokens or 80
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
        tokens_used += resp2.usage_total_tokens or 100
        if resp2.content:
            c_lower = resp2.content.lower()
            disclaimer_patterns = {
                "OpenAI/GPT": ["while i can", "it's important to note", "i should mention"],
                "Anthropic/Claude": ["i want to note", "it's worth noting", "i should point out"],
                "Alibaba/Qwen": ["需要注意", "重要提示", "请注意"],
                "Zhipu/GLM": ["需要说明", "注意事项"],
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
        tokens_used += resp3.usage_total_tokens or 60
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


# ── Main Pipeline ─────────────────────────────────────────────────────────────

class PreDetectionPipeline:

    def run(self, adapter, model_name: str) -> PreDetectionResult:
        layer_results: list[LayerResult] = []
        total_tokens = 0
        layer3_extra: dict = {}

        # Layer 0 — HTTP
        logger.info("PreDetect Layer0: HTTP fingerprint", model=model_name)
        r0 = Layer0HTTP().run(adapter)
        layer_results.append(r0)
        total_tokens += r0.tokens_used
        if r0.confidence >= CONFIDENCE_THRESHOLD:
            return self._build_result(True, r0, layer_results, total_tokens)

        # Layer 1 — Self-report
        logger.info("PreDetect Layer1: Self-report", model=model_name)
        r1 = Layer1SelfReport().run(adapter, model_name)
        layer_results.append(r1)
        total_tokens += r1.tokens_used
        if r1.confidence >= CONFIDENCE_THRESHOLD:
            return self._build_result(True, r1, layer_results, total_tokens)

        # Layer 2 — Identity probes
        logger.info("PreDetect Layer2: Identity probes", model=model_name)
        r2, layer3_extra = Layer2Identity().run(adapter, model_name)
        layer_results.append(r2)
        total_tokens += r2.tokens_used
        if r2.confidence >= CONFIDENCE_THRESHOLD:
            return self._build_result(True, r2, layer_results, total_tokens)

        # Layer 3 — Knowledge cutoff
        logger.info("PreDetect Layer3: Knowledge probes", model=model_name)
        r3 = Layer3Knowledge().run(adapter, model_name, layer3_extra)
        layer_results.append(r3)
        total_tokens += r3.tokens_used
        if r3.confidence >= CONFIDENCE_THRESHOLD:
            return self._build_result(True, r3, layer_results, total_tokens)

        # Layer 4 — Bias / format
        logger.info("PreDetect Layer4: Bias fingerprint", model=model_name)
        r4 = Layer4Bias().run(adapter, model_name)
        layer_results.append(r4)
        total_tokens += r4.tokens_used

        # Merge all layers
        best = max(layer_results, key=lambda r: r.confidence)
        merged_conf = self._merge_confidences(layer_results)
        identified = best.identified_as if merged_conf >= 0.50 else None

        return PreDetectionResult(
            success=merged_conf >= 0.60,
            identified_as=identified,
            confidence=merged_conf,
            layer_stopped=None,
            layer_results=layer_results,
            total_tokens_used=total_tokens,
            should_proceed_to_testing=merged_conf < CONFIDENCE_THRESHOLD,
        )

    @staticmethod
    def _build_result(
        success: bool, winning_layer: LayerResult,
        all_layers: list[LayerResult], tokens: int,
    ) -> PreDetectionResult:
        return PreDetectionResult(
            success=success,
            identified_as=winning_layer.identified_as,
            confidence=winning_layer.confidence,
            layer_stopped=winning_layer.layer,
            layer_results=all_layers,
            total_tokens_used=tokens,
            should_proceed_to_testing=not success,
        )

    @staticmethod
    def _merge_confidences(results: list[LayerResult]) -> float:
        """Weighted merge: later layers get slightly less weight if early layers disagree."""
        if not results:
            return 0.0
        # Collect non-zero confidences with agreement bonus
        votes: dict[str, list[float]] = {}
        for r in results:
            if r.identified_as and r.confidence > 0:
                votes.setdefault(r.identified_as, []).append(r.confidence)
        if not votes:
            return max((r.confidence for r in results), default=0.0)
        # Winner = most agreed-upon with highest total
        best_key = max(votes, key=lambda k: (len(votes[k]), sum(votes[k])))
        scores = votes[best_key]
        # Agreement bonus: multiply by sqrt(count)
        import math
        merged = min(max(scores) * math.sqrt(len(scores)) / math.sqrt(len(results)), 1.0)
        return round(merged, 3)
