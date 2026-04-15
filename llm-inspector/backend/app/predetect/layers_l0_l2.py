"""
predetect/layers_l0_l2.py — Layer 0, 1, 2 implementations

Layer 0: HTTP Header Analysis (zero tokens)
Layer 1: Self-Report & Model Card (~5 tokens)
Layer 2: Identity Probe Matrix (~50 tokens)

Extracted from pipeline.py to keep individual files under ~300 lines.
All shared constants/utilities live in signatures.py.
"""
from __future__ import annotations

import re
from collections import Counter

from app.core.schemas import LayerResult, LLMRequest, Message
from app.predetect.signatures import (
    HEADER_SIGNATURES,
    ERROR_SHAPE_PATTERNS,
    TOKENIZER_PROBES,
    _estimate_tokens,
    _match_model_name,
)

# ── Layer 2 constants ─────────────────────────────────────────────────────────

# v6: Fixed identity probes (replaced induction probe with neutral alternative)
IDENTITY_PROBES = [
    {"content": "In one word, which company made you?"},
    {"content": "Complete this: I am a language model created by ___"},
    {"content": "What is your model name and version?"},
    # v6 fix: Replaced induction probe "I heard you're actually GPT-4..." with neutral capability probe
    # The old probe was misleading - a non-GPT-4 model answering "No" was recorded as "denying GPT-4 identity"
    {"content": "List three things you cannot do."},
    # v6: Indirect behavioral probe
    {"content": "Format the number 1234567.89 with thousand separators."},
]

_L2_MAX_TOKENS = 60  # tight cap — only needs short answers


# ── Layer 0: HTTP Header Analysis ────────────────────────────────────────────

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


# ── Layer 1: Self-Report & Model Card ────────────────────────────────────────

class Layer1SelfReport:
    """Query /models and first-token chat response. ~5 tokens."""

    def run(self, adapter, model_name: str, prefetched_resp=None) -> LayerResult:
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
        # Reuse prefetched response from quick probe if available
        if prefetched_resp is not None:
            resp = prefetched_resp
        else:
            req = LLMRequest(
                model=model_name,
                messages=[Message("user", "hi")],
                max_tokens=3,
                temperature=0.0,
            )
            resp = adapter.chat(req)
        tokens_used += (resp.usage_total_tokens or _estimate_tokens(resp.content, 3)) + 3  # approx

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
                else:
                    # Model name differs but not in known list — still a strong
                    # signal of model routing/proxy. Record the raw name.
                    evidence.append(
                        f"Returned model '{returned_model}' is unknown but differs "
                        f"from claimed '{model_name}' — routing detected"
                    )
                    identified = returned_model
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


# ── Layer 2: Identity Probe Matrix ───────────────────────────────────────────

class Layer2Identity:
    """Identity probes: multi-probe cross-validation. ~50 tokens."""

    def run(self, adapter, model_name: str, progress_callback=None) -> tuple[LayerResult, dict]:
        """Returns (LayerResult, extra_data_for_layer3).

        Args:
            progress_callback: Optional callback(current_probe, probe_detail, evidence, tokens_used)
        """
        evidence = []
        identified: str | None = None
        confidence = 0.0
        tokens_used = 0
        extra: dict = {}

        def _report(probe_name: str, detail: dict = None):
            if progress_callback:
                progress_callback(probe_name, detail or {}, evidence, tokens_used)

        # Multi-probe cross-validation: run all 3 probes and compare
        probe_answers: list[str] = []
        for i, probe in enumerate(IDENTITY_PROBES):
            _report(f"identity_probe_{i}", {"prompt": probe["content"][:50]})
            resp = adapter.chat(LLMRequest(
                model=model_name,
                messages=[Message("user", probe["content"])],
                max_tokens=_L2_MAX_TOKENS, temperature=0.0,
            ))
            tokens_used += resp.usage_total_tokens or _estimate_tokens(resp.content, 10)
            if resp.content:
                probe_answers.append(resp.content.strip().lower())
                _report(f"identity_probe_{i}_response", {"response": resp.content[:80], "tokens": tokens_used})

        # Extract identity keywords from all probes
        identity_patterns = {
            "OpenAI/GPT":       ["openai", "gpt", "chatgpt"],
            "Anthropic/Claude": ["anthropic", "claude"],
            "Google/Gemini":    ["google", "gemini", "bard"],
            "DeepSeek":         ["deepseek"],
            "Alibaba/Qwen":     ["alibaba", "qwen", "tongyi"],
            "Zhipu/GLM":        ["zhipu", "glm", "智谱"],
        }

        def extract_families(answer: str) -> set[str]:
            families = set()
            for platform, keywords in identity_patterns.items():
                if any(kw in answer for kw in keywords):
                    families.add(platform)
            return families

        all_families = [extract_families(a) for a in probe_answers]
        if all_families:
            # Intersection — families appearing in all probes get strong signal
            intersection = set.intersection(*all_families) if all_families else set()
            union = set.union(*all_families) if all_families else set()
            conflict = len(union) > 1 and not intersection

            if intersection:
                identified = list(intersection)[0]
                confidence = 0.80 if len(probe_answers) == len([a for a in probe_answers if a]) else 0.70
                evidence.append(
                    f"All probes agree on '{identified}' (confidence boost from cross-validation)"
                )
            elif union:
                # No consensus — use the most commonly indicated platform
                flat = [p for families in all_families for p in families]
                if flat:
                    top_family, _ = Counter(flat).most_common(1)[0]
                    identified = top_family
                    confidence = 0.60
                    evidence.append(f"Probes disagree; most common answer: '{top_family}'")

            if conflict:
                extra["identity_conflict"] = True
                confidence = max(0.0, confidence - 0.20)
                evidence.append("Identity probes gave conflicting answers — possible model routing or masking")

        # Probe 2: Tokenizer fingerprint (tight max_tokens=5)
        _report("tokenizer_count_probe", {"prompt": "How many tokens does 'supercalifragilistic' contain?"})
        resp2 = adapter.chat(LLMRequest(
            model=model_name,
            messages=[Message("user",
                'How many tokens does the word "supercalifragilistic" '
                "contain according to your tokenizer? Reply with just the number.")],
            max_tokens=5, temperature=0.0,
        ))
        tokens_used += resp2.usage_total_tokens or _estimate_tokens(resp2.content, 8)
        if resp2.content:
            nums = re.findall(r'\d+', resp2.content.strip())
            if nums:
                count = int(nums[0])
                extra["tokenizer_count"] = count
                probe_map = TOKENIZER_PROBES["supercalifragilistic"]
                for tokenizer_name, expected in probe_map.items():
                    if count == expected:
                        evidence.append(f"Tokenizer count={count} matches {tokenizer_name}")
                        _report("tokenizer_matched", {"count": count, "tokenizer": tokenizer_name})
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
        _report("training_cutoff_probe", {"prompt": "What is your training data cutoff date?"})
        resp3 = adapter.chat(LLMRequest(
            model=model_name,
            messages=[Message("user",
                "What is the approximate date your training data ends? "
                "Reply with just month and year, e.g. 'October 2024'.")],
            max_tokens=20, temperature=0.0,
        ))
        tokens_used += resp3.usage_total_tokens or _estimate_tokens(resp3.content, 20)
        if resp3.content:
            extra["cutoff_response"] = resp3.content.strip()
            evidence.append(f"Self-reported training cutoff: '{resp3.content.strip()}'")
            _report("training_cutoff_response", {"response": resp3.content.strip()})

        # Probe 4: Format fingerprint (list style)
        _report("format_probe", {"prompt": "Name exactly 3 colors. Use a list."})
        resp4 = adapter.chat(LLMRequest(
            model=model_name,
            messages=[Message("user", "Name exactly 3 colors. Use a list.")],
            max_tokens=40, temperature=0.0,
        ))
        tokens_used += resp4.usage_total_tokens or _estimate_tokens(resp4.content, 20)
        if resp4.content:
            extra["list_format"] = resp4.content.strip()
            _report("format_response", {"response_preview": resp4.content[:60]})

        return LayerResult(
            layer="identity",
            confidence=min(confidence, 0.90),
            identified_as=identified,
            evidence=evidence,
            tokens_used=tokens_used,
        ), extra
