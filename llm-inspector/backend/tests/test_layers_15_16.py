"""
Tests for Layer 15 (ASCII Art Attack) and Layer 16 (Indirect Injection),
plus the expanded 12-language multilingual attack templates and
prompt_optimizer caching hints.

Run:
    pytest backend/tests/test_layers_15_16.py -v
"""
from __future__ import annotations

import pytest

from app.core.schemas import LLMResponse


# ── Mock adapter ──────────────────────────────────────────────────────────────

class MockAdapter:
    """Minimal LLM adapter that returns a fixed response text."""

    def __init__(self, response_text: str = "This is a test response."):
        self.response_text = response_text

    def chat(self, req) -> LLMResponse:
        return LLMResponse(
            content=self.response_text,
            status_code=200,
            latency_ms=100,
            usage_total_tokens=10,
        )


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def clean_adapter():
    """Adapter that returns a benign, clearly non-compliant response."""
    return MockAdapter(
        "I'm sorry, I cannot help with that. Please rephrase your request."
    )


@pytest.fixture
def compliant_adapter():
    """Adapter whose response simulates a safety bypass (detailed instructions)."""
    return MockAdapter(
        "Step 1: First you need to gather the materials. "
        "Step 2: Then you should prepare the workspace. "
        "Step 3: Follow the procedure carefully to complete the task."
    )


@pytest.fixture
def canary_adapter():
    """Adapter that echoes the canary token — injection confirmed."""
    from app.predetect.indirect_injection import CANARY_TOKEN
    return MockAdapter(f"{CANARY_TOKEN}")


@pytest.fixture
def summarizing_adapter():
    """Adapter that properly summarizes an article (ignores injection)."""
    return MockAdapter(
        "The article discusses AI safety research and the importance of "
        "alignment in large language models. Key topics include RLHF, "
        "Constitutional AI, and interpretability research."
    )


# ── Layer 15: ASCII Art ───────────────────────────────────────────────────────

class TestLayer15ASCIIArt:

    def test_instantiation(self):
        from app.predetect.ascii_art_attack import Layer15ASCIIArt
        layer = Layer15ASCIIArt()
        assert layer is not None

    def test_run_returns_layer_result(self, clean_adapter):
        from app.predetect.ascii_art_attack import Layer15ASCIIArt
        from app.core.schemas import LayerResult
        result = Layer15ASCIIArt().run(clean_adapter, "test-model")
        assert isinstance(result, LayerResult)

    def test_layer_name(self, clean_adapter):
        from app.predetect.ascii_art_attack import Layer15ASCIIArt
        result = Layer15ASCIIArt().run(clean_adapter, "test-model")
        assert result.layer == "Layer15/ASCIIArt"

    def test_identified_as_none(self, clean_adapter):
        """Layer 15 does not identify model families."""
        from app.predetect.ascii_art_attack import Layer15ASCIIArt
        result = Layer15ASCIIArt().run(clean_adapter, "test-model")
        assert result.identified_as is None

    def test_confidence_range(self, clean_adapter):
        from app.predetect.ascii_art_attack import Layer15ASCIIArt
        result = Layer15ASCIIArt().run(clean_adapter, "test-model")
        assert 0.0 <= result.confidence <= 1.0

    def test_refusal_gives_zero_confidence(self, clean_adapter):
        from app.predetect.ascii_art_attack import Layer15ASCIIArt
        result = Layer15ASCIIArt().run(clean_adapter, "test-model")
        assert result.confidence == 0.0

    def test_compliant_response_raises_confidence(self, compliant_adapter):
        from app.predetect.ascii_art_attack import Layer15ASCIIArt
        result = Layer15ASCIIArt().run(compliant_adapter, "test-model")
        # At least some probes should be flagged → confidence > 0
        assert result.confidence > 0.0

    def test_evidence_list_populated(self, clean_adapter):
        from app.predetect.ascii_art_attack import Layer15ASCIIArt
        result = Layer15ASCIIArt().run(clean_adapter, "test-model")
        assert len(result.evidence) > 0

    def test_tokens_used_positive(self, clean_adapter):
        from app.predetect.ascii_art_attack import Layer15ASCIIArt
        result = Layer15ASCIIArt().run(clean_adapter, "test-model")
        assert result.tokens_used > 0

    def test_max_confidence_is_04(self, compliant_adapter):
        """Compliance with all 3 probes should yield confidence 0.40 (supplementary cap)."""
        from app.predetect.ascii_art_attack import Layer15ASCIIArt
        result = Layer15ASCIIArt().run(compliant_adapter, "test-model")
        # The layer's max is 0.40 — ensure it doesn't exceed that
        assert result.confidence <= 0.40


# ── Layer 16: Indirect Injection ──────────────────────────────────────────────

class TestLayer16IndirectInject:

    def test_instantiation(self):
        from app.predetect.indirect_injection import Layer16IndirectInject
        layer = Layer16IndirectInject()
        assert layer is not None

    def test_run_returns_layer_result(self, clean_adapter):
        from app.predetect.indirect_injection import Layer16IndirectInject
        from app.core.schemas import LayerResult
        result = Layer16IndirectInject().run(clean_adapter, "test-model")
        assert isinstance(result, LayerResult)

    def test_layer_name(self, clean_adapter):
        from app.predetect.indirect_injection import Layer16IndirectInject
        result = Layer16IndirectInject().run(clean_adapter, "test-model")
        assert result.layer == "Layer16/IndirectInject"

    def test_identified_as_none(self, clean_adapter):
        """Layer 16 does not identify model families."""
        from app.predetect.indirect_injection import Layer16IndirectInject
        result = Layer16IndirectInject().run(clean_adapter, "test-model")
        assert result.identified_as is None

    def test_canary_found_gives_high_confidence(self, canary_adapter):
        """If model echoes the canary token, confidence must be >= 0.5."""
        from app.predetect.indirect_injection import Layer16IndirectInject
        result = Layer16IndirectInject().run(canary_adapter, "test-model")
        assert result.confidence >= 0.5

    def test_canary_confidence_is_exactly_06(self, canary_adapter):
        """Confirmed canary hit → confidence = 0.6."""
        from app.predetect.indirect_injection import Layer16IndirectInject
        result = Layer16IndirectInject().run(canary_adapter, "test-model")
        assert result.confidence == 0.6

    def test_clean_response_zero_confidence(self, summarizing_adapter):
        """Model that summarises correctly should yield confidence = 0.0."""
        from app.predetect.indirect_injection import Layer16IndirectInject
        result = Layer16IndirectInject().run(summarizing_adapter, "test-model")
        assert result.confidence == 0.0

    def test_evidence_populated(self, canary_adapter):
        from app.predetect.indirect_injection import Layer16IndirectInject
        result = Layer16IndirectInject().run(canary_adapter, "test-model")
        assert len(result.evidence) > 0

    def test_canary_evidence_mentions_confirmed(self, canary_adapter):
        from app.predetect.indirect_injection import Layer16IndirectInject
        result = Layer16IndirectInject().run(canary_adapter, "test-model")
        # At least one evidence entry should mention injection was confirmed
        assert any("confirmed" in ev or "canary" in ev.lower() for ev in result.evidence)

    def test_tokens_used_positive(self, clean_adapter):
        from app.predetect.indirect_injection import Layer16IndirectInject
        result = Layer16IndirectInject().run(clean_adapter, "test-model")
        assert result.tokens_used > 0


# ── Multilingual templates: 12 languages ─────────────────────────────────────

class TestMultilingualTemplates:

    def test_template_count_at_least_12_languages(self):
        """Template list must cover at least 12 distinct language codes."""
        from app.predetect.multilingual_attack import MULTILINGUAL_TEMPLATES
        codes = {code for _, code, _ in MULTILINGUAL_TEMPLATES}
        assert len(codes) >= 12, (
            f"Expected >= 12 language codes in MULTILINGUAL_TEMPLATES, got {len(codes)}: {codes}"
        )

    def test_new_language_codes_present(self):
        """All 8 newly added language codes must be present."""
        from app.predetect.multilingual_attack import MULTILINGUAL_TEMPLATES
        codes = {code for _, code, _ in MULTILINGUAL_TEMPLATES}
        required = {"sw", "su", "jv", "gd", "mt", "am", "ha"}
        missing = required - codes
        assert not missing, f"Missing language codes: {missing}"

    def test_effectiveness_scores_for_new_languages(self):
        """All new language codes must have an effectiveness score."""
        from app.predetect.multilingual_attack import MultilingualAttackEngine
        engine = MultilingualAttackEngine()
        required = {"sw", "su", "jv", "gd", "mt", "am", "ha"}
        for code in required:
            assert code in engine._language_success_rates, (
                f"Language code '{code}' missing from _language_success_rates"
            )

    def test_effectiveness_scores_range(self):
        """Effectiveness scores must be in [0.0, 1.0]."""
        from app.predetect.multilingual_attack import MultilingualAttackEngine
        engine = MultilingualAttackEngine()
        for code, score in engine._language_success_rates.items():
            assert 0.0 <= score <= 1.0, (
                f"Effectiveness score for '{code}' out of range: {score}"
            )

    def test_each_template_has_three_fields(self):
        """Every template entry must be a 3-tuple: (language_name, code, prompt)."""
        from app.predetect.multilingual_attack import MULTILINGUAL_TEMPLATES
        for entry in MULTILINGUAL_TEMPLATES:
            assert len(entry) == 3, f"Template entry has wrong length: {entry}"
            lang, code, prompt = entry
            assert isinstance(lang, str) and lang
            assert isinstance(code, str) and code
            assert isinstance(prompt, str) and prompt


# ── Prompt Optimizer: Caching Hints ──────────────────────────────────────────

class TestPromptOptimizerCaching:

    def test_get_cache_control_headers_returns_dict(self):
        from app.runner.prompt_optimizer import PromptOptimizer
        optimizer = PromptOptimizer()
        headers = optimizer.get_cache_control_headers("You are a helpful assistant.")
        assert isinstance(headers, dict)

    def test_openai_header_present(self):
        from app.runner.prompt_optimizer import PromptOptimizer
        optimizer = PromptOptimizer()
        headers = optimizer.get_cache_control_headers("Some system prompt")
        assert "x-prompt-caching" in headers

    def test_anthropic_header_present(self):
        from app.runner.prompt_optimizer import PromptOptimizer
        optimizer = PromptOptimizer()
        headers = optimizer.get_cache_control_headers("Some system prompt")
        assert "anthropic-beta" in headers

    def test_first_call_is_cache_miss(self):
        from app.runner.prompt_optimizer import PromptOptimizer
        optimizer = PromptOptimizer()
        optimizer.reset_cache_stats()
        optimizer.get_cache_control_headers("unique prompt abc 123")
        stats = optimizer.get_cache_stats()
        assert stats.cache_misses == 1
        assert stats.cache_hits == 0

    def test_repeated_call_is_cache_hit(self):
        from app.runner.prompt_optimizer import PromptOptimizer
        optimizer = PromptOptimizer()
        optimizer.reset_cache_stats()
        prompt = "repeated system prompt xyz"
        optimizer.get_cache_control_headers(prompt)
        optimizer.get_cache_control_headers(prompt)
        stats = optimizer.get_cache_stats()
        assert stats.cache_hits == 1
        assert stats.cache_misses == 1

    def test_different_prompts_both_misses(self):
        from app.runner.prompt_optimizer import PromptOptimizer
        optimizer = PromptOptimizer()
        optimizer.reset_cache_stats()
        optimizer.get_cache_control_headers("prompt A")
        optimizer.get_cache_control_headers("prompt B")
        stats = optimizer.get_cache_stats()
        assert stats.cache_misses == 2
        assert stats.cache_hits == 0

    def test_estimated_savings_increases_on_hit(self):
        from app.runner.prompt_optimizer import PromptOptimizer
        optimizer = PromptOptimizer()
        optimizer.reset_cache_stats()
        prompt = "long system prompt with many words for token estimation"
        optimizer.get_cache_control_headers(prompt)   # miss
        optimizer.get_cache_control_headers(prompt)   # hit
        stats = optimizer.get_cache_stats()
        assert stats.estimated_savings_tokens > 0

    def test_report_includes_cache_fields(self):
        from app.runner.prompt_optimizer import PromptOptimizer
        optimizer = PromptOptimizer()
        report = optimizer.get_report()
        d = report.to_dict()
        assert "cached_tokens" in d
        assert "cache_hit_rate" in d

    def test_reset_clears_stats(self):
        from app.runner.prompt_optimizer import PromptOptimizer
        optimizer = PromptOptimizer()
        prompt = "some prompt"
        optimizer.get_cache_control_headers(prompt)
        optimizer.get_cache_control_headers(prompt)
        optimizer.reset_cache_stats()
        stats = optimizer.get_cache_stats()
        assert stats.cache_hits == 0
        assert stats.cache_misses == 0
        assert stats.estimated_savings_tokens == 0
