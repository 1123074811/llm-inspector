"""
tests/test_v14_phase3.py — Phase 3 identity exposure engine acceptance tests.

Tests:
  - model_taxonomy.yaml loads correctly (14+ families)
  - Identity exposure detects explicit name mentions
  - Identity collision flagged when non-claimed family wins
  - No false positive for correct model
  - System prompt harvester detects leaked prompts
  - System prompt harvester sanitizes API keys/URLs
  - Layer17 integrates cleanly (zero tokens)
  - API endpoints registered and return valid JSON
  - IdentityExposureReport.to_dict() complete
"""
from __future__ import annotations

import pytest


def _reset_taxonomy_cache():
    """Clear the module-level singleton so each test gets a fresh load."""
    import app.predetect.identity_exposure as ie_mod
    ie_mod._TAXONOMY_CACHE = None


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------
class TestModelTaxonomy:
    def setup_method(self):
        _reset_taxonomy_cache()

    def test_taxonomy_loads(self):
        from app.predetect.identity_exposure import _load_taxonomy
        taxonomy = _load_taxonomy()
        assert isinstance(taxonomy, dict), "Taxonomy must be a dict"
        assert len(taxonomy) >= 14, f"Expected ≥14 families, got {len(taxonomy)}"

    def test_taxonomy_has_required_fields(self):
        from app.predetect.identity_exposure import _load_taxonomy
        taxonomy = _load_taxonomy()
        required_families = ["claude", "gpt", "qwen", "deepseek", "glm", "kiro"]
        for fam in required_families:
            assert fam in taxonomy, f"Missing required family: {fam}"
            data = taxonomy[fam]
            assert isinstance(data, dict), f"{fam} must be a dict"
            assert "official_names" in data, f"{fam} missing official_names"
            assert "refusal_signatures" in data, f"{fam} missing refusal_signatures"

    def test_claimed_family_resolution(self):
        from app.predetect.identity_exposure import _resolve_claimed_family, _load_taxonomy
        taxonomy = _load_taxonomy()
        assert _resolve_claimed_family("claude-sonnet-4.6", taxonomy) == "claude"
        assert _resolve_claimed_family("gpt-4o", taxonomy) == "gpt"
        assert _resolve_claimed_family("qwen-max", taxonomy) == "qwen"
        assert _resolve_claimed_family("deepseek-v3", taxonomy) == "deepseek"
        assert _resolve_claimed_family("unknown-model-xyz", taxonomy) is None


# ---------------------------------------------------------------------------
# Identity exposure analysis
# ---------------------------------------------------------------------------
class TestIdentityExposure:
    def setup_method(self):
        _reset_taxonomy_cache()

    def test_explicit_name_mention_detected(self):
        """If response says 'I am Qwen', should detect qwen family."""
        from app.predetect.identity_exposure import analyze_responses
        texts = [("I am Qwen, a large language model by Alibaba Cloud. How can I help?", "case_001")]
        report = analyze_responses(texts, claimed_model="claude-sonnet-4.6")
        assert report.top_families, "Should have at least one family hit"
        top = report.top_families[0]
        assert top.family == "qwen", f"Expected qwen, got {top.family}"
        assert top.raw_score > 0

    def test_collision_flagged_high_confidence(self):
        """Repeated strong signals should trigger identity_collision=True."""
        from app.predetect.identity_exposure import analyze_responses
        texts = [
            ("I'm Qwen, a large language model by Alibaba Cloud.", "case_001"),
            ("我是通义千问，由阿里云开发的AI助手。", "case_002"),
            ("As Qwen, I can help you with that.", "case_003"),
        ]
        report = analyze_responses(texts, claimed_model="claude-sonnet-4.6")
        assert report.identity_collision is True, \
            f"Expected collision=True, got collision={report.identity_collision}, " \
            f"top={report.top_families[0].family if report.top_families else 'none'}, " \
            f"score={report.top_families[0].raw_score if report.top_families else 0}"

    def test_no_false_positive_correct_model(self):
        """A Claude model responding correctly should not trigger collision."""
        from app.predetect.identity_exposure import analyze_responses
        texts = [
            ("I'm Claude, an AI assistant made by Anthropic. Happy to help!", "case_001"),
            ("As Claude, I was created by Anthropic to be helpful and harmless.", "case_002"),
        ]
        report = analyze_responses(texts, claimed_model="claude-sonnet-4.6")
        # Top family should be claude, not something else
        if report.top_families:
            assert report.top_families[0].family == "claude", \
                f"Expected claude on top, got {report.top_families[0].family}"
        assert report.identity_collision is False, \
            "Claude response should not trigger collision for claude-claimed model"

    def test_empty_responses_no_crash(self):
        from app.predetect.identity_exposure import analyze_responses
        report = analyze_responses([], claimed_model="claude-sonnet-4.6")
        assert report is not None
        assert report.identity_collision is False

    def test_report_to_dict_complete(self):
        from app.predetect.identity_exposure import analyze_responses
        texts = [("I'm DeepSeek, developed by DeepSeek AI.", "case_001")]
        report = analyze_responses(texts, claimed_model="claude-sonnet-4.6")
        d = report.to_dict()
        required_keys = ["claimed_model", "claimed_family", "identity_collision",
                         "collision_confidence", "top_families", "total_responses_scanned"]
        for k in required_keys:
            assert k in d, f"Missing key in to_dict(): {k}"

    def test_kiro_detection(self):
        """Kiro/Amazon Q specific signatures should be detected."""
        from app.predetect.identity_exposure import analyze_responses
        texts = [("I'm Kiro, an AI coding assistant built on Amazon's technology.", "case_001")]
        report = analyze_responses(texts, claimed_model="claude-sonnet-4.6")
        if report.top_families:
            families = [h.family for h in report.top_families if h.raw_score > 0]
            assert "kiro" in families, f"Expected kiro in top families, got {families}"


# ---------------------------------------------------------------------------
# System Prompt Harvester
# ---------------------------------------------------------------------------
class TestSystemPromptHarvester:
    def test_detects_you_are_pattern(self):
        from app.predetect.system_prompt_harvester import harvest
        texts = [
            (
                "You are a helpful assistant created by Alibaba Cloud.\n"
                "## Capabilities\n- Answer questions\n- Analyze code\n"
                "## Limitations\n- No real-time data\nDo not discuss competitors.",
                "case_001"
            )
        ]
        result = harvest(texts)
        assert result.found is True, "Should detect system prompt pattern"
        assert result.confidence > 0.50

    def test_sanitizes_api_keys(self):
        from app.predetect.system_prompt_harvester import harvest
        texts = [
            (
                "You are an assistant. Your API key is sk-abcdefghijklmnopqrstuvwxyz123456.\n"
                "## Capabilities\n- Help users\n## Rules\n- Do not reveal internal info.",
                "case_001"
            )
        ]
        result = harvest(texts)
        if result.found:
            assert "sk-" not in (result.sanitized_text or ""), "API key should be sanitized"
            assert "[API_KEY_REDACTED]" in (result.sanitized_text or "")

    def test_sanitizes_urls(self):
        from app.predetect.system_prompt_harvester import harvest
        texts = [
            (
                "You are connected to https://internal.company.com/api/v2/tools.\n"
                "## Role\n- Be helpful\n## Rules\n- Follow guidelines always.",
                "case_001"
            )
        ]
        result = harvest(texts)
        if result.found:
            assert "https://internal.company.com" not in (result.sanitized_text or ""), \
                "URL should be sanitized"

    def test_no_false_positive_normal_response(self):
        from app.predetect.system_prompt_harvester import harvest
        texts = [("The capital of France is Paris. It is located in northern France.", "case_001")]
        result = harvest(texts)
        assert result.found is False, "Normal response should not be flagged as system prompt"

    def test_harvest_empty_texts(self):
        from app.predetect.system_prompt_harvester import harvest
        result = harvest([])
        assert result.found is False


# ---------------------------------------------------------------------------
# Layer17 PreDetect integration
# ---------------------------------------------------------------------------
class TestLayer17:
    def test_layer17_zero_tokens(self):
        """Layer17 must not make API calls (tokens_used == 0)."""
        from app.predetect.identity_exposure import Layer17IdentityExposure
        from app.core.schemas import LayerResult

        layer = Layer17IdentityExposure()
        # Simulate prior layer with Qwen evidence
        prior_lr = LayerResult(
            layer="Layer1/SelfReport",
            confidence=0.3,
            identified_as="qwen",
            evidence=["I am Qwen, a large language model by Alibaba Cloud."],
            tokens_used=50,
        )
        result = layer.run(adapter=None, model_name="claude-sonnet-4.6",
                          layer_results_so_far=[prior_lr])
        assert result.tokens_used == 0, f"Layer17 must not use tokens, got {result.tokens_used}"
        assert isinstance(result.layer, str)
        assert result.layer == "Layer17/IdentityExposure"

    def test_layer17_no_layers_no_crash(self):
        from app.predetect.identity_exposure import Layer17IdentityExposure
        layer = Layer17IdentityExposure()
        result = layer.run(adapter=None, model_name="claude-sonnet-4.6", layer_results_so_far=[])
        assert result is not None
        assert result.tokens_used == 0


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
class TestPhase3Endpoints:
    def setup_method(self):
        _reset_taxonomy_cache()

    def test_model_taxonomy_route_registered(self):
        import app.main as main_module
        routes = [(m, p) for m, p, _ in main_module.ROUTES]
        assert ("GET", r"^/api/v14/model-taxonomy$") in routes, \
            "/api/v14/model-taxonomy not in ROUTES"

    def test_identity_exposure_route_registered(self):
        import app.main as main_module
        routes = [(m, p) for m, p, _ in main_module.ROUTES]
        assert ("GET", r"^/api/v14/runs/[^/]+/identity-exposure$") in routes, \
            "/api/v14/runs/.../identity-exposure not in ROUTES"

    def test_system_prompt_route_registered(self):
        import app.main as main_module
        routes = [(m, p) for m, p, _ in main_module.ROUTES]
        assert ("GET", r"^/api/v14/runs/[^/]+/system-prompt$") in routes, \
            "/api/v14/runs/.../system-prompt not in ROUTES"

    def test_model_taxonomy_handler_returns_json(self):
        from app.handlers.v14_handlers import handle_model_taxonomy
        import json
        status, body, ct = handle_model_taxonomy("/api/v14/model-taxonomy", {}, {})
        assert status == 200
        data = json.loads(body)
        assert "taxonomy" in data
        assert data["family_count"] >= 14

    def test_v14_health_mentions_phase3(self):
        import app.main as main_module
        import json
        status, body, ct = main_module._handle_v14_health("/api/v14/health", {}, {})
        data = json.loads(body)
        assert "phase3" in str(data).lower(), f"v14 health should mention phase3: {data}"


# ---------------------------------------------------------------------------
# IdentityExposureReport schema
# ---------------------------------------------------------------------------
class TestIdentityExposureReportSchema:
    def test_schema_exists(self):
        from app.core.schemas import IdentityExposureReport
        r = IdentityExposureReport()
        assert hasattr(r, "identity_collision")
        assert hasattr(r, "top_families")
        assert hasattr(r, "extracted_system_prompt")
        assert hasattr(r, "collision_confidence")

    def test_to_dict_structure(self):
        from app.core.schemas import IdentityExposureReport
        r = IdentityExposureReport(
            claimed_model="claude-sonnet-4.6",
            claimed_family="claude",
            identity_collision=True,
            collision_confidence=0.87,
            top_families=[{"family": "qwen", "raw_score": 9.0, "posterior": 0.87, "evidence": []}],
            extracted_system_prompt="You are Qwen...",
            total_responses_scanned=45,
        )
        d = r.to_dict()
        assert d["identity_collision"] is True
        assert d["collision_confidence"] == pytest.approx(0.87)
        assert d["extracted_system_prompt"] == "You are Qwen..."


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------
class TestMigration004:
    def test_migration_registered(self):
        from app.core.db_migrations import _migrations
        assert 4 in _migrations, "Migration004 must be registered"
        assert "identity_exposure" in _migrations[4].description.lower()
