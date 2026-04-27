"""
test_v16_phase4.py — v16 Phase 4 regression tests.

Validates:
  - ModelDiscovery /v1/models probe
  - ModelListReport data structure
  - compute_shell_posterior Bayesian update
  - _infer_family cross-family detection
  - system_prompt_harvester enhanced extraction
  - _SECRET_PATTERNS sanitization
  - get_extraction_prompts
  - RealModelCard / Evidence data structures
"""
import pytest


class TestModelDiscovery:
    def test_model_list_report_defaults(self):
        from app.predetect.model_discovery import ModelListReport
        r = ModelListReport()
        assert r.available == []
        assert r.claimed_present is False
        assert r.suspicious_neighbors == []
        assert r.cross_family_models == []

    def test_model_list_report_to_dict(self):
        from app.predetect.model_discovery import ModelListReport
        r = ModelListReport(
            available=["gpt-4o", "gpt-4o-mini"],
            claimed_present=True,
            http_status=200,
        )
        d = r.to_dict()
        assert d["claimed_present"] is True
        assert len(d["available"]) == 2
        assert d["http_status"] == 200

    def test_infer_family_openai(self):
        from app.predetect.model_discovery import _infer_family
        assert _infer_family("gpt-4o") == "openai"
        assert _infer_family("o1-preview") == "openai"

    def test_infer_family_anthropic(self):
        from app.predetect.model_discovery import _infer_family
        assert _infer_family("claude-3-5-sonnet") == "anthropic"

    def test_infer_family_deepseek(self):
        from app.predetect.model_discovery import _infer_family
        assert _infer_family("deepseek-chat") == "deepseek"

    def test_infer_family_unknown(self):
        from app.predetect.model_discovery import _infer_family
        assert _infer_family("my-custom-model") is None

    def test_find_cross_family(self):
        from app.predetect.model_discovery import _find_cross_family
        available = ["gpt-4o", "gpt-4o-mini", "qwen2-72b", "glm-4"]
        cross = _find_cross_family("gpt-4o", available)
        assert "qwen2-72b" in cross
        assert "glm-4" in cross
        assert "gpt-4o" not in cross

    def test_find_cross_family_no_cross(self):
        from app.predetect.model_discovery import _find_cross_family
        available = ["gpt-4o", "gpt-4o-mini", "o1-preview"]
        cross = _find_cross_family("gpt-4o", available)
        assert cross == []

    def test_find_suspicious_neighbors(self):
        from app.predetect.model_discovery import _find_suspicious_neighbors
        available = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]
        suspicious = _find_suspicious_neighbors("gpt-4o", available)
        assert "gpt-4o-mini" in suspicious

    def test_compute_shell_posterior_claimed_present(self):
        from app.predetect.model_discovery import ModelListReport, compute_shell_posterior
        r = ModelListReport(available=["gpt-4o"], claimed_present=True)
        p = compute_shell_posterior(r, "gpt-4o")
        assert p == 0.5  # No evidence of shell

    def test_compute_shell_posterior_not_present(self):
        from app.predetect.model_discovery import ModelListReport, compute_shell_posterior
        r = ModelListReport(available=["glm-4"], claimed_present=False)
        p = compute_shell_posterior(r, "gpt-4o")
        assert p >= 0.9  # +0.4 for not present

    def test_compute_shell_posterior_cross_family(self):
        from app.predetect.model_discovery import ModelListReport, compute_shell_posterior
        r = ModelListReport(
            available=["gpt-4o", "qwen2-72b"],
            claimed_present=True,
            cross_family_models=["qwen2-72b"],
        )
        p = compute_shell_posterior(r, "gpt-4o")
        assert p >= 1.0  # +0.5 for cross-family


class TestSystemPromptHarvesterV16:
    def test_secret_patterns_exist(self):
        from app.predetect.system_prompt_harvester import _SECRET_PATTERNS
        assert len(_SECRET_PATTERNS) >= 5

    def test_sanitize_github_token(self):
        from app.predetect.system_prompt_harvester import _sanitize
        text = "Your token is ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        sanitized = _sanitize(text)
        assert "ghp_" not in sanitized
        assert "[GITHUB_TOKEN_REDACTED]" in sanitized

    def test_sanitize_aws_key(self):
        from app.predetect.system_prompt_harvester import _sanitize
        text = "AWS key: AKIAIOSFODNN7EXAMPLE"
        sanitized = _sanitize(text)
        assert "AKIA" not in sanitized
        assert "[AWS_ACCESS_KEY_REDACTED]" in sanitized

    def test_sanitize_jwt(self):
        from app.predetect.system_prompt_harvester import _sanitize
        text = "Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ0ZXN0In0.abc123def456"
        sanitized = _sanitize(text)
        assert "[JWT_REDACTED]" in sanitized or "[BEARER_TOKEN_REDACTED]" in sanitized

    def test_get_extraction_prompts_all(self):
        from app.predetect.system_prompt_harvester import get_extraction_prompts
        prompts = get_extraction_prompts("all")
        assert len(prompts) >= 5  # 3 repeat_back + 2 json_mode + 1 token_economy + 1 multi_turn

    def test_get_extraction_prompts_repeat_back(self):
        from app.predetect.system_prompt_harvester import get_extraction_prompts
        prompts = get_extraction_prompts("repeat_back")
        assert len(prompts) == 3

    def test_get_extraction_prompts_json_mode(self):
        from app.predetect.system_prompt_harvester import get_extraction_prompts
        prompts = get_extraction_prompts("json_mode")
        assert len(prompts) == 2

    def test_get_extraction_prompts_multi_turn(self):
        from app.predetect.system_prompt_harvester import get_extraction_prompts
        prompts = get_extraction_prompts("multi_turn")
        assert len(prompts) == 1
        assert "Continue" in prompts[0]


class TestRealModelCard:
    def test_evidence_dataclass(self):
        from app.core.schemas import Evidence
        e = Evidence(source_layer="model_discovery", snippet="Found qwen2", confidence=0.8)
        d = e.to_dict()
        assert d["source_layer"] == "model_discovery"
        assert d["confidence"] == 0.8

    def test_real_model_card_basic(self):
        from app.core.schemas import RealModelCard
        rmc = RealModelCard(
            claimed_model="gpt-4o",
            suspected_family="qwen2",
            posterior=0.85,
        )
        assert rmc.claimed_model == "gpt-4o"
        assert rmc.suspected_family == "qwen2"
        assert rmc.is_official is False

    def test_real_model_card_official(self):
        from app.core.schemas import RealModelCard
        rmc = RealModelCard(
            claimed_model="gpt-4o",
            is_official=True,
            official_vendor="OpenAI",
            official_source_url="https://platform.openai.com/docs",
        )
        assert rmc.is_official is True
        assert rmc.official_vendor == "OpenAI"

    def test_real_model_card_to_dict(self):
        from app.core.schemas import RealModelCard, Evidence
        rmc = RealModelCard(
            claimed_model="gpt-4o",
            suspected_family="zhipu",
            posterior=0.9,
            evidence=[
                Evidence(source_layer="model_discovery", snippet="glm-4 found", confidence=0.8),
            ],
            diff_with_claimed=["claimed cutoff=2024-10 but observed 2024-04"],
        )
        d = rmc.to_dict()
        assert d["claimed_model"] == "gpt-4o"
        assert d["posterior"] == 0.9
        assert len(d["evidence"]) == 1
        assert len(d["diff_with_claimed"]) == 1
        assert "is_official" in d
        assert "official_vendor" in d
