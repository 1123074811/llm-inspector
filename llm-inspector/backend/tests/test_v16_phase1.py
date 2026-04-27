"""
test_v16_phase1.py — v16 Phase 1 + 1.5 regression tests.

Validates:
  - ErrorCode 9 new error codes exist
  - ErrorDetail has category/hint_en/source_link fields
  - make_error populates category
  - LLMResponse has error_payload/http_status fields
  - Preflight integration in predetect pipeline
  - OfficialEndpoint checker basics
"""
import pytest


class TestErrorCodeV16:
    def test_new_error_codes_exist(self):
        from app.preflight.error_taxonomy import ErrorCode
        new_codes = [
            "E_BASE_URL_INVALID", "E_INVALID_AUTH",
            "E_UPSTREAM_503", "E_BASE_URL_NOT_OPENAI_COMPATIBLE",
            "E_JSON_DECODE_ERROR", "E_TLS_CERT_FAILED",
            "E_CONTENT_FILTERED", "E_RESPONSE_TRUNCATED",
        ]
        for code_name in new_codes:
            assert hasattr(ErrorCode, code_name), f"Missing ErrorCode.{code_name}"

    def test_new_error_codes_have_details(self):
        from app.preflight.error_taxonomy import ErrorCode, _ERROR_DETAILS
        new_codes = [
            ErrorCode.E_BASE_URL_INVALID, ErrorCode.E_INVALID_AUTH,
            ErrorCode.E_UPSTREAM_503,
            ErrorCode.E_BASE_URL_NOT_OPENAI_COMPATIBLE,
            ErrorCode.E_JSON_DECODE_ERROR, ErrorCode.E_TLS_CERT_FAILED,
            ErrorCode.E_CONTENT_FILTERED, ErrorCode.E_RESPONSE_TRUNCATED,
        ]
        for code in new_codes:
            assert code in _ERROR_DETAILS, f"Missing details for {code}"
            detail = _ERROR_DETAILS[code]
            assert len(detail) == 4  # (retryable, zh, en, hint)


class TestErrorDetailV16:
    def test_category_field_exists(self):
        from app.preflight.error_taxonomy import ErrorDetail
        ed = ErrorDetail(
            code="E_TEST", retryable=False,
            user_message_zh="测试", user_message_en="test", hint_zh="",
            category="A1",
        )
        assert ed.category == "A1"

    def test_hint_en_field_exists(self):
        from app.preflight.error_taxonomy import ErrorDetail
        ed = ErrorDetail(
            code="E_TEST", retryable=False,
            user_message_zh="测试", user_message_en="test", hint_zh="",
            hint_en="Check your config",
        )
        assert ed.hint_en == "Check your config"

    def test_source_link_field_exists(self):
        from app.preflight.error_taxonomy import ErrorDetail
        ed = ErrorDetail(
            code="E_TEST", retryable=False,
            user_message_zh="测试", user_message_en="test", hint_zh="",
            source_link="https://datatracker.ietf.org/doc/html/rfc7231",
        )
        assert "rfc7231" in ed.source_link

    def test_to_dict_includes_category(self):
        from app.preflight.error_taxonomy import ErrorDetail
        ed = ErrorDetail(
            code="E_TEST", retryable=False,
            user_message_zh="测试", user_message_en="test", hint_zh="",
            category="A3",
        )
        d = ed.to_dict()
        assert d["category"] == "A3"
        assert "hint_en" in d
        assert "source_link" in d


class TestMakeErrorV16:
    def test_make_error_populates_category(self):
        from app.preflight.error_taxonomy import ErrorCode, make_error
        ed = make_error(ErrorCode.E_AUTH_INVALID_KEY)
        assert ed.category == "A3"

    def test_make_error_category_a1(self):
        from app.preflight.error_taxonomy import ErrorCode, make_error
        ed = make_error(ErrorCode.E_URL_EMPTY)
        assert ed.category == "A1"

    def test_make_error_category_a2(self):
        from app.preflight.error_taxonomy import ErrorCode, make_error
        ed = make_error(ErrorCode.E_NET_DNS_FAIL)
        assert ed.category == "A2"

    def test_make_error_category_upstream(self):
        from app.preflight.error_taxonomy import ErrorCode, make_error
        ed = make_error(ErrorCode.E_UPSTREAM_INTERNAL)
        assert ed.category == "upstream"

    def test_make_error_category_parse(self):
        from app.preflight.error_taxonomy import ErrorCode, make_error
        ed = make_error(ErrorCode.E_RESPONSE_TRUNCATED)
        assert ed.category == "parse"


class TestLLMResponseV16:
    def test_error_payload_field(self):
        from app.core.schemas import LLMResponse
        resp = LLMResponse(
            content="test",
            error_payload={"message": "invalid api key", "type": "invalid_request_error"},
            http_status=401,
        )
        assert resp.error_payload is not None
        assert resp.http_status == 401

    def test_to_dict_includes_v16_fields(self):
        from app.core.schemas import LLMResponse
        resp = LLMResponse(
            content="test",
            error_payload={"error": "test"},
            http_status=500,
        )
        d = resp.to_dict()
        assert "error_payload" in d
        assert "http_status" in d


class TestOfficialEndpoint:
    def test_registry_loads(self):
        from app.authenticity.official_endpoint import _load_registry
        reg = _load_registry()
        assert "providers" in reg
        assert "openai" in reg["providers"]

    def test_url_match_openai(self):
        from app.authenticity.official_endpoint import _match_url
        cfg = {"base_urls": ["https://api.openai.com", "https://api.openai.com/v1"]}
        assert _match_url("https://api.openai.com", cfg)
        assert _match_url("https://api.openai.com/v1", cfg)
        assert not _match_url("https://fake-proxy.com", cfg)

    def test_model_prefix_match(self):
        from app.authenticity.official_endpoint import _check_model_prefix
        cfg = {"model_prefixes": ["gpt-", "o1-", "o3-"]}
        assert _check_model_prefix("gpt-4o", cfg)
        assert _check_model_prefix("o1-preview", cfg)
        assert not _check_model_prefix("claude-3", cfg)

    def test_check_disabled(self):
        from app.authenticity.official_endpoint import check_official_endpoint
        from app.core.config import settings
        old = settings.OFFICIAL_ENDPOINT_ENABLED
        try:
            settings.OFFICIAL_ENDPOINT_ENABLED = False
            result = check_official_endpoint("https://api.openai.com", "sk-test", "gpt-4o")
            assert not result.verified
            assert result.details.get("disabled")
        finally:
            settings.OFFICIAL_ENDPOINT_ENABLED = old

    def test_result_to_dict(self):
        from app.authenticity.official_endpoint import OfficialEndpointResult
        r = OfficialEndpointResult(
            verified=True, provider="openai", display_name="OpenAI",
            confidence=0.9, url_matched=True, tls_consistent=True,
            headers_consistent=True, model_prefix_matched=True,
        )
        d = r.to_dict()
        assert d["verified"] is True
        assert d["provider"] == "openai"
        assert d["confidence"] == 0.9
