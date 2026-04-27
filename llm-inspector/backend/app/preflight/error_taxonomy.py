"""
preflight/error_taxonomy.py — Unified error classification for LLM API connections.

Error code design follows RFC 9457 "Problem Details for HTTP APIs".
References:
  - RFC 9110 HTTP Semantics (status code semantics)
  - RFC 9457 Problem Details for HTTP APIs
  - OpenAI API Reference (error.code / error.type field semantics)
  - Anthropic Messages API Reference (error structure)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


class ErrorCode(str, Enum):
    # User input errors
    E_URL_EMPTY           = "E_URL_EMPTY"
    E_URL_INVALID_FORMAT  = "E_URL_INVALID_FORMAT"
    E_API_KEY_MISSING     = "E_API_KEY_MISSING"
    E_API_KEY_FORMAT      = "E_API_KEY_FORMAT"
    # Network errors
    E_NET_DNS_FAIL        = "E_NET_DNS_FAIL"
    E_NET_CONN_REFUSED    = "E_NET_CONN_REFUSED"
    E_NET_TIMEOUT         = "E_NET_TIMEOUT"
    E_TLS_INVALID         = "E_TLS_INVALID"
    # Auth errors (upstream)
    E_AUTH_INVALID_KEY    = "E_AUTH_INVALID_KEY"
    E_AUTH_FORBIDDEN      = "E_AUTH_FORBIDDEN"
    # Model errors
    E_MODEL_NOT_FOUND     = "E_MODEL_NOT_FOUND"
    E_MODEL_NAME_EMPTY    = "E_MODEL_NAME_EMPTY"
    # Upstream service errors
    E_UPSTREAM_RATE_LIMITED       = "E_UPSTREAM_RATE_LIMITED"
    E_UPSTREAM_SERVICE_UNAVAILABLE = "E_UPSTREAM_SERVICE_UNAVAILABLE"
    E_UPSTREAM_BAD_GATEWAY        = "E_UPSTREAM_BAD_GATEWAY"
    E_UPSTREAM_INTERNAL           = "E_UPSTREAM_INTERNAL"
    # Response errors
    E_SCHEMA_INVALID_JSON         = "E_SCHEMA_INVALID_JSON"
    E_SCHEMA_MISSING_CHOICES      = "E_SCHEMA_MISSING_CHOICES"
    # Capability
    E_CAPABILITY_STREAM_UNSUPPORTED = "E_CAPABILITY_STREAM_UNSUPPORTED"
    # v16 Phase 1: Extended error codes for upstream transparency
    E_BASE_URL_INVALID           = "E_BASE_URL_INVALID"
    E_INVALID_AUTH               = "E_INVALID_AUTH"
    E_UPSTREAM_503               = "E_UPSTREAM_503"
    E_BASE_URL_NOT_OPENAI_COMPATIBLE = "E_BASE_URL_NOT_OPENAI_COMPATIBLE"
    E_JSON_DECODE_ERROR          = "E_JSON_DECODE_ERROR"
    E_TLS_CERT_FAILED            = "E_TLS_CERT_FAILED"
    E_CONTENT_FILTERED           = "E_CONTENT_FILTERED"
    E_RESPONSE_TRUNCATED         = "E_RESPONSE_TRUNCATED"
    # Generic
    E_UNKNOWN             = "E_UNKNOWN"


# Map error codes to (retryable, zh_message, en_message, hint_zh)
_ERROR_DETAILS: dict[str, tuple[bool, str, str, str]] = {
    ErrorCode.E_URL_EMPTY:           (False, "Base URL 不能为空", "Base URL is required", "请在配置页填写 API 地址"),
    ErrorCode.E_URL_INVALID_FORMAT:  (False, "Base URL 格式不合法（示例：https://api.example.com）", "Invalid Base URL format", "确认 URL 以 http:// 或 https:// 开头"),
    ErrorCode.E_API_KEY_MISSING:     (False, "API Key 不能为空", "API Key is required", "请在配置页填写 API Key"),
    ErrorCode.E_API_KEY_FORMAT:      (False, "API Key 格式疑似错误", "API Key format looks invalid", "常见格式：sk-... / Bearer ..."),
    ErrorCode.E_NET_DNS_FAIL:        (True,  "无法解析主机名，请检查 Base URL 是否正确", "DNS resolution failed", "确认主机名拼写无误，检查网络连接"),
    ErrorCode.E_NET_CONN_REFUSED:    (True,  "连接被拒绝，目标端口未开放或防火墙阻断", "Connection refused", "确认服务正在运行，检查端口和防火墙设置"),
    ErrorCode.E_NET_TIMEOUT:         (True,  "连接超时，网络延迟过大", "Connection timed out", "稍后重试；若持续出现，检查网络或上游服务状态"),
    ErrorCode.E_TLS_INVALID:         (False, "TLS/SSL 连接失败（常见原因：企业代理 SSL 审查、系统证书库过旧、网络防火墙阻断）", "TLS/SSL connection failed", "排查步骤：① pip install --upgrade certifi 更新 CA 证书库后重试 ② 若在企业内网，尝试取消勾选「验证 SSL 证书」后重试 ③ 设置代理：HTTPS_PROXY=http://代理地址:端口 后重启服务 ④ 展开「原始错误」查看具体原因"),
    ErrorCode.E_AUTH_INVALID_KEY:    (False, "API Key 无效或已过期（上游返回 401）", "Invalid API Key (401 from upstream)", "重新检查 API Key，确认未过期"),
    ErrorCode.E_AUTH_FORBIDDEN:      (False, "API Key 无此模型访问权限（上游返回 403）", "Forbidden (403 from upstream)", "确认 Key 有访问该模型的权限"),
    ErrorCode.E_MODEL_NOT_FOUND:     (False, "模型名不存在或未部署（上游返回 404）", "Model not found (404)", "检查模型名拼写，查阅上游支持的模型列表"),
    ErrorCode.E_MODEL_NAME_EMPTY:    (False, "模型名不能为空", "Model name is required", ""),
    ErrorCode.E_UPSTREAM_RATE_LIMITED:        (True,  "上游 API 请求超限（429），请稍后重试", "Rate limited by upstream (429)", "这不是 URL/Key 错误，稍后自动重试"),
    ErrorCode.E_UPSTREAM_SERVICE_UNAVAILABLE: (True,  "上游服务暂时不可用（503），请稍后重试", "Upstream service unavailable (503)", "这不是 URL/Key 错误，上游服务器临时故障"),
    ErrorCode.E_UPSTREAM_BAD_GATEWAY:         (True,  "上游网关错误（502/504），请稍后重试", "Bad gateway (502/504)", "这不是 URL/Key 错误，上游代理或负载均衡临时故障"),
    ErrorCode.E_UPSTREAM_INTERNAL:            (False, "上游服务内部错误（500）", "Upstream internal server error (500)", ""),
    ErrorCode.E_SCHEMA_INVALID_JSON:          (False, "上游返回非 JSON 格式，可能不是 OpenAI 兼容 API", "Non-JSON response from upstream", ""),
    ErrorCode.E_SCHEMA_MISSING_CHOICES:       (False, "上游 JSON 缺少 choices 字段，可能不兼容 OpenAI 格式", "Missing choices in response", ""),
    ErrorCode.E_CAPABILITY_STREAM_UNSUPPORTED:(False, "上游不支持 stream 模式", "Stream mode not supported", ""),
    # v16 Phase 1: Extended error codes
    ErrorCode.E_BASE_URL_INVALID:           (False, "Base URL 无效或不可达", "Base URL is invalid or unreachable", "检查 URL 格式和网络连通性"),
    ErrorCode.E_INVALID_AUTH:               (False, "认证失败（API Key 无效或过期）", "Authentication failed (invalid/expired API Key)", "确认 API Key 正确且未过期"),
    ErrorCode.E_UPSTREAM_503:               (True,  "上游服务暂时不可用（503）", "Upstream service unavailable (503)", "稍后重试；若持续出现，检查上游服务状态"),
    ErrorCode.E_BASE_URL_NOT_OPENAI_COMPATIBLE: (False, "上游 API 非 OpenAI 兼容格式", "Upstream API is not OpenAI-compatible", "确认 base_url 指向 OpenAI 兼容端点"),
    ErrorCode.E_JSON_DECODE_ERROR:          (False, "上游返回非 JSON 格式", "Upstream returned non-JSON response", "检查 base_url 是否正确"),
    ErrorCode.E_TLS_CERT_FAILED:            (False, "TLS 证书校验失败", "TLS certificate verification failed", "更新 CA 证书库 (pip install --upgrade certifi) 或检查网络代理"),
    ErrorCode.E_CONTENT_FILTERED:           (False, "上游内容审核拦截了响应", "Response filtered by upstream content moderation", "尝试更换测试用例或联系上游服务商"),
    ErrorCode.E_RESPONSE_TRUNCATED:         (True,  "响应被截断（max_tokens 不足）", "Response truncated (max_tokens insufficient)", "增加 max_tokens 或缩短提示词"),
    ErrorCode.E_UNKNOWN:             (True,  "未知错误", "Unknown error", ""),
}


@dataclass
class ErrorDetail:
    """Structured error detail following RFC 9457 Problem Details pattern."""
    code: str                     # ErrorCode value
    retryable: bool
    user_message_zh: str
    user_message_en: str
    hint_zh: str
    category: str = "unknown"           # v16: A1/A2/A3/A4/A5 + upstream/parse/judge
    hint_en: str = ""                   # v16: English hint
    source_link: str = ""              # v16: Reference URL (RFC / API docs)
    raw_status: int | None = None       # HTTP status code if applicable
    raw_body_excerpt: str | None = None # First 200 chars of upstream response
    source_layer: str = "preflight"

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "category": self.category,
            "retryable": self.retryable,
            "message": self.user_message_zh,
            "message_en": self.user_message_en,
            "hint": self.hint_zh,
            "hint_en": self.hint_en,
            "source_link": self.source_link,
            "raw_status": self.raw_status,
            "raw_excerpt": self.raw_body_excerpt,
            "source_layer": self.source_layer,
        }


def _error_category(code: ErrorCode) -> str:
    """v16: Map ErrorCode to preflight step category (A1-A5 / upstream / parse / judge)."""
    _CAT_MAP = {
        ErrorCode.E_URL_EMPTY: "A1", ErrorCode.E_URL_INVALID_FORMAT: "A1",
        ErrorCode.E_API_KEY_MISSING: "A1", ErrorCode.E_API_KEY_FORMAT: "A1",
        ErrorCode.E_MODEL_NAME_EMPTY: "A1", ErrorCode.E_BASE_URL_INVALID: "A1",
        ErrorCode.E_NET_DNS_FAIL: "A2", ErrorCode.E_NET_CONN_REFUSED: "A2",
        ErrorCode.E_NET_TIMEOUT: "A2", ErrorCode.E_TLS_INVALID: "A2",
        ErrorCode.E_TLS_CERT_FAILED: "A2",
        ErrorCode.E_AUTH_INVALID_KEY: "A3", ErrorCode.E_AUTH_FORBIDDEN: "A3",
        ErrorCode.E_INVALID_AUTH: "A3",
        ErrorCode.E_MODEL_NOT_FOUND: "A3",  # Original enum member
        ErrorCode.E_UPSTREAM_RATE_LIMITED: "A3", ErrorCode.E_UPSTREAM_503: "A3",
        ErrorCode.E_SCHEMA_INVALID_JSON: "A4", ErrorCode.E_SCHEMA_MISSING_CHOICES: "A4",
        ErrorCode.E_BASE_URL_NOT_OPENAI_COMPATIBLE: "A4",
        ErrorCode.E_JSON_DECODE_ERROR: "A4",
        ErrorCode.E_CAPABILITY_STREAM_UNSUPPORTED: "A5",
        ErrorCode.E_UPSTREAM_SERVICE_UNAVAILABLE: "upstream",
        ErrorCode.E_UPSTREAM_BAD_GATEWAY: "upstream",
        ErrorCode.E_UPSTREAM_INTERNAL: "upstream",
        ErrorCode.E_CONTENT_FILTERED: "upstream",
        ErrorCode.E_RESPONSE_TRUNCATED: "parse",
    }
    return _CAT_MAP.get(code, "unknown")


def make_error(code: ErrorCode,
               raw_status: int | None = None,
               raw_body: str | None = None,
               source_layer: str = "preflight") -> ErrorDetail:
    """Build an ErrorDetail from an ErrorCode."""
    retryable, zh, en, hint = _ERROR_DETAILS.get(code, (True, str(code), str(code), ""))
    excerpt = raw_body[:200] if raw_body else None
    return ErrorDetail(
        code=code.value,
        retryable=retryable,
        user_message_zh=zh,
        user_message_en=en,
        hint_zh=hint,
        category=_error_category(code),
        raw_status=raw_status,
        raw_body_excerpt=excerpt,
        source_layer=source_layer,
    )
