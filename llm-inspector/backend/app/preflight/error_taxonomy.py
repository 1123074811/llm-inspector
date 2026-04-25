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
    raw_status: int | None = None       # HTTP status code if applicable
    raw_body_excerpt: str | None = None # First 200 chars of upstream response
    source_layer: str = "preflight"

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "retryable": self.retryable,
            "message": self.user_message_zh,
            "message_en": self.user_message_en,
            "hint": self.hint_zh,
            "raw_status": self.raw_status,
            "raw_excerpt": self.raw_body_excerpt,
            "source_layer": self.source_layer,
        }


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
        raw_status=raw_status,
        raw_body_excerpt=excerpt,
        source_layer=source_layer,
    )
