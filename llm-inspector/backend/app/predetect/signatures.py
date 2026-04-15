"""
predetect/signatures.py — 常量、平台指纹数据库、工具函数

所有层共用的静态数据（HTTP 头签名、错误形态、已知模型前缀、
Tokenizer 探针词、知识截止日期映射）以及纯工具函数集中在此文件，
避免各层文件重复定义或循环引用。
"""
from __future__ import annotations

import json
import os

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# ── Early-exit confidence threshold ───────────────────────────────────────────

CONFIDENCE_THRESHOLD: float = settings.PREDETECT_CONFIDENCE_THRESHOLD

# ── Known platform HTTP-header signatures ─────────────────────────────────────

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

# ── Known model name prefixes ──────────────────────────────────────────────────

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
    ("glm-5",        "Zhipu/GLM-5"),
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
    ("minimax",      "MiniMax"),
    ("abab",         "MiniMax/Abab"),
    ("ernie",        "Baidu/ERNIE"),
    ("wenxin",       "Baidu/ERNIE"),
]

# ── Tokenizer probe words (Layer 2 & 5) ───────────────────────────────────────

TOKENIZER_PROBES: dict[str, dict[str, int]] = {
    "supercalifragilistic": {
        "OpenAI (tiktoken cl100k)": 6,
        "Anthropic (claude tokenizer)": 7,
        "Meta (LLaMA sentencepiece)": 5,
        "DeepSeek": 6,
        "Qwen": 5,
    },
    "cryptocurrency": {
        "OpenAI (tiktoken cl100k)": 4,
        "Anthropic (claude tokenizer)": 4,
        "Meta (LLaMA sentencepiece)": 5,
        "DeepSeek": 4,
        "Qwen": 4,
    },
    "antidisestablishmentarianism": {
        "OpenAI (tiktoken cl100k)": 6,
        "Anthropic (claude tokenizer)": 8,
        "Meta (LLaMA sentencepiece)": 7,
        "DeepSeek": 7,
        "Qwen": 6,
    },
    "pneumonoultramicroscopicsilicovolcanoconiosis": {
        "OpenAI (tiktoken cl100k)": 11,
        "Anthropic (claude tokenizer)": 13,
        "Meta (LLaMA sentencepiece)": 10,
        "DeepSeek": 12,
        "Qwen": 11,
    },
    "electromagnetic": {
        "OpenAI (tiktoken cl100k)": 3,
        "Anthropic (claude tokenizer)": 4,
        "Meta (LLaMA sentencepiece)": 4,
        "DeepSeek": 3,
        "Qwen": 3,
    },
    "counterintuitive": {
        "OpenAI (tiktoken cl100k)": 4,
        "Anthropic (claude tokenizer)": 5,
        "Meta (LLaMA sentencepiece)": 4,
        "DeepSeek": 4,
        "Qwen": 4,
    },
}

# Extended probes for Layer 5 multi-word fingerprinting
TOKENIZER_PROBES_EXTENDED: dict[str, dict[str, int]] = {
    "multimodality":    {"tiktoken-cl100k": 4, "tiktoken-o200k": 3, "claude": 5, "llama-spm": 6, "deepseek": 4, "qwen": 4, "chatglm": 5, "yi": 5},
    "cryptocurrency":   {"tiktoken-cl100k": 4, "tiktoken-o200k": 3, "claude": 4, "llama-spm": 5, "deepseek": 4, "qwen": 3, "chatglm": 4, "yi": 4},
    "hallucination":    {"tiktoken-cl100k": 4, "tiktoken-o200k": 3, "claude": 4, "llama-spm": 5, "deepseek": 4, "qwen": 3, "chatglm": 4, "yi": 4},
    "supercalifragilistic": {"tiktoken-cl100k": 4, "tiktoken-o200k": 3, "claude": 5, "llama-spm": 7, "deepseek": 4, "qwen": 4, "chatglm": 5, "yi": 5},
    "counterintuitive": {"tiktoken-cl100k": 4, "tiktoken-o200k": 3, "claude": 5, "llama-spm": 6, "deepseek": 4, "qwen": 4, "chatglm": 5, "yi": 5},
}

# ── Knowledge cutoff date map (Layer 3) ───────────────────────────────────────

def _load_cutoff_map() -> dict[str, list[str]]:
    """v6: Load CUTOFF_MAP from external JSON file for easy updates."""
    fixtures_dir = os.path.join(os.path.dirname(__file__), "..", "fixtures")
    cutoff_file = os.path.join(fixtures_dir, "cutoff_map.json")

    if os.path.exists(cutoff_file):
        try:
            with open(cutoff_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {k: v for k, v in data.items() if isinstance(v, list)}
        except Exception:
            pass

    return {
        "2021-09": ["OpenAI/GPT-3.5 (early)"],
        "2023-04": ["OpenAI/GPT-4 (0314/0613)"],
        "2023-12": ["Mistral AI", "Meta/LLaMA-2"],
        "2024-04": ["Anthropic/Claude-3"],
        "2024-06": ["OpenAI/GPT-4o (0513)", "Google/Gemini-1.5"],
        "2024-10": ["Anthropic/Claude-3.5 (1022)", "OpenAI/GPT-4o (1120)"],
        "2025-01": ["DeepSeek/DeepSeek-V3", "Alibaba/Qwen2.5"],
        "2025-04": ["Anthropic/Claude-4", "OpenAI/GPT-4.1"],
        "2025-10": ["Anthropic/Claude-4.5"],
        "2026-01": ["DeepSeek/DeepSeek-R2", "Alibaba/Qwen3"],
    }


# v6: Dynamic CUTOFF_MAP loaded at import time
CUTOFF_MAP: dict[str, list[str]] = _load_cutoff_map()

# ── Utility functions ──────────────────────────────────────────────────────────

def _estimate_tokens(text: str | None, base_estimate: int = 20) -> int:
    """当 API 不返回 token 数时，基于字符数粗估（每 3 个字符约 1 token）"""
    if not text:
        return base_estimate
    return max(base_estimate, len(text) // 3)


def _match_model_name(name: str) -> str | None:
    """将模型名称与 KNOWN_MODEL_PREFIXES 列表匹配，返回平台标识符。"""
    lower = name.lower()
    for prefix, platform in KNOWN_MODEL_PREFIXES:
        if prefix in lower:
            return platform
    return None
