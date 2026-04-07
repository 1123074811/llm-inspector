"""
Model behavior fingerprint database.
Used for comparing observed model behavior against known patterns
to help identify model families.
"""
from __future__ import annotations


# -- Model family behavior signatures --
# Each entry describes typical behavioral patterns for a model family.
# These are used for:
# 1. Comparing refusal style to identify the underlying model
# 2. Matching format tendencies to narrow down candidates
# 3. TTFT range validation

MODEL_BEHAVIOR_SIGNATURES: dict[str, dict] = {
    "claude": {
        "refusal_patterns": [
            "I want to be direct",
            "I appreciate you asking",
            "I'd be happy to help",
            "I should be transparent",
            "I need to be straightforward",
        ],
        "format_tendencies": {
            "xml_tags": True,
            "thinking_blocks": True,
            "bold_headers": False,
            "numbered_steps": True,
        },
        "style": {
            "hedging_rate": "high",
            "bullet_preference": "plain",
            "disclaimer_style": "upfront",
        },
        "chinese_quality": "high",
        "typical_ttft_ms": (300, 800),
        "typical_tps": (40, 120),
        "known_tokenizer": "claude",
    },
    "gpt4": {
        "refusal_patterns": [
            "As an AI language model",
            "I cannot",
            "I'm sorry, but",
            "I can't assist with",
            "It's important to note",
        ],
        "format_tendencies": {
            "xml_tags": False,
            "thinking_blocks": False,
            "bold_headers": True,
            "numbered_steps": True,
        },
        "style": {
            "hedging_rate": "medium",
            "bullet_preference": "bold+explain",
            "disclaimer_style": "inline",
        },
        "chinese_quality": "medium",
        "typical_ttft_ms": (200, 600),
        "typical_tps": (30, 100),
        "known_tokenizer": "tiktoken-cl100k",
    },
    "gpt4o": {
        "refusal_patterns": [
            "I can't help with",
            "I'm unable to",
            "That's something I can't",
        ],
        "format_tendencies": {
            "xml_tags": False,
            "thinking_blocks": False,
            "bold_headers": True,
            "numbered_steps": True,
        },
        "style": {
            "hedging_rate": "low",
            "bullet_preference": "bold+explain",
            "disclaimer_style": "minimal",
        },
        "chinese_quality": "high",
        "typical_ttft_ms": (150, 500),
        "typical_tps": (50, 150),
        "known_tokenizer": "tiktoken-o200k",
    },
    "deepseek": {
        "refusal_patterns": [
            "作为一个AI助手",
            "我无法提供",
            "作为AI，我需要",
            "这个问题涉及",
        ],
        "format_tendencies": {
            "xml_tags": False,
            "thinking_blocks": True,
            "bold_headers": False,
            "numbered_steps": True,
        },
        "style": {
            "hedging_rate": "low",
            "bullet_preference": "plain",
            "disclaimer_style": "minimal",
        },
        "chinese_quality": "native",
        "typical_ttft_ms": (400, 1200),
        "typical_tps": (20, 80),
        "known_tokenizer": "deepseek",
    },
    "qwen": {
        "refusal_patterns": [
            "需要注意的是",
            "作为AI助手",
            "很抱歉",
            "我无法帮助",
            "重要提示",
        ],
        "format_tendencies": {
            "xml_tags": False,
            "thinking_blocks": False,
            "bold_headers": True,
            "numbered_steps": True,
        },
        "style": {
            "hedging_rate": "medium",
            "bullet_preference": "numbered",
            "disclaimer_style": "footer",
        },
        "chinese_quality": "native",
        "typical_ttft_ms": (300, 900),
        "typical_tps": (30, 100),
        "known_tokenizer": "qwen",
    },
    "llama": {
        "refusal_patterns": [
            "I cannot provide",
            "I must advise against",
            "As a responsible AI",
        ],
        "format_tendencies": {
            "xml_tags": False,
            "thinking_blocks": False,
            "bold_headers": False,
            "numbered_steps": True,
        },
        "style": {
            "hedging_rate": "medium",
            "bullet_preference": "plain",
            "disclaimer_style": "inline",
        },
        "chinese_quality": "low",
        "typical_ttft_ms": (100, 500),
        "typical_tps": (30, 120),
        "known_tokenizer": "sentencepiece",
    },
    "gemini": {
        "refusal_patterns": [
            "I'm not able to",
            "I can't generate",
            "As a large language model",
        ],
        "format_tendencies": {
            "xml_tags": False,
            "thinking_blocks": False,
            "bold_headers": True,
            "numbered_steps": True,
        },
        "style": {
            "hedging_rate": "medium",
            "bullet_preference": "bold+explain",
            "disclaimer_style": "inline",
        },
        "chinese_quality": "medium",
        "typical_ttft_ms": (200, 700),
        "typical_tps": (40, 130),
        "known_tokenizer": "gemma-spm",
    },
    "minimax": {
        "refusal_patterns": [
            "作为海螺AI",
            "我是海螺",
            "MiniMax",
        ],
        "format_tendencies": {
            "xml_tags": False,
            "thinking_blocks": False,
            "bold_headers": False,
            "numbered_steps": True,
        },
        "style": {
            "hedging_rate": "low",
            "bullet_preference": "plain",
            "disclaimer_style": "minimal",
        },
        "chinese_quality": "native",
        "typical_ttft_ms": (300, 1000),
        "typical_tps": (25, 80),
        "known_tokenizer": None,
    },
    "glm": {
        "refusal_patterns": [
            "作为智谱AI",
            "很抱歉，我不能",
            "作为一个AI语言模型",
        ],
        "format_tendencies": {
            "xml_tags": False,
            "thinking_blocks": False,
            "bold_headers": True,
            "numbered_steps": True,
        },
        "style": {
            "hedging_rate": "medium",
            "bullet_preference": "numbered",
            "disclaimer_style": "footer",
        },
        "chinese_quality": "native",
        "typical_ttft_ms": (300, 800),
        "typical_tps": (30, 90),
        "known_tokenizer": "chatglm",
    },
}


def match_refusal_style(text: str) -> dict[str, float]:
    """
    Compare refusal text against known model family patterns.
    Returns a dict of {model_family: match_score} sorted by score descending.
    """
    lower_text = text.lower()
    scores: dict[str, float] = {}

    for family, sig in MODEL_BEHAVIOR_SIGNATURES.items():
        patterns = sig.get("refusal_patterns", [])
        if not patterns:
            continue
        hits = sum(1 for p in patterns if p.lower() in lower_text)
        scores[family] = hits / len(patterns) if patterns else 0.0

    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))


def get_ttft_range(model_family: str) -> tuple[int, int] | None:
    """Get the expected TTFT range for a model family."""
    sig = MODEL_BEHAVIOR_SIGNATURES.get(model_family.lower())
    if sig:
        return sig.get("typical_ttft_ms")
    return None
