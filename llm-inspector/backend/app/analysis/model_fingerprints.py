"""
Model behavior fingerprint database.
Used for comparing observed model behavior against known patterns
to help identify model families.
"""
from __future__ import annotations

import re


# -- Model family behavior signatures --
# Each entry describes typical behavioral patterns for a model family.
# These are used for:
# 1. Comparing refusal style to identify the underlying model
# 2. Matching format tendencies to narrow down candidates
# 3. TTFT range validation
# 4. Structural fingerprint matching

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
    "gpt4o_mini": {
        "refusal_patterns": [
            "I can't help with that",
            "I'm not able to assist",
            "I can't do that",
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
        "chinese_quality": "medium",
        "typical_ttft_ms": (80, 300),
        "typical_tps": (80, 200),
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
    "mistral": {
        "refusal_patterns": [
            "I'm designed to be helpful",
            "I must decline",
            "As Mistral",
            "I don't think I should",
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
        "chinese_quality": "low",
        "typical_ttft_ms": (150, 600),
        "typical_tps": (40, 130),
        "known_tokenizer": "sentencepiece",
    },
    "yi": {
        "refusal_patterns": [
            "作为零一万物",
            "作为Yi模型",
            "我是Yi",
            "很抱歉，我无法",
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
            "disclaimer_style": "inline",
        },
        "chinese_quality": "native",
        "typical_ttft_ms": (200, 700),
        "typical_tps": (30, 100),
        "known_tokenizer": "yi",
    },
    "moonshot": {
        "refusal_patterns": [
            "作为Kimi",
            "我是Kimi",
            "月之暗面",
            "作为AI助手Kimi",
            "我需要提醒您",
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
            "disclaimer_style": "upfront",
        },
        "chinese_quality": "native",
        "typical_ttft_ms": (300, 1000),
        "typical_tps": (25, 90),
        "known_tokenizer": None,
    },
    "baichuan": {
        "refusal_patterns": [
            "作为百川大模型",
            "作为百川AI",
            "我是百川",
            "很抱歉，作为AI",
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
            "disclaimer_style": "footer",
        },
        "chinese_quality": "native",
        "typical_ttft_ms": (300, 900),
        "typical_tps": (25, 85),
        "known_tokenizer": None,
    },
}


# -- CJK detection pattern (CJK Unified Ideographs + common extensions) --
_CJK_RE = re.compile(
    r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff"
    r"\U00020000-\U0002a6df\U0002a700-\U0002b73f"
    r"\U0002b740-\U0002b81f\U0002b820-\U0002ceaf]"
)


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


# -- Structural fingerprint analysis --

def _compute_cjk_ratio(text: str) -> float:
    """Return the ratio of CJK characters to total non-whitespace characters."""
    non_ws = re.sub(r"\s", "", text)
    if not non_ws:
        return 0.0
    cjk_count = len(_CJK_RE.findall(non_ws))
    return cjk_count / len(non_ws)


def _compute_markdown_density(text: str) -> float:
    """Estimate markdown syntax density as a ratio of markdown tokens to total lines."""
    lines = text.split("\n")
    if not lines:
        return 0.0
    md_patterns = re.compile(
        r"^(#{1,6}\s"        # headings
        r"|[-*+]\s"          # unordered list
        r"|\d+\.\s"          # ordered list
        r"|>\s"              # blockquote
        r"|```"              # code fence
        r"|\|.*\|"           # table row
        r"|---"              # horizontal rule / table separator
        r")"
    )
    md_line_count = sum(1 for line in lines if md_patterns.match(line.strip()))
    # Also count inline markdown: **bold**, *italic*, `code`, [links]
    inline_hits = len(re.findall(r"\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|\[[^\]]+\]\([^)]+\)", text))
    # Normalize: line-level ratio + small boost for inline density
    line_ratio = md_line_count / len(lines)
    inline_ratio = min(inline_hits / max(len(lines), 1), 1.0) * 0.3
    return min(line_ratio + inline_ratio, 1.0)


def _detect_opening_style(text: str) -> str:
    """Classify the opening style of a response."""
    first_line = text.strip().split("\n")[0].strip() if text.strip() else ""
    lower_first = first_line.lower()

    greeting_words = ["hello", "hi ", "hi!", "hey", "greetings", "你好", "您好"]
    if any(lower_first.startswith(g) for g in greeting_words):
        return "greeting"

    hedging_phrases = [
        "it depends", "that's a great question", "this is a",
        "that's an interesting", "good question", "great question",
        "well,", "so,", "这是一个", "这个问题",
    ]
    if any(phrase in lower_first for phrase in hedging_phrases):
        return "hedging"

    # Question restatement: line ends with '?' or starts by echoing back
    if first_line.endswith("?") or lower_first.startswith("you're asking"):
        return "question_restate"

    return "direct"


def _detect_list_style(text: str) -> str:
    """Detect dominant list style in the response."""
    bullet_count = len(re.findall(r"^[ \t]*[-*+]\s", text, re.MULTILINE))
    numbered_count = len(re.findall(r"^[ \t]*\d+[.)]\s", text, re.MULTILINE))

    if bullet_count == 0 and numbered_count == 0:
        return "none"
    if bullet_count > 0 and numbered_count > 0:
        # Mixed if neither dominates by more than 2:1
        ratio = max(bullet_count, numbered_count) / min(bullet_count, numbered_count)
        if ratio < 2.0:
            return "mixed"
    if bullet_count >= numbered_count:
        return "bullet"
    return "numbered"


def _detect_closing_style(text: str) -> str:
    """Classify the closing style of a response."""
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    if not lines:
        return "none"
    last_line = lines[-1].lower()

    offer_patterns = [
        "let me know", "feel free to ask", "happy to help",
        "if you have any", "hope this helps", "don't hesitate",
        "如果你有", "希望对你有帮助", "如有疑问", "随时问我",
    ]
    if any(p in last_line for p in offer_patterns):
        return "offer_help"

    summary_patterns = [
        "in summary", "to summarize", "in conclusion", "overall",
        "总结", "综上", "总之",
    ]
    if any(p in last_line for p in summary_patterns):
        return "summary"

    if last_line.endswith("?"):
        return "question"

    return "none"


def response_structure_fingerprint(text: str) -> dict:
    """
    Analyze a response text and return a dict of structural features.

    Returns a dict with keys:
        opening_style: "greeting" | "direct" | "hedging" | "question_restate"
        list_style: "bullet" | "numbered" | "none" | "mixed"
        closing_style: "summary" | "offer_help" | "none" | "question"
        uses_xml_tags: bool
        uses_thinking_blocks: bool
        uses_bold_headers: bool
        markdown_density: float (0-1)
        avg_paragraph_length: float (in characters)
        cjk_ratio: float (0-1)
    """
    if not text or not text.strip():
        return {
            "opening_style": "direct",
            "list_style": "none",
            "closing_style": "none",
            "uses_xml_tags": False,
            "uses_thinking_blocks": False,
            "uses_bold_headers": False,
            "markdown_density": 0.0,
            "avg_paragraph_length": 0.0,
            "cjk_ratio": 0.0,
        }

    # XML tags: look for paired tags like <tag>...</tag> (excluding common markdown)
    uses_xml_tags = bool(re.search(r"<[a-zA-Z_][\w-]*>.*?</[a-zA-Z_][\w-]*>", text, re.DOTALL))

    # Thinking blocks: <thinking>, <thought>, or similar patterns
    uses_thinking_blocks = bool(re.search(
        r"<(?:thinking|thought|think|reasoning|scratchpad)>", text, re.IGNORECASE
    ))

    # Bold headers: lines that are primarily **bold text** (markdown bold as section header)
    uses_bold_headers = bool(re.search(r"^\s*\*\*[^*]+\*\*\s*$", text, re.MULTILINE))

    # Average paragraph length: split on double newlines
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    avg_paragraph_length = (
        sum(len(p) for p in paragraphs) / len(paragraphs)
        if paragraphs
        else 0.0
    )

    return {
        "opening_style": _detect_opening_style(text),
        "list_style": _detect_list_style(text),
        "closing_style": _detect_closing_style(text),
        "uses_xml_tags": uses_xml_tags,
        "uses_thinking_blocks": uses_thinking_blocks,
        "uses_bold_headers": uses_bold_headers,
        "markdown_density": round(_compute_markdown_density(text), 4),
        "avg_paragraph_length": round(avg_paragraph_length, 1),
        "cjk_ratio": round(_compute_cjk_ratio(text), 4),
    }


# -- Learn fingerprint from observed results --

def learn_fingerprint_from_results(
    case_results: list[dict],
) -> dict:
    """
    Build a fingerprint profile dict from actual observed test case results.

    Args:
        case_results: list of dicts, each with at least:
            - response_content (str): the model's response text
            - judge_detail (str | dict | None): judge output / detail
            - category (str): the test case category

    Returns:
        A dict compatible with MODEL_BEHAVIOR_SIGNATURES entries, containing:
            refusal_patterns, format_tendencies, style, chinese_quality,
            avg_response_length, cjk_ratio, list_style_preference,
            formality_score.
    """
    if not case_results:
        return {
            "refusal_patterns": [],
            "format_tendencies": {
                "xml_tags": False,
                "thinking_blocks": False,
                "bold_headers": False,
                "numbered_steps": False,
            },
            "style": {
                "hedging_rate": "low",
                "bullet_preference": "plain",
                "disclaimer_style": "minimal",
            },
            "chinese_quality": "unknown",
            "avg_response_length": 0.0,
            "cjk_ratio": 0.0,
            "list_style_preference": "none",
            "formality_score": 0.5,
        }

    # -- Collect all response texts --
    all_texts: list[str] = []
    refusal_texts: list[str] = []
    style_texts: list[str] = []

    for cr in case_results:
        content = cr.get("response_content") or ""
        category = (cr.get("category") or "").lower()
        if content:
            all_texts.append(content)
        if category in ("refusal", "safety", "antispoof"):
            refusal_texts.append(content)
        if category in ("style", "instruction", "consistency"):
            style_texts.append(content)

    # -- Refusal patterns extraction --
    # Collect the first sentence from each refusal response as a candidate pattern
    observed_refusal_patterns: list[str] = []
    for rt in refusal_texts:
        first_line = rt.strip().split("\n")[0].strip() if rt.strip() else ""
        if first_line and len(first_line) < 200:
            # Deduplicate by checking near-duplicates
            already_seen = any(
                first_line.lower() in existing.lower()
                or existing.lower() in first_line.lower()
                for existing in observed_refusal_patterns
            )
            if not already_seen:
                observed_refusal_patterns.append(first_line)

    # -- Format tendencies --
    xml_count = 0
    thinking_count = 0
    bold_header_count = 0
    numbered_step_count = 0

    for t in all_texts:
        if re.search(r"<[a-zA-Z_][\w-]*>.*?</[a-zA-Z_][\w-]*>", t, re.DOTALL):
            xml_count += 1
        if re.search(r"<(?:thinking|thought|think|reasoning|scratchpad)>", t, re.IGNORECASE):
            thinking_count += 1
        if re.search(r"^\s*\*\*[^*]+\*\*\s*$", t, re.MULTILINE):
            bold_header_count += 1
        if re.search(r"^\s*\d+[.)]\s", t, re.MULTILINE):
            numbered_step_count += 1

    n = len(all_texts)
    format_tendencies = {
        "xml_tags": xml_count / n > 0.2 if n else False,
        "thinking_blocks": thinking_count / n > 0.1 if n else False,
        "bold_headers": bold_header_count / n > 0.3 if n else False,
        "numbered_steps": numbered_step_count / n > 0.3 if n else False,
    }

    # -- Average response length --
    avg_response_length = (
        sum(len(t) for t in all_texts) / n if n else 0.0
    )

    # -- CJK ratio --
    combined_text = "\n".join(all_texts)
    cjk_ratio = _compute_cjk_ratio(combined_text)

    # Chinese quality classification based on CJK ratio
    if cjk_ratio > 0.6:
        chinese_quality = "native"
    elif cjk_ratio > 0.3:
        chinese_quality = "high"
    elif cjk_ratio > 0.1:
        chinese_quality = "medium"
    else:
        chinese_quality = "low"

    # -- List style preference --
    list_votes: dict[str, int] = {"bullet": 0, "numbered": 0, "none": 0, "mixed": 0}
    for t in all_texts:
        ls = _detect_list_style(t)
        list_votes[ls] = list_votes.get(ls, 0) + 1
    list_style_preference = max(list_votes, key=list_votes.get)  # type: ignore[arg-type]

    # -- Bullet preference mapping --
    if bold_header_count / n > 0.3 if n else False:
        bullet_pref = "bold+explain"
    elif list_style_preference == "numbered":
        bullet_pref = "numbered"
    else:
        bullet_pref = "plain"

    # -- Formality score (0 = very informal, 1 = very formal) --
    # Heuristic based on several signals
    formality_signals: list[float] = []
    for t in all_texts:
        lower_t = t.lower()
        signal = 0.5  # baseline
        # Contractions lower formality
        contraction_count = len(re.findall(
            r"\b(?:i'm|don't|can't|won't|isn't|aren't|doesn't|didn't|shouldn't|couldn't|wouldn't)\b",
            lower_t,
        ))
        words_approx = max(len(lower_t.split()), 1)
        signal -= min(contraction_count / words_approx * 10, 0.3)
        # Hedging phrases raise formality
        hedge_phrases = [
            "it is important to", "please note", "it should be noted",
            "one should consider", "需要注意", "请注意", "值得注意的是",
        ]
        hedge_hits = sum(1 for h in hedge_phrases if h in lower_t)
        signal += min(hedge_hits * 0.1, 0.3)
        # Exclamation marks lower formality
        excl_count = lower_t.count("!")
        signal -= min(excl_count * 0.05, 0.2)
        formality_signals.append(max(0.0, min(1.0, signal)))

    formality_score = (
        sum(formality_signals) / len(formality_signals)
        if formality_signals
        else 0.5
    )

    # -- Hedging rate --
    hedging_indicators = [
        "perhaps", "maybe", "it depends", "generally", "typically",
        "it's worth noting", "可能", "或许", "一般来说", "通常",
    ]
    hedging_hits = sum(
        1 for t in all_texts
        for h in hedging_indicators
        if h in t.lower()
    )
    hedging_density = hedging_hits / max(n, 1)
    if hedging_density > 2.0:
        hedging_rate = "high"
    elif hedging_density > 0.8:
        hedging_rate = "medium"
    else:
        hedging_rate = "low"

    # -- Disclaimer style --
    upfront_count = 0
    footer_count = 0
    inline_count = 0
    disclaimer_kw = [
        "note", "disclaimer", "important", "注意", "提示", "请注意",
        "however", "但是", "不过",
    ]
    for t in all_texts:
        lines = t.strip().split("\n")
        if not lines:
            continue
        first_third = "\n".join(lines[: max(len(lines) // 3, 1)]).lower()
        last_third = "\n".join(lines[-(max(len(lines) // 3, 1)):]).lower()
        middle = "\n".join(lines[max(len(lines) // 3, 1): -(max(len(lines) // 3, 1)) or len(lines)]).lower()

        has_first = any(kw in first_third for kw in disclaimer_kw)
        has_last = any(kw in last_third for kw in disclaimer_kw)
        has_middle = any(kw in middle for kw in disclaimer_kw)

        if has_first:
            upfront_count += 1
        if has_last:
            footer_count += 1
        if has_middle:
            inline_count += 1

    max_disc = max(upfront_count, footer_count, inline_count, 1)
    if max_disc == 0 or (upfront_count == 0 and footer_count == 0 and inline_count == 0):
        disclaimer_style = "minimal"
    elif upfront_count >= footer_count and upfront_count >= inline_count:
        disclaimer_style = "upfront"
    elif footer_count >= inline_count:
        disclaimer_style = "footer"
    else:
        disclaimer_style = "inline"

    return {
        "refusal_patterns": observed_refusal_patterns[:8],
        "format_tendencies": format_tendencies,
        "style": {
            "hedging_rate": hedging_rate,
            "bullet_preference": bullet_pref,
            "disclaimer_style": disclaimer_style,
        },
        "chinese_quality": chinese_quality,
        "avg_response_length": round(avg_response_length, 1),
        "cjk_ratio": round(cjk_ratio, 4),
        "list_style_preference": list_style_preference,
        "formality_score": round(formality_score, 3),
    }


# -- Structural fingerprint matching --

def match_structural_fingerprint(
    observed: dict,
) -> dict[str, float]:
    """
    Compare observed structural features against all model families.

    Args:
        observed: a dict as returned by response_structure_fingerprint(),
            with keys: opening_style, list_style, closing_style,
            uses_xml_tags, uses_thinking_blocks, uses_bold_headers,
            markdown_density, avg_paragraph_length, cjk_ratio.

    Returns:
        A dict of {family: match_score} sorted by score descending.
        Scores are in range [0.0, 1.0].
    """
    scores: dict[str, float] = {}

    for family, sig in MODEL_BEHAVIOR_SIGNATURES.items():
        fmt = sig.get("format_tendencies", {})
        style = sig.get("style", {})
        chinese_q = sig.get("chinese_quality", "medium")

        score = 0.0
        max_score = 0.0

        # -- Format tendency matching (weight: 3 points each, 12 total) --
        weight_fmt = 3.0

        # XML tags
        max_score += weight_fmt
        obs_xml = observed.get("uses_xml_tags", False)
        if obs_xml == fmt.get("xml_tags", False):
            score += weight_fmt

        # Thinking blocks
        max_score += weight_fmt
        obs_think = observed.get("uses_thinking_blocks", False)
        if obs_think == fmt.get("thinking_blocks", False):
            score += weight_fmt

        # Bold headers
        max_score += weight_fmt
        obs_bold = observed.get("uses_bold_headers", False)
        if obs_bold == fmt.get("bold_headers", False):
            score += weight_fmt

        # Numbered steps (infer from list_style)
        max_score += weight_fmt
        obs_list = observed.get("list_style", "none")
        sig_numbered = fmt.get("numbered_steps", False)
        obs_has_numbered = obs_list in ("numbered", "mixed")
        if obs_has_numbered == sig_numbered:
            score += weight_fmt

        # -- List style vs bullet_preference (weight: 2 points) --
        weight_list = 2.0
        max_score += weight_list
        bullet_pref = style.get("bullet_preference", "plain")
        if bullet_pref == "bold+explain" and obs_bold and obs_list in ("numbered", "mixed", "bullet"):
            score += weight_list
        elif bullet_pref == "numbered" and obs_list in ("numbered", "mixed"):
            score += weight_list
        elif bullet_pref == "plain" and obs_list in ("bullet", "none"):
            score += weight_list

        # -- CJK ratio vs chinese_quality (weight: 3 points) --
        weight_cjk = 3.0
        max_score += weight_cjk
        obs_cjk = observed.get("cjk_ratio", 0.0)
        cjk_expected_ranges = {
            "native": (0.5, 1.0),
            "high": (0.2, 0.7),
            "medium": (0.05, 0.4),
            "low": (0.0, 0.15),
        }
        cjk_lo, cjk_hi = cjk_expected_ranges.get(chinese_q, (0.0, 1.0))
        if cjk_lo <= obs_cjk <= cjk_hi:
            score += weight_cjk
        else:
            # Partial credit based on distance
            if obs_cjk < cjk_lo:
                dist = cjk_lo - obs_cjk
            else:
                dist = obs_cjk - cjk_hi
            score += weight_cjk * max(0.0, 1.0 - dist * 3)

        # -- Markdown density (weight: 2 points) --
        # Models with bold_headers/numbered_steps tend to have higher md density
        weight_md = 2.0
        max_score += weight_md
        obs_md = observed.get("markdown_density", 0.0)
        expected_heavy_md = fmt.get("bold_headers", False) or fmt.get("numbered_steps", False)
        if expected_heavy_md and obs_md > 0.15:
            score += weight_md
        elif not expected_heavy_md and obs_md <= 0.3:
            score += weight_md
        else:
            # Partial credit
            score += weight_md * 0.3

        # -- Opening style heuristic (weight: 1 point) --
        weight_open = 1.0
        max_score += weight_open
        obs_opening = observed.get("opening_style", "direct")
        hedging_rate = style.get("hedging_rate", "medium")
        if hedging_rate == "high" and obs_opening in ("hedging", "greeting"):
            score += weight_open
        elif hedging_rate == "low" and obs_opening == "direct":
            score += weight_open
        elif hedging_rate == "medium":
            score += weight_open * 0.6  # medium matches most styles partially

        # -- Closing style heuristic (weight: 1 point) --
        weight_close = 1.0
        max_score += weight_close
        obs_closing = observed.get("closing_style", "none")
        disclaimer_style = style.get("disclaimer_style", "minimal")
        if disclaimer_style == "upfront" and obs_closing in ("offer_help", "summary"):
            score += weight_close
        elif disclaimer_style == "minimal" and obs_closing == "none":
            score += weight_close
        elif disclaimer_style == "footer" and obs_closing in ("summary", "offer_help"):
            score += weight_close
        elif disclaimer_style == "inline":
            score += weight_close * 0.5  # inline is hard to detect from closing alone

        # Normalize to [0, 1]
        final_score = score / max_score if max_score > 0 else 0.0
        scores[family] = round(final_score, 4)

    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
