"""
Judge methods — evaluate LLM responses against expected outcomes.
Each judge returns (passed: bool, detail: dict).
"""
from __future__ import annotations

import json
import re


def judge(method: str, response_text: str | None, params: dict) -> tuple[bool | None, dict]:
    """
    Dispatch to the correct judge.
    Returns (passed, detail_dict).
    passed=None means "no judgement" (heuristic/style cases).
    """
    if response_text is None:
        return False, {"error": "no response text"}

    text = response_text.strip()

    if method == "exact_match":
        return _exact_match(text, params)
    elif method == "regex_match":
        return _regex_match(text, params)
    elif method == "json_schema":
        return _json_schema(text, params)
    elif method == "line_count":
        return _line_count(text, params)
    elif method == "refusal_detect":
        return _refusal_detect(text, params)
    elif method == "heuristic_style":
        return _heuristic_style(text, params)
    else:
        return None, {"error": f"unknown judge method: {method}"}


# ── Individual judges ─────────────────────────────────────────────────────────

def _exact_match(text: str, params: dict) -> tuple[bool, dict]:
    target = str(params.get("target", ""))
    # Strip common wrapping: quotes, code blocks, extra whitespace
    clean = text.strip().strip('"').strip("'").strip("`").strip()
    passed = clean == target or text == target
    return passed, {
        "expected": target,
        "got": text[:200],
        "clean_got": clean,
    }


def _regex_match(text: str, params: dict) -> tuple[bool, dict]:
    match_means_fail = params.get("match_means_fail", False)
    match_means_pass = params.get("match_means_pass", False)

    # CJK character count check — evaluated before pattern requirement
    max_cjk = params.get("max_cjk_chars")
    if max_cjk is not None:
        cjk_count = len(re.findall(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]', text))
        passed = cjk_count <= int(max_cjk)
        return passed, {"cjk_count": cjk_count, "max_allowed": max_cjk}

    pattern = params.get("pattern") or params.get("forbidden_pattern")
    if not pattern:
        return False, {"error": "no pattern specified"}

    try:
        found = bool(re.search(pattern, text, re.MULTILINE))
    except re.error as e:
        return False, {"error": f"invalid regex: {e}"}

    if match_means_fail:
        passed = not found
    elif match_means_pass:
        passed = found
    else:
        passed = found  # default: match = pass

    return passed, {
        "pattern": pattern,
        "found": found,
        "match_means_fail": match_means_fail,
    }


def _json_schema(text: str, params: dict) -> tuple[bool, dict]:
    # Strip markdown code blocks if present
    clean = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.IGNORECASE)
    clean = re.sub(r'\s*```$', '', clean.strip())

    # Parse JSON
    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError as e:
        return False, {"error": f"invalid JSON: {e}", "got": text[:200]}

    schema = params.get("schema", {})
    if not schema:
        return True, {"parsed": str(parsed)[:200]}

    # Basic schema validation (no jsonschema library)
    errors = _validate_schema(parsed, schema)
    passed = len(errors) == 0
    return passed, {"schema_errors": errors, "parsed_type": type(parsed).__name__}


def _validate_schema(value, schema: dict) -> list[str]:
    """Minimal JSON schema validator (type + required + properties)."""
    errors = []
    expected_type = schema.get("type")
    if expected_type:
        type_map = {
            "object": dict, "array": list, "string": str,
            "integer": int, "number": (int, float), "boolean": bool,
        }
        expected_py = type_map.get(expected_type)
        if expected_py and not isinstance(value, expected_py):
            errors.append(f"Expected type '{expected_type}', got '{type(value).__name__}'")
            return errors  # can't continue without correct type

    if isinstance(value, dict):
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                errors.append(f"Missing required field: '{key}'")
        props = schema.get("properties", {})
        for key, sub_schema in props.items():
            if key in value:
                sub_errors = _validate_schema(value[key], sub_schema)
                errors.extend(f"  {key}: {e}" for e in sub_errors)
    elif isinstance(value, list):
        items_schema = schema.get("items")
        if items_schema:
            for i, item in enumerate(value):
                sub_errors = _validate_schema(item, items_schema)
                errors.extend(f"  [{i}]: {e}" for e in sub_errors)

    return errors


def _line_count(text: str, params: dict) -> tuple[bool, dict]:
    expected = int(params.get("expected_lines", 3))
    lines = [l for l in text.splitlines() if l.strip()]
    passed = len(lines) == expected
    return passed, {"expected_lines": expected, "actual_lines": len(lines)}


def _refusal_detect(text: str, params: dict) -> tuple[bool | None, dict]:
    """
    Detect if model refused the request.
    Returns passed=True if a refusal was detected (we record it, not fail it).
    """
    keywords: list[str] = params.get("refusal_keywords", [
        "i can't", "i cannot", "i'm not able", "i won't", "i'm unable",
        "not able to", "unable to", "won't be able",
        "不能", "无法", "我不能", "不方便", "不适合", "不提供", "无法提供",
    ])
    text_lower = text.lower()
    refusal_found = any(kw.lower() in text_lower for kw in keywords)

    # Check for alternative suggestion
    alt_patterns = [
        r"instead", r"alternatively", r"however", r"can help",
        r"我可以", r"但是我可以", r"替代", r"建议",
    ]
    offers_alt = any(re.search(p, text_lower) for p in alt_patterns)

    return None, {  # None = not pass/fail, just record
        "refusal_detected": refusal_found,
        "offers_alternative": offers_alt,
        "response_length": len(text),
    }


def _heuristic_style(text: str, params: dict) -> tuple[None, dict]:
    """Extract style features — no pass/fail, pure feature collection."""
    extract_keys: list[str] = params.get("extract", [])
    features: dict = {}

    # Always compute basic features
    features["length"] = len(text)
    features["word_count"] = len(text.split())
    features["line_count"] = len(text.splitlines())

    # Markdown usage
    has_headers = bool(re.search(r'^#{1,4}\s', text, re.MULTILINE))
    has_bold = "**" in text
    has_bullet = bool(re.search(r'^\s*[-*•]\s', text, re.MULTILINE))
    has_numbered = bool(re.search(r'^\s*\d+\.\s', text, re.MULTILINE))
    has_code_fence = "```" in text
    features["has_headers"] = has_headers
    features["has_bold"] = has_bold
    features["has_bullet_list"] = has_bullet
    features["has_numbered_list"] = has_numbered
    features["has_code_fence"] = has_code_fence
    features["markdown_score"] = sum([has_headers, has_bold, has_bullet, has_code_fence])

    # Opening phrase
    first_line = text.split("\n")[0][:80] if text else ""
    features["opening_phrase"] = first_line

    # Disclaimer detection
    disclaimer_patterns = [
        r"it's important to note", r"please note", r"i should mention",
        r"important:", r"warning:", r"disclaimer",
        r"需要注意", r"重要提示", r"请注意",
    ]
    features["has_disclaimer"] = any(
        re.search(p, text.lower()) for p in disclaimer_patterns
    )

    # Language detection (simple heuristic)
    cjk_count = len(re.findall(r'[\u4e00-\u9fff]', text))
    features["cjk_char_count"] = cjk_count
    features["is_primarily_chinese"] = cjk_count > len(text) * 0.3

    # List marker style
    if has_bullet:
        features["list_style"] = "bullet"
    elif has_numbered:
        features["list_style"] = "numbered"
    else:
        features["list_style"] = "none"

    return None, features
