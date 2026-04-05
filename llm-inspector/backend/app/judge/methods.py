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
        passed, detail = _exact_match(text, params)
    elif method == "regex_match":
        passed, detail = _regex_match(text, params)
    elif method == "json_schema":
        passed, detail = _json_schema(text, params)
    elif method == "line_count":
        passed, detail = _line_count(text, params)
    elif method == "refusal_detect":
        passed, detail = _refusal_detect(text, params)
    elif method == "heuristic_style":
        passed, detail = _heuristic_style(text, params)
    elif method == "code_execution":
        passed, detail = _code_execution(text, params)
    elif method == "identity_consistency":
        passed, detail = _identity_consistency(text, params)
    elif method == "any_text":
        passed, detail = True, {"length": len(text)}
    elif method == "constraint_reasoning":
        passed, detail = _constraint_reasoning(text, params)
    elif method == "text_constraints":
        passed, detail = _text_constraints(text, params)
    else:
        passed, detail = None, {"error": f"unknown judge method: {method}"}

    rubric = (params.get("_meta", {}) or {}).get("judge_rubric") or {}
    if rubric:
        detail["judge_rubric"] = rubric
        detail["judge_explanation"] = _build_judge_explanation(method, passed, detail, rubric)
    return passed, detail


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


def _constraint_reasoning(text: str, params: dict) -> tuple[bool, dict]:
    """
    Constraint-first reasoning judge:
      - Prioritizes key constraint hit and boundary-proof signals.
      - Optional final answer pattern check.
      - Does NOT reward long chain-of-thought length.
    """
    lower_text = text.lower()
    key_constraints = params.get("key_constraints", [])
    boundary_signals = params.get("boundary_signals", [])
    anti_pattern_signals = params.get("anti_pattern_signals", [])

    constraint_hits = [kw for kw in key_constraints if kw.lower() in lower_text]
    boundary_hits = [kw for kw in boundary_signals if kw.lower() in lower_text]
    anti_pattern_hits = [kw for kw in anti_pattern_signals if kw.lower() in lower_text]

    target_pattern = params.get("target_pattern")
    answer_found = None
    if target_pattern:
        try:
            answer_found = bool(re.search(target_pattern, text, re.IGNORECASE | re.MULTILINE))
        except re.error:
            answer_found = False

    min_constraints = int(params.get("min_constraint_hits", 1))
    require_boundary = bool(params.get("require_boundary", True))
    allow_anti_pattern = bool(params.get("allow_anti_pattern", False))

    constraints_ok = len(constraint_hits) >= min_constraints
    boundary_ok = (len(boundary_hits) > 0) if require_boundary else True
    anti_pattern_ok = allow_anti_pattern or (len(anti_pattern_hits) == 0)
    answer_ok = True if answer_found is None else answer_found

    passed = constraints_ok and boundary_ok and anti_pattern_ok and answer_ok

    return passed, {
        "constraint_hits": constraint_hits,
        "boundary_hits": boundary_hits,
        "anti_pattern_hits": anti_pattern_hits,
        "constraints_ok": constraints_ok,
        "boundary_ok": boundary_ok,
        "anti_pattern_ok": anti_pattern_ok,
        "answer_found": answer_found,
        "reasoning_length": len(text),
        "note": "reasoning_length is informational only; score is constraint-first",
    }


def _strip_punctuation_for_count(text: str) -> str:
    return re.sub(r"[\s\.,，。！？；：、“”‘’\-—_~`!@#$%^&*()\[\]{}<>/\\\|+=]+", "", text)


def _text_constraints(text: str, params: dict) -> tuple[bool, dict]:
    """
    Unified text constraint judge for token-control style tasks.
    - character counting uses one normalized script-wide rule.
    - can enforce exact char count and forbidden chars simultaneously.
    """
    normalized = _strip_punctuation_for_count(text)
    exact_chars = params.get("exact_chars")
    forbidden_chars = params.get("forbidden_chars", [])

    char_count = len(normalized)
    char_ok = True if exact_chars is None else (char_count == int(exact_chars))

    forbidden_hits = [ch for ch in forbidden_chars if ch in normalized]
    forbidden_ok = len(forbidden_hits) == 0

    passed = char_ok and forbidden_ok
    return passed, {
        "char_count": char_count,
        "exact_chars": exact_chars,
        "char_ok": char_ok,
        "forbidden_hits": forbidden_hits,
        "forbidden_ok": forbidden_ok,
        "normalized_text": normalized[:200],
        "counting_rule": "strip whitespace and common punctuation, then count remaining characters",
    }


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


# ── Code Execution Judge ─────────────────────────────────────────────────────

def _code_execution(text: str, params: dict) -> tuple[bool, dict]:
    """
    Extract code from response, execute test cases in a sandboxed subprocess.
    Only supports Python. Timeout enforced.
    """
    import subprocess
    import tempfile
    import os

    language = params.get("language", "python")
    test_cases = params.get("test_cases", [])

    if language != "python":
        return False, {"error": f"unsupported language: {language}"}

    if not test_cases:
        return False, {"error": "no test cases provided"}

    # Extract code — strip markdown fences if present
    code = re.sub(r'^```(?:python)?\s*', '', text.strip(), flags=re.IGNORECASE)
    code = re.sub(r'\s*```$', '', code.strip())

    passed_count = 0
    total = len(test_cases)
    results = []

    for tc in test_cases:
        call_expr = tc["call"]
        expected = tc["expected"]
        # Build test script
        test_script = f"""{code}

_result = {call_expr}
print(repr(_result))
"""
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.py', delete=False, encoding='utf-8'
            ) as f:
                f.write(test_script)
                tmp_path = f.name

            proc = subprocess.run(
                ["python", tmp_path],
                capture_output=True, text=True, timeout=5,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            os.unlink(tmp_path)

            if proc.returncode == 0:
                actual_repr = proc.stdout.strip()
                expected_repr = repr(expected)
                tc_passed = actual_repr == expected_repr
                if tc_passed:
                    passed_count += 1
                results.append({
                    "call": call_expr, "expected": expected_repr,
                    "actual": actual_repr, "passed": tc_passed,
                })
            else:
                results.append({
                    "call": call_expr, "error": proc.stderr[:200], "passed": False,
                })
        except subprocess.TimeoutExpired:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
            results.append({"call": call_expr, "error": "timeout", "passed": False})
        except Exception as e:
            results.append({"call": call_expr, "error": str(e)[:200], "passed": False})

    all_passed = passed_count == total
    return all_passed, {
        "passed_count": passed_count,
        "total": total,
        "pass_rate": passed_count / total if total else 0,
        "results": results,
    }


# ── Identity Consistency Judge ───────────────────────────────────────────────

def _identity_consistency(text: str, params: dict) -> tuple[bool | None, dict]:
    """
    Check response consistency for identity/antispoof probes.
    - If expected_answer is given: check if response matches.
    - If extract_identity is True: extract model identity keywords.
    """
    detail: dict = {"response_length": len(text)}

    expected = params.get("expected_answer")
    if expected:
        clean = text.strip().strip('"').strip("'").strip('.')
        match = expected.lower() in clean.lower()
        detail["expected"] = expected
        detail["got"] = clean[:100]
        detail["match"] = match
        return match, detail

    if params.get("extract_identity"):
        text_lower = text.lower()
        identity_signals = {
            "gpt-4o": ["gpt-4o"],
            "gpt-4": ["gpt-4", "gpt4"],
            "gpt-3.5": ["gpt-3.5", "gpt-3.5-turbo"],
            "gpt-5": ["gpt-5", "gpt5"],
            "claude": ["claude"],
            "gemini": ["gemini"],
            "llama": ["llama", "meta"],
            "deepseek": ["deepseek"],
            "qwen": ["qwen", "tongyi"],
            "glm": ["glm", "chatglm"],
            "mistral": ["mistral"],
            "yi": ["yi-"],
        }
        detected = []
        for model_family, keywords in identity_signals.items():
            if any(kw in text_lower for kw in keywords):
                detected.append(model_family)
        detail["detected_identities"] = detected

        if params.get("detect_contradiction"):
            detail["system_override_response"] = text[:300]
            detail["leaked_identity"] = detected

        return None, detail

    return None, detail


def _build_judge_explanation(
    method: str,
    passed: bool | None,
    detail: dict,
    rubric: dict,
) -> str:
    criteria = rubric.get("criteria") or []
    fail_condition = rubric.get("fail_condition") or ""
    pass_label = "PASS" if passed is True else ("FAIL" if passed is False else "INFO")
    parts = [f"[{pass_label}] {rubric.get('name', method)}"]
    if criteria:
        parts.append("criteria=" + "; ".join(str(c) for c in criteria[:4]))
    if fail_condition and passed is False:
        parts.append(f"fail_condition={fail_condition}")

    if method == "json_schema" and detail.get("schema_errors"):
        parts.append("schema_errors=" + " | ".join(detail.get("schema_errors", [])[:3]))
    if method == "regex_match":
        parts.append(f"pattern_found={detail.get('found')}")
    if method == "line_count":
        parts.append(f"line_count={detail.get('actual_lines')}/{detail.get('expected_lines')}")
    if method == "constraint_reasoning":
        parts.append(
            "constraint_hits="
            f"{len(detail.get('constraint_hits', []))}, "
            "boundary_hits="
            f"{len(detail.get('boundary_hits', []))}, "
            "anti_pattern_hits="
            f"{len(detail.get('anti_pattern_hits', []))}"
        )

    return " ; ".join(parts)
