"""
Case executor — runs a single TestCase (with n_samples) against an adapter.
"""
from __future__ import annotations

import threading
import time
from app.core.schemas import (
    TestCase, CaseResult, SampleResult, LLMRequest, Message
)
from app.judge.methods import judge
from app.judge.consensus import should_run_consensus, arbitrate_with_semantic
from app.core.logging import get_logger

logger = get_logger(__name__)

# Appended to every system prompt (or used as sole system prompt when absent)
# to discourage markdown formatting that pollutes plain-text execution logs.
# Categories whose responses legitimately use code fences / structured text are excluded.
_PLAIN_TEXT_INSTRUCTION = (
    "【格式要求】请严格用纯文本回复，禁止使用任何 Markdown 格式符号，"
    "包括但不限于：**加粗**、*斜体*、# 标题、- 列表符号、> 引用块、```代码块```。"
    "直接输出内容，不加任何格式装饰。"
    " [Format] Plain text only. No Markdown: no **bold**, *italic*, # headings, "
    "- bullets, > blockquotes, or ```code fences```."
)
_SKIP_PLAIN_TEXT_CATEGORIES = frozenset({"coding", "tool_use"})

TIMEOUT_MAP = {
    "protocol": 15,
    "instruction": 30,
    "reasoning": 60,
    "coding": 90,
    "extraction": 30,
    "consistency": 30,
}

INTER_SAMPLE_DELAY_BASE = 0.1
_delay_lock = threading.Lock()
_delay_state = {
    "current": INTER_SAMPLE_DELAY_BASE,
    "success_count": 0,
}

def _update_delay(resp):
    with _delay_lock:
        if getattr(resp, "status_code", 200) == 429 or getattr(resp, "error_type", "") == "rate_limit":
            _delay_state["current"] = min(2.0, _delay_state["current"] * 2.0)
            _delay_state["success_count"] = 0
        elif resp and resp.ok:
            _delay_state["success_count"] += 1
            if _delay_state["success_count"] >= 10:
                _delay_state["current"] = INTER_SAMPLE_DELAY_BASE
                _delay_state["success_count"] = 0

def _get_current_delay() -> float:
    with _delay_lock:
        return _delay_state["current"]


def execute_case(adapter, model_name: str, case: TestCase) -> CaseResult:
    """
    Execute a TestCase against the adapter.
    Runs case.n_samples times, judges each response.
    """
    result = CaseResult(case=case)

    # Handle param comparison cases (temperature variance test)
    if "also_run_at" in case.params:
        return _execute_param_comparison(adapter, model_name, case)

    # Adaptive sampling based on IRT information (v14 Phase 6)
    n_samples = case.n_samples
    try:
        from app.runner.adaptive_sampling import get_adaptive_n_samples
        case_dict = {
            "irt_a": getattr(case, "irt_a", None),
            "irt_b": getattr(case, "irt_b", None),
            "irt_c": getattr(case, "irt_c", None),
        }
        adaptive_n = get_adaptive_n_samples(case_dict, current_theta=None)
        if n_samples > adaptive_n:
            n_samples = adaptive_n
    except Exception:
        pass  # Non-fatal fallback to original n_samples

    consecutive_truncations = 0
    for i in range(n_samples):
        # Early-stop: if 2+ consecutive responses were truncated, skip remaining
        # samples — they will almost certainly truncate too, wasting tokens.
        if consecutive_truncations >= 2:
            logger.info(
                "Skipping remaining samples due to consecutive truncations",
                case_id=case.id, skipped_from=i, total=case.n_samples,
            )
            break

        messages = []
        if case.system_prompt:
            sys_content = case.system_prompt
            if case.category not in _SKIP_PLAIN_TEXT_CATEGORIES:
                sys_content = sys_content + "\n" + _PLAIN_TEXT_INSTRUCTION
            messages.append(Message("system", sys_content))
        elif case.category not in _SKIP_PLAIN_TEXT_CATEGORIES:
            messages.append(Message("system", _PLAIN_TEXT_INSTRUCTION))
        # Wire in PromptOptimizer (v14 Phase 6 — B5 fix)
        raw_prompt = case.user_prompt
        try:
            from app.runner.prompt_optimizer import prompt_optimizer
            compiled = prompt_optimizer.compile_prompt(
                test_prompt=raw_prompt,
                judge_method=case.judge_method or "",
                max_examples=2,
            )
            if compiled and compiled.prompt and compiled.prompt != raw_prompt:
                raw_prompt = compiled.prompt
        except Exception:
            pass  # Non-fatal: optimizer failure falls back to original prompt
        messages.append(Message("user", raw_prompt))

        # Build extra_params from case params if needed
        extra: dict = {}
        req_params = case.params.get("request_params", {})
        if req_params:
            extra.update(req_params)

        timeout = TIMEOUT_MAP.get(case.category, 60)

        req = LLMRequest(
            model=model_name,
            messages=messages,
            temperature=case.temperature,
            max_tokens=case.max_tokens,
            timeout_sec=timeout,
            extra_params=extra,
        )

        logger.info(
            "Executing case",
            case_id=case.id,
            sample=i,
            category=case.category,
        )

        # Cache check: only for deterministic (temperature=0) requests
        _cache_key: str | None = None
        _cache_hit = False
        if case.temperature == 0.0:
            try:
                from app.runner.cache_strategy import cache_strategy
                _cache_payload = {
                    "model": model_name,
                    "messages": [{"role": m.role, "content": m.content} for m in messages],
                    "max_tokens": case.max_tokens,
                }
                _cache_key = cache_strategy.build_key(
                    getattr(adapter, "base_url", ""), _cache_payload
                )
                _cached = cache_strategy.get(_cache_key)
                if _cached is not None:
                    resp = _cached
                    _cache_hit = True
                    logger.info(
                        "Cache hit",
                        case_id=case.id, sample=i, category=case.category,
                    )
            except Exception:
                pass  # Non-fatal: cache miss → proceed normally

        if not _cache_hit:
            try:
                resp = adapter.chat(req)
            except Exception as e:
                logger.error(
                    "Adapter chat failed",
                    case_id=case.id,
                    sample=i,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                # Create a mock response for error handling
                from app.core.schemas import LLMResponse
                resp = LLMResponse(
                    content="",
                    finish_reason="error",
                    status_code=500,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    latency_ms=0,
                    usage_total_tokens=0,
                )
            # Store successful deterministic response in cache
            if _cache_key and case.temperature == 0.0:
                try:
                    from app.runner.cache_strategy import cache_strategy
                    cache_strategy.set(_cache_key, resp, category=case.category or "")
                except Exception:
                    pass  # Non-fatal

        # Inter-sample delay to avoid rate limits
        _update_delay(resp)
        if i < case.n_samples - 1:
            time.sleep(_get_current_delay())

        # Detect truncated responses (finish_reason == "length")
        truncated = resp.finish_reason == "length"
        if truncated:
            consecutive_truncations += 1
            logger.warning(
                "Response truncated (finish_reason=length)",
                case_id=case.id, sample=i, max_tokens=case.max_tokens,
            )

            # v16 fix: Some APIs (e.g. deepseek) don't strictly honor max_tokens
            # and return more tokens than requested, then mark finish_reason=length.
            # For content-rich judge methods, the response may still contain
            # enough information to judge. Only skip for format-strict methods.
            _FORMAT_STRICT_JUDGES = {
                "exact_match", "regex_match", "json_schema", "line_count",
                "text_constraints", "tokenizer_fingerprint",
            }
            if case.judge_method in _FORMAT_STRICT_JUDGES:
                # Format-strict: truncation likely invalidates the result
                result.samples.append(SampleResult(
                    sample_index=i,
                    response=resp,
                    judge_passed=False,
                    judge_detail={
                        "truncated": True,
                        "reason": "response truncated (finish_reason=length), "
                                  "format-strict judge skipped",
                        "max_tokens": case.max_tokens,
                    },
                ))
            else:
                # Content-rich: still attempt judging with truncation note
                passed, detail = judge(case.judge_method, resp.content, case.params)
                if detail is None:
                    detail = {}
                detail["truncated"] = True
                detail["max_tokens"] = case.max_tokens
                result.samples.append(SampleResult(
                    sample_index=i,
                    response=resp,
                    judge_passed=passed,
                    judge_detail=detail,
                ))
        else:
            consecutive_truncations = 0
            passed, detail = judge(case.judge_method, resp.content, case.params)

            # Phase B: rule-vs-semantic consensus arbitration for complex cases
            if should_run_consensus(case.judge_method, case.params):
                try:
                    passed, detail = arbitrate_with_semantic(
                        method=case.judge_method,
                        text=resp.content or "",
                        params=case.params or {},
                        rule_passed=passed,
                        rule_detail=detail or {},
                    )
                except Exception as e:
                    logger.warning(
                        "Consensus arbitration failed, fallback to rule judge",
                        case_id=case.id,
                        sample=i,
                        error=str(e),
                    )

            result.samples.append(SampleResult(
                sample_index=i,
                response=resp,
                judge_passed=passed,
                judge_detail=detail,
            ))

    # v16 Phase 2: Rescue round — if all samples failed, try one more with adjusted params
    result = _rescue_round(adapter, model_name, case, result)

    return result


def _rescue_round(adapter, model_name: str, case: TestCase, result: CaseResult) -> CaseResult:
    """
    v16 Phase 2: Case-level rescue round.

    If ALL samples in a case failed (error or judge_failed), attempt one
    rescue retry with adjusted parameters:
      - Truncated responses: double max_tokens
      - Error responses: retry with simplified prompt
      - Content-filtered: mark excluded_from_scoring
    """
    # Check if rescue is needed
    has_any_pass = any(s.judge_passed for s in result.samples if s.judge_passed is not None)
    if has_any_pass:
        return result  # No rescue needed

    # Determine rescue strategy from failure patterns
    truncation_failures = [s for s in result.samples if s.judge_detail.get("truncated")]
    error_failures = [s for s in result.samples if s.response.error_type is not None]
    content_filtered = [s for s in result.samples
                        if s.response.finish_reason == "content_filter"
                        or (s.response.error_payload or {}).get("type") == "content_filter"]

    # Mark content-filtered samples as excluded from scoring
    for s in content_filtered:
        s.excluded_from_scoring = True
        s.retry_type = "content_filter"

    rescue_max_tokens = case.max_tokens
    rescue_prompt = case.user_prompt
    rescue_type: str | None = None

    if truncation_failures:
        # Double max_tokens for truncation rescue
        rescue_max_tokens = min(case.max_tokens * 2, 4096)
        rescue_type = "truncation"
    elif error_failures:
        # Simplify prompt for error rescue (shorter, more direct)
        rescue_prompt = case.user_prompt[:200] if len(case.user_prompt) > 200 else case.user_prompt
        rescue_type = "error"
    else:
        # All judge-failed but no clear pattern — try once with doubled tokens
        rescue_max_tokens = min(case.max_tokens * 2, 4096)
        rescue_type = "generic"

    # Perform rescue attempt
    messages = []
    if case.system_prompt:
        sys_content = case.system_prompt
        if case.category not in _SKIP_PLAIN_TEXT_CATEGORIES:
            sys_content = sys_content + "\n" + _PLAIN_TEXT_INSTRUCTION
        messages.append(Message("system", sys_content))
    elif case.category not in _SKIP_PLAIN_TEXT_CATEGORIES:
        messages.append(Message("system", _PLAIN_TEXT_INSTRUCTION))
    messages.append(Message("user", rescue_prompt))

    req = LLMRequest(
        model=model_name,
        messages=messages,
        temperature=case.temperature,
        max_tokens=rescue_max_tokens,
        timeout_sec=TIMEOUT_MAP.get(case.category, 60),
    )

    logger.info(
        "Rescue round attempt",
        case_id=case.id,
        rescue_type=rescue_type,
        max_tokens=rescue_max_tokens,
    )

    try:
        resp = adapter.chat(req)
    except Exception as e:
        logger.error(
            "Rescue round adapter failed",
            case_id=case.id,
            error=str(e),
        )
        from app.core.schemas import LLMResponse
        resp = LLMResponse(
            content="",
            finish_reason="error",
            status_code=500,
            error_type=type(e).__name__,
            error_message=str(e),
            latency_ms=0,
            usage_total_tokens=0,
        )

    if resp.finish_reason == "length":
        # Still truncated — exclude from scoring
        result.samples.append(SampleResult(
            sample_index=len(result.samples),
            response=resp,
            judge_passed=False,
            judge_detail={"truncated": True, "rescue_attempt": True, "reason": "still truncated after rescue"},
            excluded_from_scoring=True,
            retry_type=rescue_type,
        ))
    elif resp.error_type:
        # Still error — exclude from scoring
        result.samples.append(SampleResult(
            sample_index=len(result.samples),
            response=resp,
            judge_passed=False,
            judge_detail={"error": True, "rescue_attempt": True, "reason": "still error after rescue"},
            excluded_from_scoring=True,
            retry_type=rescue_type,
        ))
    else:
        # Rescue succeeded — judge normally
        passed, detail = judge(case.judge_method, resp.content, case.params)
        result.samples.append(SampleResult(
            sample_index=len(result.samples),
            response=resp,
            judge_passed=passed,
            judge_detail=detail or {},
            retry_type=rescue_type,
        ))
        if passed:
            result.rescued = True
            logger.info(
                "Rescue round succeeded",
                case_id=case.id,
                rescue_type=rescue_type,
            )

    return result


def _execute_param_comparison(adapter, model_name: str, case: TestCase) -> CaseResult:
    """
    Special handler for temperature variance tests.
    Runs case n_samples times at case.temperature,
    then also_run_at[0] n_samples times at the alt temperature.
    Judges by comparing variance between groups.
    """
    result = CaseResult(case=case)
    alt_configs = case.params.get("also_run_at", [])

    def _run_group(temp: float, n: int, group_name: str) -> list[str]:
        outputs = []
        for i in range(n):
            messages = []
            if case.system_prompt:
                sys_content = case.system_prompt
                if case.category not in _SKIP_PLAIN_TEXT_CATEGORIES:
                    sys_content = sys_content + "\n" + _PLAIN_TEXT_INSTRUCTION
                messages.append(Message("system", sys_content))
            elif case.category not in _SKIP_PLAIN_TEXT_CATEGORIES:
                messages.append(Message("system", _PLAIN_TEXT_INSTRUCTION))
            messages.append(Message("user", case.user_prompt))
            req = LLMRequest(
                model=model_name, messages=messages,
                temperature=temp, max_tokens=case.max_tokens,
            )
            try:
                resp = adapter.chat(req)
            except Exception as e:
                logger.error(
                    "Adapter chat failed in param comparison",
                    case_id=case.id,
                    temperature=temp,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                # Create a mock response for error handling
                from app.core.schemas import LLMResponse
                resp = LLMResponse(
                    content="",
                    finish_reason="error",
                    status_code=500,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    latency_ms=0,
                    usage_total_tokens=0,
                )
            if resp.content:
                outputs.append(resp.content.strip())
            result.samples.append(SampleResult(
                sample_index=len(result.samples),
                response=resp,
                judge_passed=None,
                judge_detail={"group": group_name, "temperature": temp},
            ))
            _update_delay(resp)
            time.sleep(_get_current_delay())
        return outputs

    # Primary group
    primary_outputs = _run_group(case.temperature, case.n_samples, "primary")

    # Alt groups
    for alt in alt_configs:
        alt_outputs = _run_group(alt["temperature"], alt["n_samples"], "alt")

        # Measure variance: unique outputs / total outputs
        primary_unique = len(set(primary_outputs)) / max(len(primary_outputs), 1)
        alt_unique = len(set(alt_outputs)) / max(len(alt_outputs), 1)

        param_effective = primary_unique > alt_unique or (
            case.temperature >= 1.0 and primary_unique > 0.4
        )

        # Annotate the last sample with the verdict
        if result.samples:
            last = result.samples[-1]
            last.judge_passed = param_effective
            last.judge_detail.update({
                "primary_diversity": round(primary_unique, 3),
                "alt_diversity": round(alt_unique, 3),
                "temperature_param_effective": param_effective,
                "primary_outputs": primary_outputs[:3],
                "alt_outputs": alt_outputs[:3],
            })

    return result
