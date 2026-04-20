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
            # Mark as failed with truncation detail — don't waste judge on
            # incomplete text that will almost certainly miss key signals.
            result.samples.append(SampleResult(
                sample_index=i,
                response=resp,
                judge_passed=False,
                judge_detail={
                    "truncated": True,
                    "reason": "response truncated (finish_reason=length), "
                              "output incomplete — judge skipped",
                    "max_tokens": case.max_tokens,
                },
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
