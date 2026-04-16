"""
Async Case Executor — Phase B of asyncio migration.

Mirrors case_executor.py but all I/O calls are awaited.
Supports both AsyncOpenAICompatibleAdapter (native) and sync
OpenAICompatibleAdapter (via asyncio.to_thread shim).
"""
from __future__ import annotations

import asyncio
import time
from app.core.schemas import (
    TestCase, CaseResult, SampleResult, LLMRequest, Message,
)
from app.judge.methods import judge
from app.core.logging import get_logger

logger = get_logger(__name__)

# Mirror of TIMEOUT_MAP in case_executor.py
TIMEOUT_MAP = {
    "protocol": 15,
    "instruction": 30,
    "reasoning": 60,
    "coding": 90,
    "extraction": 30,
    "consistency": 30,
}

INTER_SAMPLE_DELAY_BASE = 0.1

# Shared plain-text instruction (mirrors case_executor.py)
_PLAIN_TEXT_INSTRUCTION = (
    "【格式要求】请严格用纯文本回复，禁止使用任何 Markdown 格式符号，"
    "包括但不限于：**加粗**、*斜体*、# 标题、- 列表符号、> 引用块、```代码块```。"
    "直接输出内容，不加任何格式装饰。"
    " [Format] Plain text only. No Markdown: no **bold**, *italic*, # headings, "
    "- bullets, > blockquotes, or ```code fences```."
)
_SKIP_PLAIN_TEXT_CATEGORIES = frozenset({"coding", "tool_use"})


def _compute_new_delay(current: float, resp, success_count: int) -> tuple[float, int]:
    """Returns (new_delay, new_success_count) without mutating global state."""
    if getattr(resp, "status_code", 200) == 429 or getattr(resp, "error_type", "") == "rate_limit":
        return min(2.0, current * 2.0), 0
    elif resp and resp.ok:
        new_count = success_count + 1
        if new_count >= 10:
            return INTER_SAMPLE_DELAY_BASE, 0
        return current, new_count
    return current, success_count


async def _achat(adapter, req: LLMRequest):
    """Call adapter.achat (async) or adapter.chat (sync via to_thread)."""
    if hasattr(adapter, "achat"):
        return await adapter.achat(req)
    # Sync adapter — offload to thread pool
    return await asyncio.to_thread(adapter.chat, req)


async def async_execute_case(adapter, model_name: str, case: TestCase) -> CaseResult:
    """
    Async version of execute_case.

    Executes a TestCase against the adapter, running case.n_samples times,
    judging each response. Replaces time.sleep with asyncio.sleep so the
    event loop can schedule other coroutines during inter-sample delays.
    """
    result = CaseResult(case=case)

    if "also_run_at" in case.params:
        return await _async_execute_param_comparison(adapter, model_name, case)

    delay = INTER_SAMPLE_DELAY_BASE
    success_count = 0
    consecutive_truncations = 0

    for i in range(case.n_samples):
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
        messages.append(Message("user", case.user_prompt))

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
            "Async executing case",
            case_id=case.id, sample=i, category=case.category,
        )

        try:
            # v10: Exponential Backoff with Jitter for robust execution
            import random
            max_retries = 3
            base_delay = 1.0
            for attempt in range(max_retries):
                try:
                    resp = await _achat(adapter, req)
                    break
                except Exception as e:
                    is_rate_limit = "429" in str(e) or "rate_limit" in getattr(e, "error_type", "")
                    if attempt < max_retries - 1 and is_rate_limit:
                        sleep_time = random.uniform(0, min(60.0, base_delay * (2 ** attempt)))
                        logger.warning(
                            "Rate limited, retrying with exponential backoff",
                            case_id=case.id, attempt=attempt+1, sleep_sec=sleep_time
                        )
                        await asyncio.sleep(sleep_time)
                    else:
                        raise e
        except Exception as e:
            logger.error(
                "Async adapter chat failed after retries",
                case_id=case.id, sample=i, category=case.category,
                error=str(e), error_type=type(e).__name__,
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
                token_usage=None,
            )

        # Adaptive inter-sample delay (async sleep — doesn't block the loop)
        delay, success_count = _compute_new_delay(delay, resp, success_count)
        if i < case.n_samples - 1:
            await asyncio.sleep(delay)

        # Truncation detection
        truncated = resp.finish_reason == "length"
        if truncated:
            consecutive_truncations += 1
            logger.warning(
                "Response truncated (finish_reason=length)",
                case_id=case.id, sample=i, max_tokens=case.max_tokens,
            )
            result.samples.append(SampleResult(
                sample_index=i,
                response=resp,
                judge_passed=False,
                judge_detail={
                    "truncated": True,
                    "reason": "response truncated (finish_reason=length), output incomplete — judge skipped",
                    "max_tokens": case.max_tokens,
                },
            ))
        else:
            consecutive_truncations = 0
            # judge is pure Python (CPU-bound, no I/O) — call directly
            passed, detail = judge(case.judge_method, resp.content, case.params)
            result.samples.append(SampleResult(
                sample_index=i,
                response=resp,
                judge_passed=passed,
                judge_detail=detail,
            ))

    return result


async def _async_execute_param_comparison(adapter, model_name: str, case: TestCase) -> CaseResult:
    """Async version of temperature variance test."""
    result = CaseResult(case=case)
    alt_configs = case.params.get("also_run_at", [])

    async def _run_group(temp: float, n: int, group_name: str) -> list[str]:
        outputs = []
        delay = INTER_SAMPLE_DELAY_BASE
        success_count = 0
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
                resp = await _achat(adapter, req)
            except Exception as e:
                logger.error(
                    "Async adapter chat failed in param comparison",
                    case_id=case.id, temperature=temp,
                    error=str(e), error_type=type(e).__name__,
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
                    token_usage=None,
                )
            if resp.content:
                outputs.append(resp.content.strip())
            result.samples.append(SampleResult(
                sample_index=len(result.samples),
                response=resp,
                judge_passed=None,
                judge_detail={"group": group_name, "temperature": temp},
            ))
            delay, success_count = _compute_new_delay(delay, resp, success_count)
            await asyncio.sleep(delay)
        return outputs

    primary_outputs = await _run_group(case.temperature, case.n_samples, "primary")

    for alt in alt_configs:
        alt_outputs = await _run_group(alt["temperature"], alt["n_samples"], "alt")

        primary_unique = len(set(primary_outputs)) / max(len(primary_outputs), 1)
        alt_unique = len(set(alt_outputs)) / max(len(alt_outputs), 1)
        param_effective = primary_unique > alt_unique or (
            case.temperature >= 1.0 and primary_unique > 0.4
        )

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
