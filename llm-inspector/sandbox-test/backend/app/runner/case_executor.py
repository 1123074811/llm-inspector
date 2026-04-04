"""
Case executor — runs a single TestCase (with n_samples) against an adapter.
"""
from __future__ import annotations

import time
from app.core.schemas import (
    TestCase, CaseResult, SampleResult, LLMRequest, Message
)
from app.judge.methods import judge
from app.core.logging import get_logger

logger = get_logger(__name__)


def execute_case(adapter, model_name: str, case: TestCase) -> CaseResult:
    """
    Execute a TestCase against the adapter.
    Runs case.n_samples times, judges each response.
    """
    result = CaseResult(case=case)

    # Handle param comparison cases (temperature variance test)
    if "also_run_at" in case.params:
        return _execute_param_comparison(adapter, model_name, case)

    for i in range(case.n_samples):
        messages = []
        if case.system_prompt:
            messages.append(Message("system", case.system_prompt))
        messages.append(Message("user", case.user_prompt))

        # Build extra_params from case params if needed
        extra: dict = {}
        req_params = case.params.get("request_params", {})
        if req_params:
            extra.update(req_params)

        req = LLMRequest(
            model=model_name,
            messages=messages,
            temperature=case.temperature,
            max_tokens=case.max_tokens,
            extra_params=extra,
        )

        logger.info(
            "Executing case",
            case_id=case.id,
            sample=i,
            category=case.category,
        )

        resp = adapter.chat(req)

        # Inter-sample delay to avoid rate limits
        if i < case.n_samples - 1:
            time.sleep(0.3)

        passed, detail = judge(case.judge_method, resp.content, case.params)

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
                messages.append(Message("system", case.system_prompt))
            messages.append(Message("user", case.user_prompt))
            req = LLMRequest(
                model=model_name, messages=messages,
                temperature=temp, max_tokens=case.max_tokens,
            )
            resp = adapter.chat(req)
            if resp.content:
                outputs.append(resp.content.strip())
            result.samples.append(SampleResult(
                sample_index=len(result.samples),
                response=resp,
                judge_passed=None,
                judge_detail={"group": group_name, "temperature": temp},
            ))
            time.sleep(0.2)
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
