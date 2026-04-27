"""
runner/case_prep.py — case loading, preparation, and selection helpers

Functions for loading the test suite, managing benchmarks, preparing cases
(token capping, adaptive sampling, phase assignment), and selecting confirmatory
cases for predetect hypothesis verification.

Extracted from orchestrator.py to keep individual files under ~400 lines.
"""
from __future__ import annotations

import pathlib
import time

from app.core.schemas import TestCase, CaseResult
from app.core.eval_schemas import EvalTestCase
from app.core.config import settings
from app.core.logging import get_logger
from app.runner.compression import compressor as prompt_compressor
from app.runner.prompt_optimizer import prompt_optimizer
from app.repository import repo

logger = get_logger(__name__)

_FIXTURES_DIR = pathlib.Path(__file__).parent.parent / "fixtures"


def _load_suite(suite_version: str, test_mode: str) -> list[TestCase]:
    """Load test cases from DB (seeded from fixture JSON) and compress prompts.

    v11: Returns EvalTestCase instances for CDM/Bayesian/telemetry support.
    """
    raw_cases = repo.load_cases(suite_version, test_mode)
    cases = []
    for c in raw_cases:
        # v11: Use from_db_dict which properly parses skill_vector,
        # bayesian_prior, and eval_meta from params._meta
        eval_case = EvalTestCase.from_db_dict(c)

        # v10: Lossless prompt compression (applied after construction)
        compressed_user = prompt_compressor.compress(eval_case.user_prompt)
        compressed_system = prompt_compressor.compress(eval_case.system_prompt) if eval_case.system_prompt else None

        eval_case.user_prompt = compressed_user
        eval_case.system_prompt = compressed_system

        cases.append(eval_case)
    return cases


_benchmark_cache: dict[str, tuple[float, list[dict]]] = {}


def invalidate_benchmark_cache(suite_version: str | None = None) -> None:
    """Drop the in-process benchmark cache.

    Must be called by handlers that mutate ``golden_baselines`` (create / delete /
    activate / deactivate). Otherwise a run started within
    ``BENCHMARK_CACHE_TTL_SEC`` of a baseline change will silently use the
    *previous* baseline list and ignore the user's edit, which produced the
    "set baseline → next run doesn't see it" bug observed in v16 acceptance.
    """
    global _benchmark_cache
    if suite_version is None:
        _benchmark_cache.clear()
    else:
        _benchmark_cache.pop(suite_version, None)


def _load_benchmarks(suite_version: str) -> list[dict]:
    """Load benchmarks with a short TTL cache to avoid repeated DB reads.

    The TTL is intentionally short (default 120 s) because benchmarks change
    only when a user marks a run as baseline — and on that path we explicitly
    call ``invalidate_benchmark_cache()`` so a stale cache cannot survive a
    baseline mutation. The TTL exists only to coalesce many reads inside a
    single run (predetect → similarity → reporting all hit it).
    """
    now = time.time()
    cached = _benchmark_cache.get(suite_version)
    ttl = max(1, settings.BENCHMARK_CACHE_TTL_SEC)
    if cached:
        ts, data = cached
        if now - ts <= ttl:
            return data

    data = repo.get_benchmarks(suite_version)
    _benchmark_cache[suite_version] = (now, data)
    return data


def _save_case_results_batch(run_id: str, results: list[CaseResult]) -> None:
    if not results:
        return
    batch_rows = []
    for result in results:
        case = result.case
        for sample in result.samples:
            r = sample.response
            request_payload = {
                "messages": [{"role": "user", "content": case.user_prompt[:200]}],
                "temperature": case.temperature,
                "max_tokens": case.max_tokens,
            }
            if case.system_prompt:
                request_payload["messages"].insert(
                    0, {"role": "system", "content": case.system_prompt[:100]},
                )
            batch_rows.append({
                "run_id": run_id,
                "case_id": case.id,
                "sample_index": sample.sample_index,
                "resp_data": {
                    "request": request_payload,
                    "response_text": r.content,
                    "raw_response": r.raw_json,
                    "raw_headers": r.headers,
                    "status_code": r.status_code,
                    "latency_ms": r.latency_ms,
                    "first_token_ms": r.first_token_ms,
                    "finish_reason": r.finish_reason,
                    "usage_prompt_tokens": r.usage_prompt_tokens,
                    "usage_completion_tokens": r.usage_completion_tokens,
                    "usage_total_tokens": r.usage_total_tokens,
                    "error_type": r.error_type,
                    "error_message": r.error_message,
                    "judge_passed": sample.judge_passed,
                    "judge_detail": sample.judge_detail,
                },
            })

    if hasattr(repo, "save_response_batch"):
        repo.save_response_batch(batch_rows)
    else:
        for item in batch_rows:
            repo.save_response(
                run_id=item["run_id"],
                case_id=item["case_id"],
                sample_index=item["sample_index"],
                resp_data=item["resp_data"],
            )


def _mode_concurrency(test_mode: str) -> int:
    if test_mode == "quick":
        return max(1, settings.CONCURRENCY_QUICK)
    if test_mode == "deep":
        return max(1, settings.CONCURRENCY_DEEP)
    return max(1, settings.CONCURRENCY_STANDARD)


def _case_value(c: TestCase) -> float:
    """Information gain per token (higher is better)."""
    # High priority baseline discriminators (from optimization plan)
    PRIORITY_CATEGORIES = {
        "reasoning": 3.0,
        "coding": 3.0,
        "extraction": 2.5,
        "consistency": 2.5,
        "instruction": 2.0,
        "knowledge": 1.8,
        "protocol": 1.5,
    }

    info_gain = PRIORITY_CATEGORIES.get(c.category)
    if info_gain is None:
        info_gain = {
            "antispoof": 1.4,
            "system": 1.0,
            "param": 0.95,
            "refusal": 0.9,
            "safety": 0.9,
            "tool_use": 0.85,
            "fingerprint": 1.3,
            "style": 0.75,
            "performance": 0.7,
        }.get(c.category, 0.8)

    est_cost = max(40.0, float(c.max_tokens * max(1, c.n_samples)))
    return (info_gain * max(0.2, c.weight)) / est_cost


def _adaptive_samples(case: TestCase, mode: str, run_id: str | None = None) -> int:
    """
    v6: Adaptive sampling with historical variance check.
    High variance items need more samples for stable estimation.
    """
    judge = case.judge_method

    DETERMINISTIC_JUDGES = {
        "exact_match", "regex_match", "json_schema", "line_count",
        "text_constraints", "tokenizer_fingerprint", "spec_contradiction_check",
    }
    if judge in DETERMINISTIC_JUDGES:
        return 1

    EXTRACTION_JUDGES = {
        "prompt_leak_detect", "forbidden_word_extract", "path_leak_detect",
        "tool_config_leak_detect", "memory_leak_detect",
    }
    if judge in EXTRACTION_JUDGES:
        return 1

    # v6: Check historical variance from earlier cases in this run
    if run_id:
        try:
            from app.repository.repo import get_repository
            repo = get_repository()
            historical = repo.get_case_results_for_run(run_id, case_id=case.id)
            if historical and len(historical) >= 3:
                # Calculate pass rate variance across historical samples
                pass_results = [1 if r.get("passed") else 0 for r in historical]
                if pass_results:
                    mean_pass = sum(pass_results) / len(pass_results)
                    variance = sum((p - mean_pass) ** 2 for p in pass_results) / len(pass_results)
                    # High variance (>0.25 = 50% std dev) needs more samples
                    if variance > 0.25 and mode in ("standard", "deep"):
                        return max(case.n_samples, 3)
        except Exception:
            pass  # Fall through to mode-based logic

    if mode == "quick":
        return 1
    elif mode == "standard":
        return min(case.n_samples, 2)
    elif mode == "deep":
        return max(case.n_samples, 2)  # deep: at least 2 samples
    else:
        return case.n_samples


# -- Categories that best discriminate each model family.
# When predetect identifies a candidate, these are the test dimensions
# that differ most between the candidate and its close alternatives.
# The selection is based on known behavioural divergence patterns:
#   - identity/antispoof: directly confirms who the model claims to be
#   - consistency: multi-sample stability is family-specific
#   - reasoning: adversarial reasoning patterns are highly distinctive
#   - instruction: format compliance varies by architecture
#   - system: system prompt handling differs across families
#   - param: temperature effectiveness reveals proxy layers

_CONFIRMATORY_CATEGORIES: dict[str, list[str]] = {
    # Always include these (universal discriminators)
    "_core": ["antispoof", "consistency", "param"],
    # Family-specific additions
    "OpenAI":    ["reasoning", "instruction", "system"],
    "Anthropic": ["reasoning", "instruction", "refusal"],
    "Google":    ["reasoning", "coding", "instruction"],
    "DeepSeek":  ["reasoning", "coding", "instruction"],
    "Meta":      ["instruction", "coding", "system"],
    "Alibaba":   ["instruction", "reasoning", "system"],
    "Zhipu":     ["instruction", "reasoning", "refusal"],
    "Mistral":   ["instruction", "coding", "reasoning"],
    "Moonshot":  ["instruction", "reasoning", "system"],
    "Baichuan":  ["instruction", "reasoning", "system"],
}


def _select_confirmatory_cases(
    all_cases: list[TestCase],
    candidate: str,
) -> list[TestCase]:
    """
    Select a subset of test cases optimised for confirming/denying
    the predetect hypothesis. Returns the full suite if candidate is
    unknown (no matching family).

    Strategy:
      1. Always include core discriminator categories (antispoof, consistency, param).
      2. Add 3 family-specific categories based on known divergence patterns.
      3. Within selected categories, pick by _case_value ranking.
      4. Ensure minimum coverage: at least 12 cases, at most 20.
    """
    # Match candidate to a family key
    candidate_lower = candidate.lower()
    matched_family: str | None = None
    for family in _CONFIRMATORY_CATEGORIES:
        if family == "_core":
            continue
        if family.lower() in candidate_lower:
            matched_family = family
            break

    if not matched_family:
        # Unknown family — run full suite, no savings possible
        return all_cases

    # Build the set of target categories
    target_cats = set(_CONFIRMATORY_CATEGORIES["_core"])
    target_cats.update(_CONFIRMATORY_CATEGORIES[matched_family])

    # Select cases matching target categories
    selected = [c for c in all_cases if c.category in target_cats]

    # Ensure minimum coverage: if too few, backfill with highest-value remaining
    if len(selected) < 12:
        remaining = [c for c in all_cases if c.category not in target_cats]
        remaining.sort(key=_case_value, reverse=True)
        selected.extend(remaining[:12 - len(selected)])

    # Cap at 20 to bound cost, keeping highest-value
    if len(selected) > 20:
        selected.sort(key=_case_value, reverse=True)
        selected = selected[:20]

    return selected


def _prepare_cases(cases: list[TestCase], test_mode: str) -> tuple[list[TestCase], list[TestCase]]:
    """
    Returns (phase1_cases, phase2_cases) based on mode_level tagging.

    Mode inclusion is progressive:
      quick:    only mode_level="quick" cases
      standard: mode_level in ("quick", "standard")
      deep:     all cases (quick + standard + deep)

    Phase assignment:
      quick:    single phase (all quick-level cases)
      standard: phase1=quick-level, phase2=standard-level
      deep:     phase1=quick+standard-level, phase2=deep-level (+ multi-sampling)
    """
    # -- Helper: extract mode_level from case params._meta --
    def _get_mode_level(c: TestCase) -> str:
        meta = c.params.get("_meta", {}) if c.params else {}
        return meta.get("mode_level", "standard")

    # -- Phase 1 Optimization: Fine-grained token capping ---
    ordered = list(cases)

    _JUDGE_MAX_TOKENS: dict[str, int] = {
        "exact_match":          15,
        "json_schema":         200,
        "line_count":          200,
        "code_execution":      600,
        "regex_match":         300,
        "refusal_detect":      300,
        "constraint_reasoning": 1200,
        "text_constraints":    400,
        "identity_consistency": 400,
        "heuristic_style":     600,
        "any_text":            600,
        "prompt_leak_detect":   1500,
        "forbidden_word_extract": 800,
        "path_leak_detect":     800,
        "tool_config_leak_detect": 800,
        "memory_leak_detect":   800,
        "denial_pattern_detect": 400,
        "spec_contradiction_check": 150,
        "refusal_style_fingerprint": 500,
        "language_bias_detect": 400,
        "tokenizer_fingerprint": 30,
        # New judge methods
        "multi_step_verify":   1000,
        "yaml_csv_validate":   400,
        "hallucination_detect": 400,
        "context_overflow_detect": 1500,
        "semantic_judge":       800,
    }

    for c in ordered:
        judge_cap = _JUDGE_MAX_TOKENS.get(c.judge_method)

        if judge_cap:
            if c.category == "performance" and "latency" in c.name:
                c.max_tokens = min(c.max_tokens, 10)
            elif c.category == "performance" and "throughput" in c.name:
                c.max_tokens = min(c.max_tokens, 500)
            elif c.judge_method == "text_constraints":
                exact_chars = c.params.get("exact_chars", 0)
                derived_cap = max(100, int(exact_chars * 3))
                c.max_tokens = min(c.max_tokens, derived_cap)
            else:
                c.max_tokens = min(c.max_tokens, judge_cap)
        else:
            if c.category not in ("coding", "performance"):
                c.max_tokens = min(c.max_tokens, settings.DEFAULT_MAX_TOKENS_CAP)

        if c.id == "perf_002":
            c.max_tokens = min(c.max_tokens, settings.LONG_FORM_MAX_TOKENS_CAP)

    # Standard/Deep modes: guarantee a minimum max_tokens floor so
    # responses are not truncated.  Quick mode keeps tight caps for speed.
    if test_mode in ("standard", "deep"):
        _MIN_TOKENS_FLOOR = {
            "reasoning":    800,
            "coding":       600,
            "instruction":  400,
            "safety":       400,
            "knowledge":    400,
            "consistency":  400,
            "extraction":   600,
        }
        _DEFAULT_FLOOR = 400
        for c in ordered:
            floor = _MIN_TOKENS_FLOOR.get(c.category, _DEFAULT_FLOOR)
            # performance category keeps its tight caps (latency probes etc.)
            if c.category != "performance":
                c.max_tokens = max(c.max_tokens, floor)

    # Adaptive sampling
    for c in ordered:
        c.n_samples = _adaptive_samples(c, test_mode)

    # In quick mode, keep only top-2 core code execution cases to save budget.
    # Standard and deep modes run all assigned cases for data accuracy.
    if test_mode == "quick":
        code_cases = [c for c in ordered if c.judge_method == "code_execution"]
        code_cases.sort(key=lambda c: (-c.weight, _case_value(c)))
        keep_ids = {c.id for c in code_cases[:2]}
        if len(code_cases) > 2:
            ordered = [
                c for c in ordered
                if c.judge_method != "code_execution" or c.id in keep_ids
            ]

    # Filter by mode_level for the selected test_mode
    MODE_LEVELS = {
        "quick": {"quick"},
        "standard": {"quick", "standard"},
        "deep": {"quick", "standard", "deep"},
    }
    allowed_levels = MODE_LEVELS.get(test_mode, {"quick", "standard"})
    ordered = [c for c in ordered if _get_mode_level(c) in allowed_levels]

    # Value-first ranking
    ordered.sort(key=lambda c: (_case_value(c), c.weight), reverse=True)

    # Deep mode: multi-sampling for probabilistic judges
    if test_mode == "deep":
        for c in ordered:
            c.n_samples = max(c.n_samples, min(c.n_samples * 2, 3))

    # Phase assignment based on mode
    if test_mode == "quick":
        # Single phase: all quick-level cases
        return ordered, []

    if test_mode == "standard":
        # Phase1: quick-level (core), Phase2: standard-level (expansion)
        phase1 = [c for c in ordered if _get_mode_level(c) == "quick"]
        phase2 = [c for c in ordered if _get_mode_level(c) == "standard"]
        return phase1, phase2

    if test_mode == "deep":
        # Phase1: quick+standard (core), Phase2: deep-level (advanced)
        phase1 = [c for c in ordered if _get_mode_level(c) in ("quick", "standard")]
        phase2 = [c for c in ordered if _get_mode_level(c) == "deep"]
        return phase1, phase2

    return ordered, []
