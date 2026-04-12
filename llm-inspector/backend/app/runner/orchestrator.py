"""
RunOrchestrator — executes the full test pipeline for a single run.
Called from the task worker (thread pool or Celery).
"""
from __future__ import annotations

import pathlib
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from app.core.schemas import (
    TestCase, CaseResult, PreDetectionResult,
)
from app.core.logging import get_logger
from app.core.security import get_key_manager
from app.core.config import settings
from app.adapters.openai_compat import OpenAICompatibleAdapter
from app.predetect.pipeline import PreDetectionPipeline
from app.runner.case_executor import execute_case
from app.runner.compression import compressor as prompt_compressor
from app.analysis.pipeline import (
    FeatureExtractor, ScoreCalculator,
    SimilarityEngine, RiskEngine, ReportBuilder,
    ScoreCardCalculator, VerdictEngine,
    ThetaEstimator, UncertaintyEstimator, PercentileMapper, PairwiseEngine,
)
from app.repository import repo

logger = get_logger(__name__)

_FIXTURES_DIR = pathlib.Path(__file__).parent.parent / "fixtures"

# Model family discriminators for smart mode
FAMILY_DISCRIMINATORS: dict[str, list[str]] = {
    "openai": ["reason_001", "code_001"],
    "anthropic": ["refusal_001", "reason_001"],
    "google": ["code_001", "instr_004"],
    "deepseek": ["reason_001", "reason_candy_001"],
    "alibaba": ["instr_token_001", "style_004"],
    "zhipu": ["instr_token_001", "antispoof_001"],
    "meta": ["refusal_001", "code_001"],
    "mistral": ["instr_004", "reason_001"],
}


class SmartBudget:
    def __init__(
        self,
        token_budget: int,
        phase1_size: int,
        phase2_size: int,
        phase3_size: int,
        case_filter: list[str] | None,
        description: str,
    ):
        self.token_budget = token_budget
        self.phase1_size = phase1_size
        self.phase2_size = phase2_size
        self.phase3_size = phase3_size
        self.case_filter = case_filter
        self.description = description


class SmartModeStrategy:
    def decide_budget(self, pre_result: PreDetectionResult) -> SmartBudget:
        conf = pre_result.confidence
        identified = pre_result.identified_as or ""

        if conf >= 0.90:
            return SmartBudget(
                token_budget=8_000,
                phase1_size=6,
                phase2_size=0,
                phase3_size=0,
                case_filter=self._confirmation_cases(identified),
                description="High confidence verification mode",
            )
        elif conf >= 0.70:
            return SmartBudget(
                token_budget=15_000,
                phase1_size=10,
                phase2_size=4,
                phase3_size=0,
                case_filter=self._discriminative_cases(identified),
                description="Targeted discrimination mode",
            )
        elif conf >= 0.50:
            return SmartBudget(
                token_budget=25_000,
                phase1_size=12,
                phase2_size=8,
                phase3_size=4,
                case_filter=None,
                description="Standard detection mode",
            )
        else:
            return SmartBudget(
                token_budget=35_000,
                phase1_size=14,
                phase2_size=10,
                phase3_size=6,
                case_filter=None,
                description="Full detection mode",
            )

    def _confirmation_cases(self, model_family: str) -> list[str]:
        common = [
            "antispoof_001", "antispoof_002",
            "instr_001", "refusal_001",
        ]
        family_specific = FAMILY_DISCRIMINATORS.get(model_family.lower(), [])[:2]
        return common + family_specific

    def _discriminative_cases(self, model_family: str) -> list[str]:
        common = [
            "antispoof_001", "antispoof_002",
            "instr_001", "instr_004",
            "refusal_001", "reason_001",
            "sys_001", "consist_001",
        ]
        family_specific = FAMILY_DISCRIMINATORS.get(model_family.lower(), [])[:2]
        return common + family_specific


# ── Token Budget Guard ─────────────────────────────────────────────────────────

class TokenBudgetGuard:
    """
    Tracks cumulative token consumption during a run and gates
    low-value cases when the budget is exhausted.

    v6: Estimates remaining budget using historical consumption from past runs
    with the same model/base_url combination.
    """

    def __init__(self, budget: int, model_name: str = "", base_url: str = ""):
        self._budget = budget
        self._used = 0
        self._lock = threading.Lock()
        self._model_name = model_name
        self._base_url = base_url
        self._historical_median = self._load_historical_median()

    def _load_historical_median(self) -> int | None:
        """v6: Load median token consumption from historical runs."""
        if not self._model_name or not self._base_url:
            return None
        try:
            from app.repository.repo import get_repository
            repo = get_repository()
            # Get last 20 runs with same model/base_url
            history = repo.get_runs_by_model_base(
                self._model_name, self._base_url, limit=20, status="completed"
            )
            if not history:
                return None
            # Extract total token consumption
            consumptions = []
            for run in history:
                if isinstance(run, dict) and run.get("total_tokens"):
                    consumptions.append(run["total_tokens"])
            if len(consumptions) >= 3:
                sorted_consumptions = sorted(consumptions)
                return sorted_consumptions[len(sorted_consumptions) // 2]  # median
        except Exception:
            pass
        return None

    def estimate_tokens_needed(self, cases_count: int) -> int:
        """v6: Estimate tokens needed based on historical median or conservative default."""
        if self._historical_median:
            # Use historical median per case, with 20% buffer
            return int((self._historical_median / max(1, cases_count)) * cases_count * 1.2)
        # Conservative default: 1000 tokens per case
        return cases_count * 1000

    def consume(self, tokens: int) -> bool:
        """Returns False when budget is exhausted."""
        with self._lock:
            self._used += tokens
            return self._used <= self._budget

    def should_run_case(self, case: TestCase) -> bool:
        """v6: Decide whether to run a case based on remaining budget and case priority."""
        # Always run high-priority/anchor cases
        meta = case.params.get("_meta", {}) if case.params else {}
        if meta.get("anchor") or case.weight >= 2.0:
            return True
        # Check if we have enough budget for this case (estimated)
        est_cost = case.max_tokens * max(1, case.n_samples) * 2  # rough estimate
        return self.remaining >= est_cost

    def record_result(self, result: CaseResult) -> int:
        """Extract tokens from a CaseResult and add to running total. Returns tokens consumed."""
        tokens = 0
        for sample in result.samples:
            u = sample.response.usage_total_tokens
            if u:
                tokens += u
        self.consume(tokens)
        return tokens

    @property
    def remaining(self) -> int:
        return max(0, self._budget - self._used)

    @property
    def budget(self) -> int:
        return self._budget

    @property
    def used(self) -> int:
        return self._used


def _load_suite(suite_version: str, test_mode: str) -> list[TestCase]:
    """Load test cases from DB (seeded from fixture JSON) and compress prompts."""
    raw_cases = repo.load_cases(suite_version, test_mode)
    cases = []
    for c in raw_cases:
        params = c.get("params", {})
        meta = (params.get("_meta") or {})
        
        # v10: Lossless prompt compression
        compressed_user = prompt_compressor.compress(c["user_prompt"])
        compressed_system = prompt_compressor.compress(c.get("system_prompt", "")) if c.get("system_prompt") else None

        cases.append(TestCase(
            id=c["id"],
            category=c["category"],
            name=c["name"],
            user_prompt=compressed_user,
            expected_type=c["expected_type"],
            judge_method=c["judge_method"],
            system_prompt=compressed_system,
            dimension=meta.get("dimension") or c.get("dimension"),
            tags=meta.get("tags") or c.get("tags", []),
            judge_rubric=meta.get("judge_rubric") or c.get("judge_rubric", {}),
            params=params,
            max_tokens=c.get("max_tokens", 5),
            n_samples=c.get("n_samples", 1),
            temperature=c.get("temperature", 0.0),
            weight=c.get("weight", 1.0),
            enabled=bool(c.get("enabled", 1)),
            suite_version=c.get("suite_version", "v1"),
            difficulty=c.get("difficulty"),
        ))
    return cases


_benchmark_cache: dict[str, tuple[float, list[dict]]] = {}


def _load_benchmarks(suite_version: str) -> list[dict]:
    """Load benchmarks with a short TTL cache to avoid repeated DB reads."""
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
        "json_schema":         120,
        "line_count":          100,
        "code_execution":      300,
        "regex_match":         150,
        "refusal_detect":      150,
        "constraint_reasoning": 700,
        "text_constraints":    150,
        "identity_consistency": 150,
        "heuristic_style":     250,
        "any_text":            180,
        "prompt_leak_detect":   1200,
        "forbidden_word_extract": 500,
        "path_leak_detect":     500,
        "tool_config_leak_detect": 600,
        "memory_leak_detect":   500,
        "denial_pattern_detect": 200,
        "spec_contradiction_check": 50,
        "refusal_style_fingerprint": 300,
        "language_bias_detect": 200,
        "tokenizer_fingerprint": 20,
        # New judge methods
        "multi_step_verify":   500,
        "yaml_csv_validate":   200,
        "hallucination_detect": 200,
        "context_overflow_detect": 1200,
        "semantic_judge":       400,
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

    # Adaptive sampling
    for c in ordered:
        c.n_samples = _adaptive_samples(c, test_mode)

    # In non-deep modes, keep only top-2 core code execution cases.
    if test_mode != "deep":
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


def _adaptive_pause(backoff_state: dict) -> None:
    delay_ms = backoff_state.get("delay_ms", settings.INTER_REQUEST_DELAY_MS)
    if delay_ms > 0:
        time.sleep(delay_ms / 1000)


def _update_backoff(backoff_state: dict, result: CaseResult | None = None,
                    error: Exception | None = None) -> None:
    base_ms = settings.INTER_REQUEST_DELAY_MS
    current = int(backoff_state.get("delay_ms", base_ms))

    should_backoff = False
    if error is not None:
        should_backoff = True
    elif result is not None:
        for s in result.samples:
            status = s.response.status_code or 0
            if status == 429 or status >= 500:
                should_backoff = True
                break

    if should_backoff:
        backoff_state["delay_ms"] = min(max(base_ms, 200), max(200, current) * 2)
    else:
        backoff_state["delay_ms"] = base_ms


def _checkpoint_should_stop(test_mode: str, case_results: list[CaseResult],
                            features_cache: dict | None, sims_cache: list | None,
                            scorecard_cache=None) -> tuple[bool, dict | None, list | None, object | None]:
    if not case_results:
        return False, features_cache, sims_cache, scorecard_cache

    # v10: Computerized Adaptive Testing (CAT) early stopping
    # If standard error of measurement (SEM) is low enough, we can stop early
    total_cases = len(case_results)
    if total_cases >= 10:
        # Check CAT threshold
        from app.analysis.irt_params import get_calibrated_params
        
        dim_info = {}
        for cr in case_results:
            dim = getattr(cr.case, 'dimension', 'unknown')
            params = get_calibrated_params(cr.case.id)
            if params:
                # We don't know exact theta here without full recalculation, 
                # so we use expected information at theta=0 as a heuristic proxy
                info = params.calculate_information(0.0)
                dim_info[dim] = dim_info.get(dim, 0.0) + info
                
        # If all major dimensions have SEM < 0.3 (Info > 11.1)
        cat_satisfied = True
        major_dims = [d for d in dim_info.keys() if d != "unknown"]
        
        if len(major_dims) >= 3:
            for dim in major_dims:
                info = dim_info[dim]
                sem = 1.0 / (info ** 0.5) if info > 0 else 999.0
                if sem > 0.3:
                    cat_satisfied = False
                    break
                    
            if cat_satisfied and test_mode != "deep":
                logger.info(
                    "v10 CAT early stopping triggered (SEM < 0.3 for all dims)",
                    test_mode=test_mode,
                    cases_run=total_cases
                )
                return True, features_cache, sims_cache, scorecard_cache

    # v6: Failure rate early stop for all modes
    # If error rate > 80% after 10 cases, abort early (API likely broken)
    if total_cases >= 10:
        failed_cases = sum(1 for r in case_results if r.pass_rate == 0.0)
        fail_rate = failed_cases / total_cases
        if fail_rate > 0.80:
            logger.warning(
                "High failure rate detected, aborting early",
                fail_rate=round(fail_rate, 2),
                total_cases=total_cases,
                test_mode=test_mode,
            )
            return True, features_cache, sims_cache, scorecard_cache

    # v6: Quick mode 2-point early stop (minimum 2 cases)
    if test_mode == "quick":
        # After just 2 core cases, check if model is clearly identifiable
        if len(case_results) >= 2:
            extractor = FeatureExtractor()
            features = extractor.extract(case_results)
            similarities: list = sims_cache or []
            if not similarities:
                run_suite = case_results[0].case.suite_version if case_results else "v2"
                similarities = SimilarityEngine().compare(features, _load_benchmarks(run_suite))

            if similarities and len(similarities) >= 2:
                top = similarities[0]
                second = similarities[1]
                # If similarity gap >= 0.15, we have strong identification
                if top.similarity_score >= 0.70 and (top.similarity_score - second.similarity_score) >= 0.15:
                    return True, features, similarities, scorecard_cache
        # Otherwise continue to 6-case threshold for quick mode
        if len(case_results) < 6:
            return False, features_cache, sims_cache, scorecard_cache

    if len(case_results) < 6:
        return False, features_cache, sims_cache, scorecard_cache

    # Deep mode: no early stopping — always run full suite
    if test_mode == "deep":
        return False, features_cache, sims_cache, scorecard_cache

    extractor = FeatureExtractor()
    features = extractor.extract(case_results)
    if not features:
        return False, features_cache, sims_cache, scorecard_cache

    similarities: list = sims_cache or []
    if not similarities:
        run_suite = case_results[0].case.suite_version if case_results else "v2"
        similarities = SimilarityEngine().compare(features, _load_benchmarks(run_suite))

    if not similarities:
        return False, features, similarities, scorecard_cache

    top = similarities[0]
    second = similarities[1] if len(similarities) > 1 else None
    delta = (top.similarity_score - second.similarity_score) if second else 1.0

    if test_mode == "quick":
        return (top.similarity_score >= 0.78 and delta >= 0.10), features, similarities, scorecard_cache

    if test_mode == "standard":
        sc = scorecard_cache
        if sc is None:
            sc = ScoreCardCalculator().calculate(
                features=features,
                case_results=case_results,
                similarities=similarities,
                predetect=None,
                claimed_model=None,
            )

        if top.similarity_score >= 0.85 and delta >= 0.12:
            near_boundary = abs(sc.total_score - 70.0) <= 5.0 or abs(sc.authenticity_score - 70.0) <= 6.0
            unstable = delta < 0.05
            if not near_boundary and not unstable:
                return True, features, similarities, sc

        if top.similarity_score >= 0.92 and delta >= 0.08:
            return True, features, similarities, sc

    return False, features, similarities, scorecard_cache


def _run_cases_concurrent(adapter, model_name: str, cases: list[TestCase],
                          test_mode: str, run_id: str, phase_label: str,
                          case_results: list[CaseResult], failed_count_ref: dict,
                          backoff_state: dict,
                          budget_guard: TokenBudgetGuard | None = None) -> bool:
    if not cases:
        return False

    max_workers = _mode_concurrency(test_mode)
    idx = 0
    in_flight = {}
    lock = threading.Lock()

    # Token budget thresholds: stop dispatching new work when < 5% budget remains
    budget_threshold = (budget_guard.budget * 0.05) if budget_guard else 0
    pending_saves = []

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=f"case-{phase_label}") as pool:
        budget_exhausted = False
        while idx < len(cases) or in_flight:
            if repo.is_run_cancel_requested(run_id):
                logger.warning("Run cancel requested", run_id=run_id, phase=phase_label)
                break

            while len(in_flight) < max_workers and idx < len(cases) and not budget_exhausted:
                # Check token budget before submitting
                if budget_guard and budget_guard.remaining < budget_threshold:
                    logger.info(
                        "Token budget exhausted, skipping remaining phase cases",
                        run_id=run_id, phase=phase_label,
                        used=budget_guard.used, budget=budget_guard.budget,
                    )
                    budget_exhausted = True
                    break

                case = cases[idx]
                idx += 1
                _adaptive_pause(backoff_state)
                fut = pool.submit(execute_case, adapter, model_name, case)
                in_flight[fut] = case

            if not in_flight:
                break
            done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
            
            check_early_abort = False
            for fut in done:
                case = in_flight.pop(fut)
                try:
                    result = fut.result()
                    with lock:
                        case_results.append(result)
                        pending_saves.append(result)
                        if len(pending_saves) >= 5:
                            _save_case_results_batch(run_id, pending_saves)
                            pending_saves.clear()
                            check_early_abort = True

                    # Track token consumption for budget guard
                    if budget_guard:
                        budget_guard.record_result(result)
                    
                    _update_backoff(backoff_state, result=result)
                    logger.info(
                        "Case done",
                        run_id=run_id,
                        case_id=case.id,
                        pass_rate=round(result.pass_rate, 2),
                        phase=phase_label,
                        tokens_remaining=budget_guard.remaining if budget_guard else None,
                    )
                except Exception as e:
                    with lock:
                        failed_count_ref["count"] += 1
                    _update_backoff(backoff_state, error=e)
                    logger.warning("Case failed", run_id=run_id, case_id=case.id, error=str(e))
                    check_early_abort = True

            if check_early_abort:
                with lock:
                    total = len(case_results)
                    failed = failed_count_ref["count"]
                if total >= 10 and total > 0 and (failed / total) > 0.8:
                    logger.warning("Error rate >80%, aborting phase early", run_id=run_id, phase=phase_label)
                    break
                    
    if pending_saves:
        _save_case_results_batch(run_id, pending_saves)
        pending_saves.clear()

    return repo.is_run_cancel_requested(run_id)


def run_pipeline(run_id: str) -> None:
    """
    Full pipeline:
      1. Pre-detection (0-token identification attempt)
      2. Connectivity check
      3. Test case execution (with early-stop)
      4. Feature extraction + scoring
      5. Similarity + risk assessment
      6. Report generation
    """
    logger.info("Pipeline starting", run_id=run_id)

    # ── Load run metadata ────────────────────────────────────────────────────
    run = repo.get_run(run_id)
    if not run:
        logger.error("Run not found", run_id=run_id)
        return

    km = get_key_manager()
    try:
        api_key = km.decrypt(run["api_key_encrypted"])
    except Exception as e:
        repo.update_run_status(run_id, "failed", error_message=f"Key decrypt failed: {e}")
        return

    adapter = OpenAICompatibleAdapter(run["base_url"], api_key)
    suite_version = run.get("suite_version", "v1")
    test_mode = run.get("test_mode", "standard")

    # ── Step 1: Pre-detection ────────────────────────────────────────────────
    repo.update_run_status(run_id, "pre_detecting")
    logger.info("Running pre-detection", run_id=run_id)
    extraction_mode = test_mode == "deep"
    try:
        pre_result: PreDetectionResult = PreDetectionPipeline().run(
            adapter, run["model_name"],
            extraction_mode=extraction_mode,
            run_id=run_id,
        )
        repo.save_predetect_result(run_id, pre_result.to_dict())
        logger.info(
            "Pre-detection complete",
            run_id=run_id,
            identified=pre_result.identified_as,
            confidence=pre_result.confidence,
            tokens=pre_result.total_tokens_used,
        )
    except Exception as e:
        logger.warning("Pre-detection failed, continuing to full test", error=str(e))
        pre_result = PreDetectionResult(
            success=False, identified_as=None, confidence=0.0,
            layer_stopped=None, should_proceed_to_testing=True,
        )

    # If pre-detection is highly confident AND not extraction mode,
    # pause and let the user decide whether to continue full testing.
    # The user can POST /api/v1/runs/{id}/continue or /skip-testing.
    if not pre_result.should_proceed_to_testing and not extraction_mode:
        logger.info("Pre-detection sufficient, pausing for user decision", run_id=run_id)
        repo.update_run_status(run_id, "pre_detected")
        return

    # ── Step 2: Connectivity check ───────────────────────────────────────────
    repo.update_run_status(run_id, "running")
    conn_check = adapter.list_models()
    conn_status = conn_check.get("status_code")
    conn_error = conn_check.get("error")

    # Hard fail: network-level error (no status_code means DNS/TCP failure)
    if conn_error and not conn_status:
        msg = (
            f"无法连接到 API：{conn_error}。"
            f"请检查 base_url 是否正确（当前：{run['base_url']}），"
            f"网络是否可达，以及 API Key 是否有效。"
        )
        repo.update_run_status(run_id, "failed", error_message=msg)
        logger.error("Connectivity failed (network)", run_id=run_id, error=msg)
        return

    # Soft fail on /models (401/403/404): many providers (e.g. Baidu Qianfan
    # Coding Plan) don't expose a /models endpoint but /chat/completions works.
    # Fallback: send a minimal chat request with the actual model name.
    if conn_status and conn_status in (401, 403, 404):
        logger.info(
            "list_models returned error, probing chat endpoint as fallback",
            run_id=run_id,
            models_status=conn_status,
        )
        from app.core.schemas import LLMRequest, Message
        probe_req = LLMRequest(
            model=run["model_name"],
            messages=[Message(role="user", content="hi")],
            max_tokens=1,
            temperature=0.0,
            timeout_sec=15,
        )
        probe_resp = adapter.chat(probe_req)

        # Network-level failure on the chat endpoint
        if probe_resp.error_type and not probe_resp.status_code:
            msg = (
                f"无法连接到 API chat 端点：{probe_resp.error_message}。"
                f"请检查 base_url 是否正确（当前：{run['base_url']}），"
                f"网络是否可达，以及 API Key 是否有效。"
            )
            repo.update_run_status(run_id, "failed", error_message=msg)
            logger.error("Connectivity failed (chat probe network)", run_id=run_id, error=msg)
            return

        # Chat endpoint returns 401 → genuine auth failure
        if probe_resp.status_code == 401:
            msg = (
                f"API 鉴权失败：API Key 无效或未授权（HTTP 401）。"
                f"请检查 API Key 是否正确。base_url：{run['base_url']}"
            )
            repo.update_run_status(run_id, "failed", error_message=msg)
            logger.error("Connectivity failed (auth 401)", run_id=run_id)
            return

        # Chat endpoint returns 404 → endpoint path is wrong
        if probe_resp.status_code == 404:
            msg = (
                f"API 端点不存在（HTTP 404）。当前 base_url：{run['base_url']}，"
                f"系统将请求发送到 {run['base_url']}/chat/completions。"
                f"请确认该路径是否正确。例如百度千帆 Coding Plan 应填写 "
                f"https://qianfan.baidubce.com/v2/coding"
            )
            repo.update_run_status(run_id, "failed", error_message=msg)
            logger.error("Connectivity failed (404 on chat endpoint)", run_id=run_id)
            return

        # Any response (even 400/403/422/500) from the chat endpoint means it's
        # reachable. 403 with the real model name could mean rate-limit, model
        # access restriction, or parameter validation — not necessarily bad auth.
        # Continue to testing; the high-error-rate guard will catch persistent failures.
        logger.info(
            "Chat endpoint probe: endpoint reachable",
            run_id=run_id,
            probe_status=probe_resp.status_code,
            probe_ok=probe_resp.ok,
            probe_error=probe_resp.error_type,
        )

    # ── Step 3: Load test cases + execute ────────────────────────────────────
    cases = _load_suite(suite_version, test_mode)
    if not cases:
        logger.warning("No test cases found", suite_version=suite_version)
        repo.update_run_status(run_id, "failed", error_message="No test cases loaded")
        return

    # Targeted confirmation: when predetect has moderate confidence (0.60-0.84),
    # select only the cases with highest discriminative power for the candidate
    # model family, instead of running the full suite. This saves ~40% tokens
    # without losing detection precision — the selected cases are specifically
    # chosen to confirm or deny the predetect hypothesis.
    if (
        test_mode != "full"
        and pre_result.success
        and 0.60 <= pre_result.confidence < settings.PREDETECT_CONFIDENCE_THRESHOLD
        and pre_result.identified_as
    ):
        cases = _select_confirmatory_cases(cases, pre_result.identified_as)
        logger.info(
            "Targeted confirmation mode",
            run_id=run_id,
            candidate=pre_result.identified_as,
            selected_cases=len(cases),
        )

    phase1_cases, phase2_cases = _prepare_cases(cases, test_mode)

    # Resume support: skip already completed cases with at least one response
    existing_responses = repo.get_responses(run_id)
    completed_case_ids = {r.get("case_id") for r in existing_responses if r.get("case_id")}
    if completed_case_ids:
        phase1_cases = [c for c in phase1_cases if c.id not in completed_case_ids]
        phase2_cases = [c for c in phase2_cases if c.id not in completed_case_ids]

    case_results: list[CaseResult] = []
    failed_count_ref = {"count": 0}
    backoff_state = {"delay_ms": settings.INTER_REQUEST_DELAY_MS}

    # Token budget guard — initialized based on test mode
    budget_map = {
        "quick":      settings.TOKEN_BUDGET_QUICK,
        "standard":   settings.TOKEN_BUDGET_STANDARD,
        "deep":       settings.TOKEN_BUDGET_DEEP,
        "full":       settings.TOKEN_BUDGET_FULL,
        "extraction": settings.TOKEN_BUDGET_FULL,
        "smart":      13_000,  # Smart mode default budget
    }
    if test_mode == "smart" and pre_result.confidence > 0:
        smart_strategy = SmartModeStrategy()
        smart_budget = smart_strategy.decide_budget(pre_result)
        budget_guard = TokenBudgetGuard(smart_budget.token_budget)
        logger.info(
            "Smart mode budget allocated",
            run_id=run_id,
            budget=smart_budget.token_budget,
            description=smart_budget.description,
            phase1=smart_budget.phase1_size,
            phase2=smart_budget.phase2_size,
            phase3=smart_budget.phase3_size,
        )
    else:
        budget_guard = TokenBudgetGuard(budget_map.get(test_mode, settings.TOKEN_BUDGET_STANDARD))
    logger.info(
        "Executing test cases",
        run_id=run_id,
        total=len(phase1_cases) + len(phase2_cases),
        phase1=len(phase1_cases),
        phase2=len(phase2_cases),
        concurrency=_mode_concurrency(test_mode),
        token_budget=budget_guard.budget,
    )

    cancelled = _run_cases_concurrent(
        adapter=adapter,
        model_name=run["model_name"],
        cases=phase1_cases,
        test_mode=test_mode,
        run_id=run_id,
        phase_label="phase1",
        case_results=case_results,
        failed_count_ref=failed_count_ref,
        backoff_state=backoff_state,
        budget_guard=budget_guard,
    )

    if cancelled:
        repo.update_run_status(run_id, "failed", error_message="Cancelled by user")
        logger.info("Pipeline cancelled", run_id=run_id)
        return

    stop_now, cached_features, cached_sims, _ = _checkpoint_should_stop(
        test_mode=test_mode,
        case_results=case_results,
        features_cache=None,
        sims_cache=None,
        scorecard_cache=None,
    )

    if stop_now:
        logger.info("Early stop triggered", run_id=run_id, test_mode=test_mode)
        _build_and_save_report(
            run_id,
            run,
            pre_result,
            case_results,
            cached_features or {},
            suite_version,
            precomputed_similarities=cached_sims,
        )
        final_status = "partial_failed" if failed_count_ref["count"] > 0 else "completed"
        repo.update_run_status(run_id, final_status)
        logger.info("Pipeline complete", run_id=run_id, status=final_status)
        return

    if phase2_cases:
        # P2: Progressive result output - generate partial report after phase 1
        try:
            logger.info("Generating partial report after Phase 1", run_id=run_id)
            _build_and_save_report(
                run_id,
                run,
                pre_result,
                case_results,
                {"partial": 1.0},
                suite_version,
                precomputed_similarities=cached_sims,
            )
        except Exception as e:
            logger.warning("Failed to generate partial report", error=str(e))
            
        # Stage B targeted expansion is already value-ranked.
        cancelled = _run_cases_concurrent(
            adapter=adapter,
            model_name=run["model_name"],
            cases=phase2_cases,
            test_mode=test_mode,
            run_id=run_id,
            phase_label="phase2",
            case_results=case_results,
            failed_count_ref=failed_count_ref,
            backoff_state=backoff_state,
            budget_guard=budget_guard,
        )
        if cancelled:
            repo.update_run_status(run_id, "failed", error_message="Cancelled by user")
            logger.info("Pipeline cancelled", run_id=run_id)
            return

        # Stage C arbitration: only for standard mode when decision remains unstable.
        # Deep mode does not arbitrate — it runs all cases fully.
        if test_mode == "standard":
            stop2, f2, s2, sc2 = _checkpoint_should_stop(
                test_mode=test_mode,
                case_results=case_results,
                features_cache=None,
                sims_cache=None,
                scorecard_cache=None,
            )
            if not stop2:
                # pick arbitration cases with highest value among remaining set
                seen_ids = {c.case.id for c in case_results}
                remaining = [c for c in cases if c.id not in seen_ids]
                remaining.sort(key=_case_value, reverse=True)
                arbitration = remaining[:settings.ARBITRATION_MAX]
                if arbitration:
                    cancelled = _run_cases_concurrent(
                        adapter=adapter,
                        model_name=run["model_name"],
                        cases=arbitration,
                        test_mode=test_mode,
                        run_id=run_id,
                        phase_label="phase3",
                        case_results=case_results,
                        failed_count_ref=failed_count_ref,
                        backoff_state=backoff_state,
                        budget_guard=budget_guard,
                    )
                    if cancelled:
                        repo.update_run_status(run_id, "failed", error_message="Cancelled by user")
                        logger.info("Pipeline cancelled", run_id=run_id)
                        return

    # ── 检查：是否几乎所有请求均失败（说明是连接/配置问题而非模型问题）────────────
    if case_results:
        total_samples = sum(len(r.samples) for r in case_results)
        error_samples = sum(
            1 for r in case_results for s in r.samples if s.response.error_type
        )
        if total_samples > 0:
            error_rate = error_samples / total_samples
            if error_rate >= 0.9:
                # 收集最常见的错误类型作为提示
                error_types: dict[str, int] = {}
                error_messages: list[str] = []
                for r in case_results:
                    for s in r.samples:
                        et = s.response.error_type
                        if et:
                            error_types[et] = error_types.get(et, 0) + 1
                        em = s.response.error_message
                        if em and em not in error_messages:
                            error_messages.append(em)
                top_error = max(error_types, key=error_types.get) if error_types else "未知错误"
                sample_msg = error_messages[0][:150] if error_messages else ""
                diag_msg = (
                    f"API 连接/配置失败：{error_rate:.0%} 的请求均出错（错误类型：{top_error}）。"
                    f" 示例错误：{sample_msg}。"
                    f" 请检查：① base_url 路径是否正确（如千帆需使用 /v2/chat/completions，"
                    f"当前 base_url：{run['base_url']}）；② API Key 是否有效；③ 网络是否可达。"
                    f" 本次结果不代表模型真实能力，请修正配置后重试。"
                )
                repo.update_run_status(run_id, "failed", error_message=diag_msg)
                logger.error(
                    "Pipeline aborted: near-total request failure indicates config error",
                    run_id=run_id,
                    error_rate=error_rate,
                    top_error=top_error,
                )
                return

    # ── Steps 4-6: Analysis + report ─────────────────────────────────────────
    _build_and_save_report(run_id, run, pre_result, case_results, {}, suite_version)

    final_status = "partial_failed" if failed_count_ref["count"] > 0 else "completed"
    repo.update_run_status(run_id, final_status)
    logger.info("Pipeline complete", run_id=run_id, status=final_status)


def _build_and_save_report(
    run_id: str,
    run: dict,
    pre_result: PreDetectionResult,
    case_results: list[CaseResult],
    extra_features: dict,
    suite_version: str,
    precomputed_similarities: list | None = None,
) -> dict:
    """Extract features, score, similarity, risk, build report."""
    extractor = FeatureExtractor()
    features = extractor.extract(case_results)
    features.update(extra_features)
    if features:
        repo.save_features(run_id, features)

    scoring_profile_version = run.get("scoring_profile_version", settings.CALIBRATION_VERSION)
    calibration_tag = run.get("calibration_tag")

    # Scoring
    scorer = ScoreCalculator()
    scores = scorer.calculate(features)

    # Similarity (allow checkpoint cache injection)
    similarities = precomputed_similarities
    if similarities is None:
        benchmarks = _load_benchmarks(suite_version)
        similarity_engine = SimilarityEngine()
        similarities = similarity_engine.compare(features, benchmarks)
    if similarities:
        repo.save_similarities(run_id, [
            {
                "benchmark": s.benchmark_name,
                "score": s.similarity_score,
                "ci_95_low": s.ci_95_low,
                "ci_95_high": s.ci_95_high,
                "rank": s.rank,
            }
            for s in similarities
        ])

    # Risk assessment
    risk_engine = RiskEngine()
    risk = risk_engine.assess(features, similarities, pre_result)

    # v2 scorecard + verdict
    scorecard_calc = ScoreCardCalculator()
    scorecard = scorecard_calc.calculate(
        features=features,
        case_results=case_results,
        similarities=similarities,
        predetect=pre_result,
        claimed_model=run.get("model_name"),
    )
    verdict_engine = VerdictEngine()
    verdict = verdict_engine.assess(
        scorecard=scorecard,
        similarities=similarities,
        predetect=pre_result,
        features=features,
        case_results=case_results,
    )

    breakdowns = {
        "total": scorecard.total_score,
        "capability": scorecard.capability_score,
        "authenticity": scorecard.authenticity_score,
        "performance": scorecard.performance_score,
        "reasoning": scorecard.reasoning_score,
        "adversarial_reasoning": scorecard.adversarial_reasoning_score,
        "instruction": scorecard.instruction_score,
        "coding": scorecard.coding_score,
        "safety": scorecard.safety_score,
        "protocol": scorecard.protocol_score,
        "consistency": scorecard.consistency_score,
        "speed": scorecard.speed_score,
        "stability": scorecard.stability_score,
        "cost_efficiency": scorecard.cost_efficiency,
    }

    # Backward-compatible write path: if running mixed code version,
    # gracefully fallback to single-row API instead of failing the run.
    if hasattr(repo, "save_score_breakdowns"):
        repo.save_score_breakdowns(run_id, breakdowns)
    else:
        for dim, val in breakdowns.items():
            repo.save_score_breakdown(run_id, dim, val)

    repo.save_score_history(
        model_name=run["model_name"],
        base_url=run["base_url"],
        run_id=run_id,
        total=scorecard.total_score,
        capability=scorecard.capability_score,
        authenticity=scorecard.authenticity_score,
        performance=scorecard.performance_score,
    )

    # Theta report (relative scale)
    item_stats_rows = repo.list_item_stats()
    item_stats = {r.get("item_id"): r for r in item_stats_rows}

    theta_estimator = ThetaEstimator()
    theta_report = theta_estimator.estimate(case_results, item_stats)
    theta_report = UncertaintyEstimator().apply_ci(theta_report, case_results, theta_estimator, item_stats)

    hist = repo.get_model_theta_trend(run["model_name"], limit=200)
    theta_report = PercentileMapper().map_percentiles(theta_report, hist)

    theta_dims_payload = {d.dimension: d.to_dict() for d in theta_report.dimensions}
    pct_dims_payload = {d.dimension: d.percentile for d in theta_report.dimensions}

    repo.save_theta_history(
        run_id=run_id,
        model_name=run["model_name"],
        base_url=run["base_url"],
        theta_global=theta_report.global_theta,
        theta_global_ci_low=theta_report.global_ci_low,
        theta_global_ci_high=theta_report.global_ci_high,
        theta_dims=theta_dims_payload,
        percentile_global=theta_report.global_percentile,
        percentile_dims=pct_dims_payload,
        calibration_version=theta_report.calibration_version,
        method=theta_report.method,
    )

    baseline_theta = None
    if similarities:
        top_benchmark = similarities[0].benchmark_name
        baseline_hist = repo.get_model_theta_trend(top_benchmark, limit=1)
        if baseline_hist:
            baseline_theta = float(baseline_hist[0].get("theta_global", 0.0) or 0.0)

    pairwise = PairwiseEngine().compare_to_baseline(theta_report, baseline_theta)
    if pairwise:
        model_b = similarities[0].benchmark_name if similarities else "baseline"
        repo.save_pairwise_result(
            run_id=run_id,
            model_a=run["model_name"],
            model_b=model_b,
            delta_theta=pairwise["delta_theta"],
            win_prob_a=pairwise["win_prob"],
            method=pairwise.get("method", "bradley_terry"),
            details=pairwise,
        )
        # Update ELO standings
        from app.analysis.elo import EloLeaderboard
        try:
            elo_board = EloLeaderboard()
            new_a, new_b = elo_board.update_from_pairwise(
                model_a=run["model_name"],
                display_a=run["model_name"],
                model_b=model_b,
                display_b=model_b.title(),
                win_prob_a=pairwise["win_prob"],
                run_id=run_id,
            )
            pairwise["elo_rating_a"] = new_a
            pairwise["elo_rating_b"] = new_b
        except Exception as e:
            logger.error("Failed to update ELO rankings", error=str(e))

    # Build final report
    builder = ReportBuilder()
    report = builder.build(
        run_id=run_id,
        base_url=run["base_url"],
        model_name=run["model_name"],
        test_mode=run.get("test_mode", "standard"),
        predetect=pre_result,
        case_results=case_results,
        features=features,
        scores=scores,
        similarities=similarities,
        risk=risk,
        scorecard=scorecard,
        verdict=verdict,
        theta_report=theta_report,
        pairwise=pairwise,
        scoring_profile_version=scoring_profile_version,
        calibration_tag=calibration_tag,
    )
    repo.save_report(run_id, report)
    return report


def continue_pipeline(run_id: str) -> None:
    """Resume a pre_detected run: skip predetect, go straight to connectivity check + testing."""
    logger.info("Continue pipeline (from pre_detected)", run_id=run_id)
    run = repo.get_run(run_id)
    if not run:
        logger.error("Run not found", run_id=run_id)
        return

    km = get_key_manager()
    try:
        api_key = km.decrypt(run["api_key_encrypted"])
    except Exception as e:
        repo.update_run_status(run_id, "failed", error_message=f"Key decrypt failed: {e}")
        return

    adapter = OpenAICompatibleAdapter(run["base_url"], api_key)
    suite_version = run.get("suite_version", "v1")
    test_mode = run.get("test_mode", "standard")

    # Reload predetect result
    pre_dict = run.get("predetect_result") or {}
    pre_result = PreDetectionResult(
        success=pre_dict.get("success", False),
        identified_as=pre_dict.get("identified_as"),
        confidence=pre_dict.get("confidence", 0.0),
        layer_stopped=pre_dict.get("layer_stopped"),
        total_tokens_used=pre_dict.get("total_tokens_used", 0),
        should_proceed_to_testing=True,  # Force continue
        routing_info=pre_dict.get("routing_info", {}),
    )

    # Jump to Step 2: connectivity check + testing (same as run_pipeline from line 589)
    repo.update_run_status(run_id, "running")
    conn_check = adapter.list_models()
    conn_status = conn_check.get("status_code")
    conn_error = conn_check.get("error")

    if conn_error and not conn_status:
        msg = (
            f"无法连接到 API：{conn_error}。"
            f"请检查 base_url 是否正确（当前：{run['base_url']}），"
            f"网络是否可达，以及 API Key 是否有效。"
        )
        repo.update_run_status(run_id, "failed", error_message=msg)
        return

    if conn_status and conn_status in (401, 403, 404):
        from app.core.schemas import LLMRequest as _LR, Message as _M
        probe_resp = adapter.chat(_LR(
            model=run["model_name"],
            messages=[_M(role="user", content="hi")],
            max_tokens=1, temperature=0.0, timeout_sec=15,
        ))
        if probe_resp.error_type and not probe_resp.status_code:
            repo.update_run_status(run_id, "failed",
                                   error_message=f"无法连接到 API chat 端点：{probe_resp.error_message}")
            return
        if probe_resp.status_code == 401:
            repo.update_run_status(run_id, "failed",
                                   error_message=f"API 鉴权失败（HTTP 401）。base_url：{run['base_url']}")
            return
        if probe_resp.status_code == 404:
            repo.update_run_status(run_id, "failed",
                                   error_message=f"API 端点不存在（HTTP 404）。base_url：{run['base_url']}")
            return

    # Load and execute test cases
    cases = _load_suite(suite_version, test_mode)
    if not cases:
        repo.update_run_status(run_id, "failed", error_message="No test cases loaded")
        return

    if (
        test_mode != "full"
        and pre_result.success
        and 0.60 <= pre_result.confidence < settings.PREDETECT_CONFIDENCE_THRESHOLD
        and pre_result.identified_as
    ):
        cases = _select_confirmatory_cases(cases, pre_result.identified_as)

    phase1_cases, phase2_cases = _prepare_cases(cases, test_mode)

    existing_responses = repo.get_responses(run_id)
    completed_case_ids = {r.get("case_id") for r in existing_responses if r.get("case_id")}
    if completed_case_ids:
        phase1_cases = [c for c in phase1_cases if c.id not in completed_case_ids]
        phase2_cases = [c for c in phase2_cases if c.id not in completed_case_ids]

    case_results: list[CaseResult] = []
    failed_count_ref = {"count": 0}
    backoff_state = {"delay_ms": settings.INTER_REQUEST_DELAY_MS}

    budget_map = {
        "quick": settings.TOKEN_BUDGET_QUICK,
        "standard": settings.TOKEN_BUDGET_STANDARD,
        "full": settings.TOKEN_BUDGET_FULL,
    }
    budget_guard = TokenBudgetGuard(budget_map.get(test_mode, settings.TOKEN_BUDGET_STANDARD))

    cancelled = _run_cases_concurrent(
        adapter=adapter, model_name=run["model_name"],
        cases=phase1_cases, test_mode=test_mode, run_id=run_id,
        phase_label="phase1", case_results=case_results,
        failed_count_ref=failed_count_ref, backoff_state=backoff_state,
        budget_guard=budget_guard,
    )
    if cancelled:
        repo.update_run_status(run_id, "failed", error_message="Cancelled by user")
        return

    stop_now, cached_features, cached_sims, _ = _checkpoint_should_stop(
        test_mode=test_mode, case_results=case_results,
        features_cache=None, sims_cache=None, scorecard_cache=None,
    )
    if stop_now:
        _build_and_save_report(run_id, run, pre_result, case_results,
                               cached_features or {}, suite_version,
                               precomputed_similarities=cached_sims)
        final_status = "partial_failed" if failed_count_ref["count"] > 0 else "completed"
        repo.update_run_status(run_id, final_status)
        return

    if phase2_cases:
        cancelled = _run_cases_concurrent(
            adapter=adapter, model_name=run["model_name"],
            cases=phase2_cases, test_mode=test_mode, run_id=run_id,
            phase_label="phase2", case_results=case_results,
            failed_count_ref=failed_count_ref, backoff_state=backoff_state,
            budget_guard=budget_guard,
        )
        if cancelled:
            repo.update_run_status(run_id, "failed", error_message="Cancelled by user")
            return

    # High error rate guard
    if case_results:
        total_samples = sum(len(r.samples) for r in case_results)
        error_samples = sum(1 for r in case_results for s in r.samples if s.response.error_type)
        if total_samples > 0 and (error_samples / total_samples) >= 0.9:
            error_types: dict[str, int] = {}
            for r in case_results:
                for s in r.samples:
                    if s.response.error_type:
                        error_types[s.response.error_type] = error_types.get(s.response.error_type, 0) + 1
            top_error = max(error_types, key=error_types.get) if error_types else "unknown"
            repo.update_run_status(run_id, "failed",
                                   error_message=f"API 连接/配置失败：90%+ 请求出错（{top_error}）。请修正配置后重试。")
            return

    _build_and_save_report(run_id, run, pre_result, case_results, {}, suite_version)
    final_status = "partial_failed" if failed_count_ref["count"] > 0 else "completed"
    repo.update_run_status(run_id, final_status)
    logger.info("Continue pipeline complete", run_id=run_id, status=final_status)


def skip_testing_pipeline(run_id: str) -> None:
    """Skip full testing for a pre_detected run, generate report from predetect only."""
    logger.info("Skip testing pipeline", run_id=run_id)
    run = repo.get_run(run_id)
    if not run:
        logger.error("Run not found", run_id=run_id)
        return

    pre_dict = run.get("predetect_result") or {}
    pre_result = PreDetectionResult(
        success=pre_dict.get("success", False),
        identified_as=pre_dict.get("identified_as"),
        confidence=pre_dict.get("confidence", 0.0),
        layer_stopped=pre_dict.get("layer_stopped"),
        total_tokens_used=pre_dict.get("total_tokens_used", 0),
        should_proceed_to_testing=False,
        routing_info=pre_dict.get("routing_info", {}),
    )
    suite_version = run.get("suite_version", "v1")

    _build_and_save_report(run_id, run, pre_result, [], {}, suite_version)
    repo.update_run_status(run_id, "completed")
    logger.info("Skip testing pipeline complete", run_id=run_id)


def run_compare_pipeline(compare_id: str) -> None:
    """
    Build comparison report from two completed runs.
    compare_runs.details will include score deltas and A/B significance.
    """
    logger.info("Compare pipeline starting", compare_id=compare_id)
    compare_row = repo.get_compare_run(compare_id)
    if not compare_row:
        logger.error("Compare run not found", compare_id=compare_id)
        return

    golden_id = compare_row["golden_run_id"]
    candidate_id = compare_row["candidate_run_id"]

    repo.update_compare_run(compare_id, status="running")

    golden_report_row = repo.get_report(golden_id)
    candidate_report_row = repo.get_report(candidate_id)
    if not golden_report_row or not candidate_report_row:
        repo.update_compare_run(
            compare_id,
            status="failed",
            details={"error": "Both runs must be completed and have reports"},
        )
        return

    golden = golden_report_row.get("details") or {}
    candidate = candidate_report_row.get("details") or {}

    g_sc = (golden.get("scorecard") or {})
    c_sc = (candidate.get("scorecard") or {})

    g_total = float(g_sc.get("total_score", 0.0) or 0.0)
    c_total = float(c_sc.get("total_score", 0.0) or 0.0)
    g_cap = float(g_sc.get("capability_score", 0.0) or 0.0)
    c_cap = float(c_sc.get("capability_score", 0.0) or 0.0)
    g_auth = float(g_sc.get("authenticity_score", 0.0) or 0.0)
    c_auth = float(c_sc.get("authenticity_score", 0.0) or 0.0)

    g_sim = (golden.get("similarity") or [{}])[0]
    c_sim = (candidate.get("similarity") or [{}])[0]
    g_top_sim = float(g_sim.get("score", 0.0) or 0.0)
    c_top_sim = float(c_sim.get("score", 0.0) or 0.0)

    delta_total = round(c_total - g_total, 1)
    delta_capability = round(c_cap - g_cap, 1)
    delta_authenticity = round(c_auth - g_auth, 1)
    delta_top_similarity = round(c_top_sim - g_top_sim, 4)

    ab_stats = _compute_ab_significance(golden, candidate)

    reasons: list[str] = []
    if delta_total <= -20:
        reasons.append(f"总分低于官方基线 {abs(delta_total):.1f} 分")
    if delta_authenticity <= -15:
        reasons.append(f"真实性分低于官方基线 {abs(delta_authenticity):.1f} 分")
    if delta_top_similarity <= -0.15:
        reasons.append(f"行为相似度低于官方基线 {abs(delta_top_similarity):.2f}")

    sig_regressions = [
        s for s in ab_stats
        if s.get("significant") and s.get("delta", 0) < 0
    ]
    if sig_regressions:
        reasons.append(f"存在 {len(sig_regressions)} 项统计显著退化")

    if not reasons:
        level = "close"
        label = "接近官方基线 / Close to Baseline"
        reasons.append("候选渠道与官方基线差距可接受")
    elif delta_total <= -35 or delta_authenticity <= -30 or len(sig_regressions) >= 2:
        level = "high_risk"
        label = "高风险疑似降级/假模型 / High Risk"
    else:
        level = "suspicious"
        label = "存在可疑差距 / Suspicious Gap"

    details = {
        "compare_id": compare_id,
        "golden_run_id": golden_id,
        "candidate_run_id": candidate_id,
        "deltas": {
            "total": delta_total,
            "capability": delta_capability,
            "authenticity": delta_authenticity,
            "top_similarity": delta_top_similarity,
        },
        "golden": {
            "scorecard": g_sc,
            "top_similarity": g_sim,
        },
        "candidate": {
            "scorecard": c_sc,
            "top_similarity": c_sim,
        },
        "ab_significance": ab_stats,
        "verdict": {
            "level": level,
            "label": label,
            "reasons": reasons,
        },
    }

    repo.update_compare_run(compare_id, status="completed", details=details)
    logger.info("Compare pipeline complete", compare_id=compare_id, level=level)


def _compute_ab_significance(golden_report: dict, candidate_report: dict) -> list[dict]:
    metrics = [
        "pass_rate",
        "mean_latency_ms",
    ]

    g_cases = {c.get("case_id"): c for c in (golden_report.get("case_results") or [])}
    c_cases = {c.get("case_id"): c for c in (candidate_report.get("case_results") or [])}
    common_ids = [cid for cid in g_cases.keys() if cid in c_cases]

    out = []
    for metric in metrics:
        g_vals: list[float] = []
        c_vals: list[float] = []
        for cid in common_ids:
            gv = g_cases[cid].get(metric)
            cv = c_cases[cid].get(metric)
            if gv is None or cv is None:
                continue
            try:
                g_vals.append(float(gv))
                c_vals.append(float(cv))
            except (TypeError, ValueError):
                continue

        if len(g_vals) < 3 or len(c_vals) < 3:
            continue

        out.append(_paired_bootstrap(metric, g_vals, c_vals))

    return out


def _paired_bootstrap(metric: str, g_vals: list[float], c_vals: list[float], n: int = 1000) -> dict:
    deltas = [c - g for c, g in zip(c_vals, g_vals)]
    mean_g = sum(g_vals) / len(g_vals)
    mean_c = sum(c_vals) / len(c_vals)
    mean_delta = mean_c - mean_g

    rng = random.Random(42)
    boots = []
    for _ in range(n):
        idxs = [rng.randrange(len(deltas)) for _ in range(len(deltas))]
        sample = [deltas[i] for i in idxs]
        boots.append(sum(sample) / len(sample))

    boots.sort()
    lo = boots[int(0.025 * n)]
    hi = boots[int(0.975 * n)]

    opp_sign = sum(1 for b in boots if (b <= 0 if mean_delta > 0 else b >= 0))
    p_value = min(1.0, 2 * opp_sign / n)
    significant = (lo > 0) or (hi < 0)

    return {
        "metric": metric,
        "golden_mean": round(mean_g, 4),
        "candidate_mean": round(mean_c, 4),
        "delta": round(mean_delta, 4),
        "ci_95_low": round(lo, 4),
        "ci_95_high": round(hi, 4),
        "p_value": round(p_value, 6),
        "significant": significant,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# P3: asyncio-native concurrent loop + async pipeline
# ═══════════════════════════════════════════════════════════════════════════════

import asyncio as _asyncio

async def _run_cases_async(
    adapter,
    model_name: str,
    cases: list,
    test_mode: str,
    run_id: str,
    phase_label: str,
    case_results: list,
    failed_count_ref: dict,
    budget_guard=None,
) -> bool:
    """
    Async replacement for _run_cases_concurrent.

    Uses asyncio.Semaphore to cap concurrency (same as ThreadPoolExecutor
    max_workers) but without thread creation overhead. All case coroutines
    run in the same event loop thread — I/O waits yield the loop to peers.

    Returns True if the run was cancelled.
    """
    from app.runner.case_executor_async import async_execute_case

    if not cases:
        return False

    max_workers = _mode_concurrency(test_mode)
    semaphore = _asyncio.Semaphore(max_workers)
    budget_threshold = (budget_guard.budget * 0.05) if budget_guard else 0
    pending_saves: list = []
    budget_exhausted = False

    async def _run_one(case) -> tuple:
        if budget_exhausted:
            return case, None, None
        async with semaphore:
            try:
                result = await async_execute_case(adapter, model_name, case)
                return case, result, None
            except Exception as exc:
                return case, None, exc

    # Build and gather all tasks
    tasks = [_asyncio.create_task(_run_one(c)) for c in cases]
    check_early_abort = False

    for completed in _asyncio.as_completed(tasks):
        # Check for cancellation
        cancelled = await _asyncio.to_thread(repo.is_run_cancel_requested, run_id)
        if cancelled:
            for t in tasks:
                t.cancel()
            return True

        case, result, exc = await completed

        if exc is not None:
            failed_count_ref["count"] += 1
            logger.warning("Async case failed", run_id=run_id,
                           case_id=getattr(case, "id", "?"), error=str(exc))
            check_early_abort = True
            continue

        if result is None:
            continue  # budget_exhausted skip

        case_results.append(result)
        pending_saves.append(result)

        # Budget tracking
        if budget_guard:
            budget_guard.record_result(result)
            if budget_guard.remaining < budget_threshold:
                budget_exhausted = True
                logger.info(
                    "Token budget exhausted (async), skipping remaining",
                    run_id=run_id, phase=phase_label,
                    used=budget_guard.used, budget=budget_guard.budget,
                )

        # Batch save every 5 results (DB write offloaded to thread)
        if len(pending_saves) >= 5:
            saves_copy = pending_saves[:]
            pending_saves.clear()
            await _asyncio.to_thread(_save_case_results_batch, run_id, saves_copy)
            check_early_abort = True

        logger.info(
            "Async case done",
            run_id=run_id, case_id=case.id,
            pass_rate=round(result.pass_rate, 2), phase=phase_label,
        )

        # Early abort on >80% failure rate after 10+ results
        if check_early_abort and len(case_results) >= 10:
            total = len(case_results)
            failed = failed_count_ref["count"]
            if total > 0 and (failed / total) > 0.8:
                logger.warning("Async error rate >80%, aborting phase early",
                               run_id=run_id, phase=phase_label)
                for t in tasks:
                    t.cancel()
                break
            check_early_abort = False

    if pending_saves:
        await _asyncio.to_thread(_save_case_results_batch, run_id, pending_saves)

    return False


async def run_pipeline_async(run_id: str) -> None:
    """
    Phase D: asyncio-native full test pipeline.

    Mirrors run_pipeline() but replaces _run_cases_concurrent with
    _run_cases_async so all HTTP I/O is non-blocking.

    Entry point: call asyncio.run(run_pipeline_async(run_id)) from
    the worker thread, or await it from an existing event loop.
    """
    from app.adapters.openai_compat_async import AsyncOpenAICompatibleAdapter

    logger.info("Async pipeline starting", run_id=run_id)

    run = repo.get_run(run_id)
    if not run:
        logger.error("Run not found", run_id=run_id)
        return

    km = get_key_manager()
    try:
        api_key = km.decrypt(run["api_key_encrypted"])
    except Exception as e:
        repo.update_run_status(run_id, "failed", error_message=f"Key decrypt failed: {e}")
        return

    adapter = AsyncOpenAICompatibleAdapter(run["base_url"], api_key)
    # Sync adapter for operations that are still sync (predetect, conn check)
    sync_adapter = adapter.to_sync_adapter()

    suite_version = run.get("suite_version", "v1")
    test_mode = run.get("test_mode", "standard")

    # ── Step 1: Pre-detection (sync, stays blocking for now) ─────────────────
    repo.update_run_status(run_id, "pre_detecting")
    extraction_mode = test_mode == "deep"
    try:
        pre_result = await _asyncio.to_thread(
            lambda: PreDetectionPipeline().run(
                sync_adapter, run["model_name"],
                extraction_mode=extraction_mode,
                run_id=run_id,
            )
        )
        repo.save_predetect_result(run_id, pre_result.to_dict())
        logger.info("Async pre-detection complete", run_id=run_id,
                    identified=pre_result.identified_as, confidence=pre_result.confidence)
    except Exception as e:
        logger.warning("Async pre-detection failed, continuing", error=str(e))
        pre_result = PreDetectionResult(
            success=False, identified_as=None, confidence=0.0,
            layer_stopped=None, should_proceed_to_testing=True,
        )

    if not pre_result.should_proceed_to_testing and not extraction_mode:
        repo.update_run_status(run_id, "pre_detected")
        return

    # ── Step 2: Connectivity check (no async needed — fast probe) ────────────
    repo.update_run_status(run_id, "running")
    conn_check = await adapter.alist_models()
    conn_status = conn_check.get("status_code")
    conn_error = conn_check.get("error")

    if conn_error and not conn_status:
        msg = (
            f"无法连接到 API：{conn_error}。"
            f"请检查 base_url 是否正确（当前：{run['base_url']}），"
            f"网络是否可达，以及 API Key 是否有效。"
        )
        repo.update_run_status(run_id, "failed", error_message=msg)
        logger.error("Async connectivity failed", run_id=run_id, error=msg)
        return

    # ── Step 3: Load and execute test cases ──────────────────────────────────
    cases = _load_suite(suite_version, test_mode)
    if not cases:
        repo.update_run_status(run_id, "failed", error_message="No test cases loaded")
        return

    if (
        test_mode != "full"
        and pre_result.success
        and 0.60 <= pre_result.confidence < settings.PREDETECT_CONFIDENCE_THRESHOLD
        and pre_result.identified_as
    ):
        cases = _select_confirmatory_cases(cases, pre_result.identified_as)

    phase1_cases, phase2_cases = _prepare_cases(cases, test_mode)

    existing_responses = repo.get_responses(run_id)
    completed_case_ids = {r.get("case_id") for r in existing_responses if r.get("case_id")}
    if completed_case_ids:
        phase1_cases = [c for c in phase1_cases if c.id not in completed_case_ids]
        phase2_cases = [c for c in phase2_cases if c.id not in completed_case_ids]

    case_results: list[CaseResult] = []
    failed_count_ref = {"count": 0}
    budget_guard = TokenBudgetGuard(
        {
            "quick": settings.TOKEN_BUDGET_QUICK,
            "standard": settings.TOKEN_BUDGET_STANDARD,
            "deep": settings.TOKEN_BUDGET_DEEP,
            "full": settings.TOKEN_BUDGET_FULL,
        }.get(test_mode, settings.TOKEN_BUDGET_STANDARD)
    )

    logger.info("Async executing test cases", run_id=run_id,
                total=len(phase1_cases) + len(phase2_cases),
                phase1=len(phase1_cases), phase2=len(phase2_cases),
                concurrency=_mode_concurrency(test_mode))

    cancelled = await _run_cases_async(
        adapter=adapter, model_name=run["model_name"],
        cases=phase1_cases, test_mode=test_mode, run_id=run_id,
        phase_label="phase1", case_results=case_results,
        failed_count_ref=failed_count_ref, budget_guard=budget_guard,
    )
    if cancelled:
        repo.update_run_status(run_id, "failed", error_message="Cancelled by user")
        return

    stop_now, cached_features, cached_sims, _ = _checkpoint_should_stop(
        test_mode=test_mode, case_results=case_results,
        features_cache=None, sims_cache=None, scorecard_cache=None,
    )
    if stop_now:
        logger.info("Async early stop triggered", run_id=run_id, test_mode=test_mode)
        _build_and_save_report(run_id, run, pre_result, case_results,
                               cached_features or {}, suite_version,
                               precomputed_similarities=cached_sims)
        final_status = "partial_failed" if failed_count_ref["count"] > 0 else "completed"
        repo.update_run_status(run_id, final_status)
        return

    if phase2_cases:
        # Progressive partial report after phase 1
        try:
            _build_and_save_report(run_id, run, pre_result, case_results,
                                   {"partial": 1.0}, suite_version,
                                   precomputed_similarities=cached_sims)
        except Exception as e:
            logger.warning("Async partial report failed", error=str(e))

        cancelled = await _run_cases_async(
            adapter=adapter, model_name=run["model_name"],
            cases=phase2_cases, test_mode=test_mode, run_id=run_id,
            phase_label="phase2", case_results=case_results,
            failed_count_ref=failed_count_ref, budget_guard=budget_guard,
        )
        if cancelled:
            repo.update_run_status(run_id, "failed", error_message="Cancelled by user")
            return

    # ── Steps 4-6: Analysis + report (CPU-bound, stays sync) ─────────────────
    await _asyncio.to_thread(
        _build_and_save_report, run_id, run, pre_result, case_results, {}, suite_version
    )
    final_status = "partial_failed" if failed_count_ref["count"] > 0 else "completed"
    repo.update_run_status(run_id, final_status)
    logger.info("Async pipeline complete", run_id=run_id, status=final_status)

