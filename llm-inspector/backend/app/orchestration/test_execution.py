from __future__ import annotations
from app.core.schemas import CaseResult, PreDetectionResult
from app.core.config import settings
from app.core.logging import get_logger
from app.runner.budget_control import TokenBudgetGuard, SmartModeStrategy
from app.runner.execution import _run_cases_concurrent, _checkpoint_should_stop
from app.runner.case_prep import _mode_concurrency

logger = get_logger(__name__)

class ExecutionFlow:
    """Handles test case execution, concurrency, and budgeting."""

    def initialize_budget(self, test_mode: str, pre_result: PreDetectionResult) -> TokenBudgetGuard:
        """Initializes the token budget guard based on the test mode and pre-detection results."""
        budget_map = {
            "quick":      settings.TOKEN_BUDGET_QUICK,
            "standard":   settings.TOKEN_BUDGET_STANDARD,
            "deep":       settings.TOKEN_BUDGET_DEEP,
            "full":       settings.TOKEN_BUDGET_FULL,
            "extraction": settings.TOKEN_BUDGET_FULL,
            "smart":      13_000,
        }
        
        if test_mode == "smart" and pre_result.confidence > 0:
            strategy = SmartModeStrategy()
            smart_budget = strategy.decide_budget(pre_result)
            logger.info("Smart mode budget initialized", budget=smart_budget.token_budget)
            return TokenBudgetGuard(smart_budget.token_budget)
        
        budget = budget_map.get(test_mode, settings.TOKEN_BUDGET_STANDARD)
        logger.info("Token budget initialized", mode=test_mode, budget=budget)
        return TokenBudgetGuard(budget)

    def run_phase(
        self,
        adapter,
        model_name: str,
        cases: list,
        run_id: str,
        phase_label: str,
        test_mode: str,
        budget_guard: TokenBudgetGuard,
        base_url: str,
        case_results: list[CaseResult],
        failed_count_ref: dict,
        backoff_state: dict,
        tracer=None,
    ) -> bool:
        """Runs a specific phase of test cases. Returns True if cancelled by user."""
        logger.info(f"Starting {phase_label}", run_id=run_id, cases=len(cases))
        
        # Proxy to the existing concurrent runner
        cancelled = _run_cases_concurrent(
            adapter=adapter,
            model_name=model_name,
            cases=cases,
            test_mode=test_mode,
            run_id=run_id,
            phase_label=phase_label,
            case_results=case_results,
            failed_count_ref=failed_count_ref,
            backoff_state=backoff_state,
            budget_guard=budget_guard,
            base_url=base_url,
        )
        return cancelled

    def check_early_stop(
        self,
        test_mode: str,
        case_results: list[CaseResult],
    ):
        """Checks if the early stop condition is met."""
        stop_now, cached_features, cached_sims, _ = _checkpoint_should_stop(
            test_mode=test_mode,
            case_results=case_results,
            features_cache=None,
            sims_cache=None,
            scorecard_cache=None,
        )
        return stop_now, cached_features, cached_sims
