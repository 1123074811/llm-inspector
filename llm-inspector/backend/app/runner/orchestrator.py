"""
runner/orchestrator.py — main pipeline entry points

Contains only the three public entry points:
  run_pipeline()          — full sync pipeline
  continue_pipeline()     — resume a pre_detected run
  skip_testing_pipeline() — skip testing, generate report from predetect only

All helper functions/classes have been extracted to:
  budget_control.py  — SmartBudget, SmartModeStrategy, TokenBudgetGuard
  case_prep.py       — _load_suite, _load_benchmarks, _prepare_cases, etc.
  execution.py       — _run_cases_concurrent, _checkpoint_should_stop, etc.
  report_assembly.py — _build_and_save_report
  compare.py         — run_compare_pipeline, A/B significance
  async_runner.py    — run_pipeline_async, _run_cases_async
"""
from __future__ import annotations

import time

from app.core.schemas import (
    TestCase, CaseResult, PreDetectionResult,
)
from app.core.eval_schemas import EvalTestCase
from app.core.circuit_breaker import circuit_breaker, CircuitState
from app.core.tracer import get_tracer, remove_tracer
from app.core.logging import get_logger
from app.core.security import get_key_manager
from app.core.config import settings
from app.adapters.openai_compat import OpenAICompatibleAdapter
from app.predetect.pipeline import PreDetectionPipeline
from app.repository import repo

# ── Re-exports from split modules (for backward compatibility) ─────────────────
from app.runner.budget_control import (                    # noqa: F401
    FAMILY_DISCRIMINATORS, SmartBudget, SmartModeStrategy, TokenBudgetGuard,
)
from app.runner.case_prep import (                         # noqa: F401
    _FIXTURES_DIR, _benchmark_cache, _load_suite, _load_benchmarks,
    _save_case_results_batch, _mode_concurrency, _case_value,
    _adaptive_samples, _CONFIRMATORY_CATEGORIES, _select_confirmatory_cases,
    _prepare_cases,
)
from app.runner.execution import (                         # noqa: F401
    _adaptive_pause, _update_backoff, _checkpoint_should_stop,
    _run_cases_concurrent,
)
from app.runner.report_assembly import _build_and_save_report  # noqa: F401
from app.runner.compare import (                           # noqa: F401
    run_compare_pipeline, _compute_ab_significance, _paired_bootstrap,
)
from app.runner.async_runner import (                      # noqa: F401
    _run_cases_async, run_pipeline_async,
)

logger = get_logger(__name__)


from app.orchestration.run_lifecycle import RunLifecycleManager

def run_pipeline(run_id: str) -> None:
    """Entry point for full pipeline execution (v12 refactored)."""
    manager = RunLifecycleManager(run_id)
    manager.execute_full()


def continue_pipeline(run_id: str) -> None:
    """Entry point for continuing a pre-detected run (v12 refactored)."""
    manager = RunLifecycleManager(run_id)
    manager.execute_continue()


def skip_testing_pipeline(run_id: str) -> None:
    """Skip full testing, generate report from predetect only (v12 refactored)."""
    from app.orchestration.report_flow import ReportFlow
    from app.core.schemas import PreDetectionResult

    run = repo.get_run(run_id)
    if not run:
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
    
    reporting = ReportFlow()
    reporting.assemble_and_save_report(
        run_id, run, pre_result, [], {}, run.get("suite_version", "v1")
    )
    repo.update_run_status(run_id, "completed")
