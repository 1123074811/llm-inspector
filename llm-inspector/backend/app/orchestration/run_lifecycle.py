from __future__ import annotations
import time
from app.core.schemas import CaseResult, PreDetectionResult
from app.core.circuit_breaker import circuit_breaker
from app.core.tracer import get_tracer, remove_tracer
from app.core.logging import get_logger
from app.core.security import get_key_manager
from app.core.config import settings
from app.adapters.openai_compat import OpenAICompatibleAdapter
from app.predetect.pipeline import PreDetectionPipeline
from app.repository import repo
from app.runner.case_prep import _load_suite, _prepare_cases, _select_confirmatory_cases, _case_value
from app.orchestration.test_execution import ExecutionFlow
from app.orchestration.report_flow import ReportFlow

logger = get_logger(__name__)

class RunLifecycleManager:
    """Manages the full V12 pipeline lifecycle and state transitions."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.tracer = get_tracer(run_id)
        self.execution = ExecutionFlow()
        self.reporting = ReportFlow()
        self.run_metadata = None
        self.adapter = None
        self.base_url = None

    def execute_full(self) -> None:
        """Executes the full pipeline starting from pre-detection."""
        from app.core.events import emit, EventKind
        logger.info("Lifecycle: Starting full pipeline", run_id=self.run_id)
        self.tracer.start()
        emit(self.run_id, EventKind.RUN_STARTED)

        try:
            if not self._initialize():
                return

            # 0. Preflight — must pass before any token consumption
            pre_flight = self._step_preflight()
            if pre_flight is not None and not pre_flight.passed:
                return  # already set run status to preflight_failed

            # 1. Pre-detection
            pre_result = self._step_predetect()
            if not pre_result: # Error in pre-detect flow itself
                return

            # 2. Check if we should pause after pre-detect
            if not pre_result.should_proceed_to_testing and self.run_metadata.get("test_mode") != "deep":
                logger.info("Lifecycle: Pausing after pre-detection", run_id=self.run_id)
                repo.update_run_status(self.run_id, "pre_detected")
                remove_tracer(self.run_id)
                return

            # 3. Connectivity
            if not self._step_connectivity():
                return

            # 4. Testing & Reporting
            self._step_testing_and_reporting(pre_result)

            self.tracer.finish()
        except Exception as exc:
            logger.error("Unhandled exception in execute_full", run_id=self.run_id, error=str(exc))
            try:
                current = repo.get_run(self.run_id)
                if current and current.get("status") not in ("completed", "failed", "partial_failed", "cancelled"):
                    repo.update_run_status(self.run_id, "failed",
                        error_message=f"Unhandled error: {exc}",
                        error_code="E_UNHANDLED")
            except Exception:
                pass
            emit(self.run_id, EventKind.RUN_FAILED, error=str(exc))
        finally:
            remove_tracer(self.run_id)

    def execute_continue(self) -> None:
        """Resumes a pre-detected run."""
        from app.core.events import emit, EventKind
        logger.info("Lifecycle: Continuing pipeline", run_id=self.run_id)
        self.tracer.start()
        emit(self.run_id, EventKind.RUN_STARTED)

        try:
            if not self._initialize():
                return

            pre_dict = self.run_metadata.get("predetect_result") or {}
            pre_result = PreDetectionResult(
                success=pre_dict.get("success", False),
                identified_as=pre_dict.get("identified_as"),
                confidence=pre_dict.get("confidence", 0.0),
                layer_stopped=pre_dict.get("layer_stopped"),
                total_tokens_used=pre_dict.get("total_tokens_used", 0),
                should_proceed_to_testing=True,
                routing_info=pre_dict.get("routing_info", {}),
            )

            if not self._step_connectivity():
                return

            self._step_testing_and_reporting(pre_result)

            self.tracer.finish()
        except Exception as exc:
            logger.error("Unhandled exception in execute_continue", run_id=self.run_id, error=str(exc))
            try:
                current = repo.get_run(self.run_id)
                if current and current.get("status") not in ("completed", "failed", "partial_failed", "cancelled"):
                    repo.update_run_status(self.run_id, "failed",
                        error_message=f"Unhandled error: {exc}",
                        error_code="E_UNHANDLED")
            except Exception:
                pass
            emit(self.run_id, EventKind.RUN_FAILED, error=str(exc))
        finally:
            remove_tracer(self.run_id)

    def _initialize(self) -> bool:
        """Loads metadata and initializes adapter."""
        self.run_metadata = repo.get_run(self.run_id)
        if not self.run_metadata:
            logger.error("Run not found", run_id=self.run_id)
            # Best-effort status update so the run doesn't stay "queued" forever.
            try:
                repo.update_run_status(self.run_id, "failed",
                                       error_message="Run record missing at pipeline start",
                                       error_code="E_RUN_NOT_FOUND")
            except Exception:
                pass
            remove_tracer(self.run_id)
            return False

        km = get_key_manager()
        try:
            api_key = km.decrypt(self.run_metadata["api_key_encrypted"])
            self.base_url = self.run_metadata["base_url"]
            self.adapter = OpenAICompatibleAdapter(self.base_url, api_key)
        except Exception as e:
            repo.update_run_status(self.run_id, "failed", error_message=f"Init failed: {e}", error_code="E_INIT_FAILED")
            remove_tracer(self.run_id)
            return False

        if circuit_breaker.is_open(self.base_url):
            logger.warning("Circuit breaker OPEN", base_url=self.base_url)
            repo.update_run_status(self.run_id, "suspended", error_message="Circuit breaker open", error_code="E_CB_OPEN")
            remove_tracer(self.run_id)
            return False
        
        return True

    def _step_preflight(self):
        """Run preflight connection check before any actual testing (v15 Phase 1)."""
        from app.core.events import emit, EventKind
        from app.preflight.connection_check import run_preflight

        emit(self.run_id, EventKind.PREFLIGHT_STARTED)
        repo.update_run_status(self.run_id, "preflight_running")

        try:
            km = get_key_manager()
            api_key = km.decrypt(self.run_metadata["api_key_encrypted"])
            base_url = self.run_metadata["base_url"]
            model_name = self.run_metadata["model_name"]

            report = run_preflight(base_url, api_key, model_name)

            # Save report to DB
            import json
            repo.update_run_field(self.run_id, "preflight_report", json.dumps(report.to_dict()))

            if report.passed:
                emit(self.run_id, EventKind.PREFLIGHT_PASSED,
                     duration_ms=report.total_duration_ms)
                logger.info("Preflight passed", run_id=self.run_id,
                            duration_ms=report.total_duration_ms)
                return report
            else:
                err = report.first_error
                emit(self.run_id, EventKind.PREFLIGHT_FAILED,
                     error_code=err.code if err else "E_UNKNOWN",
                     retryable=err.retryable if err else True)
                msg = err.user_message_zh if err else "连接预检失败"
                code = err.code if err else "E_PREFLIGHT_FAILED"
                repo.update_run_status(self.run_id, "preflight_failed",
                                       error_message=msg, error_code=code)
                remove_tracer(self.run_id)
                return report
        except Exception as e:
            logger.error("Preflight exception", run_id=self.run_id, error=str(e))
            repo.update_run_status(self.run_id, "preflight_failed",
                                   error_message=f"预检测异常: {e}",
                                   error_code="E_PREFLIGHT_EXCEPTION")
            remove_tracer(self.run_id)
            return None

    def _step_predetect(self) -> PreDetectionResult | None:
        """Runs pre-detection step."""
        repo.update_run_status(self.run_id, "pre_detecting")
        test_mode = self.run_metadata.get("test_mode", "standard")
        extraction_mode = test_mode == "deep"
        
        try:
            with self.tracer.span("predetect", mode=test_mode):
                pre_result = PreDetectionPipeline().run(
                    self.adapter, self.run_metadata["model_name"],
                    extraction_mode=extraction_mode,
                    run_id=self.run_id,
                )
            repo.save_predetect_result(self.run_id, pre_result.to_dict())
            circuit_breaker.record_success(self.base_url)
            return pre_result
        except Exception as e:
            logger.warning("Pre-detection failed, falling back", error=str(e))
            circuit_breaker.record_failure(self.base_url, str(e))
            return PreDetectionResult(
                success=False, identified_as=None, confidence=0.0,
                layer_stopped=None, should_proceed_to_testing=True,
            )

    def _step_connectivity(self) -> bool:
        """Checks API connectivity."""
        repo.update_run_status(self.run_id, "running")
        with self.tracer.span("connectivity", base_url=self.base_url):
            conn_check = self.adapter.list_models()
        
        if conn_check.get("error") and not conn_check.get("status_code"):
            msg = f"API 连接故障: {conn_check.get('error')}"
            repo.update_run_status(self.run_id, "failed", error_message=msg, error_code="E_NETWORK_FAIL")
            circuit_breaker.record_failure(self.base_url, "network_error")
            remove_tracer(self.run_id)
            return False
        
        # Additional probe logic could go here if status was 401/403 etc.
        # (Keeping it simple for now, can port the full probe logic later if needed)
        return True

    def _step_testing_and_reporting(self, pre_result: PreDetectionResult) -> None:
        """Runs testing phases and final report."""
        test_mode = self.run_metadata.get("test_mode", "standard")
        suite_version = self.run_metadata.get("suite_version", "v1")
        
        # 1. Load cases
        cases = _load_suite(suite_version, test_mode)
        if not cases:
            repo.update_run_status(self.run_id, "failed", error_message="No cases found in bank", error_code="E_EMPTY_SUITE")
            return

        # 2. Filter for confirmatory if needed (quick mode only —
        #    standard/deep must run all assigned cases for data accuracy)
        if (test_mode == "quick" and pre_result.success and
            0.60 <= pre_result.confidence < settings.PREDETECT_CONFIDENCE_THRESHOLD):
            cases = _select_confirmatory_cases(cases, pre_result.identified_as)

        phase1, phase2 = _prepare_cases(cases, test_mode)

        # 3. Initialize Execution Progress
        case_results = []
        failed_count_ref = {"count": 0}
        backoff_state = {"delay_ms": settings.INTER_REQUEST_DELAY_MS}
        budget_guard = self.execution.initialize_budget(test_mode, pre_result)

        # 4. Phase 1
        with self.tracer.span("phase1", count=len(phase1)):
            cancelled = self.execution.run_phase(
                self.adapter, self.run_metadata["model_name"], phase1, self.run_id,
                "phase1", test_mode, budget_guard, self.base_url,
                case_results, failed_count_ref, backoff_state, self.tracer
            )
        
        if cancelled:
            repo.update_run_status(self.run_id, "failed", error_message="Cancelled")
            return

        # 5. Early Stop Check
        stop_now, cached_f, cached_s = self.execution.check_early_stop(test_mode, case_results)
        if stop_now:
            self.reporting.assemble_and_save_report(
                self.run_id, self.run_metadata, pre_result, case_results,
                cached_f or {}, suite_version, precomputed_similarities=cached_s
            )
            repo.update_run_status(self.run_id, "completed")
            return

        # 6. Phase 2 (and Phase 3 Arbitration if standard)
        if phase2:
            with self.tracer.span("phase2", count=len(phase2)):
                self.execution.run_phase(
                    self.adapter, self.run_metadata["model_name"], phase2, self.run_id,
                    "phase2", test_mode, budget_guard, self.base_url,
                    case_results, failed_count_ref, backoff_state, self.tracer
                )
            
            # Arbitration for standard mode
            if test_mode == "standard":
                stop2, _, _ = self.execution.check_early_stop(test_mode, case_results)
                if not stop2:
                    seen_ids = {c.case.id for c in case_results}
                    remaining = sorted([c for c in cases if c.id not in seen_ids], key=_case_value, reverse=True)
                    arbitration = remaining[:settings.ARBITRATION_MAX]
                    if arbitration:
                        self.execution.run_phase(
                            self.adapter, self.run_metadata["model_name"], arbitration, self.run_id,
                            "phase3", test_mode, budget_guard, self.base_url,
                            case_results, failed_count_ref, backoff_state, self.tracer
                        )

        # 7. Final Report
        self.reporting.assemble_and_save_report(
            self.run_id, self.run_metadata, pre_result, case_results,
            {}, suite_version
        )
        
        final_status = "partial_failed" if failed_count_ref["count"] > 0 else "completed"
        repo.update_run_status(self.run_id, final_status)
