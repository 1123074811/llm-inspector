"""
Transparent Judge Wrapper — v8.0 Judgment Transparency Layer

Wraps existing judge methods to provide:
- Detailed step-by-step logging
- Threshold source tracking
- Full input/output capture
- Performance metrics

Reference: V8_UPGRADE_PLAN.md Section 1.4.2
"""
from typing import Dict, Any, Optional, Tuple
import time
import uuid

from app.judge.plugin_interface import JudgePlugin, JudgeResult, JudgeMetadata
from app.core.structured_logger import get_structured_logger, LogEventType, LogLevel


class TransparentJudgeWrapper:
    """
    Wrapper that adds transparent logging to any judge method.
    
    Usage:
        base_plugin = ExactMatchPlugin()
        wrapped = TransparentJudgeWrapper(base_plugin)
        result = wrapped.judge(response, params)
        # Full details logged automatically
    """
    
    def __init__(self, plugin: JudgePlugin):
        self._plugin = plugin
        self._logger = get_structured_logger()
    
    @property
    def metadata(self) -> JudgeMetadata:
        return self._plugin.metadata
    
    def judge(self, response: str, params: Dict[str, Any]) -> JudgeResult:
        """
        Execute judgment with full transparency logging.
        """
        trace_id = str(uuid.uuid4())[:8]
        method = self.metadata.name
        case_id = params.get("case_id", "unknown")
        
        # Log start
        self._logger.log_judge_start(case_id, method, trace_id)
        
        start_time = time.time()
        
        try:
            # Validate parameters (with logging)
            is_valid, errors = self._validate_and_log(params, trace_id)
            if not is_valid:
                result = JudgeResult(
                    passed=None,
                    detail={"error": "Parameter validation failed", "errors": errors},
                    method=method,
                    version=self.metadata.version
                )
                self._log_completion(case_id, method, result, 0, trace_id)
                return result
            
            # Execute judgment
            result = self._plugin.judge(response, params)
            
            # Calculate metrics
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Log completion
            self._log_completion(case_id, method, result, latency_ms, trace_id)
            
            # Log threshold if present
            if result.threshold_source and result.threshold_value is not None:
                self._logger.log_threshold_apply(
                    "judge",
                    f"{method}_threshold",
                    result.threshold_value,
                    result.threshold_source,
                    {"case_id": case_id, "passed": result.passed}
                )
            
            return result
            
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Log error
            self._logger.log(
                LogEventType.JUDGE_ERROR,
                LogLevel.ERROR,
                "judge",
                f"Judgment error for case {case_id}: {str(e)}",
                {
                    "case_id": case_id,
                    "method": method,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "trace_id": trace_id
                }
            )
            
            # Return error result
            return JudgeResult(
                passed=None,
                detail={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "trace_id": trace_id
                },
                latency_ms=latency_ms,
                method=method,
                version=self.metadata.version
            )
    
    def _validate_and_log(self, params: Dict[str, Any], trace_id: str) -> Tuple[bool, list]:
        """Validate and log validation steps."""
        is_valid, errors = self._plugin.validate_params(params)
        
        if errors:
            self._logger.log(
                LogEventType.JUDGE_STEP,
                LogLevel.WARNING,
                "judge",
                "Parameter validation found issues",
                {
                    "step": "validation",
                    "errors": errors,
                    "trace_id": trace_id
                }
            )
        
        return is_valid, errors
    
    def _log_completion(self, case_id: str, method: str, 
                       result: JudgeResult, latency_ms: int, trace_id: str):
        """Log judgment completion."""
        self._logger.log_judge_complete(
            case_id=case_id,
            method=method,
            passed=result.passed if result.passed is not None else False,
            confidence=result.confidence,
            detail=result.detail,
            tokens_used=result.tokens_used,
            latency_ms=latency_ms + result.latency_ms
        )


class JudgmentLogger:
    """
    Utility class for detailed judgment logging.
    
    Provides granular logging for complex multi-step judgments.
    """
    
    def __init__(self, case_id: str, method: str):
        self.case_id = case_id
        self.method = method
        self.logger = get_structured_logger()
        self.steps: list = []
        self.start_time = time.time()
    
    def log_step(self, 
                 step_name: str,
                 input_data: Dict,
                 output_data: Dict,
                 threshold: Optional[float] = None,
                 threshold_source: Optional[str] = None):
        """Log a single judgment step."""
        step_data = {
            "step": step_name,
            "timestamp": time.time() - self.start_time,
            "input": input_data,
            "output": output_data,
        }
        
        if threshold is not None:
            step_data["threshold"] = threshold
        if threshold_source:
            step_data["threshold_source"] = threshold_source
        
        self.steps.append(step_data)
        
        # Also log to structured logger
        self.logger.log_judge_step(
            self.case_id,
            step_name,
            input_data,
            output_data,
            threshold,
            threshold_source
        )
    
    def log_coverage(self,
                    constraints: list,
                    response_keywords: list,
                    coverage: float,
                    threshold: float,
                    threshold_source: str):
        """Log keyword coverage calculation."""
        self.log_step(
            "keyword_coverage",
            {
                "constraints": constraints,
                "response_length": len(response_keywords)
            },
            {
                "coverage": coverage,
                "matched": [kw for kw in response_keywords if any(c.lower() in kw.lower() for c in constraints)],
                "missing": [c for c in constraints if not any(c.lower() in kw.lower() for kw in response_keywords)]
            },
            threshold,
            threshold_source
        )
    
    def log_quality_grade(self,
                         grade: str,
                         factors: Dict[str, Any]):
        """Log quality grade assignment."""
        self.log_step(
            "quality_grading",
            {"factors": factors},
            {"grade": grade}
        )
    
    def finalize(self,
                 passed: bool,
                 final_detail: Dict) -> Dict[str, Any]:
        """Finalize logging and return complete record."""
        total_time = time.time() - self.start_time
        
        record = {
            "case_id": self.case_id,
            "method": self.method,
            "total_steps": len(self.steps),
            "total_time_ms": int(total_time * 1000),
            "passed": passed,
            "final_detail": final_detail,
            "steps": self.steps
        }
        
        self.logger.log(
            LogEventType.JUDGE_COMPLETE,
            LogLevel.INFO,
            "judge",
            f"Judgment finalized for {self.case_id}",
            record
        )
        
        return record


def create_transparent_judge(method_name: str) -> Optional[TransparentJudgeWrapper]:
    """
    Factory function to create a transparent wrapper for a judge method.
    
    Args:
        method_name: Name of the judge method
        
    Returns:
        TransparentJudgeWrapper or None if method not found
    """
    from app.judge.plugin_manager import get_plugin_manager
    
    manager = get_plugin_manager()
    plugin = manager.get_plugin(method_name)
    
    if not plugin:
        return None
    
    return TransparentJudgeWrapper(plugin)


def judge_with_transparency(method: str, 
                            response: str, 
                            params: Dict[str, Any],
                            case_id: Optional[str] = None) -> Tuple[JudgeResult, Dict[str, Any]]:
    """
    Execute judgment with full transparency logging.
    
    Args:
        method: Judge method name
        response: Model response
        params: Judge parameters
        case_id: Optional case identifier
        
    Returns:
        Tuple of (JudgeResult, transparency_log)
    """
    if case_id:
        params = {**params, "case_id": case_id}
    
    wrapper = create_transparent_judge(method)
    if not wrapper:
        error_result = JudgeResult(
            passed=None,
            detail={"error": f"Unknown judge method: {method}"},
            method=method
        )
        return error_result, {"error": "Method not found"}
    
    result = wrapper.judge(response, params)
    
    # Retrieve logged data
    logger = get_structured_logger()
    recent_logs = logger.get_recent(10)
    
    transparency_log = {
        "method": method,
        "logs": [log.to_dict() for log in recent_logs],
        "result_summary": result.to_dict()
    }
    
    return result, transparency_log
