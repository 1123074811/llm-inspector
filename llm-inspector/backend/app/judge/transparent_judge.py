"""
Transparent Judge Wrapper — v8.0 Judgment Transparency Layer

Wraps existing judge methods to provide:
- Detailed step-by-step logging
- Threshold source tracking
- Full input/output capture
- Performance metrics

Reference: V8_UPGRADE_PLAN.md Section 1.4.2
"""
from typing import Dict, Any, Optional, Tuple, List
import time
import uuid
import json

from app.judge.plugin_interface import JudgePlugin, JudgeResult, JudgeMetadata, JudgeTier
from app.core.structured_logger import get_structured_logger, LogEventType, LogLevel
from app.adapters.openai_compat import OpenAICompatibleAdapter
from app.core.config import settings

class ChainOfVerificationJudge(JudgePlugin):
    """
    v10: Chain-of-Verification (CoVe) logic for long-chain reasoning.
    
    Instead of simply returning True/False based on the final answer,
    this judge requires the model to:
    1. Generate verification questions based on its own answer.
    2. Answer those verification questions independently.
    3. Cross-check if the original answer is consistent with the verified facts.
    
    Reference: Dhuliawala et al. (2023) "Chain-of-Verification Reduces Hallucination in Large Language Models"
    """
    
    @property
    def metadata(self) -> JudgeMetadata:
        return JudgeMetadata(
            name="chain_of_verification",
            version="1.0",
            tier=JudgeTier.LLM,
            supported_languages=["*"],
            description="Chain-of-Verification (CoVe) logic for long-chain reasoning",
            deterministic=False,
            required_params=["expected_answer"]
        )

    def judge(self, response: str, params: Dict[str, Any]) -> JudgeResult:
        """
        Execute the CoVe pipeline.
        Returns JudgeResult with detailed steps.
        """
        start_time = time.time()
        trace_id = str(uuid.uuid4())[:8]
        steps_log = []
        tokens_used = 0
        logger = get_structured_logger()
        
        # Determine model and API info
        base_url = getattr(settings, "JUDGE_API_URL", "https://api.openai.com/v1")
        api_key = getattr(settings, "JUDGE_API_KEY", "")
        model_name = getattr(settings, "JUDGE_MODEL", "gpt-4o-mini")
        
        adapter = OpenAICompatibleAdapter(base_url=base_url, api_key=api_key)
        
        original_prompt = params.get("_original_prompt", "")
        expected_answer = params.get("expected_answer", "")
        
        # Step 1: Generate Verification Questions
        q_prompt = (
            f"Original Question: {original_prompt}\n"
            f"Original Answer: {response}\n\n"
            f"Based on the above answer, generate 3 factual verification questions "
            f"that can help determine if the core logic or final conclusion is correct. "
            f"Return ONLY a JSON list of strings."
        )
        
        try:
            from app.core.schemas import LLMRequest, Message
            
            req1 = LLMRequest(
                model=model_name,
                messages=[Message("user", q_prompt)],
                temperature=0.1,
                max_tokens=300
            )
            resp1 = adapter.chat(req1)
            tokens_used += (resp1.usage_total_tokens or 0)
            
            try:
                # Extract JSON list
                content = resp1.content
                start = content.find('[')
                end = content.rfind(']') + 1
                questions = json.loads(content[start:end]) if start >= 0 else []
            except Exception:
                questions = ["Is the mathematical calculation correct?", "Does the conclusion follow the premises?"]
                
            steps_log.append({"step": "generate_questions", "questions": questions})
            
            # Step 2: Answer Verification Questions Independently
            answered_facts = []
            for q in questions[:3]:  # Limit to 3 to save tokens
                req2 = LLMRequest(
                    model=model_name,
                    messages=[Message("user", f"Answer this factual question briefly: {q}")],
                    temperature=0.0,
                    max_tokens=100
                )
                resp2 = adapter.chat(req2)
                tokens_used += (resp2.usage_total_tokens or 0)
                answered_facts.append(f"Q: {q}\nA: {resp2.content}")
                
            steps_log.append({"step": "answer_questions", "facts": answered_facts})
            
            # Step 3: Final Cross-Check
            facts_str = "\n".join(answered_facts)
            cross_prompt = (
                f"Original Question: {original_prompt}\n"
                f"Original Answer: {response}\n"
                f"Expected Ground Truth: {expected_answer}\n\n"
                f"Verified Facts:\n{facts_str}\n\n"
                f"Based on the verified facts and the expected ground truth, is the Original Answer correct? "
                f"Reply with exactly 'CORRECT' or 'INCORRECT' followed by a one-sentence reason."
            )
            
            req3 = LLMRequest(
                model=model_name,
                messages=[Message("user", cross_prompt)],
                temperature=0.0,
                max_tokens=100
            )
            resp3 = adapter.chat(req3)
            tokens_used += (resp3.usage_total_tokens or 0)
            
            final_judgment = resp3.content.strip()
            passed = final_judgment.upper().startswith("CORRECT")
            
            steps_log.append({"step": "cross_check", "judgment": final_judgment})
            
            detail = {
                "cove_executed": True,
                "steps": steps_log,
                "trace_id": trace_id
            }
            
            return JudgeResult(
                passed=passed,
                detail=detail,
                confidence=0.8,
                tokens_used=tokens_used,
                latency_ms=int((time.time() - start_time) * 1000),
                method=self.metadata.name,
                version=self.metadata.version
            )
            
        except Exception as e:
            logger.log(LogEventType.JUDGE_ERROR, LogLevel.ERROR, "cove_judge", str(e), {"trace_id": trace_id})
            return JudgeResult(
                passed=None,
                detail={"error": f"CoVe failed: {str(e)}", "trace_id": trace_id},
                latency_ms=int((time.time() - start_time) * 1000),
                method=self.metadata.name,
                version=self.metadata.version
            )


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


class JudgeChainRunner:
    """
    Multi-level judge downgrade chain (v14 Phase 4).

    Chain: External LLM → Local NLI → semantic_v2 rules → hallucination_v2 rules

    Each level is attempted in order; if a level raises an exception or returns
    None (no signal), the next level is tried.  All attempts are logged to
    judge_chain so callers can audit which level produced the final verdict.

    Reference:
        Inspired by the "cascade evaluation" approach in:
        Zheng et al. (2023) "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"
        arXiv: https://arxiv.org/abs/2306.05685
    """

    def run(self, response: str, params: dict) -> tuple[bool | None, dict]:
        """
        Execute the downgrade chain and return the first definitive verdict.

        Returns:
            (passed, detail_dict) where detail_dict includes the full chain log.
        """
        chain_log: List[Dict[str, Any]] = []
        final_passed: bool | None = None
        final_level: str = "none"

        # -- Level 1: External LLM judge (semantic_v2 with JUDGE_API_URL) -----
        try:
            from app.judge.semantic_v2 import semantic_judge_v2
            from app.core.config import settings as _settings

            has_external = bool(getattr(_settings, "JUDGE_API_URL", None))
            lvl1_params = dict(params)
            if has_external:
                passed1, detail1 = semantic_judge_v2(response, lvl1_params)
            else:
                passed1, detail1 = None, {"skipped": True, "reason": "JUDGE_API_URL not configured"}

            chain_log.append({"level": 1, "name": "external_llm", "passed": passed1, "detail": detail1})

            if passed1 is not None:
                final_passed = passed1
                final_level = "external_llm"
        except Exception as e:
            chain_log.append({"level": 1, "name": "external_llm", "error": str(e)})

        if final_level != "none":
            return final_passed, {"judge_chain": chain_log, "final_level": final_level, "passed": final_passed}

        # -- Level 2: Local NLI (semantic_entailment_judge) --------------------
        try:
            from app.judge.semantic_entailment import semantic_entailment_judge
            passed2, detail2 = semantic_entailment_judge(response, params)
            chain_log.append({"level": 2, "name": "local_nli", "passed": passed2, "detail": detail2})

            if passed2 is not None and detail2.get("backend") not in ("error",):
                final_passed = passed2
                final_level = "local_nli"
        except Exception as e:
            chain_log.append({"level": 2, "name": "local_nli", "error": str(e)})

        if final_level != "none":
            return final_passed, {"judge_chain": chain_log, "final_level": final_level, "passed": final_passed}

        # -- Level 3: semantic_v2 rule mode (local, no external API) ----------
        try:
            from app.judge.semantic_v2 import semantic_judge_v2
            rule_params = dict(params)
            # Disable external LLM for this level
            rule_params["disable_llm_judge"] = True
            passed3, detail3 = semantic_judge_v2(response, rule_params)
            chain_log.append({"level": 3, "name": "semantic_v2_rules", "passed": passed3, "detail": detail3})

            if passed3 is not None:
                final_passed = passed3
                final_level = "semantic_v2_rules"
        except Exception as e:
            chain_log.append({"level": 3, "name": "semantic_v2_rules", "error": str(e)})

        if final_level != "none":
            return final_passed, {"judge_chain": chain_log, "final_level": final_level, "passed": final_passed}

        # -- Level 4: hallucination_v2 rules (last resort) --------------------
        try:
            from app.judge.hallucination_v2 import hallucination_detect_v2
            passed4, detail4 = hallucination_detect_v2(response, params)
            chain_log.append({"level": 4, "name": "hallucination_v2_rules", "passed": passed4, "detail": detail4})
            final_passed = passed4
            final_level = "hallucination_v2_rules"
        except Exception as e:
            chain_log.append({"level": 4, "name": "hallucination_v2_rules", "error": str(e)})

        return final_passed, {"judge_chain": chain_log, "final_level": final_level, "passed": final_passed}


def run_judge_chain(response: str, params: dict) -> tuple[bool | None, dict]:
    """
    Module-level convenience wrapper for JudgeChainRunner.

    Args:
        response: Model response text.
        params:   Judge params dict (same as other judge functions).

    Returns:
        (passed, detail_dict)
    """
    return JudgeChainRunner().run(response, params)


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
