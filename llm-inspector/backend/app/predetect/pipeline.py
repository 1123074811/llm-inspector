"""
predetect/pipeline.py — PreDetectionPipeline orchestrator (16 layers)

This file is intentionally thin: it only contains PreDetectionPipeline,
which assembles and runs the layer sequence. All layer implementations and
shared data live in dedicated files:

  signatures.py      — constants & utility functions (all layers)
  layers_l0_l2.py    — Layer0HTTP, Layer1SelfReport, Layer2Identity
  layers_l3_l5.py    — Layer3Knowledge, Layer4Bias, Layer5Tokenizer
  layers_l6_l7.py    — Layer6ActiveExtraction, Layer6MultiTurnProbes, Layer7Logprobs
  semantic_fingerprint.py  — Layer8SemanticFingerprint
  extraction_v2.py         — Layer7AdvancedExtraction (Layer 9 slot)
  differential_testing.py  — Layer8DifferentialTesting (Layer 10 slot)
  tool_capability.py       — Layer9ToolCapability (Layer 11 slot)
  adversarial_analysis.py  — Layer11AdversarialAnalysis (Layer 13 slot)
  multilingual_attack.py   — Layer14MultilingualAttack (Layer 14 slot)
  ascii_art_attack.py      — Layer15ASCIIArt (Layer 15 slot)  [v13 Phase 3]
  indirect_injection.py    — Layer16IndirectInject (Layer 16 slot) [v13 Phase 3]
"""
from __future__ import annotations

import json
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from app.core.schemas import LayerResult, PreDetectionResult, LLMRequest, Message
from app.core.logging import get_logger
from app.core.config import settings

from app.predetect.signatures import CONFIDENCE_THRESHOLD

# Layer implementations
from app.predetect.layers_l0_l2 import Layer0HTTP, Layer1SelfReport, Layer2Identity
from app.predetect.layers_l3_l5 import Layer3Knowledge, Layer4Bias, Layer5Tokenizer
from app.predetect.layers_l6_l7 import (
    Layer6ActiveExtraction,
    Layer6MultiTurnProbes,
    Layer7Logprobs,
)

# v7 Phase 3: Additional detection layers
from app.predetect.semantic_fingerprint import Layer8SemanticFingerprint
from app.predetect.extraction_v2 import Layer7AdvancedExtraction       # Layer 9 slot
from app.predetect.differential_testing import Layer8DifferentialTesting  # Layer 10 slot
from app.predetect.tool_capability import Layer9ToolCapability           # Layer 11 slot
from app.predetect.adversarial_analysis import Layer11AdversarialAnalysis # Layer 13 slot

# v11 Phase 3: Multilingual translation attack
from app.predetect.multilingual_attack import Layer14MultilingualAttack  # Layer 14 slot

# v13 Phase 3: New adversarial layers
from app.predetect.ascii_art_attack import Layer15ASCIIArt            # Layer 15 slot
from app.predetect.indirect_injection import Layer16IndirectInject     # Layer 16 slot

# v14 Phase 3: Identity Exposure
from app.predetect.identity_exposure import Layer17IdentityExposure   # Layer 17 slot

# v15 Phase 7: Enhanced wrapper detection layers (Deep mode only)
from app.predetect.layer_l20_self_paradox import Layer20SelfParadox
from app.predetect.layer_l21_multistep_drift import Layer21MultiStepDrift
from app.predetect.layer_l22_prompt_reconstruct import Layer22PromptReconstruct
from app.predetect.layer_l23_adversarial_tools import Layer23AdversarialTools

logger = get_logger(__name__)


# ── JSONL trace sink ──────────────────────────────────────────────────────────

def _write_predetect_trace(run_id: str | None, layer_record: dict) -> None:
    """
    Append a layer result as a JSONL line to {DATA_DIR}/traces/{run_id}/predetect.jsonl.
    Non-fatal: any I/O error is caught and logged as warning only.
    Also emits EventKind.PREDETECT_LAYER_TRACE via the event bus.
    """
    if not run_id:
        return

    try:
        trace_dir = pathlib.Path(settings.DATA_DIR) / "traces" / run_id
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_file = trace_dir / "predetect.jsonl"

        record = {
            "layer": layer_record.get("layer"),
            "name": layer_record.get("name"),
            "started_at": layer_record.get("started_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
            "duration_ms": layer_record.get("duration_ms"),
            "tokens": layer_record.get("tokens", 0),
            "confidence": layer_record.get("confidence", 0.0),
            "skipped": layer_record.get("skipped", False),
            "evidence": layer_record.get("evidence", []),
        }
        with trace_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception as exc:
        logger.warning("Could not write predetect trace", run_id=run_id, error=str(exc))

    # Emit SSE event (best-effort)
    try:
        from app.core.events import emit, EventKind
        emit(run_id, EventKind.PREDETECT_LAYER_TRACE, **{
            k: v for k, v in layer_record.items()
            if not isinstance(v, (dict, list)) or k in ("evidence",)
        })
    except Exception:
        pass


# ── Main Pipeline ─────────────────────────────────────────────────────────────

class PreDetectionPipeline:

    def run(self, adapter, model_name: str, extraction_mode: bool = False, run_id: str = "") -> PreDetectionResult:
        layer_results: list[LayerResult] = []
        total_tokens = 0
        layer3_extra: dict = {}

        # Helper to report progress
        def _report_progress(current_layer: str, current_probe: str = None, probe_detail: dict = None, evidence: list = None, tokens_used: int = 0):
            if run_id:
                from app.repository import repo
                repo.update_predetect_progress(
                    run_id, current_layer,
                    layer_results=[r.to_dict() for r in layer_results],
                    current_probe=current_probe,
                    probe_detail=probe_detail,
                    evidence=evidence,
                    tokens_used=tokens_used,
                )

        # ── v16 Phase 1: Hard preflight before any LLM calls ──
        from app.preflight.connection_check import run_preflight
        from app.core.config import settings as _settings
        pf = run_preflight(
            base_url=getattr(adapter, "base_url", ""),
            api_key=getattr(adapter, "_api_key", ""),
            model_name=model_name,
            timeout=_settings.PREFLIGHT_TIMEOUT_S,
            verify_ssl=_settings.PREFLIGHT_VERIFY_SSL,
        )
        # Write preflight trace
        if run_id:
            try:
                import json as _json
                _trace_dir = pathlib.Path(_settings.DATA_DIR) / "traces" / run_id
                _trace_dir.mkdir(parents=True, exist_ok=True)
                with open(_trace_dir / "preflight.jsonl", "a", encoding="utf-8") as _fh:
                    for _step in pf.steps:
                        _fh.write(_json.dumps(_step.to_dict(), ensure_ascii=False) + "\n")
            except Exception:
                pass
        if not pf.passed:
            logger.warning(
                "Preflight check failed, aborting PreDetect",
                first_error=pf.first_error.to_dict() if pf.first_error else None,
            )
            return PreDetectionResult(
                success=False,
                identified_as=None,
                confidence=0.0,
                layer_stopped="preflight",
                layer_results=layer_results,
                total_tokens_used=0,
                should_proceed_to_testing=False,
                routing_info={"preflight_report": pf.to_dict()},
            )

        # ── v16 Phase 1.5: Official Endpoint Fast-Path ──
        official_result = None
        if _settings.OFFICIAL_ENDPOINT_ENABLED:
            from app.authenticity.official_endpoint import check_official_endpoint
            official_result = check_official_endpoint(
                base_url=getattr(adapter, "base_url", ""),
                api_key=getattr(adapter, "_api_key", ""),
                model_name=model_name,
            )
            if official_result.verified:
                logger.info(
                    "OfficialEndpoint verified, adding to routing_info",
                    provider=official_result.provider,
                    confidence=official_result.confidence,
                )

        # Layer 0 — HTTP
        _report_progress("Layer0/HTTP")
        logger.info("PreDetect Layer0: HTTP fingerprint", model=model_name)
        r0 = Layer0HTTP().run(adapter)
        layer_results.append(r0)
        total_tokens += r0.tokens_used
        self._log_layer_result("Layer0/HTTP", r0)
        if r0.confidence >= CONFIDENCE_THRESHOLD:
            return self._build_result(True, r0, layer_results, total_tokens)

        # Quick connectivity check: send a minimal chat request with the actual
        # model name. If this fails (error_type set), API is likely unreachable
        # or misconfigured — skip expensive identity probes.
        probe = adapter.chat(LLMRequest(
            model=model_name,
            messages=[Message(role="user", content="hi")],
            max_tokens=1,
            temperature=0.0,
            timeout_sec=15,
        ))
        if probe.error_type:
            logger.warning(
                "PreDetect quick probe failed, skipping remaining layers",
                model=model_name,
                error_type=probe.error_type,
                status_code=probe.status_code,
                error=probe.error_message,
            )
            return PreDetectionResult(
                success=False,
                identified_as=None,
                confidence=0.0,
                layer_stopped="probe",
                layer_results=layer_results,
                total_tokens_used=total_tokens,
                should_proceed_to_testing=True,
            )

        # Extract routing info from quick probe for detailed reporting
        routing_info: dict = {}
        # v16 Phase 1.5: Include official endpoint verification result
        if official_result is not None:
            routing_info["official_endpoint"] = official_result.to_dict()
        if probe.ok and probe.raw_json:
            returned_model = probe.raw_json.get("model", "")
            if returned_model:
                routing_info["returned_model"] = returned_model
                routing_info["claimed_model"] = model_name
                routing_info["is_routed"] = returned_model.lower() != model_name.lower()
            usage = probe.raw_json.get("usage", {})
            if usage:
                routing_info["probe_usage"] = usage
            if probe.latency_ms:
                routing_info["probe_latency_ms"] = probe.latency_ms

        # Layer 1 — Self-report (reuse quick probe response)
        _report_progress("Layer1/SelfReport")
        logger.info("PreDetect Layer1: Self-report", model=model_name)
        r1 = Layer1SelfReport().run(adapter, model_name, prefetched_resp=probe)
        layer_results.append(r1)
        total_tokens += r1.tokens_used
        self._log_layer_result("Layer1/SelfReport", r1)
        if r1.confidence >= CONFIDENCE_THRESHOLD:
            return self._build_result(True, r1, layer_results, total_tokens, routing_info=routing_info)

        # Layer 2 — Identity probes
        logger.info("PreDetect Layer2: Identity probes", model=model_name)

        def _l2_progress(probe_name: str, detail: dict, evidence: list, tokens: int):
            _report_progress("Layer2/Identity", probe_name, detail, evidence, tokens)

        r2, layer3_extra = Layer2Identity().run(adapter, model_name, progress_callback=_l2_progress)
        layer_results.append(r2)
        total_tokens += r2.tokens_used
        self._log_layer_result("Layer2/Identity", r2)
        if r2.confidence >= CONFIDENCE_THRESHOLD:
            return self._build_result(True, r2, layer_results, total_tokens)

        # Early skip decision: if Layer 0-2 already have high confidence, jump to Layer 6
        current_conf = self._merge_confidences(layer_results)
        if current_conf >= 0.70:
            logger.info("PreDetect: High confidence from Layer 0-2, skipping to Layer 6", confidence=current_conf)
            r6 = None
            try:
                r6 = Layer6ActiveExtraction().run(adapter, model_name, run_id=run_id)
                layer_results.append(r6)
            except Exception as e:
                logger.warning("Layer6 failed in fast-path, continuing with accumulated confidence", error=str(e))
            if r6:
                total_tokens += r6.tokens_used
                self._log_layer_result("Layer6/Extraction(early-skip)", r6)
                if r6.confidence >= CONFIDENCE_THRESHOLD:
                    return self._build_result(True, r6, layer_results, total_tokens)
            best = max(layer_results, key=lambda r: r.confidence)
            merged_conf = self._merge_confidences(layer_results)
            identified = best.identified_as if merged_conf >= 0.50 else None
            return PreDetectionResult(
                success=merged_conf >= 0.60,
                identified_as=identified,
                confidence=merged_conf,
                layer_stopped="early_skip",
                layer_results=layer_results,
                total_tokens_used=total_tokens,
                should_proceed_to_testing=merged_conf < CONFIDENCE_THRESHOLD,
                routing_info=routing_info,
            )

        # ── Parallel Group 1: L3 + L4 + L5 ──────────────────────────────────────
        # These three layers are fully independent (each only needs adapter + model_name;
        # L3 additionally uses layer3_extra from L2 but that's already available).
        # Running them in parallel reduces this phase from ~30 s to ~10 s.
        logger.info("PreDetect: Starting parallel group 1 (L3+L4+L5)", model=model_name)
        _report_progress("parallel_L3_L4_L5(0/3)")

        _parallel_tasks_g1: list[tuple[str, Callable]] = [
            ("Layer3/Knowledge",  lambda: Layer3Knowledge().run(adapter, model_name, layer3_extra)),
            ("Layer4/Bias",       lambda: Layer4Bias().run(adapter, model_name)),
            ("Layer5/Tokenizer",  lambda: Layer5Tokenizer().run(adapter, model_name)),
        ]
        _g1_done = 0
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="predetect-g1") as _ex:
            _futures_g1 = {_ex.submit(fn): name for name, fn in _parallel_tasks_g1}
            for _fut in as_completed(_futures_g1):
                _layer_name = _futures_g1[_fut]
                _g1_done += 1
                try:
                    _r = _fut.result()
                    layer_results.append(_r)
                    total_tokens += _r.tokens_used
                    self._log_layer_result(_layer_name, _r)
                except Exception as _e:
                    logger.warning(f"PreDetect parallel layer {_layer_name} failed", error=str(_e))
                _report_progress(
                    f"parallel_L3_L4_L5({_g1_done}/3)",
                    evidence=[ev for r in layer_results for ev in r.evidence],
                    tokens_used=total_tokens,
                )
        logger.info("PreDetect: Parallel group 1 complete", layers_done=_g1_done, model=model_name)

        # After G1: check early-stop threshold
        _conf_after_g1 = self._merge_confidences(layer_results)
        if _conf_after_g1 >= CONFIDENCE_THRESHOLD:
            _best = max(layer_results, key=lambda r: r.confidence)
            return self._build_result(True, _best, layer_results, total_tokens, routing_info=routing_info)

        # Legacy early-skip path (high conf after L0-L3): jump to Layer 6 confirmation
        if _conf_after_g1 >= 0.75:
            logger.info("PreDetect: High conf after G1, jumping to L6 confirmation", confidence=_conf_after_g1)
            try:
                _r6_fast = Layer6ActiveExtraction().run(adapter, model_name, run_id=run_id)
                layer_results.append(_r6_fast)
                total_tokens += _r6_fast.tokens_used
                self._log_layer_result("Layer6/Extraction(fast-path)", _r6_fast)
                if _r6_fast.confidence >= CONFIDENCE_THRESHOLD:
                    return self._build_result(True, _r6_fast, layer_results, total_tokens, routing_info=routing_info)
            except Exception as _e:
                logger.warning("Layer6 fast-path failed", error=str(_e))
            _best_fp = max(layer_results, key=lambda r: r.confidence)
            _mc_fp = self._merge_confidences(layer_results)
            return PreDetectionResult(
                success=_mc_fp >= 0.60,
                identified_as=_best_fp.identified_as if _mc_fp >= 0.50 else None,
                confidence=_mc_fp,
                layer_stopped="early_skip",
                layer_results=layer_results,
                total_tokens_used=total_tokens,
                should_proceed_to_testing=_mc_fp < CONFIDENCE_THRESHOLD,
                routing_info=routing_info,
            )

        # Layer 6 — Active Extraction (only in extraction mode or when confidence is low)
        if extraction_mode or (max(r.confidence for r in layer_results) < 0.60):
            _report_progress("Layer6/Extraction")
            logger.info("PreDetect Layer6: Active extraction", model=model_name)
            r6 = Layer6ActiveExtraction().run(adapter, model_name, run_id=run_id)
            layer_results.append(r6)
            total_tokens += r6.tokens_used
            self._log_layer_result("Layer6/Extraction", r6)
            if r6.confidence >= CONFIDENCE_THRESHOLD:
                return self._build_result(True, r6, layer_results, total_tokens)

            # ── Parallel Group 2: L6b + L7 – L16 (deep/extraction mode only) ─────
            if extraction_mode:
                logger.info("PreDetect: Starting parallel group 2 (L6b+L7-L16)", model=model_name)
                _report_progress("parallel_L6b_L16(0/11)")

                _parallel_tasks_g2: list[tuple[str, Callable]] = [
                    ("Layer6b/MultiTurn",         lambda: Layer6MultiTurnProbes().run(adapter, model_name)),
                    ("Layer7/Logprobs",            lambda: Layer7Logprobs().run(adapter, model_name)),
                    ("Layer8/SemanticFP",          lambda: Layer8SemanticFingerprint().run(adapter, model_name)),
                    ("Layer9/AdvExtractionV2",     lambda: Layer7AdvancedExtraction().run(adapter, model_name, run_id=run_id)),
                    ("Layer10/Differential",       lambda: Layer8DifferentialTesting().run(adapter, model_name)),
                    ("Layer11/ToolCapability",     lambda: Layer9ToolCapability().run(adapter, model_name)),
                    ("Layer12/MultiTurn",          lambda: Layer6MultiTurnProbes().run(adapter, model_name)),
                    ("Layer13/Adversarial",        lambda: Layer11AdversarialAnalysis().run(adapter, model_name)),
                    ("Layer14/Multilingual",       lambda: Layer14MultilingualAttack().run(adapter, model_name)),
                    ("Layer15/ASCIIArt",           lambda: Layer15ASCIIArt().run(adapter, model_name)),
                    ("Layer16/IndirectInject",     lambda: Layer16IndirectInject().run(adapter, model_name)),
                ]
                _g2_total = len(_parallel_tasks_g2)
                _g2_done = 0
                # Use max_workers=5 to balance throughput vs. server load
                with ThreadPoolExecutor(max_workers=5, thread_name_prefix="predetect-g2") as _ex2:
                    _futures_g2 = {_ex2.submit(fn): name for name, fn in _parallel_tasks_g2}
                    for _fut2 in as_completed(_futures_g2):
                        _lname2 = _futures_g2[_fut2]
                        _g2_done += 1
                        try:
                            _r2 = _fut2.result()
                            layer_results.append(_r2)
                            total_tokens += _r2.tokens_used
                            self._log_layer_result(_lname2, _r2)
                        except Exception as _e2:
                            logger.warning(f"PreDetect parallel layer {_lname2} failed", error=str(_e2))
                        _report_progress(
                            f"parallel_L6b_L16({_g2_done}/{_g2_total})",
                            evidence=[ev for r in layer_results for ev in r.evidence],
                            tokens_used=total_tokens,
                        )
                logger.info("PreDetect: Parallel group 2 complete", layers_done=_g2_done, model=model_name)

                # After G2: check if any layer hit threshold
                _conf_after_g2 = self._merge_confidences(layer_results)
                if _conf_after_g2 >= CONFIDENCE_THRESHOLD:
                    _best_g2 = max(layer_results, key=lambda r: r.confidence)
                    return self._build_result(True, _best_g2, layer_results, total_tokens)

                # v14 Phase 3: Layer 17 — Identity Exposure (zero API tokens, needs prior results)
                _report_progress("Layer17/IdentityExposure")
                logger.info("PreDetect Layer17: Identity exposure analysis", model=model_name)
                r17 = Layer17IdentityExposure().run(
                    adapter, model_name, layer_results_so_far=layer_results
                )
                layer_results.append(r17)
                total_tokens += r17.tokens_used
                self._log_layer_result("Layer17/IdentityExposure", r17)
                _write_predetect_trace(run_id, r17.to_dict() if hasattr(r17, "to_dict") else r17)
                if r17.confidence >= CONFIDENCE_THRESHOLD:
                    return self._build_result(True, r17, layer_results, total_tokens)

                # v14 Phase 5: Layer 18 — Timing Side-Channel (zero tokens, deep/extraction mode)
                _report_progress("Layer18/TimingSideChannel")
                logger.info("PreDetect Layer18: Timing side-channel analysis", model=model_name)
                from app.predetect.layers_l18_l19 import Layer18TimingSideChannel
                r18_dict = Layer18TimingSideChannel().run(adapter, model_name, layer_results)
                r18 = LayerResult(
                    layer="Layer18/TimingSideChannel",
                    confidence=r18_dict.get("confidence", 0.0),
                    identified_as=r18_dict.get("closest_family") if not r18_dict.get("skipped") else None,
                    evidence=r18_dict.get("evidence", []),
                    tokens_used=r18_dict.get("tokens", 0),
                )
                layer_results.append(r18)
                _write_predetect_trace(run_id, r18_dict)

                # v14 Phase 5: Layer 19 — Token Distribution Side-Channel (zero tokens)
                _report_progress("Layer19/TokenDistribution")
                logger.info("PreDetect Layer19: Token distribution side-channel analysis", model=model_name)
                from app.predetect.layers_l18_l19 import Layer19TokenDistribution
                r19_dict = Layer19TokenDistribution().run(adapter, model_name, layer_results)
                r19 = LayerResult(
                    layer="Layer19/TokenDistribution",
                    confidence=r19_dict.get("confidence", 0.0),
                    identified_as=r19_dict.get("closest_family") if not r19_dict.get("skipped") else None,
                    evidence=r19_dict.get("evidence", []),
                    tokens_used=r19_dict.get("tokens", 0),
                )
                layer_results.append(r19)
                _write_predetect_trace(run_id, r19_dict)

                # v15 Phase 7: Enhanced wrapper detection layers (L20-L23, Deep mode only)
                # L20 — Context Self-Report Paradox
                _report_progress("Layer20/SelfParadox")
                logger.info("PreDetect Layer20: Self-report paradox", model=model_name)
                try:
                    r20 = Layer20SelfParadox().run(adapter, model_name)
                    layer_results.append(r20)
                    total_tokens += r20.tokens_used
                    self._log_layer_result("Layer20/SelfParadox", r20)
                except Exception as e:
                    logger.warning("Layer20 failed", error=str(e))

                # L21 — Multi-Step Identity Drift
                _report_progress("Layer21/MultiStepDrift")
                logger.info("PreDetect Layer21: Multi-step identity drift", model=model_name)
                try:
                    r21 = Layer21MultiStepDrift().run(adapter, model_name)
                    layer_results.append(r21)
                    total_tokens += r21.tokens_used
                    self._log_layer_result("Layer21/MultiStepDrift", r21)
                except Exception as e:
                    logger.warning("Layer21 failed", error=str(e))

                # L22 — System Prompt Reconstruction (passive, zero tokens)
                _report_progress("Layer22/PromptReconstruct")
                logger.info("PreDetect Layer22: Prompt reconstruction", model=model_name)
                try:
                    r22 = Layer22PromptReconstruct().run(adapter, model_name, run_id=run_id)
                    layer_results.append(r22)
                    self._log_layer_result("Layer22/PromptReconstruct", r22)
                except Exception as e:
                    logger.warning("Layer22 failed", error=str(e))

                # L23 — Adversarial Tool Use Probe
                _report_progress("Layer23/AdversarialTools")
                logger.info("PreDetect Layer23: Adversarial tool use", model=model_name)
                try:
                    r23 = Layer23AdversarialTools().run(adapter, model_name)
                    layer_results.append(r23)
                    total_tokens += r23.tokens_used
                    self._log_layer_result("Layer23/AdversarialTools", r23)
                except Exception as e:
                    logger.warning("Layer23 failed", error=str(e))

        # Merge all layers
        best = max(layer_results, key=lambda r: r.confidence)
        merged_conf = self._merge_confidences(layer_results)
        identified = best.identified_as if merged_conf >= 0.50 else None
        logger.info(
            "PreDetect pipeline complete",
            model=model_name,
            layers_run=len(layer_results),
            merged_confidence=merged_conf,
            best_layer=best.layer,
            identified_as=identified,
            total_tokens=total_tokens,
        )

        return PreDetectionResult(
            success=merged_conf >= 0.60,
            identified_as=identified,
            confidence=merged_conf,
            layer_stopped=None,
            layer_results=layer_results,
            total_tokens_used=total_tokens,
            should_proceed_to_testing=merged_conf < CONFIDENCE_THRESHOLD,
            routing_info=routing_info,
        )

    @staticmethod
    def _log_layer_result(layer_label: str, result: LayerResult) -> None:
        """Log per-layer result with evidence details."""
        logger.info(
            f"PreDetect {layer_label} complete",
            identified_as=result.identified_as,
            confidence=result.confidence,
            tokens_used=result.tokens_used,
            evidence_count=len(result.evidence),
        )
        for i, ev in enumerate(result.evidence):
            logger.info(f"  [{layer_label}] evidence[{i}]: {ev}")

    @staticmethod
    def _build_result(
        success: bool, winning_layer: LayerResult,
        all_layers: list[LayerResult], tokens: int,
        routing_info: dict | None = None,
    ) -> PreDetectionResult:
        return PreDetectionResult(
            success=success,
            identified_as=winning_layer.identified_as,
            confidence=winning_layer.confidence,
            layer_stopped=winning_layer.layer,
            layer_results=all_layers,
            total_tokens_used=tokens,
            should_proceed_to_testing=not success,
            routing_info=routing_info or {},
        )

    @staticmethod
    def _merge_confidences(results: list[LayerResult]) -> float:
        """
        v6: Bayesian-style confidence merge.
        Multiple agreeing layers → confidence increases.
        Conflicting layers → confidence decreases (uncertainty).
        """
        if not results:
            return 0.0

        # Aggregate evidence by candidate model
        candidate_evidence: dict[str, list[float]] = {}
        for r in results:
            if r.identified_as and r.confidence > 0:
                candidate_evidence.setdefault(r.identified_as, []).append(r.confidence)

        if not candidate_evidence:
            return max((r.confidence for r in results), default=0.0)

        # Combined confidence for each candidate using 1 - ∏(1 - conf_i)
        # This gives higher confidence when multiple sources agree
        candidate_scores: dict[str, float] = {}
        for model, confs in candidate_evidence.items():
            combined = 1.0
            for c in confs:
                combined *= (1.0 - c)
            candidate_scores[model] = 1.0 - combined

        # Select best candidate
        best_model = max(candidate_scores, key=candidate_scores.get)
        best_score = candidate_scores[best_model]

        # If second candidate also has high confidence, reduce final confidence (uncertainty)
        sorted_scores = sorted(candidate_scores.values(), reverse=True)
        if len(sorted_scores) >= 2 and sorted_scores[1] > 0.3:
            # Two candidates with high confidence → result is unreliable
            best_score *= (1.0 - sorted_scores[1] * 0.5)

        return round(min(best_score, 0.99), 3)
