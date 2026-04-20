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

        # Layer 3 — Knowledge cutoff
        _report_progress("Layer3/Knowledge")
        logger.info("PreDetect Layer3: Knowledge probes", model=model_name)
        r3 = Layer3Knowledge().run(adapter, model_name, layer3_extra)
        layer_results.append(r3)
        total_tokens += r3.tokens_used
        self._log_layer_result("Layer3/Knowledge", r3)
        if r3.confidence >= CONFIDENCE_THRESHOLD:
            return self._build_result(True, r3, layer_results, total_tokens)

        # If Layer 0-3 merged confidence >= 0.75, skip Layer 4-5 and go to Layer 6
        current_conf = self._merge_confidences(layer_results)
        if current_conf >= 0.75:
            logger.info("PreDetect: Good confidence from Layer 0-3, skipping to Layer 6", confidence=current_conf)
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

        # Layer 4 — Bias / format
        _report_progress("Layer4/Bias")
        logger.info("PreDetect Layer4: Bias fingerprint", model=model_name)
        r4 = Layer4Bias().run(adapter, model_name)
        layer_results.append(r4)
        total_tokens += r4.tokens_used
        self._log_layer_result("Layer4/Bias", r4)
        if r4.confidence >= CONFIDENCE_THRESHOLD:
            return self._build_result(True, r4, layer_results, total_tokens)

        # Layer 5 — Tokenizer fingerprint (only in full mode or when earlier layers inconclusive)
        _report_progress("Layer5/Tokenizer")
        logger.info("PreDetect Layer5: Tokenizer fingerprint", model=model_name)
        r5 = Layer5Tokenizer().run(adapter, model_name)
        layer_results.append(r5)
        total_tokens += r5.tokens_used
        self._log_layer_result("Layer5/Tokenizer", r5)

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

            # Layer 6b — Multi-turn context overload
            if extraction_mode:
                _report_progress("Layer6b/MultiTurn")
                logger.info("PreDetect Layer6b: Multi-turn extraction", model=model_name)
                r6b = Layer6MultiTurnProbes().run(adapter, model_name)
                layer_results.append(r6b)
                total_tokens += r6b.tokens_used
                self._log_layer_result("Layer6b/MultiTurn", r6b)
                if r6b.confidence >= CONFIDENCE_THRESHOLD:
                    return self._build_result(True, r6b, layer_results, total_tokens)

                # Layer 7 — Logprobs tokenizer fingerprint (deep/extraction mode only)
                _report_progress("Layer7/Logprobs")
                logger.info("PreDetect Layer7: Logprobs fingerprint", model=model_name)
                r7 = Layer7Logprobs().run(adapter, model_name)
                layer_results.append(r7)
                total_tokens += r7.tokens_used
                self._log_layer_result("Layer7/Logprobs", r7)
                if r7.confidence >= CONFIDENCE_THRESHOLD:
                    return self._build_result(True, r7, layer_results, total_tokens)

                # v7 Phase 3: Layer 8 — Semantic Fingerprint
                _report_progress("Layer8/SemanticFP")
                logger.info("PreDetect Layer8: Semantic fingerprint", model=model_name)
                r8 = Layer8SemanticFingerprint().run(adapter, model_name)
                layer_results.append(r8)
                total_tokens += r8.tokens_used
                self._log_layer_result("Layer8/SemanticFP", r8)
                if r8.confidence >= CONFIDENCE_THRESHOLD:
                    return self._build_result(True, r8, layer_results, total_tokens)

                # v7 Phase 3: Layer 9 — Advanced Extraction v2
                _report_progress("Layer9/AdvExtractionV2")
                logger.info("PreDetect Layer9: Advanced extraction v2", model=model_name)
                r9 = Layer7AdvancedExtraction().run(adapter, model_name, run_id=run_id)
                layer_results.append(r9)
                total_tokens += r9.tokens_used
                self._log_layer_result("Layer9/AdvExtractionV2", r9)
                if r9.confidence >= CONFIDENCE_THRESHOLD:
                    return self._build_result(True, r9, layer_results, total_tokens)

                # v7 Phase 3: Layer 10 — Differential Consistency Test
                _report_progress("Layer10/Differential")
                logger.info("PreDetect Layer10: Differential testing", model=model_name)
                r10 = Layer8DifferentialTesting().run(adapter, model_name)
                layer_results.append(r10)
                total_tokens += r10.tokens_used
                self._log_layer_result("Layer10/Differential", r10)
                if r10.confidence >= CONFIDENCE_THRESHOLD:
                    return self._build_result(True, r10, layer_results, total_tokens)

                # v7 Phase 3: Layer 11 — Tool Use Capability
                _report_progress("Layer11/ToolCapability")
                logger.info("PreDetect Layer11: Tool capability", model=model_name)
                r11 = Layer9ToolCapability().run(adapter, model_name)
                layer_results.append(r11)
                total_tokens += r11.tokens_used
                self._log_layer_result("Layer11/ToolCapability", r11)
                if r11.confidence >= CONFIDENCE_THRESHOLD:
                    return self._build_result(True, r11, layer_results, total_tokens)

                # v7 Phase 3: Layer 12 — Multi-turn Context Overload (second pass)
                _report_progress("Layer12/MultiTurn")
                logger.info("PreDetect Layer12: Multi-turn context overload", model=model_name)
                r12 = Layer6MultiTurnProbes().run(adapter, model_name)
                layer_results.append(r12)
                total_tokens += r12.tokens_used
                self._log_layer_result("Layer12/MultiTurn", r12)
                if r12.confidence >= CONFIDENCE_THRESHOLD:
                    return self._build_result(True, r12, layer_results, total_tokens)

                # v7 Phase 3: Layer 13 — Adversarial Response Analysis
                _report_progress("Layer13/Adversarial")
                logger.info("PreDetect Layer13: Adversarial analysis", model=model_name)
                r13 = Layer11AdversarialAnalysis().run(adapter, model_name)
                layer_results.append(r13)
                total_tokens += r13.tokens_used
                self._log_layer_result("Layer13/Adversarial", r13)
                if r13.confidence >= CONFIDENCE_THRESHOLD:
                    return self._build_result(True, r13, layer_results, total_tokens)

                # v11 Phase 3: Layer 14 — Multilingual Translation Attack
                _report_progress("Layer14/Multilingual")
                logger.info("PreDetect Layer14: Multilingual attack", model=model_name)
                r14 = Layer14MultilingualAttack().run(adapter, model_name)
                layer_results.append(r14)
                total_tokens += r14.tokens_used
                self._log_layer_result("Layer14/Multilingual", r14)
                if r14.confidence >= CONFIDENCE_THRESHOLD:
                    return self._build_result(True, r14, layer_results, total_tokens)

                # v13 Phase 3: Layer 15 — ASCII Art attack (deep/extraction mode)
                _report_progress("Layer15/ASCIIArt")
                logger.info("PreDetect Layer15: ASCII Art attack", model=model_name)
                r15 = Layer15ASCIIArt().run(adapter, model_name)
                layer_results.append(r15)
                total_tokens += r15.tokens_used
                self._log_layer_result("Layer15/ASCIIArt", r15)
                if r15.confidence >= CONFIDENCE_THRESHOLD:
                    return self._build_result(True, r15, layer_results, total_tokens)

                # v13 Phase 3: Layer 16 — Indirect injection (deep/extraction mode)
                _report_progress("Layer16/IndirectInject")
                logger.info("PreDetect Layer16: Indirect injection", model=model_name)
                r16 = Layer16IndirectInject().run(adapter, model_name)
                layer_results.append(r16)
                total_tokens += r16.tokens_used
                self._log_layer_result("Layer16/IndirectInject", r16)
                if r16.confidence >= CONFIDENCE_THRESHOLD:
                    return self._build_result(True, r16, layer_results, total_tokens)

                # v14 Phase 3: Layer 17 — Identity Exposure (zero API tokens)
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
