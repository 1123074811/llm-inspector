"""
Analysis pipeline:
  FeatureExtractor  — raw responses → named features
  ScoreCalculator   — features → 4 scores (0-100)
  SimilarityEngine  — features vs benchmarks → cosine + bootstrap CI
  RiskEngine        — features + similarity → risk level
  ReportBuilder     — everything → final report dict
"""
from __future__ import annotations

import math
import random
from app.core.schemas import (
    CaseResult, PreDetectionResult, Scores, SimilarityResult, RiskAssessment,
    ScoreCard, TrustVerdict, ThetaReport, ThetaDimensionEstimate,
)
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


# ── Feature Extractor ─────────────────────────────────────────────────────────

class FeatureExtractor:
    """
    Extracts a flat dict of named numeric features from all case results.
    """

    def extract(self, case_results: list[CaseResult]) -> dict[str, float]:
        features: dict[str, float] = {}
        dim_stats = self._dimension_stats(case_results)
        tag_stats = self._tag_stats(case_results)
        failure_stats = self._failure_attribution(case_results)

        # --- Protocol features (from protocol category) ---
        proto_cases = [r for r in case_results if r.case.category == "protocol"]
        if proto_cases:
            features["protocol_success_rate"] = self._pass_rate(proto_cases)
            usage_cases = [
                r for r in proto_cases
                if any(s.response.usage_total_tokens for s in r.samples)
            ]
            features["has_usage_fields"] = 1.0 if usage_cases else 0.0
            finish_cases = [
                r for r in proto_cases
                if any(s.response.finish_reason for s in r.samples)
            ]
            features["has_finish_reason"] = 1.0 if finish_cases else 0.0

        # --- Instruction following ---
        instr_cases = [r for r in case_results if r.case.category == "instruction"]
        if instr_cases:
            features["instruction_pass_rate"] = self._pass_rate(instr_cases)
            exact = [r for r in instr_cases if r.case.judge_method == "exact_match"]
            features["exact_match_rate"] = self._pass_rate(exact) if exact else 0.0
            json_c = [r for r in instr_cases if r.case.judge_method == "json_schema"]
            features["json_valid_rate"] = self._pass_rate(json_c) if json_c else 0.0
            line_c = [r for r in instr_cases if r.case.judge_method == "line_count"]
            features["line_count_follow_rate"] = self._pass_rate(line_c) if line_c else 0.0
            regex_c = [r for r in instr_cases if r.case.judge_method == "regex_match"]
            features["format_follow_rate"] = self._pass_rate(regex_c) if regex_c else 0.0

        # --- System prompt obedience ---
        sys_cases = [r for r in case_results if r.case.category == "system"]
        if sys_cases:
            features["system_obedience_rate"] = self._pass_rate(sys_cases)

        # --- Parameter compliance ---
        param_cases = [r for r in case_results if r.case.category == "param"]
        if param_cases:
            features["param_compliance_rate"] = self._pass_rate(
                [r for r in param_cases if r.samples and r.samples[-1].judge_passed is not None]
            )
            temp_cases = [r for r in param_cases if "temperature" in r.case.id]
            if temp_cases:
                for r in temp_cases:
                    for s in r.samples:
                        if "temperature_param_effective" in s.judge_detail:
                            features["temperature_param_effective"] = (
                                1.0 if s.judge_detail["temperature_param_effective"] else 0.0
                            )

        # --- Style features ---
        style_cases = [r for r in case_results if r.case.category == "style"]
        if style_cases:
            markdown_scores = []
            lengths = []
            has_disclaimer_count = 0
            for r in style_cases:
                for s in r.samples:
                    d = s.judge_detail
                    if "markdown_score" in d:
                        markdown_scores.append(d["markdown_score"])
                    if "length" in d:
                        lengths.append(d["length"])
                    if d.get("has_disclaimer"):
                        has_disclaimer_count += 1
            if markdown_scores:
                features["avg_markdown_score"] = sum(markdown_scores) / len(markdown_scores)
            if lengths:
                features["avg_response_length"] = sum(lengths) / len(lengths)
            total_style_samples = sum(len(r.samples) for r in style_cases)
            if total_style_samples > 0:
                features["disclaimer_rate"] = has_disclaimer_count / total_style_samples

        # --- Refusal features ---
        refusal_cases = [r for r in case_results if r.case.category == "refusal"]
        if refusal_cases:
            refusal_count = 0
            alt_count = 0
            total = 0
            for r in refusal_cases:
                for s in r.samples:
                    total += 1
                    if s.judge_detail.get("refusal_detected"):
                        refusal_count += 1
                    if s.judge_detail.get("offers_alternative"):
                        alt_count += 1
            if total > 0:
                features["refusal_rate"] = refusal_count / total
                features["alt_suggestion_rate"] = alt_count / total

        # --- Latency features ---
        all_latencies = [
            s.response.latency_ms
            for r in case_results
            for s in r.samples
            if s.response.latency_ms is not None
        ]
        if all_latencies:
            features["latency_mean_ms"] = sum(all_latencies) / len(all_latencies)
            sorted_lat = sorted(all_latencies)
            idx_p95 = max(0, int(len(sorted_lat) * 0.95) - 1)
            features["latency_p95_ms"] = sorted_lat[idx_p95]

        features.update(dim_stats)
        features.update(tag_stats)
        features.update(failure_stats)

        # --- Adversarial spoof signal ---
        # Aggregate anti-pattern hit rate from adversarial_reasoning dimension cases.
        # Also cross-check paired variants for template-matching behaviour.
        adv_cases = [
            r for r in case_results
            if (r.case.dimension or r.case.category or "").lower() == "adversarial_reasoning"
        ]
        if adv_cases:
            features["adversarial_spoof_signal_rate"] = self._adversarial_spoof_rate(
                adv_cases, case_results,
            )

        return {k: round(v, 4) for k, v in features.items()}

    @staticmethod
    def _pass_rate(results: list[CaseResult]) -> float:
        total = 0
        passed = 0
        for r in results:
            for s in r.samples:
                if s.judge_passed is not None:
                    total += 1
                    if s.judge_passed:
                        passed += 1
        return (passed / total) if total > 0 else 0.0

    @staticmethod
    def _dimension_stats(case_results: list[CaseResult]) -> dict[str, float]:
        by_dim: dict[str, list[CaseResult]] = {}
        for r in case_results:
            dim = (r.case.dimension or r.case.category or "unknown").strip().lower()
            by_dim.setdefault(dim, []).append(r)

        out: dict[str, float] = {}
        for dim, items in by_dim.items():
            out[f"dim_{dim}_pass_rate"] = FeatureExtractor._pass_rate(items)
            out[f"dim_{dim}_coverage"] = len(items) / max(len(case_results), 1)
        return out

    @staticmethod
    def _tag_stats(case_results: list[CaseResult]) -> dict[str, float]:
        tag_to_samples: dict[str, list[bool]] = {}
        for r in case_results:
            tags = r.case.tags or []
            for tag in tags:
                t = str(tag).strip().lower()
                if not t:
                    continue
                tag_to_samples.setdefault(t, [])
                for s in r.samples:
                    if s.judge_passed is not None:
                        tag_to_samples[t].append(bool(s.judge_passed))

        out: dict[str, float] = {}
        for tag, vals in tag_to_samples.items():
            if not vals:
                continue
            out[f"tag_{tag}_pass_rate"] = sum(1 for v in vals if v) / len(vals)
        return out

    @staticmethod
    def _adversarial_spoof_rate(
        adv_cases: list[CaseResult],
        all_cases: list[CaseResult],
    ) -> float:
        """
        Compute spoof signal rate from adversarial reasoning cases.

        Two signal sources:
        1. Anti-pattern hits in adversarial cases (template-matching detected).
        2. Paired-variant cross-check: if a model gives the WRONG answer on a
           variant that flips the expected outcome, it's likely memorising
           rather than reasoning.
        """
        if not adv_cases:
            return 0.0

        # Build lookup for paired cross-checking
        all_by_id = {r.case.id: r for r in all_cases}

        spoof_signals = 0
        total_checks = 0

        for r in adv_cases:
            meta = (r.case.params.get("_meta") or {})
            spoof_cfg = meta.get("spoof_detection") or {}
            paired_id = meta.get("paired_with")

            for s in r.samples:
                total_checks += 1
                d = s.judge_detail or {}

                # Signal 1: anti-pattern hits on the adversarial case itself
                if d.get("anti_pattern_hits"):
                    spoof_signals += 1
                    continue

                # Signal 2: case failed (wrong answer on variant)
                if s.judge_passed is False:
                    # Check if the paired original was passed — if so, the model
                    # "knows" the original answer but can't adapt to the variant,
                    # which is a strong spoof signal.
                    if paired_id and paired_id in all_by_id:
                        paired_result = all_by_id[paired_id]
                        if paired_result.pass_rate >= 0.5:
                            spoof_signals += 1
                            continue

        if total_checks == 0:
            return 0.0
        return spoof_signals / total_checks

    @staticmethod
    def _failure_attribution(case_results: list[CaseResult]) -> dict[str, float]:
        counts = {
            "error_response": 0,
            "format_violation": 0,
            "safety_violation": 0,
            "reasoning_failure": 0,
            "unknown": 0,
        }
        total_fail = 0

        for r in case_results:
            for s in r.samples:
                if s.judge_passed is not False:
                    continue
                total_fail += 1
                if s.response.error_type:
                    counts["error_response"] += 1
                    continue

                jm = (r.case.judge_method or "").lower()
                cat = (r.case.category or "").lower()
                detail = s.judge_detail or {}

                if jm in {"json_schema", "regex_match", "line_count", "exact_match"}:
                    counts["format_violation"] += 1
                elif jm == "refusal_detect" or cat == "refusal":
                    counts["safety_violation"] += 1
                elif cat in {"reasoning", "coding", "consistency"}:
                    counts["reasoning_failure"] += 1
                elif detail.get("schema_errors") or detail.get("found") is False:
                    counts["format_violation"] += 1
                else:
                    counts["unknown"] += 1

        if total_fail == 0:
            return {f"failure_{k}_rate": 0.0 for k in counts}

        return {f"failure_{k}_rate": v / total_fail for k, v in counts.items()}


# ── Score Calculator ──────────────────────────────────────────────────────────

class ScoreCalculator:

    def calculate(self, features: dict[str, float]) -> Scores:
        def f(key: str, default: float = 0.0) -> float:
            return features.get(key, default)

        # Protocol score (0-100)
        protocol = (
            f("protocol_success_rate") * 40
            + f("has_usage_fields") * 20
            + f("has_finish_reason") * 20
            + f("param_compliance_rate", 0.5) * 20
        )

        # Instruction score (0-100)
        instruction = (
            f("instruction_pass_rate") * 30
            + f("exact_match_rate") * 25
            + f("json_valid_rate") * 25
            + f("format_follow_rate") * 20
        )

        # System obedience (0-100)
        system_obedience = f("system_obedience_rate") * 100

        # Param compliance (0-100)
        param = (
            f("param_compliance_rate") * 60
            + f("temperature_param_effective", 0.5) * 40
        )

        return Scores(
            protocol_score=min(100.0, round(protocol, 1)),
            instruction_score=min(100.0, round(instruction, 1)),
            system_obedience_score=min(100.0, round(system_obedience, 1)),
            param_compliance_score=min(100.0, round(param, 1)),
        )


# ── ScoreCard Calculator (v2) ────────────────────────────────────────────────

class ScoreCardCalculator:
    """
    v2 三维评分体系 (updated with adversarial_reasoning split):
      CapabilityScore  = 0.15×reasoning + 0.15×adversarial_reasoning + 0.25×instruction
                         + 0.20×coding + 0.12×safety + 0.13×protocol
      AuthenticityScore = 0.40×similarity + 0.25×predetect + 0.15×consistency + 0.10×temp + 0.10×usage
      PerformanceScore = 0.40×speed + 0.30×stability + 0.30×cost_efficiency
      TotalScore = 0.45×Capability + 0.35×Authenticity + 0.20×Performance
    """

    def calculate(
        self,
        features: dict[str, float],
        case_results: list[CaseResult],
        similarities: list[SimilarityResult],
        predetect: PreDetectionResult | None,
        claimed_model: str | None = None,
    ) -> ScoreCard:
        card = ScoreCard()

        # ── Capability sub-scores ──
        card.reasoning_score = self._reasoning_score(case_results)
        card.adversarial_reasoning_score = self._adversarial_reasoning_score(case_results)
        card.instruction_score = self._instruction_score(features)
        card.coding_score = self._coding_score(case_results)
        card.safety_score = self._safety_score(features)
        card.protocol_score = self._protocol_score(features)

        card.capability_score = min(100.0, round(
            0.15 * card.reasoning_score
            + 0.15 * card.adversarial_reasoning_score
            + 0.25 * card.instruction_score
            + 0.20 * card.coding_score
            + 0.12 * card.safety_score
            + 0.13 * card.protocol_score,
            1,
        ))

        # ── Authenticity sub-scores ──
        card.similarity_to_claimed = self._similarity_to_claimed(
            similarities, claimed_model
        )
        card.predetect_confidence = (
            predetect.confidence * 100 if predetect and predetect.success else 0.0
        )
        card.consistency_score = self._consistency_score(case_results)
        card.temperature_effectiveness = (
            features.get("temperature_param_effective", 0.5) * 100
        )
        card.usage_fingerprint_match = self._usage_fingerprint_score(features)

        card.authenticity_score = min(100.0, round(
            0.40 * card.similarity_to_claimed
            + 0.25 * card.predetect_confidence
            + 0.15 * card.consistency_score
            + 0.10 * card.temperature_effectiveness
            + 0.10 * card.usage_fingerprint_match,
            1,
        ))

        # ── Performance sub-scores ──
        card.speed_score = self._speed_score(features)
        card.stability_score = self._stability_score(case_results)
        card.cost_efficiency = self._cost_efficiency(features, case_results)

        card.performance_score = min(100.0, round(
            0.40 * card.speed_score
            + 0.30 * card.stability_score
            + 0.30 * card.cost_efficiency,
            1,
        ))

        # ── Total ──
        card.total_score = round(
            0.45 * card.capability_score
            + 0.35 * card.authenticity_score
            + 0.20 * card.performance_score,
            1,
        )

        return card

    # ── Sub-score implementations ──

    def _reasoning_score(self, case_results: list[CaseResult]) -> float:
        """Basic reasoning score — excludes adversarial_reasoning dimension."""
        cases = [
            r for r in case_results
            if r.case.category == "reasoning"
            and (r.case.dimension or "").lower() != "adversarial_reasoning"
        ]
        if not cases:
            return 50.0  # no data, neutral

        base = self._weighted_pass_rate(cases) * 100

        # Constraint-first bonus: reward key-constraint hit + boundary proof,
        # and penalize anti-pattern/template slip.
        total_samples = 0
        constraint_hit_samples = 0
        boundary_hit_samples = 0
        anti_pattern_samples = 0

        for r in cases:
            for s in r.samples:
                d = s.judge_detail or {}
                total_samples += 1
                if d.get("constraint_hits"):
                    if len(d.get("constraint_hits", [])) > 0:
                        constraint_hit_samples += 1
                if d.get("boundary_hits"):
                    if len(d.get("boundary_hits", [])) > 0:
                        boundary_hit_samples += 1
                if d.get("anti_pattern_hits"):
                    if len(d.get("anti_pattern_hits", [])) > 0:
                        anti_pattern_samples += 1

        if total_samples == 0:
            return base

        constraint_rate = constraint_hit_samples / total_samples
        boundary_rate = boundary_hit_samples / total_samples
        anti_pattern_rate = anti_pattern_samples / total_samples

        adjusted = base + 12.0 * constraint_rate + 12.0 * boundary_rate - 10.0 * anti_pattern_rate
        return max(0.0, min(100.0, round(adjusted, 1)))

    def _adversarial_reasoning_score(self, case_results: list[CaseResult]) -> float:
        """
        Adversarial reasoning score — paired-variant cases only.

        Scoring logic:
        - Base: weighted pass rate × 100
        - Bonus: +15 per paired cross-check that succeeds (model differentiates
          solvable vs unsolvable, or adapts base-encoding to round count)
        - Penalty: -20 for anti-pattern hits (template-matching detected)
        """
        cases = [
            r for r in case_results
            if (r.case.dimension or "").lower() == "adversarial_reasoning"
        ]
        if not cases:
            return 50.0  # no data, neutral

        base = self._weighted_pass_rate(cases) * 100

        total_samples = 0
        anti_pattern_samples = 0
        constraint_hit_samples = 0

        for r in cases:
            for s in r.samples:
                d = s.judge_detail or {}
                total_samples += 1
                if d.get("anti_pattern_hits") and len(d["anti_pattern_hits"]) > 0:
                    anti_pattern_samples += 1
                if d.get("constraint_hits") and len(d["constraint_hits"]) > 0:
                    constraint_hit_samples += 1

        if total_samples == 0:
            return base

        constraint_rate = constraint_hit_samples / total_samples
        anti_pattern_rate = anti_pattern_samples / total_samples

        adjusted = base + 15.0 * constraint_rate - 20.0 * anti_pattern_rate
        return max(0.0, min(100.0, round(adjusted, 1)))

    def _instruction_score(self, features: dict[str, float]) -> float:
        f = features.get
        return min(100.0, (
            f("instruction_pass_rate", 0.5) * 30
            + f("exact_match_rate", 0.5) * 25
            + f("json_valid_rate", 0.5) * 25
            + f("format_follow_rate", 0.5) * 10
            + f("system_obedience_rate", 0.5) * 10
        ))

    def _coding_score(self, case_results: list[CaseResult]) -> float:
        cases = [r for r in case_results if r.case.category == "coding"]
        if not cases:
            return 50.0
        return self._weighted_pass_rate(cases) * 100

    def _safety_score(self, features: dict[str, float]) -> float:
        refusal = features.get("refusal_rate", 0.5)
        disclaimer = features.get("disclaimer_rate", 0.3)
        # Higher refusal on dangerous content = safer = higher score
        return min(100.0, refusal * 60 + disclaimer * 40)

    def _protocol_score(self, features: dict[str, float]) -> float:
        f = features.get
        return min(100.0, (
            f("protocol_success_rate", 0.5) * 40
            + f("has_usage_fields", 0.5) * 20
            + f("has_finish_reason", 0.5) * 20
            + f("param_compliance_rate", 0.5) * 20
        ))

    def _similarity_to_claimed(
        self, similarities: list[SimilarityResult],
        claimed_model: str | None,
    ) -> float:
        if not similarities:
            return 50.0
        if claimed_model:
            claimed_lower = claimed_model.lower()
            for s in similarities:
                if s.benchmark_name.lower() in claimed_lower or \
                   claimed_lower in s.benchmark_name.lower():
                    return min(100.0, s.similarity_score * 100)
        # Fallback: use top similarity
        return min(100.0, similarities[0].similarity_score * 100)

    def _consistency_score(self, case_results: list[CaseResult]) -> float:
        cases = [r for r in case_results if r.case.category == "consistency"]
        if not cases:
            # Fallback: measure consistency from multi-sample cases
            multi = [r for r in case_results if len(r.samples) >= 3]
            if not multi:
                return 70.0
            consistent = 0
            total = 0
            for r in multi:
                texts = [s.response.content for s in r.samples
                         if s.response.content]
                if len(texts) >= 2:
                    total += 1
                    # Check if all are identical
                    if len(set(t.strip().lower()[:50] for t in texts)) == 1:
                        consistent += 1
            return (consistent / total * 100) if total else 70.0
        return self._weighted_pass_rate(cases) * 100

    def _usage_fingerprint_score(self, features: dict[str, float]) -> float:
        has_usage = features.get("has_usage_fields", 0.0)
        has_finish = features.get("has_finish_reason", 0.0)
        return (has_usage * 50 + has_finish * 50)

    def _speed_score(self, features: dict[str, float]) -> float:
        mean_lat = features.get("latency_mean_ms", 2000)
        p95_lat = features.get("latency_p95_ms", 5000)
        # Lower latency = higher score. Scale: 0-200ms=100, 5000ms+=0
        mean_score = max(0, min(100, 100 - (mean_lat / 50)))
        p95_score = max(0, min(100, 100 - (p95_lat / 80)))
        return round(mean_score * 0.6 + p95_score * 0.4, 1)

    def _stability_score(self, case_results: list[CaseResult]) -> float:
        if not case_results:
            return 50.0
        total_samples = 0
        error_samples = 0
        for r in case_results:
            for s in r.samples:
                total_samples += 1
                if s.response.error_type:
                    error_samples += 1
        if total_samples == 0:
            return 50.0
        error_rate = error_samples / total_samples
        return max(0, min(100, round((1 - error_rate) * 100, 1)))

    def _cost_efficiency(self, features: dict[str, float],
                         case_results: list[CaseResult]) -> float:
        # Based on avg response length vs latency
        avg_len = features.get("avg_response_length", 300)
        mean_lat = features.get("latency_mean_ms", 2000)
        if mean_lat <= 0:
            return 80.0
        # Chars per second
        cps = (avg_len / (mean_lat / 1000))
        # Scale: 500 cps = 100, 0 cps = 0
        return max(0, min(100, round(cps / 5, 1)))

    @staticmethod
    def _weighted_pass_rate(results: list[CaseResult]) -> float:
        total_weight = 0.0
        weighted_pass = 0.0
        for r in results:
            w = r.case.weight
            total_weight += w
            weighted_pass += w * r.pass_rate
        return (weighted_pass / total_weight) if total_weight > 0 else 0.0


# ── Verdict Engine (v2) ──────────────────────────────────────────────────────

class VerdictEngine:
    """
    Maps ScoreCard to a trust verdict:
      85+   → trusted
      70-85 → suspicious
      50-70 → high_risk
      <50   → fake
    """

    def assess(
        self,
        scorecard: ScoreCard,
        similarities: list[SimilarityResult],
        predetect: PreDetectionResult | None,
        features: dict[str, float],
    ) -> TrustVerdict:
        reasons: list[str] = []

        # Signal 1: Authenticity score
        auth = scorecard.authenticity_score
        if auth >= 85:
            reasons.append(f"真实性分 {auth:.1f} — 行为高度匹配声称模型")
        elif auth >= 70:
            reasons.append(f"真实性分 {auth:.1f} — 行为存在轻度偏差")
        elif auth >= 50:
            reasons.append(f"真实性分 {auth:.1f} — 行为偏差明显")
        else:
            reasons.append(f"真实性分 {auth:.1f} — 行为严重不匹配")

        # Signal 2: Pre-detection
        if predetect and predetect.success:
            reasons.append(
                f"预检测识别为 {predetect.identified_as} "
                f"(置信度 {predetect.confidence:.0%})"
            )

        # Signal 3: Temperature
        if features.get("temperature_param_effective", 1.0) < 0.5:
            reasons.append("temperature 参数不生效 — 可能是代理/包装层屏蔽")

        # Signal 4: Consistency anomaly
        if scorecard.consistency_score < 50:
            reasons.append(f"一致性分仅 {scorecard.consistency_score:.1f} — 多次采样结果不稳定")

        # Signal 5: Top similarity details
        if similarities:
            top = similarities[0]
            reasons.append(
                f"最相似基准模型: {top.benchmark_name} "
                f"(相似度 {top.similarity_score:.2f})"
            )

        # Signal 6: Adversarial spoof signal
        adv_spoof = features.get("adversarial_spoof_signal_rate")
        if adv_spoof is not None and adv_spoof > 0.3:
            reasons.append(
                f"对抗推理检测到模板套用 (信号率 {adv_spoof:.0%})"
            )

        # Determine level based on authenticity + overall score
        # Adversarial spoof > 0.7 forces at least high_risk regardless of other scores.
        score = scorecard.total_score
        adv_override = adv_spoof is not None and adv_spoof > 0.7

        if adv_override:
            if auth < 50 or score < 45:
                level, label = "fake", "疑似假模型 / Likely Fake"
            else:
                level, label = "high_risk", "高风险 / High Risk"
        elif auth >= 85 and score >= 75:
            level, label = "trusted", "可信 / Trusted"
        elif auth >= 70 or score >= 65:
            level, label = "suspicious", "轻度可疑 / Suspicious"
        elif auth >= 50 or score >= 45:
            level, label = "high_risk", "高风险 / High Risk"
        else:
            level, label = "fake", "疑似假模型 / Likely Fake"

        if not reasons:
            reasons.append("没有检测到明显异常信号")

        return TrustVerdict(
            level=level,
            label=label,
            total_score=score,
            reasons=reasons,
        )


# ── Similarity Engine ─────────────────────────────────────────────────────────

FEATURE_ORDER = [
    "protocol_success_rate", "instruction_pass_rate", "exact_match_rate",
    "json_valid_rate", "system_obedience_rate", "param_compliance_rate",
    "temperature_param_effective", "refusal_rate", "disclaimer_rate",
    "avg_markdown_score", "avg_response_length",
    "adversarial_spoof_signal_rate", "latency_mean_ms",
]


class SimilarityEngine:

    def compare(
        self,
        target_features: dict[str, float],
        benchmark_profiles: list[dict],
    ) -> list[SimilarityResult]:
        """
        Returns similarity results ranked by score.
        Each benchmark_profile has: {name, suite_version, feature_vector: {k: v}}
        """
        if not benchmark_profiles:
            return []

        target_vec = self._to_vector(target_features)
        results: list[SimilarityResult] = []

        for bp in benchmark_profiles:
            bench_vec = self._to_vector(bp["feature_vector"])
            sim = self._cosine_similarity(target_vec, bench_vec)
            ci_low, ci_high = self._bootstrap_ci(target_vec, bench_vec)
            results.append(SimilarityResult(
                benchmark_name=bp["benchmark_name"],
                similarity_score=round(sim, 4),
                ci_95_low=round(ci_low, 4),
                ci_95_high=round(ci_high, 4),
                rank=0,
            ))

        results.sort(key=lambda r: r.similarity_score, reverse=True)
        for i, r in enumerate(results):
            r.rank = i + 1

        return results

    def _to_vector(self, features: dict[str, float]) -> list[float]:
        """Build a fixed-length vector from features, normalised 0-1."""
        vec = []
        for key in FEATURE_ORDER:
            val = features.get(key)
            if val is None:
                vec.append(0.0)  # Unknown data doesn't contribute to similarity
                continue

            # Feature-specific normalization
            if key == "avg_response_length":
                # Normalize length: 0-1200 range
                val = val / 1200.0
            elif key == "avg_markdown_score":
                # Normalize score: 0-5 range
                val = val / 5.0
            elif key == "latency_mean_ms":
                # Normalize latency: 0-5000ms range, inverted (lower is 1.0)
                val = 1.0 - (val / 5000.0)
            
            # Clamp to [0,1]
            vec.append(max(0.0, min(1.0, float(val))))
        return vec

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        if len(a) != len(b):
            # Pad shorter
            n = max(len(a), len(b))
            a = a + [0.0] * (n - len(a))
            b = b + [0.0] * (n - len(b))
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @classmethod
    def _bootstrap_ci(
        cls, a: list[float], b: list[float], n: int = 200
    ) -> tuple[float, float]:
        """Bootstrap 95% confidence interval for cosine similarity."""
        length = len(a)
        if length == 0:
            return 0.0, 0.0
        sims = []
        rng = random.Random(42)
        for _ in range(n):
            indices = [rng.randrange(length) for _ in range(length)]
            a2 = [a[i] for i in indices]
            b2 = [b[i] for i in indices]
            sims.append(cls._cosine_similarity(a2, b2))
        sims.sort()
        lo = sims[int(n * 0.025)]
        hi = sims[int(n * 0.975)]
        return lo, hi


# ── Risk Engine ───────────────────────────────────────────────────────────────

class RiskEngine:
    """
    Combines pre-detection + similarity + feature signals into a risk level.
    Levels: low | medium | high | very_high
    """

    def assess(
        self,
        features: dict[str, float],
        similarities: list[SimilarityResult],
        predetect: PreDetectionResult | None,
    ) -> RiskAssessment:
        reasons: list[str] = []
        risk_score = 0.0  # 0.0 to 1.0

        # Signal 1: Pre-detection confidence
        if predetect and predetect.success:
            conf = predetect.confidence
            risk_score += conf * 0.35
            reasons.append(
                f"预检测识别为 {predetect.identified_as}，置信度 {conf:.0%}"
                f" / Pre-detection identified as {predetect.identified_as} "
                f"(confidence {conf:.0%})"
            )

        # Signal 2: Similarity to top benchmark (reduced from 0.35 to 0.25)
        if similarities:
            top = similarities[0]
            if top.similarity_score >= 0.85:
                risk_score += 0.25
                reasons.append(
                    f"与基准模型 {top.benchmark_name} 相似度极高 ({top.similarity_score:.2f})"
                    f" / Very high similarity to {top.benchmark_name} ({top.similarity_score:.2f})"
                )
            elif top.similarity_score >= 0.70:
                risk_score += 0.15
                reasons.append(
                    f"与基准模型 {top.benchmark_name} 相似度较高 ({top.similarity_score:.2f})"
                )

        # Signal 3: Temperature not effective
        if features.get("temperature_param_effective", 1.0) < 0.5:
            risk_score += 0.10
            reasons.append(
                "temperature 参数似乎不生效（响应多样性异常低）"
                " / temperature parameter appears ineffective (low output diversity)"
            )

        # Signal 4: System obedience unusually low (may be locked)
        sys_obey = features.get("system_obedience_rate", 0.5)
        if sys_obey < 0.3:
            risk_score += 0.10
            reasons.append(
                f"system prompt 遵循率异常低 ({sys_obey:.0%})，可能存在覆盖层"
                f" / System prompt obedience abnormally low ({sys_obey:.0%})"
            )

        # Signal 5: Adversarial spoof signal — highest single-indicator
        # precision for detecting proxy/wrapper models.
        adv_spoof = features.get("adversarial_spoof_signal_rate")
        if adv_spoof is not None and adv_spoof > 0.3:
            risk_score += adv_spoof * 0.20
            reasons.append(
                f"对抗推理套壳信号率 {adv_spoof:.0%}（配对变体检测到模板套用）"
                f" / Adversarial reasoning spoof signal {adv_spoof:.0%} "
                f"(template-matching detected in paired variants)"
            )

        # Determine level
        if risk_score >= 0.70:
            level, label = "very_high", "很高 / Very High"
        elif risk_score >= 0.45:
            level, label = "high", "高 / High"
        elif risk_score >= 0.25:
            level, label = "medium", "中 / Medium"
        else:
            level, label = "low", "低 / Low"

        if not reasons:
            reasons.append("没有检测到明显的套壳信号 / No strong proxy signals detected")

        return RiskAssessment(level=level, label=label, reasons=reasons)


# ── Theta Estimation (relative scale) ───────────────────────────────────────

class ThetaEstimator:
    """Simple Rasch-like 1PL estimator from pass/fail case results."""

    def estimate(self, case_results: list[CaseResult], item_stats: dict[str, dict]) -> ThetaReport:
        by_dim: dict[str, list[float]] = {}
        for r in case_results:
            dim = (r.case.dimension or r.case.category or "unknown").lower()
            st = item_stats.get(r.case.id, {})
            b = float(st.get("irt_b", 0.0) or 0.0)
            for s in r.samples:
                if s.judge_passed is None:
                    continue
                x = 1.0 if s.judge_passed else 0.0
                by_dim.setdefault(dim, []).append((x, b))

        dims: list[ThetaDimensionEstimate] = []
        all_thetas: list[float] = []
        for dim, obs in by_dim.items():
            theta = self._estimate_theta_1pl(obs)
            dims.append(ThetaDimensionEstimate(
                dimension=dim,
                theta=theta,
                ci_low=theta - 0.25,
                ci_high=theta + 0.25,
                n_items=len(obs),
            ))
            all_thetas.append(theta)

        if not all_thetas:
            return ThetaReport(
                global_theta=0.0,
                global_ci_low=-0.3,
                global_ci_high=0.3,
                dimensions=[],
                calibration_version=settings.CALIBRATION_VERSION,
                method=settings.THETA_METHOD,
                notes=["insufficient judged samples"],
            )

        g = sum(all_thetas) / len(all_thetas)
        return ThetaReport(
            global_theta=g,
            global_ci_low=g - 0.25,
            global_ci_high=g + 0.25,
            dimensions=sorted(dims, key=lambda d: d.theta, reverse=True),
            calibration_version=settings.CALIBRATION_VERSION,
            method=settings.THETA_METHOD,
        )

    def _estimate_theta_1pl(self, obs: list[tuple[float, float]]) -> float:
        if not obs:
            return 0.0
        theta = 0.0
        for _ in range(25):
            p_vals = [1.0 / (1.0 + math.exp(-(theta - b))) for _, b in obs]
            grad = sum((x - p) for (x, _), p in zip(obs, p_vals))
            hess = -sum(p * (1.0 - p) for p in p_vals)
            if abs(hess) < 1e-6:
                break
            step = grad / hess
            theta -= step
            if abs(step) < 1e-4:
                break
        return max(-4.0, min(4.0, theta))


class UncertaintyEstimator:
    def apply_ci(self, theta_report: ThetaReport, case_results: list[CaseResult],
                 estimator: ThetaEstimator, item_stats: dict[str, dict]) -> ThetaReport:
        boot_n = max(10, settings.THETA_BOOTSTRAP_B)
        if not case_results:
            return theta_report

        samples_global: list[float] = []
        samples_dims: dict[str, list[float]] = {}

        flat = []
        for r in case_results:
            for s in r.samples:
                if s.judge_passed is None:
                    continue
                flat.append((r, s))

        if len(flat) < 4:
            return theta_report

        for _ in range(boot_n):
            picked = [flat[random.randint(0, len(flat) - 1)] for _ in range(len(flat))]
            by_case: dict[str, CaseResult] = {}
            for r, s in picked:
                rid = r.case.id
                if rid not in by_case:
                    by_case[rid] = CaseResult(case=r.case, samples=[])
                by_case[rid].samples.append(s)

            rep = estimator.estimate(list(by_case.values()), item_stats)
            samples_global.append(rep.global_theta)
            for d in rep.dimensions:
                samples_dims.setdefault(d.dimension, []).append(d.theta)

        theta_report.global_ci_low, theta_report.global_ci_high = self._percentile_ci(samples_global)
        dim_map = {d.dimension: d for d in theta_report.dimensions}
        for dim, vals in samples_dims.items():
            if dim in dim_map and vals:
                lo, hi = self._percentile_ci(vals)
                dim_map[dim].ci_low = lo
                dim_map[dim].ci_high = hi
        return theta_report

    @staticmethod
    def _percentile_ci(values: list[float]) -> tuple[float, float]:
        if not values:
            return (-0.3, 0.3)
        arr = sorted(values)
        n = len(arr)
        lo = arr[max(0, int(0.025 * n) - 1)]
        hi = arr[min(n - 1, int(0.975 * n))]
        return (round(lo, 4), round(hi, 4))


class PercentileMapper:
    def map_percentiles(self, theta_report: ThetaReport, historical: list[dict]) -> ThetaReport:
        if not historical:
            return theta_report
        globals_hist = sorted(float(r.get("theta_global", 0.0) or 0.0) for r in historical)
        theta_report.global_percentile = self._pct(theta_report.global_theta, globals_hist)

        dim_hist: dict[str, list[float]] = {}
        for r in historical:
            dims = r.get("theta_dims_json") or {}
            for k, v in dims.items():
                dim_hist.setdefault(k, []).append(float((v or {}).get("theta", 0.0) or 0.0))
        for d in theta_report.dimensions:
            h = sorted(dim_hist.get(d.dimension, []))
            d.percentile = self._pct(d.theta, h) if h else None
        return theta_report

    @staticmethod
    def _pct(v: float, arr: list[float]) -> float:
        if not arr:
            return 50.0
        rank = sum(1 for x in arr if x <= v)
        return round(rank * 100.0 / len(arr), 2)


class PairwiseEngine:
    def compare_to_baseline(self, theta_report: ThetaReport, baseline_theta: float | None) -> dict | None:
        if baseline_theta is None:
            return None
        delta = theta_report.global_theta - baseline_theta
        scale = max(0.1, settings.THETA_SCALE_FOR_WIN_PROB)
        win_prob = 1.0 / (1.0 + math.exp(-delta / scale))
        return {
            "delta_theta": round(delta, 4),
            "win_prob": round(win_prob, 4),
            "baseline_theta": round(baseline_theta, 4),
            "method": "bradley_terry",
        }


# ── Report Builder ────────────────────────────────────────────────────────────

class ReportBuilder:

    def build(
        self,
        run_id: str,
        base_url: str,
        model_name: str,
        test_mode: str,
        predetect: PreDetectionResult | None,
        case_results: list[CaseResult],
        features: dict[str, float],
        scores: Scores,
        similarities: list[SimilarityResult],
        risk: RiskAssessment,
        scorecard: ScoreCard | None = None,
        verdict: TrustVerdict | None = None,
        theta_report: ThetaReport | None = None,
        pairwise: dict | None = None,
    ) -> dict:
        dimensions = {
            k.replace("dim_", "").replace("_pass_rate", ""): v
            for k, v in features.items()
            if k.startswith("dim_") and k.endswith("_pass_rate")
        }
        tag_breakdown = {
            k.replace("tag_", "").replace("_pass_rate", ""): v
            for k, v in features.items()
            if k.startswith("tag_") and k.endswith("_pass_rate")
        }
        failure_attribution = {
            k.replace("failure_", "").replace("_rate", ""): v
            for k, v in features.items()
            if k.startswith("failure_") and k.endswith("_rate")
        }

        report = {
            "run_id": run_id,
            "target": {
                "base_url": base_url,
                "model": model_name,
                "test_mode": test_mode,
            },
            "predetection": predetect.to_dict() if predetect else None,
            "scores": {
                "protocol_score": scores.protocol_score,
                "instruction_score": scores.instruction_score,
                "system_obedience_score": scores.system_obedience_score,
                "param_compliance_score": scores.param_compliance_score,
            },
            "similarity": [
                {
                    "rank": s.rank,
                    "benchmark": s.benchmark_name,
                    "score": s.similarity_score,
                    "ci_95_low": s.ci_95_low,
                    "ci_95_high": s.ci_95_high,
                }
                for s in similarities
            ],
            "risk": {
                "level": risk.level,
                "label": risk.label,
                "reasons": risk.reasons,
                "disclaimer": risk.disclaimer,
            },
            "dimensions": dimensions,
            "tag_breakdown": tag_breakdown,
            "failure_attribution": failure_attribution,
            "features": features,
            "case_results": [
                {
                    "case_id": r.case.id,
                    "category": r.case.category,
                    "dimension": r.case.dimension or r.case.category,
                    "tags": r.case.tags,
                    "name": r.case.name,
                    "judge_rubric": r.case.judge_rubric,
                    "pass_rate": round(r.pass_rate, 3),
                    "mean_latency_ms": r.mean_latency_ms,
                    "samples": [
                        {
                            "sample_index": s.sample_index,
                            "output": (s.response.content or "")[:500],
                            "passed": s.judge_passed,
                            "latency_ms": s.response.latency_ms,
                            "error_type": s.response.error_type,
                            "judge_detail": s.judge_detail,
                        }
                        for s in r.samples
                    ],
                }
                for r in case_results
            ],
        }

        # v2 scorecard & verdict
        if scorecard:
            report["scorecard"] = scorecard.to_dict()
        if verdict:
            report["verdict"] = verdict.to_dict()
        if theta_report:
            report["theta"] = theta_report.to_dict()
        if pairwise:
            report["pairwise_rank"] = pairwise

        return report
