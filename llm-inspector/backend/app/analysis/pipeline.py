"""
Analysis pipeline:
  FeatureExtractor  — raw responses → named features
  ScoreCalculator   — features → 4 scores (0-100)
  SimilarityEngine  — features vs benchmarks → cosine + bootstrap CI
  RiskEngine        — features + similarity → risk level
  ReportBuilder     — everything → final report dict
"""
from __future__ import annotations

import json
import math
import random
from app.core.schemas import (
    CaseResult, PreDetectionResult, Scores, SimilarityResult, RiskAssessment
)
from app.core.logging import get_logger

logger = get_logger(__name__)


# ── Feature Extractor ─────────────────────────────────────────────────────────

class FeatureExtractor:
    """
    Extracts a flat dict of named numeric features from all case results.
    """

    def extract(self, case_results: list[CaseResult]) -> dict[str, float]:
        features: dict[str, float] = {}

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


# ── Similarity Engine ─────────────────────────────────────────────────────────

FEATURE_ORDER = [
    "protocol_success_rate", "instruction_pass_rate", "exact_match_rate",
    "json_valid_rate", "system_obedience_rate", "param_compliance_rate",
    "temperature_param_effective", "refusal_rate", "disclaimer_rate",
    "avg_markdown_score", "avg_response_length",
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
            val = features.get(key, 0.5)  # 0.5 = unknown/missing
            # Normalise latency (0-5000ms range → 0-1, inverted)
            if "latency" in key:
                val = max(0.0, 1.0 - val / 5000.0)
            # Clamp to [0,1]
            vec.append(max(0.0, min(1.0, float(val))))
        return vec

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        if len(a) != len(b):
            # Pad shorter
            n = max(len(a), len(b))
            a = a + [0.5] * (n - len(a))
            b = b + [0.5] * (n - len(b))
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
            risk_score += conf * 0.40
            reasons.append(
                f"预检测识别为 {predetect.identified_as}，置信度 {conf:.0%}"
                f" / Pre-detection identified as {predetect.identified_as} "
                f"(confidence {conf:.0%})"
            )

        # Signal 2: Similarity to top benchmark
        if similarities:
            top = similarities[0]
            if top.similarity_score >= 0.85:
                risk_score += 0.35
                reasons.append(
                    f"与基准模型 {top.benchmark_name} 相似度极高 ({top.similarity_score:.2f})"
                    f" / Very high similarity to {top.benchmark_name} ({top.similarity_score:.2f})"
                )
            elif top.similarity_score >= 0.70:
                risk_score += 0.20
                reasons.append(
                    f"与基准模型 {top.benchmark_name} 相似度较高 ({top.similarity_score:.2f})"
                )

        # Signal 3: Temperature not effective
        if features.get("temperature_param_effective", 1.0) < 0.5:
            risk_score += 0.15
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
    ) -> dict:
        return {
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
            "features": features,
            "case_results": [
                {
                    "case_id": r.case.id,
                    "category": r.case.category,
                    "name": r.case.name,
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
