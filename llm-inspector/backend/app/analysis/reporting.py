"""
analysis/reporting.py — NarrativeBuilder, ProxyLatencyAnalyzer,
                         ExtractionAuditBuilder, ReportBuilder

Human-readable narrative generation and final report assembly.
Extracted from pipeline.py to keep individual files under ~900 lines.
"""
from __future__ import annotations

import math
import re

import numpy as np

from app.core.schemas import (
    CaseResult, PreDetectionResult, Scores, SimilarityResult, RiskAssessment,
    ScoreCard, TrustVerdict, ThetaReport, ThetaDimensionEstimate,
)
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Lazy imports to avoid circular dependencies - resolved at runtime
# from app.analysis.feature_engine import FeatureExtractor
# from app.analysis.scoring import ScoreCardCalculator, ScoreCalculator
# from app.analysis.verdicts import VerdictEngine
# from app.analysis.similarity import SimilarityEngine
# from app.analysis.estimation import RiskEngine, ThetaEstimator, UncertaintyEstimator, PercentileMapper, PairwiseEngine


class NarrativeBuilder:
    """
    Generates human-readable narrative summaries from structured report data.
    Zero token cost — pure rule-based text generation.
    """

    def build(
        self,
        model_name: str,
        verdict: TrustVerdict | None,
        scorecard: ScoreCard | None,
        similarities: list[SimilarityResult],
        predetect: PreDetectionResult | None,
        features: dict[str, float],
        case_results: list[CaseResult],
    ) -> dict:
        return {
            "executive_summary": self._executive_summary(model_name, verdict, scorecard, similarities),
            "detection_process": self._detection_process(predetect),
            "dimension_analysis": self._dimension_analysis(scorecard, features, case_results),
            "similarity_narrative": self._similarity_narrative(similarities),
            "risk_narrative": self._risk_narrative(verdict, scorecard),
            "recommendations": self._recommendations(verdict, scorecard, features),
            "confidence_statement": self._confidence_statement(similarities, predetect),
        }

    @staticmethod
    def _executive_summary(
        model_name: str,
        verdict: TrustVerdict | None,
        scorecard: ScoreCard | None,
        similarities: list[SimilarityResult],
    ) -> str:
        top_match = similarities[0] if similarities else None
        trust_cn = {
            "trusted": "可信", "suspicious": "可疑",
            "high_risk": "高风险", "fake": "疑似套壳",
        }.get(verdict.level if verdict else "", verdict.level if verdict else "unknown")

        top_match_str = ""
        if top_match and top_match.similarity_score > 0.7:
            ci_str = ""
            if top_match.ci_95_low is not None and top_match.ci_95_high is not None:
                ci_str = f"（95% CI: {top_match.ci_95_low:.1%}–{top_match.ci_95_high:.1%}）。"
            top_match_str = (
                f"行为特征与 **{top_match.benchmark_name}** 的相似度最高，"
                f"达到 {top_match.similarity_score:.1%}{ci_str}"
            )

        sc_str = ""
        if scorecard:
            sc_str = (
                f"能力维度得分 {scorecard.capability_score:.1f}，"
                f"真实性维度得分 {scorecard.authenticity_score:.1f}，"
                f"性能维度得分 {scorecard.performance_score:.1f}。"
            )

        total = f"总分 **{scorecard.total_score:.1f}/100**" if scorecard else "综合评分"

        return (
            f"检测目标声称为 **{model_name}**。"
            f"{total}，"
            f"信任等级判定为 **{trust_cn}**。"
            f"{top_match_str}"
            f"{sc_str}"
        )

    @staticmethod
    def _detection_process(predetect: PreDetectionResult | None) -> str:
        if not predetect:
            return "预检测阶段未执行或无结果。"

        layers_passed = [l.layer for l in (predetect.layer_results or []) if l.confidence > 0]
        confidence = predetect.confidence or 0.0
        candidate = predetect.identified_as or "未能确定"

        if confidence >= 0.85:
            result_str = f"预检测在 {len(layers_passed)} 层后提前终止（置信度 {confidence:.0%} ≥ 阈值）"
        else:
            result_str = f"预检测完成全部层次，最终置信度 {confidence:.0%}"

        return (
            f"{result_str}，候选模型为 **{candidate}**。"
            f"有效信号层：{', '.join(layers_passed) if layers_passed else '无'}。"
        )

    @staticmethod
    def _dimension_analysis(
        scorecard: ScoreCard | None,
        features: dict[str, float],
        case_results: list[CaseResult],
    ) -> str:
        parts = []

        failed_cases = [r for r in case_results if r.pass_rate < 0.5]
        failed_dims: dict[str, int] = {}
        for r in failed_cases:
            dim = r.case.dimension or r.case.category or "unknown"
            failed_dims[dim] = failed_dims.get(dim, 0) + 1

        if failed_dims:
            worst = sorted(failed_dims.items(), key=lambda x: -x[1])[:3]
            parts.append(
                "失败用例集中于："
                + "、".join(f"**{d}**（{n}个）" for d, n in worst) + "。"
            )

        instr = features.get("instruction_pass_rate")
        if instr is not None:
            parts.append(f"指令遵循通过率 {instr:.0%}。")

        consist = features.get("dim_consistency_pass_rate")
        if consist is not None:
            level = "良好" if consist > 0.8 else ("一般" if consist > 0.5 else "较差")
            parts.append(f"多采样一致性{level}（{consist:.0%}）。")

        temp_eff = features.get("temperature_param_effective")
        if temp_eff is not None:
            parts.append(
                f"Temperature 参数{'有效响应' if temp_eff > 0.5 else '无效（疑似参数透传缺失）'}。"
            )

        return " ".join(parts) if parts else "维度分析数据不足。"

    @staticmethod
    def _similarity_narrative(similarities: list[SimilarityResult]) -> str:
        if not similarities:
            return "未找到相似基准模型。"

        top3 = similarities[:3]
        lines = []
        for s in top3:
            ci_width = (s.ci_95_high - s.ci_95_low) if (s.ci_95_high is not None and s.ci_95_low is not None) else None
            confidence_desc = "高置信度" if ci_width is not None and ci_width < 0.1 else ("中等置信度" if ci_width is not None and ci_width < 0.2 else "低置信度")
            ci_str = f"（{confidence_desc}，CI: {s.ci_95_low:.1%}–{s.ci_95_high:.1%}）" if ci_width is not None else f"（{confidence_desc}）"
            lines.append(
                f"  - **#{s.rank} {s.benchmark_name}**：相似度 {s.similarity_score:.1%}{ci_str}"
            )

        top = top3[0]
        if top.similarity_score > 0.85:
            conclusion = f"与 {top.benchmark_name} 的行为高度一致，有较强证据支持。"
        elif top.similarity_score > 0.65:
            conclusion = f"与 {top.benchmark_name} 存在中等相似性，但无法排除其他可能。"
        else:
            conclusion = "与所有已知基准模型相似度偏低，可能为未收录模型或行为受到干预。"

        return "最相似的基准模型（Top-3）：\n" + "\n".join(lines) + f"\n\n{conclusion}"

    @staticmethod
    def _risk_narrative(
        verdict: TrustVerdict | None,
        scorecard: ScoreCard | None,
    ) -> str:
        level_text = {
            "trusted":   "当前 API 行为与声称模型一致，未发现明显欺骗信号。",
            "suspicious":"检测发现若干异常信号，建议进一步验证或谨慎使用。",
            "high_risk": "检测发现多项高风险信号，真实模型与声称模型存在显著差异。",
            "fake":      "综合评估认为该 API 极有可能在使用与声称不同的底层模型（套壳）。",
        }.get(verdict.level if verdict else "", "无法确定风险等级。")

        reasons = "\n".join(f"  - {r}" for r in (verdict.reasons if verdict else []))
        return f"{level_text}\n\n主要风险信号：\n{reasons}" if reasons else level_text

    @staticmethod
    def _recommendations(
        verdict: TrustVerdict | None,
        scorecard: ScoreCard | None,
        features: dict[str, float],
    ) -> list[str]:
        recs = []

        if verdict and verdict.level in ("fake", "high_risk"):
            recs.append("⚠️ 建议向 API 提供商要求提供模型版本证明，或切换至可信赖的直连端点。")

        temp_eff = features.get("temperature_param_effective", 1.0)
        if temp_eff < 0.5:
            recs.append("🔧 Temperature 参数未生效，若您的应用依赖随机性，请确认 API 提供商是否支持该参数透传。")

        if scorecard and scorecard.performance_score < 50:
            lat = features.get("latency_mean_ms", 0)
            recs.append(f"⏱️ 平均延迟 {lat:.0f}ms，性能得分偏低，不建议用于延迟敏感型应用。")

        if scorecard and scorecard.capability_score < 60:
            recs.append("📉 能力得分低于基准，复杂推理和指令遵循任务可能表现不稳定。")

        if not recs:
            recs.append("✅ 未发现明显问题，该 API 端点行为与预期一致。")

        return recs

    @staticmethod
    def _confidence_statement(
        similarities: list[SimilarityResult],
        predetect: PreDetectionResult | None,
    ) -> str:
        notes = []

        if predetect and (predetect.confidence or 0) < 0.85:
            notes.append("预检测置信度未达阈值，最终判定主要依赖行为测试而非指纹识别。")

        if similarities:
            top = similarities[0]
            ci_width = (top.ci_95_high - top.ci_95_low) if (top.ci_95_high is not None and top.ci_95_low is not None) else None
            if ci_width is not None and ci_width > 0.2:
                notes.append(
                    f"最高相似度的置信区间较宽（±{ci_width / 2:.1%}），"
                    f"建议增加采样量（切换至 full 模式）以提高置信度。"
                )

        return " ".join(notes) if notes else "检测样本充足，结论具有较高可信度。"


# ── Proxy Latency Analyzer ───────────────────────────────────────────────────

class ProxyLatencyAnalyzer:
    """
    基于延迟分布特征推断代理层存在性。
    直连 API 的 TTFT 分布集中；二次转发代理有双峰特征。
    """

    KNOWN_TTFT_BASELINES: dict[str, dict] = {
        "claude-opus-4":    {"p50": 800,  "p95": 2000, "mean": 1000},
        "claude-sonnet-4":  {"p50": 400,  "p95": 1200, "mean": 600},
        "gpt-4o":           {"p50": 500,  "p95": 1500, "mean": 700},
        "gpt-4o-mini":      {"p50": 200,  "p95": 600,  "mean": 300},
        "deepseek-v3":      {"p50": 600,  "p95": 2000, "mean": 900},
        "minimax":          {"p50": 300,  "p95": 900,  "mean": 500},
        "qwen-max":         {"p50": 400,  "p95": 1200, "mean": 600},
    }

    def analyze(
        self,
        case_results: list[CaseResult],
        claimed_model: str,
    ) -> dict:
        ttft_samples = [
            s.response.first_token_ms
            for r in case_results
            for s in r.samples
            if s.response.first_token_ms is not None and s.response.first_token_ms > 0
        ]

        if len(ttft_samples) < 5:
            return {"status": "insufficient_samples", "sample_count": len(ttft_samples)}

        ttft_sorted = sorted(ttft_samples)
        n = len(ttft_sorted)
        mean_ttft = sum(ttft_samples) / n
        p50 = ttft_sorted[int(n * 0.50)]
        p95 = ttft_sorted[int(n * 0.95)]

        p25 = ttft_sorted[int(n * 0.25)]
        p75 = ttft_sorted[int(n * 0.75)]
        iqr = p75 - p25
        dispersion = iqr / max(p50, 1)

        # Baseline deviation
        baseline_deviation = None
        model_key = self._match_model_key(claimed_model)
        if model_key and model_key in self.KNOWN_TTFT_BASELINES:
            baseline = self.KNOWN_TTFT_BASELINES[model_key]
            baseline_deviation = (mean_ttft - baseline["mean"]) / max(baseline["mean"], 1)

        bimodal_score = self._detect_bimodal(ttft_sorted)

        proxy_signals: list[str] = []
        proxy_confidence = 0.0

        if dispersion > 1.5:
            proxy_signals.append(f"High TTFT dispersion (IQR/median={dispersion:.2f}) suggests proxy layer")
            proxy_confidence = max(proxy_confidence, 0.65)

        if baseline_deviation is not None and baseline_deviation > 1.0:
            proxy_signals.append(
                f"Mean TTFT {mean_ttft:.0f}ms is {baseline_deviation:.0%} above "
                f"{model_key} baseline — consistent with proxy forwarding overhead"
            )
            proxy_confidence = max(proxy_confidence, 0.70)

        if bimodal_score > 0.6:
            proxy_signals.append(
                f"Bimodal TTFT distribution detected (score={bimodal_score:.2f}) — "
                f"classic signature of two-stage forwarding"
            )
            proxy_confidence = max(proxy_confidence, 0.75)

        return {
            "ttft_mean_ms": round(mean_ttft, 1),
            "ttft_p50_ms": p50,
            "ttft_p95_ms": p95,
            "ttft_dispersion": round(dispersion, 3),
            "bimodal_score": round(bimodal_score, 3),
            "baseline_deviation": round(baseline_deviation, 3) if baseline_deviation is not None else None,
            "proxy_signals": proxy_signals,
            "proxy_confidence": round(proxy_confidence, 3),
            "sample_count": n,
        }

    def _match_model_key(self, model_name: str) -> str | None:
        model_lower = model_name.lower()
        for key in self.KNOWN_TTFT_BASELINES:
            if key in model_lower:
                return key
        return None

    @staticmethod
    def _detect_bimodal(sorted_samples: list) -> float:
        if len(sorted_samples) < 10:
            return 0.0
        total_range = sorted_samples[-1] - sorted_samples[0]
        if total_range < 100:
            return 0.0
        gaps = [sorted_samples[i + 1] - sorted_samples[i] for i in range(len(sorted_samples) - 1)]
        max_gap = max(gaps)
        gap_ratio = max_gap / total_range
        return min(gap_ratio / 0.3, 1.0)


# ── Extraction Audit Builder ─────────────────────────────────────────────────

class ExtractionAuditBuilder:
    """整合 Layer6 提取结果与用例结果，生成提取审计报告"""

    def build(
        self,
        predetect: PreDetectionResult | None,
        case_results: list[CaseResult],
    ) -> dict:
        audit: dict = {
            "prompt_leaked": False,
            "real_model_exposed": False,
            "real_model_names": [],
            "forbidden_words_leaked": [],
            "file_paths_leaked": [],
            "spec_contradictions": [],
            "language_bias_detected": False,
            "tokenizer_mismatch": False,
            "overall_severity": "NONE",
            "evidence_chain": [],
        }

        # Integrate Layer6 results from predetect
        if predetect:
            for lr in predetect.layer_results:
                if lr.layer in ("active_extraction", "multi_turn_extraction"):
                    if lr.confidence > 0.5:
                        audit["prompt_leaked"] = True
                    if lr.identified_as:
                        audit["real_model_exposed"] = True
                        audit["evidence_chain"].append(
                            f"[{lr.layer}] {lr.identified_as}"
                        )
                    for ev in lr.evidence:
                        if "[CRITICAL]" in ev:
                            audit["evidence_chain"].append(ev)

        # Integrate extraction suite case results
        ext_cases = [r for r in case_results if r.case.category == "extraction"]
        for r in ext_cases:
            for s in r.samples:
                d = s.judge_detail or {}
                severity = d.get("severity", "NONE")

                if d.get("real_models_in_forbidden_list"):
                    audit["forbidden_words_leaked"].extend(d["real_models_in_forbidden_list"])
                    audit["evidence_chain"].append(
                        f"[{r.case.id}] Forbidden word list leaked: {d['real_models_in_forbidden_list']}"
                    )

                if d.get("real_models_found"):
                    audit["real_model_exposed"] = True
                    audit["real_model_names"].extend(d["real_models_found"])
                    audit["evidence_chain"].append(
                        f"[{r.case.id}] Real model exposed: {d['real_models_found']}"
                    )

                if d.get("paths_found"):
                    audit["file_paths_leaked"].extend(d["paths_found"][:5])

                if d.get("actual_model_match"):
                    audit["spec_contradictions"].append({
                        "case": r.case.id,
                        "reported": d.get("reported_value"),
                        "expected": d.get("expected_value"),
                        "actual_match": d.get("actual_model_match"),
                    })
                    audit["evidence_chain"].append(
                        f"[{r.case.id}] Spec contradiction: reported {d.get('reported_value')}, "
                        f"matches {d.get('actual_model_match')} not claimed model"
                    )

                if r.case.judge_method == "language_bias_detect" and s.judge_passed:
                    audit["language_bias_detected"] = True

                if r.case.judge_method == "tokenizer_fingerprint" and s.judge_passed:
                    audit["tokenizer_mismatch"] = True

        # Compute extraction resistance rate for consistency with pipeline features
        LEAK_JUDGES = {
            "prompt_leak_detect", "forbidden_word_extract",
            "path_leak_detect", "tool_config_leak_detect", "memory_leak_detect",
        }
        resisted = 0
        total_ext_samples = 0
        for r in ext_cases:
            for s in r.samples:
                if s.judge_passed is not None:
                    total_ext_samples += 1
                    if r.case.judge_method in LEAK_JUDGES:
                        if not s.judge_passed:  # no leak = good
                            resisted += 1
                    else:
                        if s.judge_passed:  # resistance = good
                            resisted += 1
        resist_rate = resisted / total_ext_samples if total_ext_samples > 0 else None
        audit["extraction_resist_rate"] = round(resist_rate, 3) if resist_rate is not None else None

        # Overall severity — also considers low extraction resistance
        if audit["real_model_exposed"] or audit["forbidden_words_leaked"]:
            audit["overall_severity"] = "CRITICAL"
        elif audit["spec_contradictions"] or audit["file_paths_leaked"]:
            audit["overall_severity"] = "HIGH"
        elif audit["language_bias_detected"] or audit["prompt_leaked"]:
            audit["overall_severity"] = "MEDIUM"
        elif resist_rate is not None and resist_rate < 0.3:
            audit["overall_severity"] = "LOW"

        audit["real_model_names"] = list(set(audit["real_model_names"]))
        audit["forbidden_words_leaked"] = list(set(audit["forbidden_words_leaked"]))

        return audit


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
        scoring_profile_version: str = "v1",
        calibration_tag: str | None = None,
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

        DIMENSION_MIN_SAMPLES = {
            "adversarial_reasoning": 10,
            "coding": 10,
            "safety": 6,
            "consistency": 6,
            "knowledge": 3,
            "tool_use": 3,
        }

        dimension_warnings = []
        for dim, min_n in DIMENSION_MIN_SAMPLES.items():
            dim_cases = [r for r in case_results
                         if (r.case.dimension or r.case.category) == dim]
            actual_samples = sum(len(r.samples) for r in dim_cases)
            if actual_samples < min_n:
                dimension_warnings.append({
                    "dimension": dim,
                    "actual_samples": actual_samples,
                    "required_samples": min_n,
                    "warning": f"{dim} 维度样本量不足（{actual_samples}/{min_n}），分数置信度低",
                })

        # Overall completeness check
        MODE_EXPECTED = {"quick": 18, "standard": 62, "deep": 87}
        expected_total = MODE_EXPECTED.get(test_mode, 87)
        actual_total = len(case_results)
        completeness_ratio = actual_total / expected_total if expected_total > 0 else 1.0
        if completeness_ratio < 0.8:
            dimension_warnings.insert(0, {
                "dimension": "_overall",
                "actual_samples": actual_total,
                "required_samples": expected_total,
                "warning": (
                    f"仅完成 {actual_total}/{expected_total} 题（{completeness_ratio:.0%}），"
                    f"部分维度数据不足，分数仅供参考"
                ),
            })

        report = {
            "run_id": run_id,
            "target": {
                "base_url": base_url,
                "model": model_name,
                "test_mode": test_mode,
            },
            "scoring_profile_version": scoring_profile_version,
            "calibration_tag": calibration_tag,
            "uncertainty_flags": [],
            "warnings": dimension_warnings,
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

        # Extraction audit (for extraction mode or whenever extraction cases exist)
        ext_cases = [r for r in case_results if r.case.category == "extraction"]
        if ext_cases or (predetect and any(
            lr.layer in ("active_extraction", "multi_turn_extraction")
            for lr in (predetect.layer_results or [])
        )):
            extraction_audit = ExtractionAuditBuilder().build(predetect, case_results)
            report["extraction_audit"] = extraction_audit

        # Proxy latency analysis
        proxy_analysis = ProxyLatencyAnalyzer().analyze(
            case_results=case_results,
            claimed_model=model_name,
        )
        if proxy_analysis.get("status") != "insufficient_samples":
            report["proxy_latency_analysis"] = proxy_analysis
            proxy_conf = proxy_analysis.get("proxy_confidence", 0.0)
            if proxy_conf > 0:
                features["proxy_latency_confidence"] = proxy_conf
            if proxy_conf > 0.65 and "extraction_audit" in report:
                report["extraction_audit"]["evidence_chain"].append(
                    f"[TTFT] Proxy layer detected: confidence={proxy_analysis['proxy_confidence']}, "
                    f"signals={proxy_analysis['proxy_signals']}"
                )

        # Narrative summary (human-readable text, zero token cost)
        narrative = NarrativeBuilder().build(
            model_name=model_name,
            verdict=verdict,
            scorecard=scorecard,
            similarities=similarities,
            predetect=predetect,
            features=features,
            case_results=case_results,
        )
        report["narrative"] = narrative

        # Add run_id to report for export tools and other uses
        report["run_id"] = run_id

        report["evidence_chain"] = self._build_evidence_chain(
            predetect_result=predetect,
            case_results=case_results,
            features=features,
            verdict=verdict,
        )

        # Failed cases detail with failure reason attribution
        failed_detail = []
        for r in case_results:
            if r.pass_rate >= 1.0:
                continue
            detail = self._summarize_failure(r)
            failed_detail.append({
                "case_id": r.case.id,
                "name": r.case.name,
                "category": r.case.category,
                "judge_method": r.case.judge_method,
                "pass_rate": round(r.pass_rate, 3),
                "failure_reason": detail,
            })
        report["failed_cases_detail"] = failed_detail

        # Phase B: token ROI billing for cost/benefit transparency
        report["token_roi"] = self._build_token_roi(case_results)

        # v15 Phase 10: Token audit — aggregate token metrics with cache & budget info
        report["token_audit"] = self._build_token_audit(case_results, predetect)

        return report

    @staticmethod
    def _build_token_roi(case_results: list[CaseResult]) -> dict:
        """Build per-case and aggregate token ROI summary.

        ROI definition:
            roi = information_gain / max(total_tokens, 1)
        where information_gain ~= abs(pass_rate - 0.5) * 2 in [0, 1].
        """
        rows = []
        total_tokens = 0
        total_info = 0.0

        for r in case_results:
            case_tokens = 0
            for s in r.samples:
                t = s.response.usage_total_tokens
                if isinstance(t, (int, float)) and t > 0:
                    case_tokens += int(t)
            info_gain = abs((r.pass_rate or 0.0) - 0.5) * 2.0
            roi = info_gain / max(case_tokens, 1)

            total_tokens += case_tokens
            total_info += info_gain
            rows.append({
                "case_id": r.case.id,
                "name": r.case.name,
                "category": r.case.category,
                "dimension": r.case.dimension or r.case.category,
                "samples": len(r.samples),
                "pass_rate": round(r.pass_rate, 4),
                "information_gain": round(info_gain, 4),
                "total_tokens": case_tokens,
                "roi": round(roi, 6),
            })

        rows.sort(key=lambda x: x["roi"], reverse=True)
        for idx, row in enumerate(rows, start=1):
            row["roi_rank"] = idx

        avg_roi = (total_info / total_tokens) if total_tokens > 0 else 0.0
        return {
            "summary": {
                "total_cases": len(case_results),
                "total_tokens": total_tokens,
                "total_information_gain": round(total_info, 4),
                "average_roi": round(avg_roi, 6),
            },
            "per_case": rows,
        }

    @staticmethod
    def _build_token_audit(
        case_results: list[CaseResult],
        predetect: PreDetectionResult | None,
    ) -> dict:
        """v15 Phase 10: Build token audit with phase breakdown, cache metrics, budget."""
        total_tokens = 0
        predetect_tokens = 0
        testing_tokens = 0
        phase_tokens: dict[str, int] = {}
        per_category: dict[str, int] = {}

        for r in case_results:
            case_tokens = 0
            for s in r.samples:
                t = s.response.usage_total_tokens
                if isinstance(t, (int, float)) and t > 0:
                    case_tokens += int(t)
            total_tokens += case_tokens
            testing_tokens += case_tokens

            cat = r.case.category or "unknown"
            per_category[cat] = per_category.get(cat, 0) + case_tokens

        # Predetect token usage
        if predetect:
            predetect_tokens = predetect.total_tokens_used or 0

        # Try to get cache metrics from the global cache strategy
        cache_metrics = None
        try:
            from app.runner.cache_strategy import cache_strategy
            cache_metrics = cache_strategy.snapshot().to_dict()
        except Exception:
            pass

        # Phase breakdown
        phase_tokens["predetect"] = predetect_tokens
        phase_tokens["testing"] = testing_tokens
        phase_tokens["total"] = total_tokens + predetect_tokens

        return {
            "phase_breakdown": phase_tokens,
            "per_category": per_category,
            "cache_metrics": cache_metrics,
            "early_stop_info": None,  # Populated by pipeline if early stop triggered
        }

    def _build_evidence_chain(
        self,
        predetect_result: PreDetectionResult | None,
        case_results: list[CaseResult],
        features: dict[str, float],
        verdict: TrustVerdict | None,
    ) -> list[dict]:
        chain = []

        if predetect_result:
            for lr in predetect_result.layer_results:
                for ev in lr.evidence:
                    chain.append({
                        "phase": "predetect",
                        "layer": lr.layer,
                        "signal": ev,
                        "confidence": round((lr.confidence or 0) * 100, 1),
                        "severity": "critical" if (lr.confidence or 0) >= 0.85 else "warn" if (lr.confidence or 0) >= 0.6 else "info",
                    })

        NOTABLE_CASES = {
            "system_override_resist": ("身份系统提示覆盖抵抗", True),
            "model_name_probe": ("模型名称自报", True),
            "candy_shape_pool_original": ("约束推理（抓糖题）", True),
            "mice_two_rounds_original": ("约束推理（毒鼠题）", True),
            "python_function": ("Python 代码执行", True),
            "temperature_variance": ("Temperature 参数有效性", True),
        }
        results_by_name = {r.case.name: r for r in case_results}
        for case_name, (display, pass_is_good) in NOTABLE_CASES.items():
            r = results_by_name.get(case_name)
            if r:
                passed = r.pass_rate >= 0.5
                chain.append({
                    "phase": "testing",
                    "signal": display,
                    "case_id": case_name,
                    "pass_rate": round((r.pass_rate or 0) * 100, 1),
                    "severity": "info" if (passed == pass_is_good) else "warn",
                })

        ttft_proxy = features.get("ttft_proxy_signal", 0.0)
        if ttft_proxy > 0:
            chain.append({
                "phase": "timing",
                "signal": "首Token时延双峰分布",
                "value": features.get("ttft_cluster_gap_ms", 0),
                "unit": "ms_gap",
                "severity": "warn",
            })
        if features.get("latency_length_correlated", 1.0) < 0.5:
            chain.append({
                "phase": "timing",
                "signal": "延迟与输出长度无相关性",
                "severity": "warn",
            })

        if verdict:
            chain.append({
                "phase": "verdict",
                "signal": verdict.label,
                "confidence_real": verdict.confidence_real,
                "level": verdict.level,
                "severity": "critical" if verdict.level in ("fake", "high_risk")
                            else "warn" if verdict.level == "suspicious"
                            else "info",
            })

        return chain

    @staticmethod
    def _summarize_failure(result: CaseResult) -> str:
        samples = [s for s in result.samples if not s.judge_passed]
        if not samples:
            return "偶发失败（部分采样通过）"

        detail = samples[0].judge_detail or {}
        method = result.case.judge_method or ""

        if method == "exact_match":
            expected = detail.get("expected", "")
            got = str(detail.get("got", ""))[:60]
            return f"期望精确匹配 '{expected}'，实际输出：'{got}'"
        elif method == "regex_match":
            pattern = detail.get("pattern", "")
            return f"正则 '{pattern}' 未匹配"
        elif method == "constraint_reasoning":
            return detail.get("failure_reason", "约束推理未满足关键条件")
        elif method == "code_execution":
            return detail.get("error", "代码执行失败")
        elif method == "semantic_judge":
            kc = detail.get("keyword_coverage") or 0
            return f"语义评判未通过，关键覆盖率 {kc:.0%}"
        elif detail.get("error"):
            return f"错误：{detail['error']}"

        return "判定未通过（详见 judge_detail）"
