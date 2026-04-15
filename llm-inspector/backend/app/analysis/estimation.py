"""
analysis/estimation.py — RiskEngine, ThetaEstimator, UncertaintyEstimator,
                          PercentileMapper, PairwiseEngine

Risk assessment and theta (IRT ability) estimation.
Extracted from pipeline.py to keep individual files under ~280 lines.
"""
from __future__ import annotations

import math
import random

import numpy as np

from app.core.schemas import (
    CaseResult, RiskAssessment, ThetaReport, ThetaDimensionEstimate, SimilarityResult,
)
from app.core.config import settings
from app.core.logging import get_logger
from app.core.provenance import get_provenance_tracker, DataProvenance

logger = get_logger(__name__)


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
        predetect,
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

    def estimate(self, case_results, item_stats: dict[str, dict]) -> ThetaReport:
        tracker = get_provenance_tracker()
        by_dim: dict[str, list[float]] = {}
        total_samples = 0
        calibrated_items = 0
        
        for r in case_results:
            dim = (r.case.dimension or r.case.category or "unknown").lower()
            st = item_stats.get(r.case.id, {})
            b = float(st.get("irt_b", 0.0) or 0.0)
            
            # Check if item has IRT calibration data
            if st.get("irt_a") and st.get("irt_b"):
                calibrated_items += 1
            
            for s in r.samples:
                if s.judge_passed is None:
                    continue
                x = 1.0 if s.judge_passed else 0.0
                by_dim.setdefault(dim, []).append((x, b))
                total_samples += 1

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
            
            # Register dimension theta provenance for v12
            theta_key = f"theta_{dim}_{settings.CALIBRATION_VERSION}"
            dim_provenance = DataProvenance.from_irt_calibration(
                case_id=dim,
                calibration_version=settings.CALIBRATION_VERSION,
                sample_size=len(obs),
                confidence=0.8 if len(obs) >= 5 else 0.6
            )
            tracker.register(theta_key, dim_provenance)

        if not all_thetas:
            # Register fallback provenance for global theta
            fallback_provenance = DataProvenance.create_fallback(
                "theta_global", 
                "insufficient judged samples"
            )
            tracker.register(f"theta_global_{settings.CALIBRATION_VERSION}", fallback_provenance)
            
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
        
        # Register global theta provenance for v12
        global_theta_key = f"theta_global_{settings.CALIBRATION_VERSION}"
        global_provenance = DataProvenance.from_irt_calibration(
            case_id="global",
            calibration_version=settings.CALIBRATION_VERSION,
            sample_size=total_samples,
            confidence=0.9 if total_samples >= 30 else 0.7
        )
        tracker.register(global_theta_key, global_provenance)
        
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
    def apply_ci(self, theta_report: ThetaReport, case_results,
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
