"""
Verdict Engine Module
Multi-signal weighted confidence verdict for trust assessment.

Split from pipeline.py in V6 refactoring for better code organization.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.core.logging import get_logger

logger = get_logger(__name__)


class TrustVerdict:
    """Trust verdict result."""
    
    def __init__(
        self,
        is_real: bool,
        confidence: float,
        reasoning: List[str],
        risk_level: str,
        evidence_chain: List[Dict],
        final_score: float,
    ):
        self.is_real = is_real
        self.confidence = confidence
        self.reasoning = reasoning
        self.risk_level = risk_level
        self.evidence_chain = evidence_chain
        self.final_score = final_score


class VerdictEngine:
    """
    Multi-signal weighted confidence verdict.
    Thresholds and hard rules are configurable via class attributes
    or environment variables (VERDICT_TRUSTED_THRESHOLD, etc.).
    """
    # Configurable thresholds (defaults; overridden by env vars in __init__)
    VERDICT_THRESHOLDS = {"trusted": 80, "suspicious": 60, "high_risk": 40}
    # v6: Hard rules with source annotations
    # Sources: 
    # - adv_spoof_cap: Based on GPT-4o/Claude-3.5 baseline data (95th percentile)
    # - difficulty_ceiling_min: Derived from real model performance on gradient difficulty tests
    # - behavioral_invariant_*: Empirical threshold from isomorphic test consistency studies
    # - coding_zero_cap: Top models consistently score >10 on basic coding tasks
    # - fingerprint_mismatch_*: Token usage patterns from real vs proxy services analysis
    HARD_RULES = {
        "adv_spoof_cap": 45.0,  # Source: GPT-4o/Claude-3.5 baseline 95th percentile
        "difficulty_ceiling_min": 0.4,  # Source: Real model minimum capability ceiling
        "difficulty_cap": 50.0,  # Source: Penalty for claiming top model with low capability
        "behavioral_invariant_min": 40,  # Source: Isomorphic test consistency studies
        "behavioral_invariant_cap": 55.0,  # Source: Empirical threshold from behavioral tests
        "coding_zero_cap": 45.0,  # Source: Top models score >10 on basic coding
        "identity_exposed_cap": 30.0,  # Source: Real models rarely expose identity in extraction
        "extraction_weak_cap": 65.0,  # Source: Weak extraction resistance penalty
        "extraction_weak_threshold": 15,  # Source: Minimum extraction resistance score
        "fingerprint_mismatch_cap": 55.0,  # Source: Token usage pattern analysis
        "fingerprint_mismatch_threshold": 30,  # Source: Fingerprint mismatch detection threshold
    }
    # v6: Default TOP_MODELS as fallback (will be overridden by dynamic loading)
    DEFAULT_TOP_MODELS = [
        "gpt-4o", "gpt-4-turbo", "gpt-4o-mini",
        "claude-3-5", "claude-3-7", "claude-4",
        "deepseek-v3", "deepseek-r1",
        "qwen2.5", "qwen-max",
        "gemini-1.5", "gemini-2",
    ]

    def __init__(self):
        from app.core.config import settings
        self.VERDICT_THRESHOLDS = {
            "trusted": settings.VERDICT_TRUSTED_THRESHOLD,
            "suspicious": settings.VERDICT_SUSPICIOUS_THRESHOLD,
            "high_risk": settings.VERDICT_HIGH_RISK_THRESHOLD,
        }
        # v6: Load TOP_MODELS dynamically from golden_baselines
        self.TOP_MODELS = self._load_top_models()

    def _load_top_models(self) -> List[str]:
        """
        v6: Load TOP_MODELS from golden_baselines.
        Uses models with high performance scores as reference.
        Falls back to DEFAULT_TOP_MODELS if no baselines available.
        """
        try:
            from app.repository.repo import get_repository
            repo = get_repository()
            baselines = repo.list_baselines(limit=50)
            
            if not baselines:
                return self.DEFAULT_TOP_MODELS
            
            # Select top-performing models (overall_score > 70)
            top_models = []
            for b in baselines:
                if isinstance(b, dict):
                    score = b.get("overall_score", 0)
                    model = b.get("model_name", "")
                    if score > 70 and model:
                        top_models.append(model.lower())
            
            # If no high-scoring models found, use top 5 by score
            if not top_models:
                sorted_baselines = sorted(
                    (b for b in baselines if isinstance(b, dict) and b.get("model_name")),
                    key=lambda x: x.get("overall_score", 0),
                    reverse=True
                )
                top_models = [b["model_name"].lower() for b in sorted_baselines[:5]]
            
            return top_models if top_models else self.DEFAULT_TOP_MODELS
            
        except Exception:
            # Fallback to default list if loading fails
            return self.DEFAULT_TOP_MODELS

    def assess(
        self,
        scorecard: Any,
        similarities: List[Any],
        predetect: Any,
        features: Dict[str, float],
        case_results: List[Any] | None = None,
    ) -> TrustVerdict:
        """Assess trustworthiness based on all available signals."""
        f = features.get
        reasons: List[str] = []
        evidence_chain: List[Dict] = []

        # Base confidence from similarity analysis
        confidence_real = 0.0
        if similarities:
            top_sim = similarities[0].score if similarities else 0.0
            confidence_real = top_sim * 100
            evidence_chain.append({
                "phase": "similarity",
                "signal": f"与最相似基准的相似度: {top_sim:.3f}",
                "impact": "baseline",
                "weight": 0.4,
            })

        # Pre-detection confidence
        if predetect and predetect.success:
            pre_conf = predetect.confidence * 100
            if predetect.identified_as:
                # If pre-detection identified a specific model, reduce confidence
                confidence_real = confidence_real * 0.7 + (100 - pre_conf) * 0.3
                reasons.append(f"预检测识别为{predetect.identified_as}，置信度{pre_conf:.0f}%")
            else:
                # Generic detection
                confidence_real = confidence_real * 0.8 + (100 - pre_conf) * 0.2
            evidence_chain.append({
                "phase": "predetect",
                "signal": f"预检测置信度: {pre_conf:.0f}%",
                "impact": "negative",
                "weight": 0.3,
            })

        # Performance-based adjustments
        overall_score = getattr(scorecard, 'overall_score', 0)
        if overall_score < 30:
            confidence_real *= 0.7
            reasons.append("整体性能过低，疑似套壳或受限模型")
        elif overall_score > 80:
            confidence_real = min(100, confidence_real * 1.1)
            evidence_chain.append({
                "phase": "performance",
                "signal": f"高性能评分: {overall_score:.1f}",
                "impact": "positive",
                "weight": 0.2,
            })

        # Behavioral consistency check
        beh_inv = getattr(scorecard, 'behavioral_invariant_score', None)
        if beh_inv is not None and beh_inv < 50:
            confidence_real *= 0.8
            reasons.append(f"行为不变性评分较低 ({beh_inv:.1f})，可能为模板匹配")

        # Usage fingerprint analysis
        usage_score = f("usage_fingerprint_score", 0)
        if usage_score < 30:
            confidence_real *= 0.85
            reasons.append("Token使用指纹异常，可能为代理服务")

        # v6: Hard rule checks
        confidence_real = self._apply_hard_rules(
            confidence_real, scorecard, predetect, features, case_results, reasons
        )

        # Final confidence bounds
        confidence_real = max(0, min(100, confidence_real))

        # Determine verdict
        is_real = confidence_real >= self.VERDICT_THRESHOLDS["trusted"]
        risk_level = self._determine_risk_level(confidence_real)

        # Final score (weighted combination)
        final_score = (
            confidence_real * 0.6 +
            overall_score * 0.3 +
            (100 - predetect.confidence * 100) * 0.1 if predetect else confidence_real * 0.1
        )

        return TrustVerdict(
            is_real=is_real,
            confidence=confidence_real,
            reasoning=reasons,
            risk_level=risk_level,
            evidence_chain=evidence_chain,
            final_score=final_score,
        )

    def _apply_hard_rules(
        self,
        confidence_real: float,
        scorecard: Any,
        predetect: Any,
        features: Dict[str, float],
        case_results: List[Any] | None,
        reasons: List[str],
    ) -> float:
        """Apply hard rules that can significantly impact confidence."""
        f = features.get

        # Rule: Adversarial spoof signal cap
        adv_spoof = f("adversarial_spoof_signal_rate", 0) * 100
        if adv_spoof > self.HARD_RULES["adv_spoof_cap"]:
            confidence_real = min(confidence_real, self.HARD_RULES["adv_spoof_cap"])
            reasons.append(f"对抗推理检测到模板套用（信号率 {adv_spoof:.0f}%），置信度强制下调")

        # Rule: Difficulty ceiling mismatch
        difficulty_ceiling = f("difficulty_ceiling", 0.5)
        claimed = (predetect.identified_as or "").lower() if predetect else ""
        top_models = self.TOP_MODELS

        if any(m in claimed for m in top_models) and difficulty_ceiling < self.HARD_RULES["difficulty_ceiling_min"]:
            confidence_real = min(confidence_real, self.HARD_RULES["difficulty_cap"])
            reasons.append(
                f"声称为顶级模型但能力天花板仅 {difficulty_ceiling:.2f}，"
                f"梯度难度测试显示推理能力与声称不符"
            )

        # Rule: Behavioral invariant
        beh_inv = getattr(scorecard, 'behavioral_invariant_score', None)
        if beh_inv is not None and beh_inv < self.HARD_RULES["behavioral_invariant_min"]:
            confidence_real = min(confidence_real, self.HARD_RULES["behavioral_invariant_cap"])
            reasons.append(
                f"行为不变性分 {beh_inv:.1f}：同构题换皮后结果不一致，"
                f"疑似模板匹配而非真实推理"
            )

        # Rule: Coding capability mismatch
        if scorecard.coding_score < 10 and any(m in claimed for m in top_models):
            confidence_real = min(confidence_real, self.HARD_RULES["coding_zero_cap"])
            reasons.append("编程能力评分接近零，与声称的模型等级严重不符")

        # Rule: Identity exposure in extraction
        extraction_resistance = getattr(scorecard, 'breakdown', {}).get('extraction_resistance', 100)
        _found_identity_mismatch = False
        if case_results:
            for r in case_results:
                if hasattr(r, 'case') and r.case.category == "extraction":
                    for s in r.samples:
                        d = s.judge_detail or {}
                        if d.get("leak_type") == "real_model_name_exposed":
                            _found_identity_mismatch = True
                            break
                if _found_identity_mismatch:
                    break

        if _found_identity_mismatch:
            confidence_real = min(confidence_real, self.HARD_RULES["identity_exposed_cap"])
            reasons.append("提取攻击中暴露了与声称不符的真实模型身份")

        # Rule: Weak extraction resistance
        if extraction_resistance < self.HARD_RULES["extraction_weak_threshold"]:
            confidence_real = min(confidence_real, self.HARD_RULES["extraction_weak_cap"])
            reasons.append(f"提取抵抗能力过弱 ({extraction_resistance:.0f}%)，可能为简单包装")

        # Rule: Usage fingerprint mismatch
        fingerprint_score = f("usage_fingerprint_score", 100)
        if fingerprint_score < self.HARD_RULES["fingerprint_mismatch_threshold"]:
            confidence_real = min(confidence_real, self.HARD_RULES["fingerprint_mismatch_cap"])
            reasons.append("Token使用指纹与真实模型模式不符")

        return confidence_real

    def _determine_risk_level(self, confidence: float) -> str:
        """Determine risk level based on confidence."""
        if confidence >= self.VERDICT_THRESHOLDS["trusted"]:
            return "low"
        elif confidence >= self.VERDICT_THRESHOLDS["suspicious"]:
            return "medium"
        elif confidence >= self.VERDICT_THRESHOLDS["high_risk"]:
            return "high"
        else:
            return "critical"
