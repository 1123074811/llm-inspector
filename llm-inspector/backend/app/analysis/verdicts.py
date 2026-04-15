"""
analysis/verdicts.py — VerdictEngine

Hard-rule verdict assessment engine.
Extracted from pipeline.py to keep individual files under ~400 lines.
"""
from __future__ import annotations

from app.core.schemas import CaseResult, TrustVerdict, ScoreCard, PreDetectionResult, SimilarityResult
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


# ── Verdict Engine (v2) ──────────────────────────────────────────────────────

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

    def _load_top_models(self) -> list[str]:
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
        scorecard: ScoreCard,
        similarities: list[SimilarityResult],
        predetect: PreDetectionResult | None,
        features: dict[str, float],
        case_results: list[CaseResult] | None = None,
    ) -> TrustVerdict:
        f = features.get
        reasons: list[str] = []
        signal_details: dict = {}

        sim_score = 0.0
        if similarities:
            top_sim = similarities[0].similarity_score
            sim_score = min(100.0, top_sim * 100)
            signal_details["behavioral_similarity"] = round(sim_score, 1)
            if sim_score >= 85:
                reasons.append(f"行为向量与 {similarities[0].benchmark_name} 高度吻合（{sim_score:.1f}分）")
            elif sim_score >= 65:
                reasons.append(f"行为向量与 {similarities[0].benchmark_name} 部分吻合（{sim_score:.1f}分）")
            else:
                reasons.append(f"行为向量与所有基准模型相似度均较低（最高 {sim_score:.1f}分）")

        cap_score = scorecard.capability_score
        signal_details["capability_score"] = round(cap_score, 1)

        ttft_proxy = f("ttft_proxy_signal", 0.0)
        lat_corr = f("latency_length_correlated", 1.0)
        temp_ok = f("temperature_param_effective", 1.0)
        proxy_lat_conf = f("proxy_latency_confidence", 0.0)
        timing_score = (
            (1.0 - max(ttft_proxy, proxy_lat_conf)) * 40
            + lat_corr * 35
            + temp_ok * 25
        )
        signal_details["timing_fingerprint"] = round(timing_score, 1)
        if ttft_proxy > 0:
            ttft_gap = f("ttft_cluster_gap_ms", 0.0)
            reasons.append(f"首Token时延分布异常，检测到可能的中转路由层（两簇间距 {ttft_gap:.0f}ms）")
        if temp_ok < 0.5:
            reasons.append("temperature 参数无效，可能存在代理层屏蔽参数")

        consistency = scorecard.consistency_score
        signal_details["consistency_score"] = round(consistency, 1)
        if consistency < 50:
            reasons.append(f"一致性分 {consistency:.1f}：确定性模式下结果不稳定，疑似路由到不同后端")

        protocol = scorecard.protocol_score
        token_consistent = f("token_count_consistent", 0.5) * 100
        proto_score = round(protocol * 0.6 + token_consistent * 0.4, 1)
        signal_details["protocol_compliance"] = proto_score
        if token_consistent < 40:
            reasons.append("Token 计数与声称模型 tokenizer 不符")

        predetect_score = 50.0
        if predetect and predetect.success:
            predetect_score = (predetect.confidence or 0) * 100
            reasons.append(
                f"预检测识别为 {predetect.identified_as}（置信度 {predetect.confidence:.0%}）"
            )
        signal_details["predetect_identity"] = round(predetect_score, 1)

        WEIGHTS = {
            "behavioral_similarity": 0.30,
            "capability_score":      0.20,
            "timing_fingerprint":    0.20,
            "consistency_score":     0.15,
            "protocol_compliance":   0.10,
            "predetect_identity":    0.05,
        }
        scores_map = {
            "behavioral_similarity": sim_score,
            "capability_score":      cap_score,
            "timing_fingerprint":    timing_score,
            "consistency_score":     consistency,
            "protocol_compliance":   proto_score,
            "predetect_identity":    predetect_score,
        }
        confidence_real = sum(
            scores_map[k] * w for k, w in WEIGHTS.items()
        )
        confidence_real = round(min(100.0, max(0.0, confidence_real)), 1)

        # ── 信号矛盾调和：预检测 vs 行为相似度 ──
        predetect_name = ""
        if predetect and predetect.success and predetect.identified_as:
            predetect_name = predetect.identified_as.lower()
        sim_name = similarities[0].benchmark_name.lower() if similarities else ""

        if predetect_name and sim_name and sim_score >= 70:
            # Check if the two signals agree on the same model family
            _family_aliases = {
                "deepseek": ["deepseek"],
                "minimax": ["minimax", "abab"],
                "claude": ["claude", "anthropic"],
                "gpt": ["gpt", "openai", "chatgpt"],
                "qwen": ["qwen", "tongyi"],
                "gemini": ["gemini", "bard"],
                "llama": ["llama", "meta"],
                "glm": ["glm", "chatglm", "zhipu"],
                "mistral": ["mistral", "mixtral"],
                "yi": ["yi", "零一"],
                "moonshot": ["moonshot", "kimi"],
                "baichuan": ["baichuan"],
            }

            def _to_family(name: str) -> str:
                for fam, aliases in _family_aliases.items():
                    if any(a in name for a in aliases):
                        return fam
                return name

            pre_fam = _to_family(predetect_name)
            sim_fam = _to_family(sim_name)

            if pre_fam != sim_fam:
                signal_details["signal_conflict"] = {
                    "predetect": predetect.identified_as if predetect else None,
                    "similarity": similarities[0].benchmark_name if similarities else None,
                }
                reasons.append(
                    f"⚠ 信号矛盾：预检测识别为 {predetect.identified_as}，"
                    f"但行为特征最匹配 {similarities[0].benchmark_name}（{sim_score:.1f}分）"
                    f"——可能存在多层路由或模型混合部署"
                )

        adv_spoof = f("adversarial_spoof_signal_rate", 0.0)
        if adv_spoof > 0.5:
            confidence_real = min(confidence_real, self.HARD_RULES["adv_spoof_cap"])
            reasons.append(f"对抗推理检测到模板套用（信号率 {adv_spoof:.0%}），置信度强制下调")

        # ── 硬规则：能力-声称不匹配检测 ──
        difficulty_ceiling = f("difficulty_ceiling", 0.5)
        claimed = (predetect.identified_as or "").lower() if predetect else ""
        top_models = self.TOP_MODELS

        if any(m in claimed for m in top_models) and difficulty_ceiling < self.HARD_RULES["difficulty_ceiling_min"]:
            confidence_real = min(confidence_real, self.HARD_RULES["difficulty_cap"])
            reasons.append(
                f"声称为顶级模型但能力天花板仅 {difficulty_ceiling:.2f}，"
                f"梯度难度测试显示推理能力与声称不符"
            )

        # ── 硬规则：行为不变性检测 ──
        beh_inv = scorecard.behavioral_invariant_score
        if beh_inv is not None and beh_inv < self.HARD_RULES["behavioral_invariant_min"]:
            confidence_real = min(confidence_real, self.HARD_RULES["behavioral_invariant_cap"])
            reasons.append(
                f"行为不变性分 {beh_inv:.1f}：同构题换皮后结果不一致，"
                f"疑似模板匹配而非真实推理"
            )

        # ── 硬规则：编程能力与声称等级不符 ──
        if scorecard.coding_score is not None and scorecard.coding_score < 10 and any(m in claimed for m in top_models):
            confidence_real = min(confidence_real, self.HARD_RULES["coding_zero_cap"])
            reasons.append("编程能力评分接近零，与声称的模型等级严重不符")

        # ── v3 硬规则：提取攻击泄露真实模型名 ──
        # Only trigger if extraction probes exposed a DIFFERENT real model identity
        # (not just leaking our test system prompt, which any model can do).
        extraction_resistance = getattr(scorecard, 'breakdown', {}).get('extraction_resistance', 100)
        if extraction_resistance is None:
            extraction_resistance = 100
        _found_identity_mismatch = False
        for r in (case_results or []):
            if hasattr(r, 'case') and r.case.category == "extraction":
                for s in r.samples:
                    d = s.judge_detail or {}
                    if d.get("leak_type") == "real_model_name_exposed":
                        _found_identity_mismatch = True
                        break

        if _found_identity_mismatch:
            confidence_real = min(confidence_real, self.HARD_RULES["identity_exposed_cap"])
            reasons.append("提取攻击暴露了与声称不同的真实模型身份，置信度强制下调")
        elif extraction_resistance < self.HARD_RULES["extraction_weak_threshold"]:
            # Very weak resistance but no identity mismatch — suspicious but not definitive
            confidence_real = min(confidence_real, self.HARD_RULES["extraction_weak_cap"])
            reasons.append(f"提取攻击抵抗度极低（{extraction_resistance:.0f}分），系统提示词容易被提取")

        # ── v3 硬规则：tokenizer 指纹不匹配 ──
        fingerprint_match = getattr(scorecard, 'breakdown', {}).get('fingerprint_match', 100)
        if fingerprint_match < self.HARD_RULES["fingerprint_mismatch_threshold"] and claimed:
            confidence_real = min(confidence_real, self.HARD_RULES["fingerprint_mismatch_cap"])
            reasons.append("tokenizer/行为指纹与声称模型不符")

        t = self.VERDICT_THRESHOLDS
        if confidence_real >= t["trusted"]:
            level, label = "trusted", "可信 / Trusted"
        elif confidence_real >= t["suspicious"]:
            level, label = "suspicious", "轻度可疑 / Suspicious"
        elif confidence_real >= t["high_risk"]:
            level, label = "high_risk", "高风险 / High Risk"
        else:
            level, label = "fake", "疑似假模型 / Likely Fake"

        if not reasons:
            reasons.append("没有检测到明显异常信号")

        return TrustVerdict(
            level=level,
            label=label,
            total_score=scorecard.total_score,
            reasons=reasons,
            confidence_real=confidence_real,
            signal_details=signal_details,
        )
