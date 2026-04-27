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
    # v13: Hard-rule thresholds are loaded via `_rule()` → SRC[...] with fallback.
    # Source-backed rules live in SOURCES.yaml (ids prefixed with `verdict.`).
    # Derived caps (e.g. the 50.0 penalty applied when a top-claimed model fails
    # a threshold) are kept as hard-coded fallbacks below pending data fitting.
    _RULE_FALLBACKS = {
        "adv_spoof_cap": 45.0,                  # SRC: verdict.adv_spoof_cap
        "difficulty_ceiling_min": 0.4,          # SRC: verdict.difficulty_ceiling_min
        "difficulty_cap": 50.0,                 # SRC: verdict.difficulty_cap
        "behavioral_invariant_min": 40,         # SRC: verdict.behavioral_invariant_min
        "behavioral_invariant_cap": 55.0,       # SRC: verdict.behavioral_invariant_cap
        "coding_zero_cap": 45.0,                # SRC: verdict.coding_zero_cap
        "identity_exposed_cap": 30.0,           # SRC: verdict.identity_exposed_cap
        "extraction_weak_cap": 65.0,            # SRC: verdict.extraction_weak_cap
        "extraction_weak_threshold": 15,        # SRC: verdict.extraction_weak_threshold
        "fingerprint_mismatch_cap": 55.0,       # SRC: verdict.fingerprint_mismatch_cap
        "fingerprint_mismatch_threshold": 30,   # SRC: verdict.fingerprint_mismatch_threshold
    }
    _SRC_KEY_MAP = {
        "adv_spoof_cap": "verdict.adv_spoof_cap",
        "difficulty_ceiling_min": "verdict.difficulty_ceiling_min",
        "difficulty_cap": "verdict.difficulty_cap",
        "behavioral_invariant_min": "verdict.behavioral_invariant_min",
        "behavioral_invariant_cap": "verdict.behavioral_invariant_cap",
        "coding_zero_cap": "verdict.coding_zero_cap",
        "identity_exposed_cap": "verdict.identity_exposed_cap",
        "extraction_weak_threshold": "verdict.extraction_weak_threshold",
        "extraction_weak_cap": "verdict.extraction_weak_cap",
        "fingerprint_mismatch_threshold": "verdict.fingerprint_mismatch_threshold",
        "fingerprint_mismatch_cap": "verdict.fingerprint_mismatch_cap",
    }
    # v6: Default TOP_MODELS as fallback (will be overridden by dynamic loading)
    # v15 fix: expanded to cover 2025 model releases (deepseek-v4, qwen3, etc.)
    DEFAULT_TOP_MODELS = [
        # OpenAI
        "gpt-4o", "gpt-4-turbo", "gpt-4o-mini", "gpt-4.1", "gpt-5",
        # Anthropic
        "claude-3-5", "claude-3-7", "claude-4", "claude-4-5",
        # DeepSeek
        "deepseek-v3", "deepseek-r1", "deepseek-v4", "deepseek-r2",
        # Qwen
        "qwen2.5", "qwen-max", "qwen3", "qwen3-235b",
        # Google
        "gemini-1.5", "gemini-2", "gemini-2.5",
        # Other frontier
        "llama-3.1", "llama-3.3", "llama-4", "mistral-large",
        "glm-4", "glm-4.5", "yi-large", "moonshot",
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

    @property
    def HARD_RULES(self) -> dict:
        """
        v13 compat: materialize the rule dict from SRC + fallbacks.
        Preserved so existing callers / tests that introspect
        ``engine.HARD_RULES`` continue to work.
        """
        return {name: self._rule(name) for name in self._RULE_FALLBACKS}

    def _rule(self, name: str):
        """
        v13: Resolve a hard-rule threshold, preferring SRC[...] from SOURCES.yaml
        with a hard-coded fallback so the engine remains usable if the registry
        is unavailable or an entry is missing.
        """
        fallback = self._RULE_FALLBACKS.get(name)
        src_key = self._SRC_KEY_MAP.get(name)
        if src_key is None:
            return fallback
        try:
            from app._data import SRC
            return SRC[src_key].value
        except Exception:
            return fallback

    def _sim_threshold(self, band: str, fallback: float) -> float:
        """
        v13: Similarity band thresholds from SRC (0-100 scale).
        band="match" → SRC.similarity.match_cosine_threshold (fallback 0.90)
        band="suspicious" → SRC.similarity.suspicious_cosine_threshold (fallback 0.75)
        """
        key = {
            "match": "similarity.match_cosine_threshold",
            "suspicious": "similarity.suspicious_cosine_threshold",
        }.get(band)
        if key is None:
            return fallback
        try:
            from app._data import SRC
            return float(SRC[key].value) * 100.0
        except Exception:
            return fallback

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
            # v13: Similarity bands from SRC (fallbacks preserve v12 behaviour band shape,
            # but numeric cutoffs follow SOURCES.yaml: match=90, suspicious=75).
            _match_cut = self._sim_threshold("match", 90.0)
            _susp_cut = self._sim_threshold("suspicious", 75.0)
            if sim_score >= _match_cut:
                reasons.append(f"行为向量与 {similarities[0].benchmark_name} 高度吻合（{sim_score:.1f}分）")
            elif sim_score >= _susp_cut:
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

        # ── v16 Phase 1.5: Official Endpoint Fast-Path ──
        # If the API endpoint was verified as official via TLS triple-consistency,
        # this is strong evidence that the model is genuine. Boost confidence and
        # relax hard-rule caps accordingly.
        official_endpoint_info = None
        if predetect and hasattr(predetect, 'routing_info') and predetect.routing_info:
            official_endpoint_info = predetect.routing_info.get("official_endpoint")

        official_verified = False
        official_provider = None
        official_confidence = 0.0
        if official_endpoint_info and isinstance(official_endpoint_info, dict):
            official_verified = official_endpoint_info.get("verified", False)
            official_provider = official_endpoint_info.get("provider")
            official_confidence = official_endpoint_info.get("confidence", 0.0)

        if official_verified:
            # Official endpoint verified: boost confidence_real
            boost = official_confidence * 15  # max 15 points boost at confidence=1.0
            confidence_real = min(100.0, confidence_real + boost)
            signal_details["official_endpoint"] = {
                "verified": True,
                "provider": official_provider,
                "confidence": round(official_confidence, 3),
                "boost": round(boost, 1),
            }
            reasons.append(
                f"✅ 官方API端点验证通过（{official_provider}，置信度 {official_confidence:.0%}），"
                f"置信度提升 {boost:.1f} 分"
            )
        else:
            signal_details["official_endpoint"] = {
                "verified": False,
                "provider": official_provider,
                "confidence": round(official_confidence, 3),
            }

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
            confidence_real = min(confidence_real, float(self._rule("adv_spoof_cap")))
            reasons.append(f"对抗推理检测到模板套用（信号率 {adv_spoof:.0%}），置信度强制下调")

        # ── 硬规则：能力-声称不匹配检测 ──
        difficulty_ceiling = f("difficulty_ceiling", 0.5)
        claimed = (predetect.identified_as or "").lower() if predetect else ""
        top_models = self.TOP_MODELS
        # v15: track whether any evidence-backed hard rule fires (used for uncertain verdict)
        _hard_rule_fired = False

        if any(m in claimed for m in top_models) and difficulty_ceiling < float(self._rule("difficulty_ceiling_min")):
            confidence_real = min(confidence_real, float(self._rule("difficulty_cap")))
            _hard_rule_fired = True
            reasons.append(
                f"声称为顶级模型但能力天花板仅 {difficulty_ceiling:.2f}，"
                f"梯度难度测试显示推理能力与声称不符"
            )

        # ── 硬规则：行为不变性检测 ──
        beh_inv = scorecard.behavioral_invariant_score
        if beh_inv is not None and beh_inv < float(self._rule("behavioral_invariant_min")):
            # v16: Relax cap when official endpoint is verified — low invariance
            # on an official API is more likely ability deficit than spoofing
            inv_cap = float(self._rule("behavioral_invariant_cap"))
            if official_verified:
                inv_cap = min(100.0, inv_cap + 15.0)  # 55 → 70
            confidence_real = min(confidence_real, inv_cap)
            _hard_rule_fired = True
            cap_note = "（官方API已验证，cap放宽）" if official_verified else ""
            reasons.append(
                f"行为不变性分 {beh_inv:.1f}：同构题换皮后结果不一致，"
                f"疑似模板匹配而非真实推理{cap_note}"
            )

        # ── 硬规则：编程能力与声称等级不符 ──
        if scorecard.coding_score is not None and scorecard.coding_score < 10 and any(m in claimed for m in top_models):
            confidence_real = min(confidence_real, float(self._rule("coding_zero_cap")))
            _hard_rule_fired = True
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
            confidence_real = min(confidence_real, float(self._rule("identity_exposed_cap")))
            _hard_rule_fired = True
            reasons.append("提取攻击暴露了与声称不同的真实模型身份，置信度强制下调")
        elif extraction_resistance < float(self._rule("extraction_weak_threshold")):
            # Very weak resistance but no identity mismatch — suspicious but not definitive
            # v16: Relax cap when official endpoint is verified
            ext_cap = float(self._rule("extraction_weak_cap"))
            if official_verified:
                ext_cap = min(100.0, ext_cap + 15.0)  # 65 → 80
            confidence_real = min(confidence_real, ext_cap)
            _hard_rule_fired = True
            cap_note = "（官方API已验证，cap放宽）" if official_verified else ""
            reasons.append(f"提取攻击抵抗度极低（{extraction_resistance:.0f}分），系统提示词容易被提取{cap_note}")

        # ── v3 硬规则：tokenizer 指纹不匹配 ──
        # v15 fix: only apply when claimed model is in TOP_MODELS; for library-unknown models
        # a low fingerprint_match simply means we have no reference, not that it's fake.
        fingerprint_match = getattr(scorecard, 'breakdown', {}).get('fingerprint_match', 100)
        _claimed_is_known = bool(claimed) and any(m in claimed for m in top_models)
        if fingerprint_match < float(self._rule("fingerprint_mismatch_threshold")) and _claimed_is_known:
            confidence_real = min(confidence_real, float(self._rule("fingerprint_mismatch_cap")))
            _hard_rule_fired = True
            reasons.append("tokenizer/行为指纹与声称模型不符")

        t = self.VERDICT_THRESHOLDS

        # ── v15 fix: uncertain verdict when evidence is insufficient ──
        # Condition: no hard rule fired AND similarity has no strong match AND
        # predetect couldn't identify the model. This means "don't know" ≠ "high risk".
        _no_baseline_match = sim_score < 40.0 and not similarities
        _sim_no_match = sim_score < 40.0  # even with baselines, very low similarity
        _predetect_no_id = not (predetect and predetect.success and predetect.confidence
                                and predetect.confidence >= 0.5)
        _is_uncertain = (
            not _hard_rule_fired
            and _sim_no_match
            and _predetect_no_id
            and confidence_real < t["suspicious"]
        )

        if _is_uncertain:
            level, label = "uncertain", "证据不足 / Insufficient Evidence"
            signal_details["uncertain_reason"] = (
                "相似度库中无匹配基准，预检测无法识别模型家族，"
                "无法给出可信度判断（非高风险信号）"
            )
            if not any("证据不足" in r or "无法识别" in r for r in reasons):
                reasons.append(
                    "参考库中无该模型的基准数据，无法完成相似度比对；"
                    "建议将本次结果标记为基准后重新评测，或等待库更新"
                )
        elif confidence_real >= t["trusted"]:
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


# ── v16 Phase 11: Bayesian Evidence-Weighted Verdict Engine ──────────────────

import math
from dataclasses import dataclass, field as dc_field


@dataclass
class EvidenceItem:
    """A single piece of evidence in the Bayesian log-odds model."""
    rule_id: str
    direction: str                        # "up" (toward real) | "down" (toward fake)
    log_odds_delta: float
    sources: list[str] = dc_field(default_factory=list)
    confidence: float = 1.0
    source_url: str = ""
    fired: bool = False
    corroboration_count: int = 0
    corroboration_min: int = 2

    def effective_delta(self) -> float:
        if not self.fired:
            return 0.0
        corrob_factor = min(1.0, self.corroboration_count / max(1, self.corroboration_min))
        sign = -1.0 if self.direction == "up" else 1.0
        return sign * self.log_odds_delta * self.confidence * corrob_factor

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "direction": self.direction,
            "log_odds_delta": self.log_odds_delta,
            "effective_delta": round(self.effective_delta(), 4),
            "sources": self.sources,
            "confidence": round(self.confidence, 3),
            "fired": self.fired,
            "corroboration_count": self.corroboration_count,
            "corroboration_min": self.corroboration_min,
        }


@dataclass
class VerdictReport:
    """v16 Phase 11: Probabilistic verdict with CI and evidence chain."""
    tier: str
    tier_probabilities: dict[str, float] = dc_field(default_factory=dict)
    log_odds_fake: float = 0.0
    log_odds_ci95: tuple[float, float] = (0.0, 0.0)
    is_borderline: bool = False
    dominant_evidence: list[EvidenceItem] = dc_field(default_factory=list)
    coverage: float = 1.0
    p_fake: float = 0.5
    sample_count: int = 0
    test_mode: str = ""

    def to_dict(self) -> dict:
        return {
            "tier": self.tier,
            "tier_probabilities": {k: round(v, 4) for k, v in self.tier_probabilities.items()},
            "log_odds_fake": round(self.log_odds_fake, 4),
            "log_odds_ci95": [round(self.log_odds_ci95[0], 4), round(self.log_odds_ci95[1], 4)],
            "is_borderline": self.is_borderline,
            "p_fake": round(self.p_fake, 4),
            "coverage": round(self.coverage, 4),
            "sample_count": self.sample_count,
            "test_mode": self.test_mode,
            "dominant_evidence": [e.to_dict() for e in self.dominant_evidence],
        }


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)


# v16 Symmetric rule set (up and down rules balanced)
_V16_RULES: list[dict] = [
    # ── UP rules (evidence toward REAL) ──
    {"rule_id": "official_endpoint_match", "direction": "up", "log_odds_delta": 2.0,
     "sources": ["tls", "url", "/v1/models"], "corroboration_min": 3,
     "source_url": "RFC8446/RFC6125"},
    {"rule_id": "tokenizer_fingerprint_match", "direction": "up", "log_odds_delta": 0.8,
     "sources": ["L5"], "corroboration_min": 1,
     "source_url": "SOURCES.yaml:verdict.tokenizer_fingerprint"},
    {"rule_id": "knowledge_cutoff_match_claimed", "direction": "up", "log_odds_delta": 0.5,
     "sources": ["L3", "self_report"], "corroboration_min": 2,
     "source_url": "SOURCES.yaml:verdict.cutoff_match"},
    {"rule_id": "behavioral_invariant_high", "direction": "up", "log_odds_delta": 0.6,
     "sources": ["L7", "L8"], "corroboration_min": 2,
     "source_url": "SOURCES.yaml:verdict.behavioral_invariant"},
    {"rule_id": "extraction_resistance_high", "direction": "up", "log_odds_delta": 0.5,
     "sources": ["L9", "L22"], "corroboration_min": 2,
     "source_url": "SOURCES.yaml:verdict.extraction_resistance"},
    # ── DOWN rules (evidence toward FAKE) ──
    {"rule_id": "difficulty_ceiling_low", "direction": "down", "log_odds_delta": 0.7,
     "sources": ["capability"], "corroboration_min": 1,
     "source_url": "SOURCES.yaml:verdict.difficulty_ceiling"},
    {"rule_id": "coding_score_low", "direction": "down", "log_odds_delta": 0.6,
     "sources": ["coding"], "corroboration_min": 1,
     "source_url": "SOURCES.yaml:verdict.coding_zero"},
    {"rule_id": "adversarial_spoof_rate_high", "direction": "down", "log_odds_delta": 0.8,
     "sources": ["L13", "L23"], "corroboration_min": 2,
     "source_url": "SOURCES.yaml:verdict.adv_spoof"},
    {"rule_id": "extraction_leaked_real_identity", "direction": "down", "log_odds_delta": 1.5,
     "sources": ["L17", "L22"], "corroboration_min": 2,
     "source_url": "SOURCES.yaml:verdict.identity_exposed"},
    {"rule_id": "fingerprint_mismatch", "direction": "down", "log_odds_delta": 0.5,
     "sources": ["L5"], "corroboration_min": 1,
     "source_url": "SOURCES.yaml:verdict.fingerprint_mismatch"},
]

# Tier thresholds on P(fake)
_TIER_THRESHOLDS = {"trusted": 0.15, "suspicious": 0.40, "high_risk": 0.70}


class BayesianEvidenceVerdictEngine:
    """
    v16 Phase 11: Bayesian evidence-weighted verdict.

    log-odds(fake) = log(prior/(1-prior)) + Σ effective_delta_i
    P(fake) = sigmoid(log-odds)

    Rule fires only if corroboration >= corroboration_min.
    Prevents single-point false positives (root cause of user complaints).
    """

    PRIOR_LOG_ODDS = 0.0  # log(0.5/0.5) = 0 → uniform prior

    def __init__(self, prior_log_odds: float = 0.0):
        self.prior_log_odds = prior_log_odds
        self.rules = [
            EvidenceItem(
                rule_id=r["rule_id"], direction=r["direction"],
                log_odds_delta=r["log_odds_delta"], sources=list(r["sources"]),
                source_url=r.get("source_url", ""),
                corroboration_min=r.get("corroboration_min", 2),
            ) for r in _V16_RULES
        ]

    def assess_evidence(
        self,
        fired_rules: list[dict],
        coverage: float = 1.0,
        sample_count: int = 0,
        test_mode: str = "",
    ) -> VerdictReport:
        """
        Assess verdict from fired rules using Bayesian log-odds stacking.

        Args:
            fired_rules: List of dicts with keys: rule_id, corroboration_count, confidence (optional)
            coverage: Effective sample ratio (0-1).
            sample_count: Number of test cases executed.
            test_mode: "quick" / "standard" / "deep"
        """
        fired_map: dict[str, dict] = {fr["rule_id"]: fr for fr in fired_rules}

        # Apply fired state to rules
        for rule in self.rules:
            fr = fired_map.get(rule.rule_id)
            if fr and fr.get("corroboration_count", 0) >= rule.corroboration_min:
                rule.fired = True
                rule.corroboration_count = fr.get("corroboration_count", 0)
                rule.confidence = fr.get("confidence", rule.confidence)
            else:
                rule.fired = False

        # Small-sample protection: difficulty_ceiling_low needs >= 12 samples
        for rule in self.rules:
            if rule.rule_id == "difficulty_ceiling_low" and sample_count < 12:
                rule.fired = False
            if rule.rule_id == "coding_score_low" and sample_count < 5:
                rule.fired = False

        # Compute log-odds
        log_odds = self.prior_log_odds + sum(r.effective_delta() for r in self.rules)
        p_fake = _sigmoid(log_odds)

        # Bootstrap CI (simplified: ±1.96 * SE via delta method)
        fired_count = sum(1 for r in self.rules if r.fired)
        se = math.sqrt(max(0.01, sum(r.effective_delta() ** 2 for r in self.rules if r.fired)))
        ci_lo = log_odds - 1.96 * se
        ci_hi = log_odds + 1.96 * se
        p_lo = _sigmoid(ci_lo)
        p_hi = _sigmoid(ci_hi)

        # Determine tier
        tier = self._p_to_tier(p_fake, test_mode, sample_count)

        # Coverage gate: insufficient samples → inconclusive
        if coverage < 0.7:
            tier = "inconclusive"

        # Borderline check: CI crosses any threshold
        is_borderline = self._is_borderline(p_lo, p_hi)

        # Tier probabilities
        tier_probs = self._compute_tier_probabilities(p_lo, p_hi)

        # Dominant evidence (top 5 by |effective_delta|)
        dominant = sorted(
            [r for r in self.rules if r.fired],
            key=lambda r: abs(r.effective_delta()),
            reverse=True,
        )[:5]

        return VerdictReport(
            tier=tier,
            tier_probabilities=tier_probs,
            log_odds_fake=log_odds,
            log_odds_ci95=(ci_lo, ci_hi),
            is_borderline=is_borderline,
            dominant_evidence=dominant,
            coverage=coverage,
            p_fake=p_fake,
            sample_count=sample_count,
            test_mode=test_mode,
        )

    def _p_to_tier(self, p_fake: float, test_mode: str, sample_count: int) -> str:
        """Map P(fake) to tier with small-sample protection."""
        # Quick mode: cap at suspicious (no high_risk/fake)
        if test_mode == "quick" and sample_count < 18:
            if p_fake >= _TIER_THRESHOLDS["suspicious"]:
                return "suspicious"
        if p_fake < _TIER_THRESHOLDS["trusted"]:
            return "trusted"
        elif p_fake < _TIER_THRESHOLDS["suspicious"]:
            return "suspicious"
        elif p_fake < _TIER_THRESHOLDS["high_risk"]:
            return "high_risk"
        else:
            return "fake"

    def _is_borderline(self, p_lo: float, p_hi: float) -> bool:
        """Check if CI crosses any tier threshold."""
        for threshold in _TIER_THRESHOLDS.values():
            if p_lo < threshold < p_hi:
                return True
        return False

    def _compute_tier_probabilities(self, p_lo: float, p_hi: float) -> dict[str, float]:
        """Approximate tier probabilities from CI."""
        p_mid = (p_lo + p_hi) / 2
        probs = {}
        probs["trusted"] = max(0, _TIER_THRESHOLDS["trusted"] - p_lo) / max(0.01, p_hi - p_lo)
        probs["suspicious"] = max(0, min(_TIER_THRESHOLDS["suspicious"], p_hi) - max(_TIER_THRESHOLDS["trusted"], p_lo)) / max(0.01, p_hi - p_lo)
        probs["high_risk"] = max(0, min(_TIER_THRESHOLDS["high_risk"], p_hi) - max(_TIER_THRESHOLDS["suspicious"], p_lo)) / max(0.01, p_hi - p_lo)
        probs["fake"] = max(0, p_hi - _TIER_THRESHOLDS["high_risk"]) / max(0.01, p_hi - p_lo)
        # Normalize
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}
        return probs

    def check_symmetry(self) -> dict:
        """Verify up/down rule balance (Phase 11 validation)."""
        up_count = sum(1 for r in self.rules if r.direction == "up")
        down_count = sum(1 for r in self.rules if r.direction == "down")
        return {
            "up_rules": up_count,
            "down_rules": down_count,
            "difference": abs(up_count - down_count),
            "is_balanced": abs(up_count - down_count) <= 1,
        }
