"""
Shapley Attribution Engine — v11 score attribution analysis.

Implements Shapley Value-based feature attribution for the scoring system,
as specified in v11_upgrade_plan.md §2.6.

Reference: Lundberg & Lee (2017) "A Unified Approach to Interpreting
Model Predictions" (SHAP values)

The Shapley Value answers: "该模型丢失 15 分，其中 60% 归因于无法遵循
JSON 格式，40% 归因于幻觉" — by fairly distributing the total score
change among contributing features.

Key properties of Shapley Values:
1. Efficiency: Σ φ_i = f(x) - f(∅)  (attribution sums to score difference)
2. Symmetry: equally contributing features get equal attribution
3. Dummy: features with no contribution get zero attribution
4. Additivity: works for combined scoring functions

Implementation uses KernelSHAP approximation (sampling-based) rather than
exact Shapley (exponential in features) for tractability.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable

from app.core.schemas import ScoreCard, TrustVerdict
from app.core.logging import get_logger

logger = get_logger(__name__)


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class FeatureAttribution:
    """Shapley value attribution for a single feature/dimension."""
    feature_name: str
    shapley_value: float       # Attribution value (can be negative)
    contribution_pct: float    # % of total score change attributable to this feature
    direction: str = "neutral" # "positive" | "negative" | "neutral"
    baseline_value: float = 0.0
    actual_value: float = 0.0
    impact_description: str = ""

    def to_dict(self) -> dict:
        return {
            "feature_name": self.feature_name,
            "shapley_value": round(self.shapley_value, 3),
            "contribution_pct": round(self.contribution_pct, 1),
            "direction": self.direction,
            "baseline_value": round(self.baseline_value, 3),
            "actual_value": round(self.actual_value, 3),
            "impact_description": self.impact_description,
        }


@dataclass
class AttributionReport:
    """Complete Shapley attribution report for a run."""
    target_score: float = 0.0
    baseline_score: float = 0.0
    score_delta: float = 0.0
    attributions: list[FeatureAttribution] = field(default_factory=list)
    top_positive: list[str] = field(default_factory=list)
    top_negative: list[str] = field(default_factory=list)
    narrative: str = ""
    method: str = "kernel_shap"

    def to_dict(self) -> dict:
        return {
            "target_score": round(self.target_score, 1),
            "baseline_score": round(self.baseline_score, 1),
            "score_delta": round(self.score_delta, 1),
            "attributions": [a.to_dict() for a in self.attributions],
            "top_positive": self.top_positive,
            "top_negative": self.top_negative,
            "narrative": self.narrative,
            "method": self.method,
        }


# ── Scoring Features for Attribution ────────────────────────────────────────

# The features we attribute scores to. These map to ScoreCard sub-scores.
ATTRIBUTABLE_FEATURES = [
    "reasoning",
    "adversarial_reasoning",
    "instruction",
    "coding",
    "safety",
    "protocol",
    "consistency",
    "behavioral_invariant",
    "extraction_resistance",
    "fingerprint_match",
    "speed",
    "stability",
]

# Mapping from feature name to ScoreCard attribute path
FEATURE_SCORE_MAP: dict[str, str] = {
    "reasoning": "reasoning_score",
    "adversarial_reasoning": "adversarial_reasoning_score",
    "instruction": "instruction_score",
    "coding": "coding_score",
    "safety": "safety_score",
    "protocol": "protocol_score",
    "consistency": "consistency_score",
    "behavioral_invariant": "behavioral_invariant_score",
    "extraction_resistance": "breakdown.extraction_resistance",
    "fingerprint_match": "breakdown.fingerprint_match",
    "speed": "speed_score",
    "stability": "stability_score",
}


def _get_score_value(scorecard: ScoreCard, feature: str) -> float:
    """Extract a sub-score value from a ScoreCard by feature name."""
    attr_path = FEATURE_SCORE_MAP.get(feature)
    if not attr_path:
        return 50.0  # Neutral baseline for unknown features

    if "." in attr_path:
        # Navigate nested: e.g. "breakdown.extraction_resistance"
        parts = attr_path.split(".", 1)
        breakdown = getattr(scorecard, "breakdown", {})
        if isinstance(breakdown, dict):
            val = breakdown.get(parts[1])
            return float(val) if val is not None else 50.0
    else:
        val = getattr(scorecard, attr_path, None)
        if val is not None:
            return float(val)

    return 50.0  # Neutral


def _compute_total_score(
    feature_values: dict[str, float],
    capability_weight: float = 0.45,
    authenticity_weight: float = 0.30,
    performance_weight: float = 0.25,
) -> float:
    """
    Compute total score from feature values using ScoreCard formula.

    This is the "value function" v(S) for the Shapley computation.
    When a feature is "absent", we use its baseline (neutral) value.
    """
    # Capability sub-score (weighted average)
    cap_weights = {
        "reasoning": 0.20, "adversarial_reasoning": 0.15,
        "instruction": 0.20, "coding": 0.20, "safety": 0.10, "protocol": 0.05,
    }
    cap_score = 0.0
    cap_weight_sum = 0.0
    for feat, w in cap_weights.items():
        if feat in feature_values:
            cap_score += w * feature_values[feat]
            cap_weight_sum += w
    if cap_weight_sum > 0:
        cap_score = cap_score / cap_weight_sum

    # Authenticity sub-score
    auth_weights = {
        "consistency": 0.15, "behavioral_invariant": 0.20,
        "extraction_resistance": 0.10, "fingerprint_match": 0.15,
    }
    auth_score = 0.0
    auth_weight_sum = 0.0
    for feat, w in auth_weights.items():
        if feat in feature_values:
            auth_score += w * feature_values[feat]
            auth_weight_sum += w
    if auth_weight_sum > 0:
        auth_score = auth_score / auth_weight_sum

    # Performance sub-score
    perf_weights = {"speed": 0.35, "stability": 0.25}
    perf_score = 0.0
    perf_weight_sum = 0.0
    for feat, w in perf_weights.items():
        if feat in feature_values:
            perf_score += w * feature_values[feat]
            perf_weight_sum += w
    if perf_weight_sum > 0:
        perf_score = perf_score / perf_weight_sum

    # Total
    return (
        capability_weight * cap_score
        + authenticity_weight * auth_score
        + performance_weight * perf_score
    )


# ── KernelSHAP Approximation ────────────────────────────────────────────────

class ShapleyAttributor:
    """
    Shapley Value attribution engine using KernelSHAP approximation.

    For N features, exact Shapley requires 2^N evaluations.
    KernelSHAP approximates by sampling subsets and solving a weighted
    linear regression, giving O(N × M) complexity where M is sample count.
    """

    def __init__(self, n_samples: int = 500, seed: int = 42):
        self._n_samples = n_samples
        self._rng = random.Random(seed)

    def attribute(
        self,
        scorecard: ScoreCard,
        verdict: TrustVerdict | None = None,
        features: dict[str, float] | None = None,
    ) -> AttributionReport:
        """
        Compute Shapley Value attributions for the given scorecard.

        Answers: "What contributed to the total score, and by how much?"

        Args:
            scorecard: The scored result
            verdict: Optional verdict for narrative enhancement
            features: Raw features dict for additional context

        Returns:
            AttributionReport with per-feature Shapley values
        """
        # 1. Get actual feature values
        actual_values: dict[str, float] = {}
        for feat in ATTRIBUTABLE_FEATURES:
            actual_values[feat] = _get_score_value(scorecard, feat)

        # 2. Define baseline (neutral/average model)
        baseline_values = {feat: 50.0 for feat in ATTRIBUTABLE_FEATURES}

        # 3. Compute actual and baseline total scores
        actual_total = _compute_total_score(actual_values)
        baseline_total = _compute_total_score(baseline_values)
        score_delta = actual_total - baseline_total

        # 4. Compute Shapley values via KernelSHAP
        shapley_values = self._kernel_shap(actual_values, baseline_values)

        # 5. Build attribution list
        attributions = self._build_attributions(
            shapley_values, actual_values, baseline_values, score_delta,
        )

        # 6. Sort by absolute Shapley value
        attributions.sort(key=lambda a: abs(a.shapley_value), reverse=True)

        # 7. Build narrative
        narrative = self._build_narrative(attributions, score_delta)

        # 8. Top positive/negative
        top_positive = [a.feature_name for a in attributions if a.direction == "positive"][:5]
        top_negative = [a.feature_name for a in attributions if a.direction == "negative"][:5]

        return AttributionReport(
            target_score=actual_total,
            baseline_score=baseline_total,
            score_delta=score_delta,
            attributions=attributions,
            top_positive=top_positive,
            top_negative=top_negative,
            narrative=narrative,
        )

    def _kernel_shap(
        self,
        actual: dict[str, float],
        baseline: dict[str, float],
    ) -> dict[str, float]:
        """
        KernelSHAP approximation of Shapley values.

        Algorithm:
        1. Sample random subsets S of features
        2. For each S, compute v(S) using actual values for S, baseline for others
        3. Solve weighted least squares to approximate Shapley values

        The kernel weight for subset S of size |S| is:
            w(S) = (C(n-1, |S|) × |S| × (n - |S|)) ^ (-1)
        """
        n = len(ATTRIBUTABLE_FEATURES)
        feature_list = ATTRIBUTABLE_FEATURES
        shapley_values = {f: 0.0 for f in feature_list}

        # Sample subsets and compute value function
        samples = []
        for _ in range(self._n_samples):
            # Random subset: each feature included with 50% probability
            subset = set()
            for feat in feature_list:
                if self._rng.random() < 0.5:
                    subset.add(feat)

            # Compute v(S): use actual values for features in S, baseline otherwise
            mixed_values = {}
            for feat in feature_list:
                mixed_values[feat] = actual[feat] if feat in subset else baseline[feat]

            v_S = _compute_total_score(mixed_values)

            # Weight based on subset size (KernelSHAP kernel)
            s_size = len(subset)
            if s_size == 0 or s_size == n:
                weight = 1e10  # Very high weight for empty and full subsets
            else:
                # Kernel weight: proportional to 1 / (C(n-1,s) × s × (n-s))
                try:
                    from math import comb
                    weight = (comb(n - 1, s_size) * s_size * (n - s_size))
                except (ValueError, OverflowError):
                    weight = 1.0
                weight = 1.0 / max(weight, 1.0)

            samples.append((subset, v_S, weight))

        # Add empty set and full set with high weight
        v_empty = _compute_total_score(baseline)
        v_full = _compute_total_score(actual)
        samples.append((set(), v_empty, 1e10))
        samples.append((set(feature_list), v_full, 1e10))

        # Solve weighted least squares: v(S) - v(∅) ≈ Σ φ_i × z_i
        # where z_i = 1 if feature i ∈ S, 0 otherwise
        # This gives the Shapley value approximation

        # Simple iterative approach: for each feature, compute marginal contribution
        # across all samples weighted by kernel weights
        total_weight = sum(w for _, _, w in samples)
        if total_weight == 0:
            return shapley_values

        for feat in feature_list:
            weighted_contribution = 0.0
            for subset, v_S, weight in samples:
                # Marginal contribution of this feature to this subset
                if feat in subset:
                    # v(S) - v(S \ {feat})
                    subset_without = subset - {feat}
                    mixed_without = {}
                    for f in feature_list:
                        mixed_without[f] = actual[f] if f in subset_without else baseline[f]
                    v_without = _compute_total_score(mixed_without)
                    marginal = v_S - v_without
                    weighted_contribution += weight * marginal

            shapley_values[feat] = weighted_contribution / total_weight

        # Normalize to ensure efficiency: Σ φ_i = v(full) - v(empty)
        total_shapley = sum(shapley_values.values())
        expected_total = v_full - v_empty
        if abs(total_shapley) > 1e-10 and abs(expected_total) > 1e-10:
            # Scale to match efficiency property
            scale = expected_total / total_shapley
            shapley_values = {f: v * scale for f, v in shapley_values.items()}

        return shapley_values

    def _build_attributions(
        self,
        shapley_values: dict[str, float],
        actual_values: dict[str, float],
        baseline_values: dict[str, float],
        score_delta: float,
    ) -> list[FeatureAttribution]:
        """Build FeatureAttribution list from computed Shapley values."""
        attributions = []
        total_abs = sum(abs(v) for v in shapley_values.values())

        for feat in ATTRIBUTABLE_FEATURES:
            sv = shapley_values.get(feat, 0.0)

            # Contribution percentage
            if total_abs > 0:
                contrib_pct = (abs(sv) / total_abs) * 100
            else:
                contrib_pct = 0.0

            # Direction
            if sv > 1.0:
                direction = "positive"
            elif sv < -1.0:
                direction = "negative"
            else:
                direction = "neutral"

            # Human-readable impact description
            actual_v = actual_values.get(feat, 50.0)
            baseline_v = baseline_values.get(feat, 50.0)
            delta_v = actual_v - baseline_v

            if direction == "positive":
                desc = f"{feat} 高于基准 (+{abs(delta_v):.1f})，贡献 +{abs(sv):.1f} 分"
            elif direction == "negative":
                desc = f"{feat} 低于基准 ({delta_v:.1f})，损失 {abs(sv):.1f} 分"
            else:
                desc = f"{feat} 接近基准，影响中性"

            attributions.append(FeatureAttribution(
                feature_name=feat,
                shapley_value=sv,
                contribution_pct=contrib_pct,
                direction=direction,
                baseline_value=baseline_v,
                actual_value=actual_v,
                impact_description=desc,
            ))

        return attributions

    def _build_narrative(
        self,
        attributions: list[FeatureAttribution],
        score_delta: float,
    ) -> str:
        """Build a human-readable attribution narrative."""
        positive = [a for a in attributions if a.direction == "positive"]
        negative = [a for a in attributions if a.direction == "negative"]

        parts = []

        if score_delta > 0:
            parts.append(f"总分高于基准 {abs(score_delta):.1f} 分。")
        elif score_delta < 0:
            parts.append(f"总分低于基准 {abs(score_delta):.1f} 分。")

        if positive:
            top_pos = positive[:3]
            pos_desc = "、".join(
                f"{a.feature_name}（+{abs(a.shapley_value):.1f}分，{a.contribution_pct:.0f}%）"
                for a in top_pos
            )
            parts.append(f"主要正向贡献：{pos_desc}。")

        if negative:
            top_neg = negative[:3]
            neg_desc = "、".join(
                f"{a.feature_name}（{a.shapley_value:.1f}分，{a.contribution_pct:.0f}%）"
                for a in top_neg
            )
            parts.append(f"主要负向影响：{neg_desc}。")

        return " ".join(parts) if parts else "各项指标均接近基准水平。"


# ── Global singleton ────────────────────────────────────────────────────────

shapley_attributor = ShapleyAttributor()
