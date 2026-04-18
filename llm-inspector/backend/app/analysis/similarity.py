"""
analysis/similarity.py — SimilarityEngine + feature constants

Weighted cosine similarity with bootstrap CI.
Extracted from pipeline.py to keep individual files under ~320 lines.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np

from app.core.schemas import SimilarityResult
from app.core.logging import get_logger

logger = get_logger(__name__)


# ── Similarity Engine ─────────────────────────────────────────────────────────

FEATURE_ORDER = [
    "protocol_success_rate", "instruction_pass_rate", "exact_match_rate",
    "json_valid_rate", "system_obedience_rate", "param_compliance_rate",
    "temperature_param_effective", "refusal_rate", "disclaimer_rate",
    "identity_consistency_pass_rate", "antispoof_identity_detect_rate",
    "antispoof_override_leak_rate", "avg_markdown_score", "avg_response_length",
    "adversarial_spoof_signal_rate", "latency_mean_ms",
    "reasoning_pass_rate", "coding_pass_rate", "adversarial_pass_rate",
    "latency_cv", "first_token_mean_ms", "tokens_per_second",
    "refusal_verbosity", "safety_alternative_style",
    "avg_sentence_count", "avg_words_per_sentence",
    "bullet_list_rate", "numbered_list_rate", "code_block_rate", "heading_rate",
    "max_passed_difficulty", "min_failed_difficulty", "difficulty_ceiling", "difficulty_dropoff",
    "tool_use_pass_rate", "avg_formality_score",
    # v3 new features
    "dim_knowledge_pass_rate", "dim_safety_pass_rate",
    "dim_tool_use_pass_rate", "dim_consistency_pass_rate",
    "hallucination_resist_rate", "extraction_resist_rate",
]

# v6 fix: Removed GLOBAL_FEATURE_MEANS (was hardcoded fake data with no statistical basis)
# If feature statistics are needed, use FeatureStatsRepo from app.repository.feature_stats

# Feature importance weights for similarity computation.
# Higher weight = more discriminative for model identification.
FEATURE_IMPORTANCE: dict[str, float] = {
    # High discrimination power
    "reasoning_pass_rate": 2.0,
    "coding_pass_rate": 2.0,
    "adversarial_pass_rate": 2.0,
    "difficulty_ceiling": 2.5,
    "difficulty_dropoff": 1.8,
    "refusal_verbosity": 1.8,
    "safety_alternative_style": 1.5,
    "avg_response_length": 1.5,
    "avg_words_per_sentence": 1.5,
    "tokens_per_second": 1.8,
    "first_token_mean_ms": 1.5,
    "hallucination_resist_rate": 1.5,
    "extraction_resist_rate": 1.5,
    # Medium discrimination
    "instruction_pass_rate": 1.3,
    "exact_match_rate": 1.3,
    "identity_consistency_pass_rate": 1.5,
    "adversarial_spoof_signal_rate": 1.5,
    "bullet_list_rate": 1.3,
    "numbered_list_rate": 1.3,
    "code_block_rate": 1.3,
    "heading_rate": 1.3,
    "avg_markdown_score": 1.3,
    "latency_cv": 1.3,
    "tool_use_pass_rate": 1.3,
    # Low discrimination (most models similar)
    "protocol_success_rate": 0.5,
    "json_valid_rate": 0.8,
    "system_obedience_rate": 0.7,
    "param_compliance_rate": 0.7,
    "disclaimer_rate": 1.0,
    "latency_mean_ms": 1.0,
    "avg_formality_score": 1.0,
}


class SimilarityEngine:

    @staticmethod
    def compute_feature_importance_from_baselines(baselines: list[dict]) -> dict[str, float]:
        """
        v6: Compute feature importance from baseline standard deviations.
        Higher standard deviation = more discriminative = higher weight.

        Args:
            baselines: List of baseline profiles with 'feature_vector' field

        Returns:
            Dictionary mapping feature names to importance weights (0.5-3.0)
        """
        if len(baselines) < 3:
            return FEATURE_IMPORTANCE  # Not enough data, use defaults

        import numpy as np

        # Collect feature values across all baselines
        feature_values: dict[str, list[float]] = {}
        for bp in baselines:
            fv = bp.get("feature_vector", {})
            for key in FEATURE_ORDER:
                if key in fv and fv[key] is not None:
                    feature_values.setdefault(key, []).append(float(fv[key]))

        # Calculate importance from standard deviation
        importance: dict[str, float] = {}
        for key, values in feature_values.items():
            if len(values) >= 3:
                std = float(np.std(values))
                # Higher std = more discriminative = higher weight
                # Scale to 0.5-3.0 range
                importance[key] = max(0.5, min(3.0, 1.0 + std * 5.0))
            else:
                importance[key] = FEATURE_IMPORTANCE.get(key, 1.0)

        return importance

    def compare(
        self,
        target_features: dict[str, float],
        benchmark_profiles: list[dict],
    ) -> list[SimilarityResult]:
        """
        Returns similarity results ranked by score.
        Each benchmark_profile has: {name, suite_version, feature_vector: {k: v}}
        Uses sparse vector similarity (only valid dimensions are compared).
        """
        # 过滤掉 estimated 类型的基准
        benchmark_profiles = [bp for bp in benchmark_profiles if bp.get("data_source") != "estimated"]
        if not benchmark_profiles:
            # v13 Phase 5: Try reference embeddings as fallback when no user baselines exist
            ref_profiles = _load_reference_profiles()
            if ref_profiles:
                benchmark_profiles = ref_profiles
            else:
                return []  # 无真实基准时返回空，前端显示"暂无基准数据"

        target_vec, target_mask = self._to_vector_with_mask(target_features)
        results: list[SimilarityResult] = []

        for bp in benchmark_profiles:
            bench_vec, bench_mask = self._to_vector_with_mask(bp["feature_vector"])
            sim, valid_count = self._masked_cosine_similarity(target_vec, bench_vec, target_mask, bench_mask)
            ci_low, ci_high, _ = self._bootstrap_ci(target_vec, bench_vec)
            bm_name = bp.get("benchmark_name") or bp.get("name", "unknown")

            # 判定可信度等级
            if valid_count >= 30:
                confidence_level = "high"
            elif valid_count >= 20:
                confidence_level = "medium"
            elif valid_count >= 12:
                confidence_level = "low"
            else:
                confidence_level = "insufficient"

            results.append(SimilarityResult(
                benchmark_name=bm_name,
                similarity_score=round(sim, 4),
                ci_95_low=round(ci_low, 4) if ci_low is not None else None,
                ci_95_high=round(ci_high, 4) if ci_high is not None else None,
                rank=0,
                confidence_level=confidence_level,
                valid_feature_count=valid_count,
            ))

        results.sort(key=lambda r: r.similarity_score, reverse=True)
        for i, r in enumerate(results):
            r.rank = i + 1

        return results

    def _to_vector_with_mask(
        self,
        features: dict[str, float],
        normalization_params: dict[str, float] | None = None,
    ) -> tuple[list[float], list[bool]]:
        """
        Build a fixed-length vector from features with a mask indicating valid dimensions.

        v6 improvements:
        - Accepts dynamic normalization_params from benchmark statistics
        - Falls back to conservative defaults if no stats provided

        Returns (vector, mask) where mask[i]=True means the dimension has real data.
        Missing features are set to 0 and marked as False in mask (sparse vector support).
        """
        # v6: Default normalization parameters (conservative)
        defaults = {
            "avg_response_length_max": 1200.0,
            "avg_markdown_score_max": 5.0,
            "latency_mean_ms_max": 5000.0,
            "tokens_per_second_max": 200.0,
            "refusal_verbosity_max": 200.0,
            "avg_sentence_count_max": 15.0,
            "avg_words_per_sentence_max": 30.0,
        }
        norms = normalization_params or defaults

        vec, mask = [], []
        for key in FEATURE_ORDER:
            val = features.get(key)
            if val is None:
                # Sparse vector: missing feature = 0, mask=False
                vec.append(0.0)
                mask.append(False)
                continue

            # v6: Feature-specific normalization with configurable params
            if key == "avg_response_length":
                max_val = norms.get("avg_response_length_max", 1200.0)
                val = val / max_val if max_val > 0 else val
            elif key == "avg_markdown_score":
                max_val = norms.get("avg_markdown_score_max", 5.0)
                val = val / max_val if max_val > 0 else val
            elif key == "latency_mean_ms":
                # Normalize latency: inverted (lower is 1.0)
                max_val = norms.get("latency_mean_ms_max", 5000.0)
                val = 1.0 - (val / max_val) if max_val > 0 else val
            elif key == "tokens_per_second":
                max_val = norms.get("tokens_per_second_max", 200.0)
                val = val / max_val if max_val > 0 else val
            elif key == "refusal_verbosity":
                max_val = norms.get("refusal_verbosity_max", 200.0)
                val = val / max_val if max_val > 0 else val
            elif key == "avg_sentence_count":
                max_val = norms.get("avg_sentence_count_max", 15.0)
                val = val / max_val if max_val > 0 else val
            elif key == "avg_words_per_sentence":
                max_val = norms.get("avg_words_per_sentence_max", 30.0)
                val = val / max_val if max_val > 0 else val

            # Clamp to [0,1] then apply feature importance weight
            weight = FEATURE_IMPORTANCE.get(key, 1.0)
            vec.append(max(0.0, min(1.0, float(val))) * weight)
            mask.append(True)
        return vec, mask

    def _to_vector(self, features: dict[str, float]) -> list[float]:
        """Build a fixed-length vector from features, normalised 0-1.
        Legacy method - now delegates to _to_vector_with_mask.
        """
        vec, _ = self._to_vector_with_mask(features)
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

    @staticmethod
    def _masked_cosine_similarity(a: list[float], b: list[float], mask_a: list[bool], mask_b: list[bool]) -> tuple[float, int]:
        """Compute cosine similarity only on dimensions where both vectors have valid data.
        Returns (similarity_score, valid_feature_count).
        """
        valid = [i for i in range(len(a)) if mask_a[i] and mask_b[i]]
        valid_count = len(valid)
        # v6 fix: Lowered minimum threshold from 8 to 5 for quick mode support
        if valid_count < 5:
            return 0.0, valid_count  # Insufficient features
        a_v = [a[i] for i in valid]
        b_v = [b[i] for i in valid]
        dot = sum(x * y for x, y in zip(a_v, b_v))
        norm_a = math.sqrt(sum(x * x for x in a_v))
        norm_b = math.sqrt(sum(y * y for y in b_v))
        if norm_a == 0 or norm_b == 0:
            return 0.0, valid_count
        return dot / (norm_a * norm_b), valid_count

    @classmethod
    def _bootstrap_ci(
        cls, a: list[float], b: list[float], n: int = 200
    ) -> tuple[float | None, float | None, int]:
        """Bootstrap 95% confidence interval for cosine similarity.
        Returns (ci_low, ci_high, valid_feature_count).
        """
        from app.core.config import settings
        length = len(a)
        if length == 0:
            return 0.0, 0.0, 0

        # v6 fix: Lowered from 12 to 5 for quick mode compatibility
        MIN_BOOTSTRAP_FEATURES = 5
        valid_features = [x for x in a + b if x != 0.0]
        valid_count = len(valid_features)
        if valid_count < MIN_BOOTSTRAP_FEATURES:
            return None, None, valid_count

        raw_sim = cls._cosine_similarity(a, b)
        # v6 fix: High similarity needs more samples for precise CI estimation
        if raw_sim >= 0.90:
            n_bootstrap = settings.THETA_BOOTSTRAP_B  # default 200
        elif raw_sim >= 0.75:
            n_bootstrap = 150
        else:
            n_bootstrap = 100  # Low similarity can use fewer samples

        sims = []
        rng = random.Random(42)
        for _ in range(n_bootstrap):
            indices = [rng.randrange(length) for _ in range(length)]
            a2 = [a[i] for i in indices]
            b2 = [b[i] for i in indices]
            sims.append(cls._cosine_similarity(a2, b2))
        sims.sort()
        lo = sims[int(n_bootstrap * 0.025)]
        hi = sims[int(n_bootstrap * 0.975)]
        return lo, hi, valid_count


# ── Reference embedding helpers (v13 Phase 5) ─────────────────────────────────

def _load_reference_embeddings() -> dict:
    """
    Load reference embeddings from app/_data/reference_embeddings.json.

    Returns the raw ``models`` dict keyed by model name, or {} if the file
    cannot be found or parsed.
    """
    try:
        path = Path(__file__).parent.parent / "_data" / "reference_embeddings.json"
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("models", {})
    except Exception:
        return {}


def _load_reference_profiles() -> list[dict]:
    """
    Convert reference_embeddings.json into benchmark_profile dicts compatible
    with ``SimilarityEngine.compare()``.

    Each profile has:
      benchmark_name, name, feature_vector, data_source="reference",
      baseline_source="reference", scores (for AnalysisPipeline.compare_with_baseline)
    """
    raw = _load_reference_embeddings()
    if not raw:
        return []
    profiles = []
    for model_name, data in raw.items():
        profiles.append({
            "benchmark_name": model_name,
            "name": model_name,
            "feature_vector": data.get("features", {}),
            "data_source": "reference",
            "baseline_source": "reference",
            "scores": data.get("scores", {}),
            "family": data.get("family", ""),
            "sample_count": 1,
        })
    return profiles
