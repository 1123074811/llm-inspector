"""
Similarity Engine Module
Calculate similarity between runs and manage baseline comparisons.

Split from pipeline.py in V6 refactoring for better code organization.
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

from app.core.logging import get_logger

logger = get_logger(__name__)

# Feature order for vectorization (must match across modules)
FEATURE_ORDER = [
    "avg_response_length",
    "avg_markdown_score", 
    "latency_mean_ms",
    "tokens_per_second",
    "refusal_verbosity",
    "avg_sentence_count",
    "avg_words_per_sentence",
    "has_usage_fields",
    "has_finish_reason",
    "param_compliance_rate",
    "format_compliance_score",
    "protocol_success_rate",
    "instruction_pass_rate",
    "exact_match_rate",
    "json_valid_rate",
    "format_follow_rate",
    "line_count_rate",
    "response_quality_basic_rate",
    "response_quality_strict_rate",
    "code_execution_rate",
    "identity_consistency_rate",
    "hallucination_detection_rate",
    "topic_relevance_rate",
    "refusal_rate",
    "over_refusal_rate",
    "safety_alternative_style",
    "adversarial_spoof_signal_rate",
    "extraction_resist_rate",
    "token_rounding_anomaly",
    "zero_completion_anomaly", 
    "identical_token_anomaly",
    "temp_zero_diversity",
    "token_count_consistent",
    "latency_length_correlated",
    "first_token_ratio_plausible",
    "usage_fingerprint_score",
]

# Feature importance weights (v6: will be overridden by data-driven calculation)
FEATURE_IMPORTANCE = {
    "avg_response_length": 1.0,
    "avg_markdown_score": 0.5,
    "latency_mean_ms": 1.5,
    "tokens_per_second": 1.2,
    "refusal_verbosity": 0.8,
    "avg_sentence_count": 0.6,
    "avg_words_per_sentence": 0.6,
    "has_usage_fields": 0.5,
    "has_finish_reason": 0.5,
    "param_compliance_rate": 1.0,
    "format_compliance_score": 1.0,
    "protocol_success_rate": 1.2,
    "instruction_pass_rate": 1.5,
    "exact_match_rate": 1.3,
    "json_valid_rate": 1.0,
    "format_follow_rate": 1.0,
    "line_count_rate": 0.8,
    "response_quality_basic_rate": 1.2,
    "response_quality_strict_rate": 1.0,
    "code_execution_rate": 1.5,
    "identity_consistency_rate": 1.3,
    "hallucination_detection_rate": 1.4,
    "topic_relevance_rate": 1.2,
    "refusal_rate": 1.1,
    "over_refusal_rate": 0.9,
    "safety_alternative_style": 1.0,
    "adversarial_spoof_signal_rate": 1.3,
    "extraction_resist_rate": 1.2,
    "token_rounding_anomaly": 2.0,
    "zero_completion_anomaly": 2.0,
    "identical_token_anomaly": 2.0,
    "temp_zero_diversity": 1.5,
    "token_count_consistent": 1.8,
    "latency_length_correlated": 1.6,
    "first_token_ratio_plausible": 1.7,
    "usage_fingerprint_score": 1.5,
}


class SimilarityEngine:
    """Calculate similarity between runs and manage baseline comparisons."""

    @staticmethod
    def compute_feature_importance_from_baselines(baselines: List[Dict]) -> Dict[str, float]:
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

        # Collect feature vectors from baselines
        vectors = []
        for b in baselines:
            if isinstance(b, dict) and b.get("feature_vector"):
                vectors.append(b["feature_vector"])

        if len(vectors) < 3:
            return FEATURE_IMPORTANCE

        # Convert to numpy array for easier calculation
        arr = np.array(vectors)
        
        # Calculate standard deviation for each feature
        std_devs = np.std(arr, axis=0)
        
        # Map to feature names
        importance = {}
        for i, feature in enumerate(FEATURE_ORDER):
            if i < len(std_devs):
                # Normalize to range 0.5-3.0
                # Higher std dev = higher importance
                std = std_devs[i]
                # Scale std dev to importance range
                importance[feature] = max(0.5, min(3.0, 0.5 + std * 2))
            else:
                importance[feature] = 1.0

        return importance

    @staticmethod
    def calculate_similarity(
        run_features: Dict[str, float],
        baseline_features: Dict[str, float],
        feature_importance: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, int]:
        """
        Calculate cosine similarity between two feature vectors.
        
        Args:
            run_features: Features from the current run
            baseline_features: Features from baseline run
            feature_importance: Optional feature importance weights
            
        Returns:
            (similarity_score, valid_feature_count)
        """
        # Build vectors with masks
        vec1, mask1 = SimilarityEngine._to_vector_with_mask(run_features, None)
        vec2, mask2 = SimilarityEngine._to_vector_with_mask(baseline_features, None)
        
        # Calculate similarity with mask
        similarity, valid_count = SimilarityEngine._cosine_similarity_with_mask(vec1, vec2, mask1, mask2)
        
        # Apply feature importance weights if provided
        if feature_importance and valid_count >= 5:
            weighted_similarity = SimilarityEngine._apply_feature_weights(
                run_features, baseline_features, feature_importance
            )
            # Blend weighted and unweighted similarities
            similarity = similarity * 0.7 + weighted_similarity * 0.3
        
        return similarity, valid_count

    @staticmethod
    def calculate_similarity_with_ci(
        run_features: Dict[str, float],
        baseline_features: Dict[str, float],
        feature_importance: Optional[Dict[str, float]] = None,
        rng: Optional[random.Random] = None,
    ) -> Tuple[float, float, float, int]:
        """
        Calculate similarity with confidence intervals using bootstrap.
        
        Args:
            run_features: Features from the current run
            baseline_features: Features from baseline run
            feature_importance: Optional feature importance weights
            rng: Random number generator for bootstrap
            
        Returns:
            (similarity_score, ci_95_low, ci_95_high, valid_feature_count)
        """
        if rng is None:
            rng = random.Random(42)
            
        similarity, valid_count = SimilarityEngine.calculate_similarity(
            run_features, baseline_features, feature_importance
        )
        
        if valid_count < 5:
            return similarity, None, None, valid_count
            
        # Bootstrap CI calculation
        vec1, mask1 = SimilarityEngine._to_vector_with_mask(run_features, None)
        vec2, mask2 = SimilarityEngine._to_vector_with_mask(baseline_features, None)
        
        ci_low, ci_high = SimilarityEngine._cosine_similarity_with_bootstrap_ci(
            vec1, vec2, rng
        )
        
        return similarity, ci_low, ci_high, valid_count

    @staticmethod
    def _to_vector_with_mask(
        features: Dict[str, float],
        normalization_params: Dict[str, float] | None = None,
    ) -> Tuple[List[float], List[bool]]:
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
                val = 1.0 - min(1.0, val / max_val) if max_val > 0 else val
            elif key == "tokens_per_second":
                max_val = norms.get("tokens_per_second_max", 200.0)
                val = min(1.0, val / max_val) if max_val > 0 else val
            elif key == "refusal_verbosity":
                max_val = norms.get("refusal_verbosity_max", 200.0)
                val = min(1.0, val / max_val) if max_val > 0 else val
            elif key == "avg_sentence_count":
                max_val = norms.get("avg_sentence_count_max", 15.0)
                val = min(1.0, val / max_val) if max_val > 0 else val
            elif key == "avg_words_per_sentence":
                max_val = norms.get("avg_words_per_sentence_max", 30.0)
                val = min(1.0, val / max_val) if max_val > 0 else val
            else:
                # For rate-based features, assume already in [0,1] range
                val = max(0.0, min(1.0, val))

            vec.append(val)
            mask.append(True)

        return vec, mask

    @staticmethod
    def _cosine_similarity_with_mask(
        vec1: List[float], 
        vec2: List[float], 
        mask1: Optional[List[bool]] = None,
        mask2: Optional[List[bool]] = None
    ) -> Tuple[float, int]:
        """
        Calculate cosine similarity with feature mask support.
        
        Returns (similarity_score, valid_feature_count).
        """
        if mask1 is None:
            mask1 = [True] * len(vec1)
        if mask2 is None:
            mask2 = [True] * len(vec2)
            
        valid = [i for i in range(len(vec1)) if mask1[i] and mask2[i]]
        valid_count = len(valid)
        # v6 fix: Lowered minimum threshold from 8 to 5 for quick mode support
        if valid_count < 5:
            return 0.0, valid_count  # Insufficient features
            
        vec1_v = [vec1[i] for i in valid]
        vec2_v = [vec2[i] for i in valid]
        
        dot = sum(x * y for x, y in zip(vec1_v, vec2_v))
        norm1 = math.sqrt(sum(x * x for x in vec1_v))
        norm2 = math.sqrt(sum(y * y for y in vec2_v))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0, valid_count
            
        return dot / (norm1 * norm2), valid_count

    @staticmethod
    def _cosine_similarity_with_bootstrap_ci(
        vec1: List[float], 
        vec2: List[float], 
        rng: random.Random,
        n_bootstrap: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Calculate cosine similarity with bootstrap confidence intervals.
        
        Returns (ci_95_low, ci_95_high).
        """
        if n_bootstrap is None:
            # v6 fix: High similarity needs more samples for precise CI estimation
            raw_sim = SimilarityEngine._cosine_similarity_with_mask(vec1, vec2)[0]
            if raw_sim >= 0.90:
                n_bootstrap = 200  # default 200
            elif raw_sim >= 0.75:
                n_bootstrap = 150
            else:
                n_bootstrap = 100  # Low similarity can use fewer samples

        length = len(vec1)
        sims = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample with replacement
            indices = [rng.randrange(length) for _ in range(length)]
            vec1_boot = [vec1[i] for i in indices]
            vec2_boot = [vec2[i] for i in indices]
            
            sim, _ = SimilarityEngine._cosine_similarity_with_mask(vec1_boot, vec2_boot)
            sims.append(sim)

        # Calculate 95% confidence interval
        sims.sort()
        lower_idx = int(0.025 * n_bootstrap)
        upper_idx = int(0.975 * n_bootstrap)
        
        return sims[lower_idx], sims[upper_idx]

    @staticmethod
    def _apply_feature_weights(
        features1: Dict[str, float],
        features2: Dict[str, float],
        weights: Dict[str, float]
    ) -> float:
        """
        Apply feature importance weights to similarity calculation.
        """
        weighted_similarities = []
        weight_sum = 0.0
        
        for feature, weight in weights.items():
            if feature in features1 and feature in features2:
                val1 = features1[feature]
                val2 = features2[feature]
                
                if val1 is not None and val2 is not None:
                    # Simple similarity for this feature
                    if val1 == 0 and val2 == 0:
                        feature_sim = 1.0
                    else:
                        max_val = max(abs(val1), abs(val2))
                        if max_val > 0:
                            feature_sim = 1.0 - abs(val1 - val2) / max_val
                        else:
                            feature_sim = 1.0
                    
                    weighted_similarities.append(feature_sim * weight)
                    weight_sum += weight
        
        if weight_sum == 0:
            return 0.0
            
        return sum(weighted_similarities) / weight_sum

    @staticmethod
    def rank_similarities(
        run_features: Dict[str, float],
        baselines: List[Dict],
        feature_importance: Optional[Dict[str, float]] = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        Rank baselines by similarity to the given run.
        
        Args:
            run_features: Features from the current run
            baselines: List of baseline dictionaries with 'features' field
            feature_importance: Optional feature importance weights
            limit: Maximum number of results to return
            
        Returns:
            List of similarity results sorted by descending similarity
        """
        results = []
        
        for baseline in baselines:
            if not isinstance(baseline, dict) or not baseline.get("features"):
                continue
                
            baseline_features = baseline["features"]
            similarity, ci_low, ci_high, valid_count = SimilarityEngine.calculate_similarity_with_ci(
                run_features, baseline_features, feature_importance
            )
            
            if valid_count >= 5:  # Only include results with sufficient features
                results.append({
                    "benchmark": baseline.get("model_name", "Unknown"),
                    "similarity": similarity,
                    "ci_95_low": ci_low,
                    "ci_95_high": ci_high,
                    "valid_features": valid_count,
                    "run_id": baseline.get("run_id"),
                    "overall_score": baseline.get("overall_score"),
                })
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        return results[:limit]
