"""
analysis/pipeline.py — thin shim + AnalysisPipeline utility class

This file used to be a 3345-line monolith.  It is now a re-export shim so
that every existing import of the form

    from app.analysis.pipeline import FeatureExtractor

continues to work without change, while the actual implementations live in
smaller, focused modules:

  feature_engine.py   — FeatureExtractor
  scoring.py          — ScoreCalculator, ScoreCardCalculator
  verdicts.py         — VerdictEngine
  similarity.py       — FEATURE_ORDER, FEATURE_IMPORTANCE, SimilarityEngine
  estimation.py       — RiskEngine, ThetaEstimator, UncertaintyEstimator,
                        PercentileMapper, PairwiseEngine
  reporting.py        — NarrativeBuilder, ProxyLatencyAnalyzer,
                        ExtractionAuditBuilder, ReportBuilder
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from app.core.schemas import (
    ScoreCard,
    Scores,           # noqa: F401  — re-exported for `from app.analysis.pipeline import Scores`
    RiskAssessment,   # noqa: F401  — re-exported for `from app.analysis.pipeline import RiskAssessment`
)
from app.core.config import settings
from app.core.logging import get_logger

# ── Re-exports from split modules ─────────────────────────────────────────────

from app.analysis.feature_engine import FeatureExtractor                        # noqa: F401
from app.analysis.scoring import ScoreCalculator, ScoreCardCalculator           # noqa: F401
from app.analysis.verdicts import VerdictEngine                                 # noqa: F401
from app.analysis.similarity import (                                           # noqa: F401
    FEATURE_ORDER, FEATURE_IMPORTANCE, SimilarityEngine,
)
from app.analysis.estimation import (                                           # noqa: F401
    RiskEngine, ThetaEstimator, UncertaintyEstimator, PercentileMapper,
    PairwiseEngine,
)
from app.analysis.reporting import (                                            # noqa: F401
    NarrativeBuilder, ProxyLatencyAnalyzer, ExtractionAuditBuilder, ReportBuilder,
)

logger = get_logger(__name__)


# ── Reference embedding helpers (v13 Phase 5) ─────────────────────────────────

def _load_reference_embeddings() -> dict:
    """Load reference embeddings from _data/reference_embeddings.json."""
    try:
        path = Path(__file__).parent.parent / "_data" / "reference_embeddings.json"
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("models", {})
    except Exception:
        return {}


# ── AnalysisPipeline — baseline comparison utility ────────────────────────────

class AnalysisPipeline:
    @staticmethod
    def build_similarity_comparisons(
        current_features: dict[str, float],
        current_card: ScoreCard,
        golden_baselines: list[dict],
    ) -> list[dict]:
        """
        Build baseline comparison dicts for all available baselines.

        When ``golden_baselines`` is empty, falls back to reference embeddings
        from ``_data/reference_embeddings.json`` (v13 Phase 5).

        Each returned dict includes a ``baseline_source`` key:
          - ``"user"``      — real user-marked golden baseline
          - ``"reference"`` — embedded reference data (HELM/Arena-derived)
        """
        if golden_baselines:
            results = []
            for bl in golden_baselines:
                fv = bl.get("feature_vector") or {}
                bl_scores = {
                    "total_score":        bl.get("total_score", 0.0),
                    "capability_score":   bl.get("capability_score", 0.0),
                    "authenticity_score": bl.get("authenticity_score", 0.0),
                    "performance_score":  bl.get("performance_score", 0.0),
                }
                comp = AnalysisPipeline.compare_with_baseline(
                    current_features, fv, current_card, bl_scores
                )
                comp["model_name"] = bl.get("model_name", "unknown")
                comp["baseline_source"] = "user"
                results.append(comp)
            return results

        # Fallback: reference embeddings
        ref_embeddings = _load_reference_embeddings()
        if ref_embeddings:
            similarities = []
            for model_name, ref_data in ref_embeddings.items():
                comp = AnalysisPipeline.compare_with_baseline(
                    current_features,
                    ref_data.get("features", {}),
                    current_card,
                    ref_data.get("scores", {}),
                )
                comp["model_name"] = model_name
                comp["baseline_source"] = "reference"
                similarities.append(comp)
            return sorted(
                similarities,
                key=lambda x: x.get("cosine_similarity", 0.0),
                reverse=True,
            )
        return []

    @staticmethod
    def compare_with_baseline(
        current_features: dict[str, float],
        baseline_feature_vector: dict[str, float],
        current_card: ScoreCard,
        baseline_scores: dict[str, float],
    ) -> dict:
        """
        计算当前运行与基准模型的差异。
        baseline_scores 为内部 0-100 单位。
        """
        try:
            all_keys = sorted(set(current_features) | set(baseline_feature_vector))
            vec_curr = [float(current_features.get(k, 0.0) or 0.0) for k in all_keys]
            vec_base = [float(baseline_feature_vector.get(k, 0.0) or 0.0) for k in all_keys]

            dot = sum(x * y for x, y in zip(vec_curr, vec_base))
            norm_curr = math.sqrt(sum(x * x for x in vec_curr))
            norm_base = math.sqrt(sum(y * y for y in vec_base))
            denom = norm_curr * norm_base
            cosine_sim = (dot / denom) if denom > 0 else 0.0

            delta_total = float(current_card.total_score) - float(baseline_scores.get("total_score", 0.0))
            delta_cap   = float(current_card.capability_score) - float(baseline_scores.get("capability_score", 0.0))
            delta_auth  = float(current_card.authenticity_score) - float(baseline_scores.get("authenticity_score", 0.0))
            delta_perf  = float(current_card.performance_score) - float(baseline_scores.get("performance_score", 0.0))

            feature_drift = {}
            for k in all_keys:
                base_val = float(baseline_feature_vector.get(k, 0.0) or 0.0)
                curr_val = float(current_features.get(k, 0.0) or 0.0)
                if base_val != 0:
                    pct = (curr_val - base_val) / abs(base_val) * 100
                else:
                    pct = 0.0
                feature_drift[k] = {
                    "baseline":   round(base_val, 4),
                    "current":    round(curr_val, 4),
                    "delta_pct":  round(pct, 2),
                }
            top5 = dict(
                sorted(feature_drift.items(), key=lambda x: abs(x[1]["delta_pct"]), reverse=True)[:5]
            )

            abs_delta_total_display = abs(delta_total) * 100
            if (
                cosine_sim >= settings.BASELINE_MATCH_COSINE_THRESHOLD
                and abs_delta_total_display <= settings.BASELINE_MATCH_SCORE_DELTA_MAX
            ):
                verdict = "match"
            elif cosine_sim >= 0.85 or abs_delta_total_display <= 1500:
                verdict = "suspicious"
            else:
                verdict = "mismatch"

            return {
                "cosine_similarity": round(cosine_sim, 4),
                "score_delta": {
                    "total":        round(delta_total * 100),
                    "capability":   round(delta_cap   * 100),
                    "authenticity": round(delta_auth  * 100),
                    "performance":  round(delta_perf  * 100),
                },
                "feature_drift_top5": top5,
                "verdict": verdict,
            }
        except KeyError as e:
            return {
                "cosine_similarity": 0.0,
                "score_delta": {"total": 0.0, "capability": 0.0, "authenticity": 0.0, "performance": 0.0},
                "feature_drift_top5": {},
                "verdict": "mismatch",
                "error": f"Missing key: {str(e)}",
            }
        except Exception as e:
            return {
                "cosine_similarity": 0.0,
                "score_delta": {"total": 0.0, "capability": 0.0, "authenticity": 0.0, "performance": 0.0},
                "feature_drift_top5": {},
                "verdict": "mismatch",
                "error": f"Comparison error: {str(e)}",
            }
