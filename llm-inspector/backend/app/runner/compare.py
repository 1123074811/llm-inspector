"""
runner/compare.py — run_compare_pipeline, A/B significance testing

Builds comparison reports between two completed runs.
Extracted from orchestrator.py to keep individual files under ~200 lines.
"""
from __future__ import annotations

import random

from app.core.schemas import CaseResult
from app.core.config import settings
from app.core.logging import get_logger
from app.analysis.pipeline import (
    FeatureExtractor, ScoreCardCalculator, SimilarityEngine,
    AnalysisPipeline,
)
from app.repository import repo

logger = get_logger(__name__)


def run_compare_pipeline(compare_id: str) -> None:
    """
    Build comparison report from two completed runs.
    compare_runs.details will include score deltas and A/B significance.
    """
    logger.info("Compare pipeline starting", compare_id=compare_id)
    compare_row = repo.get_compare_run(compare_id)
    if not compare_row:
        logger.error("Compare run not found", compare_id=compare_id)
        return

    golden_id = compare_row["golden_run_id"]
    candidate_id = compare_row["candidate_run_id"]

    repo.update_compare_run(compare_id, status="running")

    golden_report_row = repo.get_report(golden_id)
    candidate_report_row = repo.get_report(candidate_id)
    if not golden_report_row or not candidate_report_row:
        repo.update_compare_run(
            compare_id,
            status="failed",
            details={"error": "Both runs must be completed and have reports"},
        )
        return

    golden = golden_report_row.get("details") or {}
    candidate = candidate_report_row.get("details") or {}

    g_sc = (golden.get("scorecard") or {})
    c_sc = (candidate.get("scorecard") or {})

    g_total = float(g_sc.get("total_score", 0.0) or 0.0)
    c_total = float(c_sc.get("total_score", 0.0) or 0.0)
    g_cap = float(g_sc.get("capability_score", 0.0) or 0.0)
    c_cap = float(c_sc.get("capability_score", 0.0) or 0.0)
    g_auth = float(g_sc.get("authenticity_score", 0.0) or 0.0)
    c_auth = float(c_sc.get("authenticity_score", 0.0) or 0.0)

    g_sim = (golden.get("similarity") or [{}])[0]
    c_sim = (candidate.get("similarity") or [{}])[0]
    g_top_sim = float(g_sim.get("score", 0.0) or 0.0)
    c_top_sim = float(c_sim.get("score", 0.0) or 0.0)

    delta_total = round(c_total - g_total, 1)
    delta_capability = round(c_cap - g_cap, 1)
    delta_authenticity = round(c_auth - g_auth, 1)
    delta_top_similarity = round(c_top_sim - g_top_sim, 4)

    ab_stats = _compute_ab_significance(golden, candidate)

    reasons: list[str] = []
    if delta_total <= -20:
        reasons.append(f"总分低于官方基线 {abs(delta_total):.1f} 分")
    if delta_authenticity <= -15:
        reasons.append(f"真实性分低于官方基线 {abs(delta_authenticity):.1f} 分")
    if delta_top_similarity <= -0.15:
        reasons.append(f"行为相似度低于官方基线 {abs(delta_top_similarity):.2f}")

    sig_regressions = [
        s for s in ab_stats
        if s.get("significant") and s.get("delta", 0) < 0
    ]
    if sig_regressions:
        reasons.append(f"存在 {len(sig_regressions)} 项统计显著退化")

    if not reasons:
        level = "close"
        label = "接近官方基线 / Close to Baseline"
        reasons.append("候选渠道与官方基线差距可接受")
    elif delta_total <= -35 or delta_authenticity <= -30 or len(sig_regressions) >= 2:
        level = "high_risk"
        label = "高风险疑似降级/假模型 / High Risk"
    else:
        level = "suspicious"
        label = "存在可疑差距 / Suspicious Gap"

    details = {
        "compare_id": compare_id,
        "golden_run_id": golden_id,
        "candidate_run_id": candidate_id,
        "deltas": {
            "total": delta_total,
            "capability": delta_capability,
            "authenticity": delta_authenticity,
            "top_similarity": delta_top_similarity,
        },
        "golden": {
            "scorecard": g_sc,
            "top_similarity": g_sim,
        },
        "candidate": {
            "scorecard": c_sc,
            "top_similarity": c_sim,
        },
        "ab_significance": ab_stats,
        "verdict": {
            "level": level,
            "label": label,
            "reasons": reasons,
        },
    }

    repo.update_compare_run(compare_id, status="completed", details=details)
    logger.info("Compare pipeline complete", compare_id=compare_id, level=level)


def _compute_ab_significance(golden_report: dict, candidate_report: dict) -> list[dict]:
    metrics = [
        "pass_rate",
        "mean_latency_ms",
    ]

    g_cases = {c.get("case_id"): c for c in (golden_report.get("case_results") or [])}
    c_cases = {c.get("case_id"): c for c in (candidate_report.get("case_results") or [])}
    common_ids = [cid for cid in g_cases.keys() if cid in c_cases]

    out = []
    for metric in metrics:
        g_vals: list[float] = []
        c_vals: list[float] = []
        for cid in common_ids:
            gv = g_cases[cid].get(metric)
            cv = c_cases[cid].get(metric)
            if gv is None or cv is None:
                continue
            try:
                g_vals.append(float(gv))
                c_vals.append(float(cv))
            except (TypeError, ValueError):
                continue

        if len(g_vals) < 3 or len(c_vals) < 3:
            continue

        out.append(_paired_bootstrap(metric, g_vals, c_vals))

    return out


def _paired_bootstrap(metric: str, g_vals: list[float], c_vals: list[float], n: int = 1000) -> dict:
    deltas = [c - g for c, g in zip(c_vals, g_vals)]
    mean_g = sum(g_vals) / len(g_vals)
    mean_c = sum(c_vals) / len(c_vals)
    mean_delta = mean_c - mean_g

    rng = random.Random(42)
    boots = []
    for _ in range(n):
        idxs = [rng.randrange(len(deltas)) for _ in range(len(deltas))]
        sample = [deltas[i] for i in idxs]
        boots.append(sum(sample) / len(sample))

    boots.sort()
    lo = boots[int(0.025 * n)]
    hi = boots[int(0.975 * n)]

    opp_sign = sum(1 for b in boots if (b <= 0 if mean_delta > 0 else b >= 0))
    p_value = min(1.0, 2 * opp_sign / n)
    significant = (lo > 0) or (hi < 0)

    return {
        "metric": metric,
        "golden_mean": round(mean_g, 4),
        "candidate_mean": round(mean_c, 4),
        "delta": round(mean_delta, 4),
        "ci_95_low": round(lo, 4),
        "ci_95_high": round(hi, 4),
        "p_value": round(p_value, 6),
        "significant": significant,
    }
