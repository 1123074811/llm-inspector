"""
handlers/v14_handlers.py — v14 API handlers.

Phase 2: Bradley-Terry leaderboard (pairwise strength estimation).
Phase 3+: Identity Exposure Engine endpoints (planned).

Bradley-Terry model reference:
    Bradley & Terry (1952) "Rank Analysis of Incomplete Block Designs"
    Biometrika, 39(3/4), 324-345.
    URL: https://doi.org/10.1093/biomet/39.3-4.324
"""
from __future__ import annotations

import math
from app.handlers.helpers import _json, _error
from app.core.logging import get_logger

logger = get_logger(__name__)


def _compute_bradley_terry(
    comparisons: list[dict],
    max_iter: int = 200,
    tol: float = 1e-6,
) -> dict[str, float]:
    """
    Iterative ML estimation of Bradley-Terry strengths.

    Each comparison is {"winner": model_a, "loser": model_b}.
    Returns {model_name: strength} where strengths are normalised so
    sum(exp(s)) = N (number of models).

    Reference:
        Bradley & Terry (1952) Biometrika 39:324.
        Hunter (2004) Ann. Stat. 32(1):384-406 (MM algorithm).
    """
    if not comparisons:
        return {}

    # Collect all models
    models = sorted({c["winner"] for c in comparisons} | {c["loser"] for c in comparisons})
    n = len(models)
    if n < 2:
        return {models[0]: 1.0} if models else {}

    idx = {m: i for i, m in enumerate(models)}

    # Win / comparison counts
    wins = [0.0] * n
    played = [[0.0] * n for _ in range(n)]
    for c in comparisons:
        w = idx[c["winner"]]
        l = idx[c["loser"]]
        wins[w] += 1
        played[w][l] += 1
        played[l][w] += 1

    # Initial strengths = 1.0
    s = [1.0] * n

    for _ in range(max_iter):
        s_new = [0.0] * n
        for i in range(n):
            denom = sum(
                played[i][j] / (s[i] + s[j])
                for j in range(n) if played[i][j] > 0
            )
            s_new[i] = wins[i] / denom if denom > 0 else s[i]

        # Normalise: geometric mean = 1
        log_mean = sum(math.log(max(x, 1e-12)) for x in s_new) / n
        scale = math.exp(log_mean)
        s_new = [x / scale for x in s_new]

        # Convergence check
        delta = max(abs(s_new[i] - s[i]) for i in range(n))
        s = s_new
        if delta < tol:
            break

    return {models[i]: round(s[i], 4) for i in range(n)}


def handle_bt_leaderboard(path: str, qs: dict, body: dict):
    """
    GET /api/v14/bt-leaderboard

    Returns Bradley-Terry strength estimates from compare-run pairwise data.
    If no compare-run data is available, falls back to ELO data.

    Response:
    {
      "model": "bradley_terry",
      "reference": "Bradley & Terry (1952) Biometrika 39:324",
      "models": [
        {"rank": 1, "model": "gpt-4o", "bt_strength": 2.341, "wins": 12, "losses": 3},
        ...
      ],
      "total_comparisons": 45,
      "note": "..."
    }
    """
    try:
        import app.repository.repo as repo_module

        # Load pairwise comparisons from compare_runs
        comparisons = []
        try:
            compare_runs = repo_module.list_compare_runs(limit=500)
            for cr in compare_runs:
                r = cr if isinstance(cr, dict) else (cr.__dict__ if hasattr(cr, "__dict__") else {})
                winner = r.get("winner_model") or r.get("model_a") if (r.get("result") == "a_wins") else (
                    r.get("model_b") if r.get("result") == "b_wins" else None
                )
                loser = r.get("model_b") if r.get("result") == "a_wins" else (
                    r.get("model_a") if r.get("result") == "b_wins" else None
                )
                if winner and loser:
                    comparisons.append({"winner": winner, "loser": loser})
        except Exception as e:
            logger.warning("Could not load compare_runs for BT leaderboard", error=str(e))

        # Fallback: synthesise comparisons from ELO leaderboard scores
        if not comparisons:
            try:
                lb = repo_module.get_leaderboard(limit=100)
                # Build pseudo-comparisons from ELO score ordering
                models_with_scores = []
                for entry in (lb if isinstance(lb, list) else []):
                    e = entry if isinstance(entry, dict) else {}
                    m = e.get("model_name") or e.get("model")
                    s = e.get("elo_score") or e.get("total_score") or 0
                    if m:
                        models_with_scores.append((m, float(s)))
                models_with_scores.sort(key=lambda x: x[1], reverse=True)
                # Each pair: higher-ranked model "wins" by score delta
                for i in range(len(models_with_scores)):
                    for j in range(i + 1, min(i + 5, len(models_with_scores))):
                        m_a, s_a = models_with_scores[i]
                        m_b, s_b = models_with_scores[j]
                        if s_a > s_b:
                            comparisons.append({"winner": m_a, "loser": m_b})
                        elif s_b > s_a:
                            comparisons.append({"winner": m_b, "loser": m_a})
            except Exception as e:
                logger.warning("Could not load ELO data for BT fallback", error=str(e))

        strengths = _compute_bradley_terry(comparisons)
        if not strengths:
            return _json({
                "model": "bradley_terry",
                "reference": "Bradley & Terry (1952) Biometrika 39:324. DOI:10.1093/biomet/39.3-4.324",
                "models": [],
                "total_comparisons": 0,
                "note": "No pairwise comparison data available yet. Run compare-runs to populate.",
            })

        # Build win/loss stats
        win_count: dict[str, int] = {}
        loss_count: dict[str, int] = {}
        for c in comparisons:
            win_count[c["winner"]] = win_count.get(c["winner"], 0) + 1
            loss_count[c["loser"]] = loss_count.get(c["loser"], 0) + 1

        ranked = sorted(strengths.items(), key=lambda x: x[1], reverse=True)
        models_list = [
            {
                "rank": i + 1,
                "model": m,
                "bt_strength": s,
                "wins": win_count.get(m, 0),
                "losses": loss_count.get(m, 0),
            }
            for i, (m, s) in enumerate(ranked)
        ]

        return _json({
            "model": "bradley_terry",
            "reference": "Bradley & Terry (1952) Biometrika 39:324. DOI:10.1093/biomet/39.3-4.324",
            "models": models_list,
            "total_comparisons": len(comparisons),
            "note": "Strengths estimated via MM algorithm (Hunter 2004). Higher = stronger model.",
        })

    except Exception as e:
        logger.error("BT leaderboard error", error=str(e))
        return _error(f"Bradley-Terry leaderboard error: {e}", 500)
