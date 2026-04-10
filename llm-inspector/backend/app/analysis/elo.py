"""
ELO Ranking Engine for LLM Inspector.

Implements the standard ELO formula used by LMSYS Chatbot Arena,
adapted for automated pairwise comparisons derived from the
Bradley-Terry model (PairwiseEngine win_prob output).

ELO update rules:
  Ra' = Ra + K * (Sa - Ea)
  Ea  = 1 / (1 + 10^((Rb - Ra) / 400))
  Sa  = 1 (win), 0.5 (draw), 0 (loss)
  K   = 32  when games_played < 10  (provisional)
      = 16  when games_played >= 10 (established)
      = 12  when games_played >= 30  (elite)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from app.core.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_ELO   = 1500.0
_K_PROVISIONAL = 32.0   # < 10 games
_K_ESTABLISHED = 16.0   # 10–29 games
_K_ELITE       = 12.0   # 30+ games

# Win probability threshold: if |win_prob - 0.5| < DRAW_THRESHOLD → draw
_DRAW_THRESHOLD = 0.08


@dataclass
class EloRecord:
    model_name:   str
    display_name: str
    elo_rating:   float = _DEFAULT_ELO
    games_played: int   = 0
    wins:         int   = 0
    losses:       int   = 0
    draws:        int   = 0
    peak_elo:     float = _DEFAULT_ELO
    last_run_id:  str | None = None


def _k_factor(games_played: int) -> float:
    if games_played < 10:
        return _K_PROVISIONAL
    if games_played < 30:
        return _K_ESTABLISHED
    return _K_ELITE


def expected_score(rating_a: float, rating_b: float) -> float:
    """ELO expected score for player A vs player B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _outcome_from_win_prob(win_prob_a: float) -> tuple[float, float]:
    """
    Convert a win probability into a (score_a, score_b) pair.
    Returns (1,0) for a win, (0.5,0.5) for draw, (0,1) for loss.
    """
    if abs(win_prob_a - 0.5) < _DRAW_THRESHOLD:
        return 0.5, 0.5
    if win_prob_a > 0.5:
        return 1.0, 0.0
    return 0.0, 1.0


def update_elo(
    record_a: EloRecord,
    record_b: EloRecord,
    win_prob_a: float,
    run_id_a: str | None = None,
) -> tuple[EloRecord, EloRecord]:
    """
    Update ELO ratings for model_a and model_b given win_prob_a.

    win_prob_a: probability that model_a beats model_b (from PairwiseEngine).
    Returns updated (record_a, record_b).
    """
    score_a, score_b = _outcome_from_win_prob(win_prob_a)
    ea = expected_score(record_a.elo_rating, record_b.elo_rating)
    eb = 1.0 - ea

    ka = _k_factor(record_a.games_played)
    kb = _k_factor(record_b.games_played)

    new_ra = record_a.elo_rating + ka * (score_a - ea)
    new_rb = record_b.elo_rating + kb * (score_b - eb)

    # Build updated records
    def _updated(rec: EloRecord, new_r: float, score: float, run_id: str | None) -> EloRecord:
        wins   = rec.wins   + (1 if score == 1.0 else 0)
        losses = rec.losses + (1 if score == 0.0 else 0)
        draws  = rec.draws  + (1 if score == 0.5 else 0)
        return EloRecord(
            model_name=rec.model_name,
            display_name=rec.display_name,
            elo_rating=round(new_r, 2),
            games_played=rec.games_played + 1,
            wins=wins,
            losses=losses,
            draws=draws,
            peak_elo=max(rec.peak_elo, new_r),
            last_run_id=run_id or rec.last_run_id,
        )

    ua = _updated(record_a, new_ra, score_a, run_id_a)
    ub = _updated(record_b, new_rb, score_b, None)

    logger.info(
        "ELO updated",
        model_a=record_a.model_name,
        model_b=record_b.model_name,
        win_prob_a=round(win_prob_a, 3),
        score_a=score_a,
        old_elo_a=record_a.elo_rating,
        new_elo_a=ua.elo_rating,
        old_elo_b=record_b.elo_rating,
        new_elo_b=ub.elo_rating,
    )
    return ua, ub


class EloLeaderboard:
    """
    Façade that reads/writes from the DB via the repo layer.
    All persistence is delegated; this class is pure logic + orchestration.
    """

    def update_from_pairwise(
        self,
        model_a: str,
        display_a: str,
        model_b: str,
        display_b: str,
        win_prob_a: float,
        run_id: str | None = None,
    ) -> tuple[float, float]:
        """
        Load ELO records for both models, update ratings, persist.
        Returns (new_elo_a, new_elo_b).
        """
        from app.repository import repo

        def _load(name: str, display: str) -> EloRecord:
            row = repo.get_elo(name)
            if row:
                return EloRecord(
                    model_name=row["model_name"],
                    display_name=row.get("display_name", name),
                    elo_rating=float(row["elo_rating"]),
                    games_played=int(row["games_played"]),
                    wins=int(row["wins"]),
                    losses=int(row["losses"]),
                    draws=int(row["draws"]),
                    peak_elo=float(row["peak_elo"]),
                    last_run_id=row.get("last_run_id"),
                )
            return EloRecord(model_name=name, display_name=display)

        rec_a = _load(model_a, display_a)
        rec_b = _load(model_b, display_b)

        ua, ub = update_elo(rec_a, rec_b, win_prob_a, run_id_a=run_id)

        repo.upsert_elo(ua)
        repo.upsert_elo(ub)

        return ua.elo_rating, ub.elo_rating

    def get_rankings(self, limit: int = 100) -> list[dict]:
        """Return sorted ELO leaderboard from DB."""
        from app.repository import repo
        return repo.list_elo(limit=limit)
