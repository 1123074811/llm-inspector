"""
ELO and Glicko-2 Ranking Engine for LLM Inspector.

v10 Update: Implements Glicko-2 rating system which accounts for Rating Deviation (RD)
and Volatility (sigma). The classic ELO formula is kept for backward compatibility.

Reference: Glickman (1999) "The Glicko system" & (2001) "Glicko-2 system"
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from app.core.logging import get_logger

logger = get_logger(__name__)

# ELO Constants
_DEFAULT_ELO   = 1500.0
_K_PROVISIONAL = 32.0   # < 10 games
_K_ESTABLISHED = 16.0   # 10–29 games
_K_ELITE       = 12.0   # 30+ games

# Glicko-2 Constants
_GLICKO_TAU = 0.5  # System constant
_GLICKO_SCALE = 173.7178

# Win probability threshold: if |win_prob - 0.5| < DRAW_THRESHOLD → draw
_DRAW_THRESHOLD = 0.08


@dataclass
class EloRecord:
    model_name:   str
    display_name: str
    elo_rating:   float = _DEFAULT_ELO
    # Glicko-2 specific parameters
    rd:           float = 350.0  # Rating Deviation
    volatility:   float = 0.06   # Volatility
    
    games_played: int   = 0
    wins:         int   = 0
    losses:       int   = 0
    draws:        int   = 0
    peak_elo:     float = _DEFAULT_ELO
    last_run_id:  str | None = None

    def get_glicko_mu(self) -> float:
        """Convert ELO scale to Glicko scale."""
        return (self.elo_rating - 1500) / _GLICKO_SCALE

    def get_glicko_phi(self) -> float:
        """Convert RD scale to Glicko scale."""
        return self.rd / _GLICKO_SCALE

def _glicko2_g(phi: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * phi ** 2 / (math.pi ** 2))

def _glicko2_E(mu: float, mu_j: float, phi_j: float) -> float:
    return 1.0 / (1.0 + math.exp(-_glicko2_g(phi_j) * (mu - mu_j)))

def update_glicko2(
    record_a: EloRecord,
    record_b: EloRecord,
    win_prob_a: float,
    run_id_a: str | None = None,
) -> tuple[EloRecord, EloRecord]:
    """
    v10: Update ratings using Glicko-2 algorithm.
    """
    score_a, score_b = _outcome_from_win_prob(win_prob_a)
    
    def _update_single(player: EloRecord, opponent: EloRecord, score: float, run_id: str | None) -> EloRecord:
        mu = player.get_glicko_mu()
        phi = player.get_glicko_phi()
        sigma = player.volatility
        
        mu_j = opponent.get_glicko_mu()
        phi_j = opponent.get_glicko_phi()
        
        g_j = _glicko2_g(phi_j)
        E_j = _glicko2_E(mu, mu_j, phi_j)
        
        # Estimated variance v
        v = 1.0 / (g_j ** 2 * E_j * (1.0 - E_j))
        
        # Estimated improvement delta
        delta = v * g_j * (score - E_j)
        
        # Step 5: update volatility (simplified iteration)
        a = math.log(sigma ** 2)
        
        def f(x: float) -> float:
            tmp = phi**2 + v + math.exp(x)
            part1 = math.exp(x) * (delta**2 - tmp) / (2 * tmp**2)
            part2 = (x - a) / (_GLICKO_TAU**2)
            return part1 - part2

        A = a
        if delta**2 > phi**2 + v:
            B = math.log(delta**2 - phi**2 - v)
        else:
            k = 1
            while f(a - k * _GLICKO_TAU) < 0:
                k += 1
            B = a - k * _GLICKO_TAU
            
        f_A = f(A)
        f_B = f(B)
        
        # Illinois algorithm for root finding
        for _ in range(20):
            C = A + (A - B) * f_A / (f_B - f_A)
            f_C = f(C)
            if abs(f_C) < 0.000001:
                break
            if f_C * f_B < 0:
                A, f_A = B, f_B
            else:
                f_A /= 2
            B, f_B = C, f_C
            
        new_sigma = math.exp(A / 2)
        
        # Step 6: Update RD and rating
        phi_star = math.sqrt(phi**2 + new_sigma**2)
        new_phi = 1.0 / math.sqrt(1.0 / phi_star**2 + 1.0 / v)
        new_mu = mu + new_phi**2 * g_j * (score - E_j)
        
        # Convert back to ELO scale
        new_rating = 1500 + new_mu * _GLICKO_SCALE
        new_rd = new_phi * _GLICKO_SCALE
        
        wins   = player.wins   + (1 if score == 1.0 else 0)
        losses = player.losses + (1 if score == 0.0 else 0)
        draws  = player.draws  + (1 if score == 0.5 else 0)
        
        return EloRecord(
            model_name=player.model_name,
            display_name=player.display_name,
            elo_rating=round(new_rating, 2),
            rd=round(new_rd, 2),
            volatility=round(new_sigma, 6),
            games_played=player.games_played + 1,
            wins=wins,
            losses=losses,
            draws=draws,
            peak_elo=max(player.peak_elo, new_rating),
            last_run_id=run_id or player.last_run_id,
        )

    ua = _update_single(record_a, record_b, score_a, run_id_a)
    ub = _update_single(record_b, record_a, score_b, None)

    logger.info(
        "Glicko-2 updated",
        model_a=record_a.model_name,
        model_b=record_b.model_name,
        score_a=score_a,
        old_rating_a=record_a.elo_rating,
        new_rating_a=ua.elo_rating,
        old_rd_a=record_a.rd,
        new_rd_a=ua.rd
    )
    return ua, ub


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
                    rd=float(row.get("rd", 350.0)),
                    volatility=float(row.get("volatility", 0.06)),
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

        # v10: Switch to Glicko-2
        ua, ub = update_glicko2(rec_a, rec_b, win_prob_a, run_id_a=run_id)

        repo.upsert_elo(ua)
        repo.upsert_elo(ub)

        return ua.elo_rating, ub.elo_rating

    def get_rankings(self, limit: int = 100) -> list[dict]:
        """Return sorted ELO leaderboard from DB."""
        from app.repository import repo
        return repo.list_elo(limit=limit)
