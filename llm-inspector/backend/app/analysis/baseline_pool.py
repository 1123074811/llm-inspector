"""
analysis/baseline_pool.py — v17 Phase 11: eligible-baseline pool selection.

Background
----------
Before v17 the similarity engine compared each run against a *static*
``_data/reference_embeddings.json`` curated by hand.  That meant
deprecated and sunset models stayed in the comparison set forever.

This module surfaces the model_registry table (Phase 5) as the new
single source of truth for "who is eligible to act as a baseline" and
exposes two helpers consumed by both the report assembly path and the
front-end's "baseline pool transparency" card:

  * ``filter_eligible_baselines(baselines)`` — accepts the legacy list of
    baseline dicts and returns only those whose ``model_name`` matches
    an active, recent, official-source registry row.

  * ``baseline_pool_summary()`` — returns an at-a-glance breakdown that
    reports.py renders in the "基线池透明化" panel:

      {
        "active_count":        int,
        "sources_breakdown":   {"openai_api": 12, "openrouter": 8, ...},
        "min_age_days":        30,
        "freshness_window":    14,
        "synced_at":           int (epoch),
        "recently_deprecated": ["claude-3-haiku", "gpt-4-turbo", ...],
      }

The filter intentionally **falls back to the full input list** when the
registry is empty (cold-start) — failing closed would break every run
on a fresh database.  Users are warned via a structured log line.
"""
from __future__ import annotations

import time
from typing import Any, Iterable

from app.core.logging import get_logger
from app.repository import registry_repo

logger = get_logger(__name__)


# ── Tunables (sync them with registry_repo defaults if you change one) ──────


_MIN_AGE_DAYS = 30
_FRESHNESS_WINDOW_DAYS = 14
_RECENTLY_DEPRECATED_WINDOW_DAYS = 14


# ── Public API ──────────────────────────────────────────────────────────────


def _baseline_model_name(b: dict) -> str | None:
    """Pluck the model identifier from a baseline dict shape-tolerantly."""
    if not isinstance(b, dict):
        return None
    for k in ("model_name", "model_id", "name"):
        v = b.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def list_eligible_model_ids(now: int | None = None) -> set[str]:
    """Return ids of registry rows that satisfy the Phase 11 filter.

    Empty set means "registry has nothing to say" — callers should treat
    that as a cold-start signal and pass through their input list rather
    than dropping every baseline.
    """
    now = int(now or time.time())
    try:
        rows = registry_repo.list_eligible_baselines(
            now=now,
            min_age_days=_MIN_AGE_DAYS,
            freshness_window_days=_FRESHNESS_WINDOW_DAYS,
        )
    except Exception as e:
        logger.warning("baseline_pool: registry_repo failed", error=str(e))
        return set()
    return {r["model_id"] for r in rows if r and r.get("model_id")}


def filter_eligible_baselines(
    baselines: Iterable[dict],
    now: int | None = None,
) -> list[dict]:
    """Return ``baselines`` filtered to registry-eligible models.

    Cold-start fallback: when the registry knows zero eligible models we
    pass the full list through unchanged so the system remains usable
    on a fresh deployment. Callers wanting to surface that fallback in
    the report can use :func:`filter_eligible_baselines_with_status`
    instead.
    """
    out, _ = filter_eligible_baselines_with_status(baselines, now=now)
    return out


def filter_eligible_baselines_with_status(
    baselines: Iterable[dict],
    now: int | None = None,
) -> tuple[list[dict], bool]:
    """Same as :func:`filter_eligible_baselines` but also returns
    ``is_fallback``: True when the registry was empty and the input list
    was passed through unchanged. Reports should display a
    ``baseline pool not yet initialized — comparing against legacy
    reference embeddings`` notice when this is True.
    """
    baselines = list(baselines or [])
    eligible = list_eligible_model_ids(now=now)
    if not eligible:
        if baselines:
            logger.info(
                "baseline_pool: registry empty, passing baselines through unchanged",
                input_count=len(baselines),
            )
        return baselines, True

    out: list[dict] = []
    dropped: list[str] = []
    for b in baselines:
        name = _baseline_model_name(b)
        if name and name in eligible:
            out.append(b)
        else:
            if name:
                dropped.append(name)
    if dropped:
        logger.info(
            "baseline_pool: filtered baselines",
            kept=len(out),
            dropped=len(dropped),
            sample_dropped=dropped[:5],
        )
    return out, False


def baseline_pool_summary(now: int | None = None) -> dict[str, Any]:
    """Aggregate registry stats for the "基线池透明化" report card."""
    now = int(now or time.time())
    try:
        rows = registry_repo.list_eligible_baselines(
            now=now,
            min_age_days=_MIN_AGE_DAYS,
            freshness_window_days=_FRESHNESS_WINDOW_DAYS,
        )
    except Exception as e:
        logger.warning("baseline_pool_summary: registry_repo failed", error=str(e))
        rows = []

    sources: dict[str, int] = {}
    last_synced = 0
    for r in rows:
        src = r.get("data_source") or "unknown"
        sources[src] = sources.get(src, 0) + 1
        if r.get("last_synced_at") and int(r["last_synced_at"]) > last_synced:
            last_synced = int(r["last_synced_at"])

    return {
        "active_count": len(rows),
        "sources_breakdown": dict(sorted(sources.items(), key=lambda kv: -kv[1])),
        "min_age_days": _MIN_AGE_DAYS,
        "freshness_window_days": _FRESHNESS_WINDOW_DAYS,
        "synced_at": last_synced or None,
        "recently_deprecated": _list_recently_deprecated(now),
    }


def _list_recently_deprecated(now: int) -> list[str]:
    """Names of models that flipped to ``deprecated`` in the last 14 days."""
    from app.core.db import get_conn
    cutoff = now - _RECENTLY_DEPRECATED_WINDOW_DAYS * 86400
    try:
        rows = get_conn().execute(
            "SELECT model_id FROM model_registry "
            "WHERE status='deprecated' AND deprecated_at IS NOT NULL "
            "AND deprecated_at > ? "
            "ORDER BY deprecated_at DESC LIMIT 20",
            (cutoff,),
        ).fetchall()
    except Exception as e:
        logger.debug("baseline_pool: recently_deprecated lookup failed", error=str(e))
        return []
    return [r["model_id"] for r in rows]


__all__ = [
    "filter_eligible_baselines",
    "filter_eligible_baselines_with_status",
    "list_eligible_model_ids",
    "baseline_pool_summary",
]
