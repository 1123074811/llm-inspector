"""
repository/registry_repo.py — v17 Phase 5: model_registry data access.

Single-source-of-truth for the table that downstream phases (6 sync, 7
changelog harvester, 8 self-probe) keep up-to-date and that Phase 11's
similarity engine reads as its eligible-baseline pool.

The schema is defined in ``app.core.db_migrations.Migration008V17ModelRegistry``.

Public surface (kept small to avoid coupling):
  upsert_model(record)               — INSERT OR UPDATE; auto-audits diffs
  mark_deprecated(model_id, ts)      — flip status='deprecated'
  mark_sunset(model_id, ts)          — flip status='sunset'
  get_model_card(model_id)           — full record dict or None
  get_official_price(model_id)       — {input,output,cache_read} USD/Mtok or None
  list_eligible_baselines(now)       — Phase 11 query
  audit_log(model_id, field, old, new, source)
  count_models()                     — total active rows (debug/monitoring)

Time fields (``first_seen_at``, ``last_seen_at``, ``deprecated_at``,
``sunset_at``, ``last_synced_at``, ``changed_at``) are stored as **Unix
epoch seconds**; callers should pass ``int(time.time())`` rather than ISO
strings.

Source-priority rules (used by ``upsert_model`` to decide whether an
incoming record may overwrite an existing one):

  official_api > openrouter > changelog > self_probed > manual

Lower-priority sources may *fill in* fields that are NULL on the existing
row but never overwrite higher-priority values.  Equal-or-higher priority
always overwrites.
"""
from __future__ import annotations

import json
import time
from typing import Any, Iterable

from app.core.db import get_conn
from app.core.logging import get_logger

logger = get_logger(__name__)


# Lower number = higher priority.
_SOURCE_PRIORITY: dict[str, int] = {
    "openai_api":    10,
    "anthropic_api": 10,
    "google_api":    10,
    "xai_api":       10,
    "deepseek_api":  10,
    "mistral_api":   10,
    "openrouter":    20,
    "changelog":     30,
    "self_probed":   40,
    "manual":        50,
}

_DEFAULT_PRIORITY = 60   # unknown source = lowest


def _priority(source: str | None) -> int:
    return _SOURCE_PRIORITY.get((source or "").lower(), _DEFAULT_PRIORITY)


# Columns that may be touched by upserts (omitting model_id, which is the PK).
_UPSERTABLE_COLUMNS: tuple[str, ...] = (
    "vendor", "family", "status",
    "first_seen_at", "last_seen_at", "deprecated_at", "sunset_at",
    "cutoff_date", "context_window", "max_output_tokens", "modality",
    "supports_thinking", "supports_tools",
    "input_price_usd", "output_price_usd", "cache_read_price_usd",
    "ttft_p50_ms", "tps_p50",
    "tokenizer_id", "self_report_id", "fingerprint_sha256",
    "data_source", "confidence", "last_synced_at", "raw_metadata_json",
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _row_to_dict(row) -> dict[str, Any]:
    """Convert sqlite3.Row → dict, decoding raw_metadata_json on the fly."""
    if row is None:
        return None  # type: ignore[return-value]
    d = dict(row)
    raw = d.get("raw_metadata_json")
    if raw and isinstance(raw, str):
        try:
            d["raw_metadata"] = json.loads(raw)
        except Exception:
            d["raw_metadata"] = None
    else:
        d["raw_metadata"] = None
    return d


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    """Coerce/normalize a caller-supplied record before upsert."""
    if not record.get("model_id"):
        raise ValueError("registry_repo.upsert_model: model_id is required")
    if not record.get("vendor"):
        raise ValueError("registry_repo.upsert_model: vendor is required")
    if not record.get("data_source"):
        raise ValueError("registry_repo.upsert_model: data_source is required")
    rec = dict(record)
    # Booleans → 0/1 for SQLite INTEGER columns
    for k in ("supports_thinking", "supports_tools"):
        if k in rec and rec[k] is not None:
            rec[k] = 1 if bool(rec[k]) else 0
    # Confidence default
    rec.setdefault("confidence", 1.0)
    # Status default
    rec.setdefault("status", "active")
    # Serialize raw_metadata if caller passed a dict
    if "raw_metadata" in rec and "raw_metadata_json" not in rec:
        rec["raw_metadata_json"] = json.dumps(rec.pop("raw_metadata"), ensure_ascii=False)
    return rec


# ── Public API ───────────────────────────────────────────────────────────────


def upsert_model(record: dict[str, Any]) -> dict[str, Any]:
    """INSERT OR UPDATE a registry row, with priority-aware merging.

    Returns the post-upsert row (dict).  Diff'd fields are written to
    ``model_registry_audit`` automatically.
    """
    rec = _normalize_record(record)
    model_id = rec["model_id"]
    now = int(rec.get("last_synced_at") or time.time())
    rec["last_synced_at"] = now

    conn = get_conn()
    # Concurrent maintenance jobs (registry_sync + changelog_harvester +
    # dataset_sync) can race on the same model_id. SQLite's deferred default
    # would let two SELECT-then-INSERT sequences both see "no row" and both
    # try to INSERT, with the second one hitting the UNIQUE constraint.
    # BEGIN IMMEDIATE acquires the write lock up front so the read side of
    # the upsert sees a consistent view. ON CONFLICT(model_id) is not used
    # here because the priority-aware merge below needs the existing row
    # values, which an UPSERT clause cannot easily express in pure SQL.
    try:
        conn.execute("BEGIN IMMEDIATE")
    except Exception:
        # If a transaction is already open (nested call) just continue —
        # the outer caller owns the lock.
        pass
    existing_row = conn.execute(
        "SELECT * FROM model_registry WHERE model_id = ?", (model_id,)
    ).fetchone()

    if existing_row is None:
        # Fresh insert
        rec.setdefault("first_seen_at", now)
        rec.setdefault("last_seen_at", now)
        cols = [c for c in _UPSERTABLE_COLUMNS if c in rec]
        placeholders = ",".join("?" for _ in cols)
        sql = (
            f"INSERT INTO model_registry (model_id, {', '.join(cols)}) "
            f"VALUES (?, {placeholders})"
        )
        conn.execute(sql, [model_id] + [rec[c] for c in cols])
        conn.commit()
        # No audit entries for first insert
        return _row_to_dict(
            conn.execute(
                "SELECT * FROM model_registry WHERE model_id = ?", (model_id,)
            ).fetchone()
        )

    existing = dict(existing_row)
    incoming_priority = _priority(rec["data_source"])
    existing_priority = _priority(existing.get("data_source"))

    updates: dict[str, Any] = {}
    audit_entries: list[tuple[str, Any, Any]] = []

    # Always advance last_seen_at and last_synced_at
    updates["last_seen_at"] = now
    updates["last_synced_at"] = now

    for col in _UPSERTABLE_COLUMNS:
        if col in {"last_seen_at", "last_synced_at", "first_seen_at"}:
            continue
        if col not in rec:
            continue
        new_val = rec[col]
        old_val = existing.get(col)
        if new_val is None:
            continue                      # never blank-overwrite with NULL
        # Priority-aware overwrite
        if incoming_priority <= existing_priority:
            if old_val != new_val:
                updates[col] = new_val
                audit_entries.append((col, old_val, new_val))
        else:
            # Lower priority: only fill in when existing is NULL
            if old_val in (None, "") and new_val not in (None, ""):
                updates[col] = new_val
                audit_entries.append((col, old_val, new_val))

    if not updates:
        return _row_to_dict(existing_row)

    set_clause = ", ".join(f"{c} = ?" for c in updates)
    conn.execute(
        f"UPDATE model_registry SET {set_clause} WHERE model_id = ?",
        list(updates.values()) + [model_id],
    )

    if audit_entries:
        conn.executemany(
            """
            INSERT INTO model_registry_audit
                (model_id, field, old_value, new_value, data_source, changed_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    model_id,
                    field,
                    None if old is None else json.dumps(old, ensure_ascii=False, default=str),
                    None if new is None else json.dumps(new, ensure_ascii=False, default=str),
                    rec["data_source"],
                    now,
                )
                for field, old, new in audit_entries
            ],
        )

    conn.commit()
    return _row_to_dict(
        conn.execute(
            "SELECT * FROM model_registry WHERE model_id = ?", (model_id,)
        ).fetchone()
    )


def mark_deprecated(model_id: str, ts: int | None = None) -> bool:
    """Flip status='deprecated' and stamp deprecated_at.  Returns True if a row was updated."""
    ts = int(ts or time.time())
    conn = get_conn()
    cursor = conn.execute(
        "UPDATE model_registry SET status='deprecated', deprecated_at=?, last_synced_at=? "
        "WHERE model_id=? AND status != 'deprecated'",
        (ts, ts, model_id),
    )
    conn.commit()
    return cursor.rowcount > 0


def mark_sunset(model_id: str, ts: int | None = None) -> bool:
    """Flip status='sunset' and stamp sunset_at.  Returns True if a row was updated."""
    ts = int(ts or time.time())
    conn = get_conn()
    cursor = conn.execute(
        "UPDATE model_registry SET status='sunset', sunset_at=?, last_synced_at=? "
        "WHERE model_id=? AND status != 'sunset'",
        (ts, ts, model_id),
    )
    conn.commit()
    return cursor.rowcount > 0


def get_model_card(model_id: str) -> dict[str, Any] | None:
    """Return the full registry row as a dict, or None if not found."""
    if not model_id:
        return None
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM model_registry WHERE model_id = ?", (model_id,)
    ).fetchone()
    return _row_to_dict(row) if row else None


def get_official_price(model_id: str) -> dict[str, Any] | None:
    """Return {input, output, cache_read} USD/Mtok for ``model_id`` or None.

    Phase 3's ``authenticity/price_evidence`` layer prefers its YAML seed
    but may fall back to this function once Phase 6 sync has populated
    pricing in the registry.
    """
    card = get_model_card(model_id)
    if card is None:
        return None
    if all(card.get(k) is None for k in ("input_price_usd", "output_price_usd")):
        return None
    return {
        "model_id": model_id,
        "vendor": card.get("vendor"),
        "input_per_mtok_usd": card.get("input_price_usd"),
        "output_per_mtok_usd": card.get("output_price_usd"),
        "cache_read_per_mtok_usd": card.get("cache_read_price_usd"),
        "data_source": card.get("data_source"),
        "last_synced_at": card.get("last_synced_at"),
    }


def list_eligible_baselines(
    now: int | None = None,
    min_age_days: int = 30,
    freshness_window_days: int = 14,
    sources: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Phase 11: list models eligible to act as similarity baselines.

    Eligibility criteria (mirrors UPGRADE_PLAN_V17.md §14):
      * status='active'
      * last_seen_at > NOW − freshness_window_days  (recently confirmed)
      * first_seen_at < NOW − min_age_days          (stable for at least N days)
      * data_source ∈ allowlist (default: official_api / openrouter / changelog)
    """
    now = int(now or time.time())
    fresh_cutoff = now - freshness_window_days * 86400
    age_cutoff = now - min_age_days * 86400
    if sources is None:
        sources = (
            "openai_api", "anthropic_api", "google_api",
            "xai_api", "deepseek_api", "mistral_api",
            "openrouter", "changelog",
        )
    placeholders = ",".join("?" for _ in sources)
    sql = (
        f"SELECT * FROM model_registry "
        f"WHERE status='active' "
        f"  AND last_seen_at > ? "
        f"  AND first_seen_at < ? "
        f"  AND data_source IN ({placeholders}) "
        f"ORDER BY vendor, model_id"
    )
    conn = get_conn()
    rows = conn.execute(sql, [fresh_cutoff, age_cutoff, *sources]).fetchall()
    return [_row_to_dict(r) for r in rows]


def audit_log(
    model_id: str,
    field: str,
    old_value: Any,
    new_value: Any,
    data_source: str,
    ts: int | None = None,
) -> None:
    """Manually emit an audit row (most callers don't need this; upsert
    handles diff-auditing automatically)."""
    ts = int(ts or time.time())
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO model_registry_audit
            (model_id, field, old_value, new_value, data_source, changed_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            model_id,
            field,
            None if old_value is None else json.dumps(old_value, ensure_ascii=False, default=str),
            None if new_value is None else json.dumps(new_value, ensure_ascii=False, default=str),
            data_source,
            ts,
        ),
    )
    conn.commit()


def count_models(status: str | None = "active") -> int:
    """Return number of models with the given ``status`` (or all if None)."""
    conn = get_conn()
    if status is None:
        row = conn.execute("SELECT COUNT(*) AS n FROM model_registry").fetchone()
    else:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM model_registry WHERE status = ?", (status,)
        ).fetchone()
    return int(row["n"]) if row else 0


__all__ = [
    "upsert_model",
    "mark_deprecated",
    "mark_sunset",
    "get_model_card",
    "get_official_price",
    "list_eligible_baselines",
    "audit_log",
    "count_models",
]
