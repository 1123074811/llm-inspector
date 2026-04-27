"""
v17 Phase 5 — model_registry schema + registry_repo tests.

Covers:
  * Migration 008 idempotency (table & indexes exist after re-run)
  * upsert_model: insert, equal-priority overwrite, lower-priority fill-in,
    higher-priority overwrite, audit log entries
  * mark_deprecated / mark_sunset
  * list_eligible_baselines (active ∧ fresh ∧ aged ∧ data_source allowlist)
  * get_official_price
"""
from __future__ import annotations

import time

import pytest


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    """Each test gets a fresh sqlite DB via DATABASE_URL override."""
    db_path = tmp_path / "v17p5.sqlite"
    # Reset thread-local conn so the new URL is picked up
    from app.core import db as _db_mod

    monkeypatch.setattr(
        _db_mod.settings, "DATABASE_URL", f"sqlite:///{db_path.as_posix()}"
    )
    monkeypatch.setattr(_db_mod, "_DB_PATH", db_path)
    if hasattr(_db_mod._local, "conn") and _db_mod._local.conn is not None:
        try:
            _db_mod._local.conn.close()
        except Exception:
            pass
        _db_mod._local.conn = None

    # Initialise schema + apply all migrations on fresh DB
    _db_mod.init_db()
    from app.core.db_migrations import migrate
    migrate(_db_mod.get_conn())

    yield

    # Tear down: close conn before tmp_path is removed
    if hasattr(_db_mod._local, "conn") and _db_mod._local.conn is not None:
        try:
            _db_mod._local.conn.close()
        except Exception:
            pass
        _db_mod._local.conn = None


# ── Migration ───────────────────────────────────────────────────────────────


def test_migration_creates_tables_and_indexes():
    from app.core.db import get_conn
    conn = get_conn()
    tables = {row["name"] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    assert "model_registry" in tables
    assert "model_registry_audit" in tables
    indexes = {row["name"] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index'"
    ).fetchall()}
    assert "idx_registry_vendor_status" in indexes
    assert "idx_registry_last_seen" in indexes


def test_migration_is_idempotent():
    from app.core.db import get_conn
    from app.core.db_migrations import migrate
    # Re-run; should be a no-op (no exception).
    migrate(get_conn())
    migrate(get_conn())


# ── upsert_model: fresh insert ──────────────────────────────────────────────


def test_upsert_inserts_new_row():
    from app.repository import registry_repo as rr
    rec = {
        "model_id": "gpt-4o",
        "vendor": "openai",
        "family": "gpt",
        "context_window": 128_000,
        "input_price_usd": 2.5,
        "output_price_usd": 10.0,
        "data_source": "openai_api",
    }
    out = rr.upsert_model(rec)
    assert out["model_id"] == "gpt-4o"
    assert out["status"] == "active"
    assert out["confidence"] == 1.0
    assert out["first_seen_at"] == out["last_seen_at"]


def test_upsert_requires_model_id_vendor_data_source():
    from app.repository import registry_repo as rr
    with pytest.raises(ValueError):
        rr.upsert_model({"vendor": "openai", "data_source": "openai_api"})
    with pytest.raises(ValueError):
        rr.upsert_model({"model_id": "x", "data_source": "openai_api"})
    with pytest.raises(ValueError):
        rr.upsert_model({"model_id": "x", "vendor": "openai"})


# ── upsert_model: priority-aware merging ────────────────────────────────────


def test_upsert_higher_priority_overwrites():
    from app.repository import registry_repo as rr
    rr.upsert_model({
        "model_id": "claude-3-5-sonnet",
        "vendor": "anthropic",
        "context_window": 100_000,           # wrong — placeholder
        "data_source": "self_probed",
        "confidence": 0.85,
    })
    rr.upsert_model({
        "model_id": "claude-3-5-sonnet",
        "vendor": "anthropic",
        "context_window": 200_000,           # corrected by official source
        "data_source": "anthropic_api",
    })
    out = rr.get_model_card("claude-3-5-sonnet")
    assert out["context_window"] == 200_000
    assert out["data_source"] == "anthropic_api"


def test_upsert_lower_priority_only_fills_blanks():
    from app.repository import registry_repo as rr
    rr.upsert_model({
        "model_id": "claude-x",
        "vendor": "anthropic",
        "context_window": 200_000,
        "data_source": "anthropic_api",
    })
    rr.upsert_model({
        "model_id": "claude-x",
        "vendor": "anthropic",
        "context_window": 50_000,        # MUST NOT overwrite (lower priority)
        "tokenizer_id": "claude",        # SHOULD fill in (was NULL)
        "data_source": "self_probed",
    })
    out = rr.get_model_card("claude-x")
    assert out["context_window"] == 200_000     # not clobbered
    assert out["tokenizer_id"] == "claude"      # filled in


def test_upsert_writes_audit_entries_on_change():
    from app.repository import registry_repo as rr
    from app.core.db import get_conn

    rr.upsert_model({
        "model_id": "gpt-4o-mini",
        "vendor": "openai",
        "context_window": 100_000,
        "data_source": "openai_api",
    })
    rr.upsert_model({
        "model_id": "gpt-4o-mini",
        "vendor": "openai",
        "context_window": 128_000,
        "data_source": "openai_api",
    })

    rows = get_conn().execute(
        "SELECT field, old_value, new_value FROM model_registry_audit "
        "WHERE model_id=? ORDER BY id",
        ("gpt-4o-mini",),
    ).fetchall()
    fields = [r["field"] for r in rows]
    assert "context_window" in fields


def test_upsert_no_audit_for_first_insert():
    from app.repository import registry_repo as rr
    from app.core.db import get_conn
    rr.upsert_model({
        "model_id": "fresh-model",
        "vendor": "openai",
        "data_source": "openai_api",
    })
    n = get_conn().execute(
        "SELECT COUNT(*) AS n FROM model_registry_audit WHERE model_id=?",
        ("fresh-model",),
    ).fetchone()["n"]
    assert n == 0


# ── mark_deprecated / mark_sunset ───────────────────────────────────────────


def test_mark_deprecated_sets_status_and_timestamp():
    from app.repository import registry_repo as rr
    rr.upsert_model({
        "model_id": "gpt-4-turbo",
        "vendor": "openai",
        "data_source": "openai_api",
    })
    assert rr.mark_deprecated("gpt-4-turbo") is True
    out = rr.get_model_card("gpt-4-turbo")
    assert out["status"] == "deprecated"
    assert out["deprecated_at"] is not None


def test_mark_deprecated_returns_false_if_already_deprecated():
    from app.repository import registry_repo as rr
    rr.upsert_model({
        "model_id": "claude-2",
        "vendor": "anthropic",
        "data_source": "anthropic_api",
    })
    rr.mark_deprecated("claude-2")
    assert rr.mark_deprecated("claude-2") is False


def test_mark_sunset_sets_status_and_timestamp():
    from app.repository import registry_repo as rr
    rr.upsert_model({
        "model_id": "deprecated-old",
        "vendor": "openai",
        "data_source": "openai_api",
    })
    assert rr.mark_sunset("deprecated-old") is True
    out = rr.get_model_card("deprecated-old")
    assert out["status"] == "sunset"
    assert out["sunset_at"] is not None


# ── list_eligible_baselines ─────────────────────────────────────────────────


def test_eligible_baselines_apply_all_filters():
    from app.repository import registry_repo as rr
    now = int(time.time())
    aged = now - 60 * 86400          # 60d old
    fresh = now - 1 * 86400          # 1d old → fresh
    long_ago = now - 60 * 86400      # 60d ago → not fresh

    # 1. Active, aged, fresh, official → eligible
    rr.upsert_model({
        "model_id": "good-1",
        "vendor": "openai",
        "data_source": "openai_api",
        "first_seen_at": aged,
        "last_seen_at": fresh,
        "last_synced_at": fresh,
    })
    # 2. Too new (first_seen_at < 30d cutoff) → ineligible
    rr.upsert_model({
        "model_id": "too-new",
        "vendor": "openai",
        "data_source": "openai_api",
        "first_seen_at": now - 5 * 86400,
        "last_seen_at": fresh,
        "last_synced_at": fresh,
    })
    # 3. Stale (last_seen too old) → ineligible
    rr.upsert_model({
        "model_id": "stale",
        "vendor": "openai",
        "data_source": "openai_api",
        "first_seen_at": aged,
        "last_seen_at": long_ago,
        "last_synced_at": long_ago,
    })
    # 4. Self-probed → ineligible regardless of dates
    rr.upsert_model({
        "model_id": "self-probed",
        "vendor": "anthropic",
        "data_source": "self_probed",
        "first_seen_at": aged,
        "last_seen_at": fresh,
        "last_synced_at": fresh,
        "confidence": 0.85,
    })
    # 5. Deprecated → ineligible
    rr.upsert_model({
        "model_id": "deprecated-1",
        "vendor": "openai",
        "data_source": "openai_api",
        "first_seen_at": aged,
        "last_seen_at": fresh,
        "last_synced_at": fresh,
    })
    rr.mark_deprecated("deprecated-1")

    eligible_ids = {m["model_id"] for m in rr.list_eligible_baselines(now=now)}
    assert eligible_ids == {"good-1"}


# ── get_official_price ──────────────────────────────────────────────────────


def test_get_official_price_falls_back_to_none_when_unset():
    from app.repository import registry_repo as rr
    rr.upsert_model({
        "model_id": "no-price",
        "vendor": "openai",
        "data_source": "openai_api",
    })
    assert rr.get_official_price("no-price") is None


def test_get_official_price_returns_pricing_when_set():
    from app.repository import registry_repo as rr
    rr.upsert_model({
        "model_id": "with-price",
        "vendor": "openai",
        "input_price_usd": 2.5,
        "output_price_usd": 10.0,
        "cache_read_price_usd": 1.25,
        "data_source": "openai_api",
    })
    pricing = rr.get_official_price("with-price")
    assert pricing["input_per_mtok_usd"] == 2.5
    assert pricing["output_per_mtok_usd"] == 10.0
    assert pricing["cache_read_per_mtok_usd"] == 1.25
    assert pricing["data_source"] == "openai_api"


def test_count_models():
    from app.repository import registry_repo as rr
    assert rr.count_models() == 0
    for i in range(3):
        rr.upsert_model({
            "model_id": f"m-{i}",
            "vendor": "openai",
            "data_source": "openai_api",
        })
    assert rr.count_models("active") == 3
    rr.mark_deprecated("m-1")
    assert rr.count_models("active") == 2
    assert rr.count_models("deprecated") == 1
    assert rr.count_models(None) == 3
