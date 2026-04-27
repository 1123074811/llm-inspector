"""
v17 Phase 11 — baseline_pool tests.

Covers:
  * filter_eligible_baselines drops deprecated / self_probed / too-new rows
  * cold-start fallback: empty registry → input passed through
  * baseline_pool_summary aggregates source breakdown + last_synced
  * recently_deprecated lists models within the 14-day window
"""
from __future__ import annotations

import time

import pytest


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    db_path = tmp_path / "v17p11.sqlite"
    from app.core import db as _db_mod

    monkeypatch.setattr(_db_mod.settings, "DATABASE_URL", f"sqlite:///{db_path.as_posix()}")
    monkeypatch.setattr(_db_mod, "_DB_PATH", db_path)
    if hasattr(_db_mod._local, "conn") and _db_mod._local.conn is not None:
        try:
            _db_mod._local.conn.close()
        except Exception:
            pass
        _db_mod._local.conn = None

    _db_mod.init_db()
    from app.core.db_migrations import migrate
    migrate(_db_mod.get_conn())

    yield

    if hasattr(_db_mod._local, "conn") and _db_mod._local.conn is not None:
        try:
            _db_mod._local.conn.close()
        except Exception:
            pass
        _db_mod._local.conn = None


def _seed_eligible(model_id, source="openai_api", *, fresh=True, aged=True):
    from app.repository import registry_repo as rr
    now = int(time.time())
    first_seen = now - 60 * 86400 if aged else now - 5 * 86400
    last_seen = now - 1 * 86400 if fresh else now - 60 * 86400
    rr.upsert_model({
        "model_id": model_id,
        "vendor": "openai",
        "data_source": source,
        "first_seen_at": first_seen,
        "last_seen_at": last_seen,
        "last_synced_at": last_seen,
    })


# ── filter_eligible_baselines ───────────────────────────────────────────────


def test_filter_drops_non_registered_baselines():
    from app.analysis.baseline_pool import filter_eligible_baselines
    _seed_eligible("gpt-4o")
    _seed_eligible("claude-3-5-sonnet", source="anthropic_api")

    baselines = [
        {"model_name": "gpt-4o", "features": {}},
        {"model_name": "deprecated-old", "features": {}},
        {"model_name": "claude-3-5-sonnet", "features": {}},
        {"model_name": "totally-unknown", "features": {}},
    ]
    out = filter_eligible_baselines(baselines)
    names = [b["model_name"] for b in out]
    assert "gpt-4o" in names
    assert "claude-3-5-sonnet" in names
    assert "deprecated-old" not in names
    assert "totally-unknown" not in names
    assert len(out) == 2


def test_filter_drops_self_probed_models():
    from app.analysis.baseline_pool import filter_eligible_baselines
    _seed_eligible("gpt-4o")
    _seed_eligible("self-probed-model", source="self_probed")
    out = filter_eligible_baselines([
        {"model_name": "gpt-4o", "features": {}},
        {"model_name": "self-probed-model", "features": {}},
    ])
    names = [b["model_name"] for b in out]
    assert names == ["gpt-4o"]


def test_filter_drops_deprecated_models():
    from app.analysis.baseline_pool import filter_eligible_baselines
    from app.repository import registry_repo as rr
    _seed_eligible("gpt-4o")
    _seed_eligible("retired-model")
    rr.mark_deprecated("retired-model")
    out = filter_eligible_baselines([
        {"model_name": "gpt-4o", "features": {}},
        {"model_name": "retired-model", "features": {}},
    ])
    assert [b["model_name"] for b in out] == ["gpt-4o"]


def test_filter_drops_too_new_models():
    """Models seen for less than 30 days are not yet stable as baselines."""
    from app.analysis.baseline_pool import filter_eligible_baselines
    _seed_eligible("gpt-4o")
    _seed_eligible("brand-new-model", aged=False)
    out = filter_eligible_baselines([
        {"model_name": "gpt-4o", "features": {}},
        {"model_name": "brand-new-model", "features": {}},
    ])
    assert [b["model_name"] for b in out] == ["gpt-4o"]


def test_filter_drops_stale_models():
    """Models not seen in the last 14 days fall out of the eligible pool."""
    from app.analysis.baseline_pool import filter_eligible_baselines
    _seed_eligible("gpt-4o")
    _seed_eligible("forgotten", fresh=False)
    out = filter_eligible_baselines([
        {"model_name": "gpt-4o", "features": {}},
        {"model_name": "forgotten", "features": {}},
    ])
    assert [b["model_name"] for b in out] == ["gpt-4o"]


def test_filter_cold_start_passes_through():
    """Empty registry must NOT cause the run to lose all baselines."""
    from app.analysis.baseline_pool import filter_eligible_baselines
    baselines = [
        {"model_name": "gpt-4o", "features": {}},
        {"model_name": "anything", "features": {}},
    ]
    out = filter_eligible_baselines(baselines)
    assert out == baselines


def test_filter_handles_alternate_id_keys():
    """Some baselines use 'model_id' or 'name' rather than 'model_name'."""
    from app.analysis.baseline_pool import filter_eligible_baselines
    _seed_eligible("gpt-4o")
    out = filter_eligible_baselines([
        {"model_id": "gpt-4o", "features": {}},
        {"name": "gpt-4o", "features": {}},     # also accepted
        {"name": "stranger", "features": {}},
    ])
    assert len(out) == 2


def test_filter_skips_malformed_entries():
    from app.analysis.baseline_pool import filter_eligible_baselines
    _seed_eligible("gpt-4o")
    out = filter_eligible_baselines([
        {"model_name": "gpt-4o"},
        None,
        "not-a-dict",
        {"missing_id_keys": True},
    ])
    assert len(out) == 1


# ── baseline_pool_summary ──────────────────────────────────────────────────


def test_summary_counts_sources_breakdown():
    from app.analysis.baseline_pool import baseline_pool_summary
    _seed_eligible("openai-1", source="openai_api")
    _seed_eligible("openai-2", source="openai_api")
    _seed_eligible("anthropic-1", source="anthropic_api")
    _seed_eligible("openrouter-1", source="openrouter")

    summary = baseline_pool_summary()
    assert summary["active_count"] == 4
    assert summary["sources_breakdown"]["openai_api"] == 2
    assert summary["sources_breakdown"]["anthropic_api"] == 1
    assert summary["sources_breakdown"]["openrouter"] == 1
    assert summary["min_age_days"] == 30
    assert summary["freshness_window_days"] == 14
    assert summary["synced_at"] is not None


def test_summary_recently_deprecated_lists_recent_only():
    from app.analysis.baseline_pool import baseline_pool_summary
    from app.repository import registry_repo as rr
    _seed_eligible("active-1")
    _seed_eligible("retired-recent")
    _seed_eligible("retired-old")
    rr.mark_deprecated("retired-recent")
    # Retire old by setting deprecated_at far in the past
    now = int(time.time())
    far_past = now - 60 * 86400
    rr.mark_deprecated("retired-old", ts=far_past)

    summary = baseline_pool_summary()
    assert "retired-recent" in summary["recently_deprecated"]
    assert "retired-old" not in summary["recently_deprecated"]


def test_summary_handles_empty_registry():
    from app.analysis.baseline_pool import baseline_pool_summary
    summary = baseline_pool_summary()
    assert summary["active_count"] == 0
    assert summary["sources_breakdown"] == {}
    assert summary["synced_at"] is None
    assert summary["recently_deprecated"] == []
