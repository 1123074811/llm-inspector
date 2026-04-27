"""
v17 Phase 6 — model_registry_sync tests.

All HTTP is mocked via monkeypatching ``_http_get_json``; no real network
traffic.  Covers:

  * OpenAI-compatible adapter (id/created)
  * Anthropic adapter (ISO created_at)
  * Google adapter (models/<id> name + token limits)
  * OpenRouter adapter (per-token → per-Mtok price conversion + vendor inference)
  * Skipped sources when env keys are absent
  * deprecate_stale_models sweep
  * full_sync orchestration aggregates per-source counts
"""
from __future__ import annotations

import time

import pytest


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    db_path = tmp_path / "v17p6.sqlite"
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


# ── Helpers ─────────────────────────────────────────────────────────────────


def _patch_http(monkeypatch, response_map: dict[str, object]):
    """Replace ``_http_get_json`` with a lookup against ``response_map``."""
    from app.runner import model_registry_sync as mrs

    def _fake(url, headers, timeout=20):
        # Match by URL prefix to be lenient with query strings.
        for prefix, payload in response_map.items():
            if url.startswith(prefix):
                return payload
        return None

    monkeypatch.setattr(mrs, "_http_get_json", _fake)


# ── _to_per_mtok / _vendor_from_openrouter_id ───────────────────────────────


def test_to_per_mtok_handles_strings_and_zero():
    from app.runner.model_registry_sync import _to_per_mtok
    assert _to_per_mtok("0.0000025") == 2.5
    assert _to_per_mtok("0.000010") == 10.0
    assert _to_per_mtok(None) is None
    assert _to_per_mtok("") is None
    assert _to_per_mtok("0") is None
    assert _to_per_mtok("not-a-number") is None


def test_vendor_from_openrouter_id_normalises_prefixes():
    from app.runner.model_registry_sync import _vendor_from_openrouter_id
    assert _vendor_from_openrouter_id("openai/gpt-4o") == "openai"
    assert _vendor_from_openrouter_id("anthropic/claude-3-5-sonnet") == "anthropic"
    assert _vendor_from_openrouter_id("x-ai/grok-2") == "xai"
    assert _vendor_from_openrouter_id("meta-llama/llama-3.1-405b") == "meta"
    assert _vendor_from_openrouter_id("mistralai/mistral-large") == "mistral"
    assert _vendor_from_openrouter_id("moonshotai/kimi") == "moonshot"
    assert _vendor_from_openrouter_id("01-ai/yi-large") == "yi"
    assert _vendor_from_openrouter_id("singletoken") == "unknown"


# ── Tier 1 adapters ─────────────────────────────────────────────────────────


def test_sync_openai_compat_inserts_models(monkeypatch):
    from app.runner import model_registry_sync as mrs
    from app.repository import registry_repo as rr

    _patch_http(monkeypatch, {
        "https://api.openai.com/v1/models": {
            "data": [
                {"id": "gpt-4o", "created": 1700000000, "owned_by": "openai"},
                {"id": "gpt-4o-mini", "created": 1710000000, "owned_by": "openai"},
            ],
        },
    })
    res = mrs._sync_openai_compat(
        "openai", "https://api.openai.com/v1", "sk-fake", "openai_api"
    )
    assert res.fetched == 2
    assert res.inserted == 2
    assert res.errors == 0
    assert rr.get_model_card("gpt-4o")["data_source"] == "openai_api"
    assert rr.get_model_card("gpt-4o")["first_seen_at"] == 1700000000


def test_sync_openai_compat_handles_http_failure(monkeypatch):
    from app.runner import model_registry_sync as mrs
    _patch_http(monkeypatch, {})        # all requests return None
    res = mrs._sync_openai_compat(
        "openai", "https://api.openai.com/v1", "sk-fake", "openai_api"
    )
    assert res.fetched == 0
    assert res.errors == 1


def test_sync_openai_compat_handles_unexpected_payload(monkeypatch):
    from app.runner import model_registry_sync as mrs
    _patch_http(monkeypatch, {
        "https://api.openai.com/v1/models": {"unexpected": "shape"},
    })
    res = mrs._sync_openai_compat(
        "openai", "https://api.openai.com/v1", "sk-fake", "openai_api"
    )
    assert res.errors == 1
    assert res.fetched == 0


def test_sync_anthropic_parses_iso_created_at(monkeypatch):
    from app.runner import model_registry_sync as mrs
    from app.repository import registry_repo as rr

    _patch_http(monkeypatch, {
        "https://api.anthropic.com/v1/models": {
            "data": [
                {
                    "id": "claude-3-5-sonnet-20241022",
                    "display_name": "Claude 3.5 Sonnet",
                    "created_at": "2024-10-22T00:00:00Z",
                },
            ],
        },
    })
    res = mrs._sync_anthropic("sk-ant-fake")
    assert res.fetched == 1
    card = rr.get_model_card("claude-3-5-sonnet-20241022")
    assert card["vendor"] == "anthropic"
    assert card["data_source"] == "anthropic_api"
    assert card["first_seen_at"] is not None


def test_sync_google_strips_models_prefix(monkeypatch):
    from app.runner import model_registry_sync as mrs
    from app.repository import registry_repo as rr

    _patch_http(monkeypatch, {
        "https://generativelanguage.googleapis.com/v1beta/models": {
            "models": [
                {
                    "name": "models/gemini-2.0-flash",
                    "displayName": "Gemini 2.0 Flash",
                    "inputTokenLimit": 1_048_576,
                    "outputTokenLimit": 8192,
                },
            ],
        },
    })
    res = mrs._sync_google("AIza-fake")
    assert res.fetched == 1
    card = rr.get_model_card("gemini-2.0-flash")
    assert card is not None
    assert card["vendor"] == "google"
    assert card["context_window"] == 1_048_576


# ── Tier 2: OpenRouter ──────────────────────────────────────────────────────


def test_sync_openrouter_converts_pricing(monkeypatch):
    from app.runner import model_registry_sync as mrs
    from app.repository import registry_repo as rr

    _patch_http(monkeypatch, {
        "https://openrouter.ai/api/v1/models": {
            "data": [
                {
                    "id": "openai/gpt-4o",
                    "context_length": 128000,
                    "pricing": {
                        "prompt": "0.0000025",          # $2.5 / Mtok
                        "completion": "0.00001",        # $10 / Mtok
                        "input_cache_read": "0.00000125",
                    },
                    "architecture": {
                        "modality": "text+image->text",
                        "tokenizer": "GPT",
                    },
                },
                {
                    "id": "anthropic/claude-3-5-sonnet",
                    "context_length": 200000,
                    "pricing": {"prompt": "0.000003", "completion": "0.000015"},
                    "architecture": {"modality": "text", "tokenizer": "Claude"},
                },
            ],
        },
    })
    res = mrs._sync_openrouter()         # unauthenticated path
    assert res.fetched == 2
    assert res.inserted == 2
    gpt = rr.get_model_card("openai/gpt-4o")
    assert gpt["input_price_usd"] == 2.5
    assert gpt["output_price_usd"] == 10.0
    assert gpt["cache_read_price_usd"] == 1.25
    assert gpt["context_window"] == 128_000
    assert gpt["vendor"] == "openai"
    cla = rr.get_model_card("anthropic/claude-3-5-sonnet")
    assert cla["vendor"] == "anthropic"


def test_sync_openrouter_with_api_key_sets_auth_header(monkeypatch):
    """Smoke-check: passing an api_key arg adds Authorization header."""
    from app.runner import model_registry_sync as mrs
    captured = {}

    def _fake(url, headers, timeout=20):
        captured["headers"] = dict(headers)
        return {"data": []}

    monkeypatch.setattr(mrs, "_http_get_json", _fake)
    mrs._sync_openrouter(api_key="or-fake-token")
    assert captured["headers"].get("Authorization") == "Bearer or-fake-token"


# ── deprecate_stale_models ──────────────────────────────────────────────────


def test_deprecate_stale_models_flips_old_active_rows():
    from app.runner import model_registry_sync as mrs
    from app.repository import registry_repo as rr
    now = int(time.time())
    long_ago = now - 30 * 86400

    rr.upsert_model({
        "model_id": "old-1",
        "vendor": "openai",
        "data_source": "openai_api",
        "first_seen_at": long_ago,
        "last_seen_at": long_ago,        # well past the 14d window
        "last_synced_at": long_ago,
    })
    rr.upsert_model({
        "model_id": "fresh-1",
        "vendor": "openai",
        "data_source": "openai_api",
        "first_seen_at": long_ago,
        "last_seen_at": now,
        "last_synced_at": now,
    })
    rr.upsert_model({
        "model_id": "manual-only",
        "vendor": "openai",
        "data_source": "manual",
        "first_seen_at": long_ago,
        "last_seen_at": long_ago,
        "last_synced_at": long_ago,
    })

    transitioned = mrs.deprecate_stale_models(now=now, miss_window_days=14)
    assert "old-1" in transitioned
    assert "fresh-1" not in transitioned        # too recent
    assert "manual-only" not in transitioned    # manual is exempt
    assert rr.get_model_card("old-1")["status"] == "deprecated"
    assert rr.get_model_card("fresh-1")["status"] == "active"


# ── run_full_sync orchestration ────────────────────────────────────────────


def test_full_sync_skips_sources_without_keys(monkeypatch):
    from app.runner import model_registry_sync as mrs

    _patch_http(monkeypatch, {
        "https://openrouter.ai/api/v1/models": {"data": []},
    })
    report = mrs.run_full_sync(env={})       # no keys at all
    by_src = {r.source: r for r in report.per_source}
    # Tier-1 sources without keys are skipped
    for src in ("openai_api", "anthropic_api", "google_api", "xai_api",
                "deepseek_api", "mistral_api"):
        assert by_src[src].skipped_reason
        assert by_src[src].fetched == 0
    # OpenRouter is always attempted
    assert by_src["openrouter"].errors == 0


def test_full_sync_aggregates_counts(monkeypatch):
    from app.runner import model_registry_sync as mrs

    _patch_http(monkeypatch, {
        "https://api.openai.com/v1/models": {
            "data": [
                {"id": "gpt-4o", "created": 1700000000},
                {"id": "gpt-4o-mini", "created": 1710000000},
            ],
        },
        "https://openrouter.ai/api/v1/models": {
            "data": [
                {"id": "anthropic/claude-3-5-sonnet",
                 "context_length": 200000,
                 "pricing": {"prompt": "0.000003", "completion": "0.000015"},
                 "architecture": {}},
            ],
        },
    })
    report = mrs.run_full_sync(env={"OPENAI_API_KEY": "sk-fake"})
    d = report.to_dict()
    assert d["total_fetched"] == 3
    assert d["total_inserted"] == 3
    assert d["total_errors"] == 0
    by_src = {r["source"]: r for r in d["per_source"]}
    assert by_src["openai_api"]["fetched"] == 2
    assert by_src["openrouter"]["fetched"] == 1
    assert by_src["anthropic_api"]["skipped_reason"]
