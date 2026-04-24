"""
tests/test_v15_phase5.py — v15 Phase 5/10: Cache Strategy

Covers:
- CacheStrategy can be instantiated
- cache_strategy.build_key(url, payload) returns a 64-char hex string
- cache_strategy.snapshot() returns a CacheMetricsSnapshot with all expected fields
- cache_strategy.snapshot().to_dict() has correct keys
- cache_strategy.reset_metrics() zeroes counters
- cache_strategy.evict_expired() returns an int (may be 0)
- cache_strategy.cache_size() returns an int >= 0
- Category TTL resolution: "tokenizer" → 168h, "timing" → 4h short_hours, unknown → DEFAULT_TTL
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# CacheMetricsSnapshot
# ---------------------------------------------------------------------------

class TestCacheMetricsSnapshot:
    def test_defaults(self):
        from app.runner.cache_strategy import CacheMetricsSnapshot
        snap = CacheMetricsSnapshot()
        assert snap.total_requests == 0
        assert snap.hits == 0
        assert snap.misses == 0
        assert snap.tokens_saved == 0
        assert snap.tokens_spent_on_misses == 0
        assert snap.hit_rate == 0.0
        assert snap.estimated_cost_savings == 0.0
        assert snap.cache_size == 0
        assert snap.expired_entries == 0
        assert snap.warming_requests == 0

    def test_to_dict_keys(self):
        from app.runner.cache_strategy import CacheMetricsSnapshot
        snap = CacheMetricsSnapshot(
            total_requests=10,
            hits=7,
            misses=3,
            tokens_saved=1000,
            tokens_spent_on_misses=500,
            hit_rate=0.7,
            estimated_cost_savings=0.01,
            cache_size=5,
            expired_entries=1,
            warming_requests=2,
        )
        d = snap.to_dict()
        expected_keys = {
            "total_requests", "hits", "misses", "tokens_saved",
            "tokens_spent_on_misses", "hit_rate", "estimated_cost_savings",
            "cache_size", "expired_entries", "warming_requests",
        }
        assert expected_keys == set(d.keys())

    def test_to_dict_values(self):
        from app.runner.cache_strategy import CacheMetricsSnapshot
        snap = CacheMetricsSnapshot(
            total_requests=100,
            hits=80,
            misses=20,
            hit_rate=0.8,
            estimated_cost_savings=0.123456,
        )
        d = snap.to_dict()
        assert d["total_requests"] == 100
        assert d["hits"] == 80
        assert d["misses"] == 20
        assert d["hit_rate"] == round(0.8, 4)
        assert d["estimated_cost_savings"] == round(0.123456, 6)

    def test_to_dict_rounding(self):
        from app.runner.cache_strategy import CacheMetricsSnapshot
        snap = CacheMetricsSnapshot(hit_rate=0.123456789, estimated_cost_savings=0.0000001234)
        d = snap.to_dict()
        assert d["hit_rate"] == round(0.123456789, 4)
        assert d["estimated_cost_savings"] == round(0.0000001234, 6)


# ---------------------------------------------------------------------------
# CacheStrategy instantiation
# ---------------------------------------------------------------------------

class TestCacheStrategyInstantiation:
    def test_can_be_instantiated(self):
        from app.runner.cache_strategy import CacheStrategy
        cs = CacheStrategy()
        assert cs is not None

    def test_singleton_exists(self):
        from app.runner.cache_strategy import cache_strategy, CacheStrategy
        assert isinstance(cache_strategy, CacheStrategy)

    def test_get_cache_strategy_returns_singleton(self):
        from app.runner.cache_strategy import get_cache_strategy, cache_strategy
        assert get_cache_strategy() is cache_strategy


# ---------------------------------------------------------------------------
# build_key
# ---------------------------------------------------------------------------

class TestBuildKey:
    def test_returns_64_char_hex(self):
        from app.runner.cache_strategy import CacheStrategy
        cs = CacheStrategy()
        key = cs.build_key("https://api.example.com", {"model": "gpt-4", "messages": []})
        assert isinstance(key, str)
        assert len(key) == 64
        # Valid hex
        int(key, 16)

    def test_same_inputs_same_key(self):
        from app.runner.cache_strategy import CacheStrategy
        cs = CacheStrategy()
        payload = {"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]}
        key1 = cs.build_key("https://api.example.com", payload)
        key2 = cs.build_key("https://api.example.com", payload)
        assert key1 == key2

    def test_different_url_different_key(self):
        from app.runner.cache_strategy import CacheStrategy
        cs = CacheStrategy()
        payload = {"model": "gpt-4"}
        key1 = cs.build_key("https://api1.example.com", payload)
        key2 = cs.build_key("https://api2.example.com", payload)
        assert key1 != key2

    def test_different_payload_different_key(self):
        from app.runner.cache_strategy import CacheStrategy
        cs = CacheStrategy()
        url = "https://api.example.com"
        key1 = cs.build_key(url, {"model": "gpt-4"})
        key2 = cs.build_key(url, {"model": "gpt-3.5-turbo"})
        assert key1 != key2

    def test_key_is_deterministic_regardless_of_dict_order(self):
        """build_key uses sort_keys=True so dict ordering doesn't matter."""
        from app.runner.cache_strategy import CacheStrategy
        cs = CacheStrategy()
        url = "https://api.example.com"
        key1 = cs.build_key(url, {"b": 2, "a": 1})
        key2 = cs.build_key(url, {"a": 1, "b": 2})
        assert key1 == key2


# ---------------------------------------------------------------------------
# snapshot()
# ---------------------------------------------------------------------------

class TestSnapshot:
    def test_snapshot_returns_cache_metrics_snapshot(self):
        from app.runner.cache_strategy import CacheStrategy, CacheMetricsSnapshot
        cs = CacheStrategy()
        snap = cs.snapshot()
        assert isinstance(snap, CacheMetricsSnapshot)

    def test_snapshot_has_all_fields(self):
        from app.runner.cache_strategy import CacheStrategy
        cs = CacheStrategy()
        snap = cs.snapshot()
        assert hasattr(snap, "total_requests")
        assert hasattr(snap, "hits")
        assert hasattr(snap, "misses")
        assert hasattr(snap, "tokens_saved")
        assert hasattr(snap, "tokens_spent_on_misses")
        assert hasattr(snap, "hit_rate")
        assert hasattr(snap, "estimated_cost_savings")
        assert hasattr(snap, "cache_size")
        assert hasattr(snap, "expired_entries")
        assert hasattr(snap, "warming_requests")

    def test_fresh_instance_snapshot_zeroes(self):
        from app.runner.cache_strategy import CacheStrategy
        cs = CacheStrategy()
        snap = cs.snapshot()
        assert snap.total_requests == 0
        assert snap.hits == 0
        assert snap.misses == 0
        assert snap.tokens_saved == 0
        assert snap.warming_requests == 0

    def test_snapshot_to_dict_has_correct_keys(self):
        from app.runner.cache_strategy import CacheStrategy
        cs = CacheStrategy()
        snap = cs.snapshot()
        d = snap.to_dict()
        expected_keys = {
            "total_requests", "hits", "misses", "tokens_saved",
            "tokens_spent_on_misses", "hit_rate", "estimated_cost_savings",
            "cache_size", "expired_entries", "warming_requests",
        }
        assert expected_keys == set(d.keys())


# ---------------------------------------------------------------------------
# reset_metrics()
# ---------------------------------------------------------------------------

class TestResetMetrics:
    def test_reset_zeroes_counters(self):
        from app.runner.cache_strategy import CacheStrategy
        cs = CacheStrategy()
        # Manually increment internal state
        cs._total_requests = 100
        cs._hits = 80
        cs._misses = 20
        cs._tokens_saved = 50000
        cs._tokens_spent_on_misses = 10000
        cs._warming_count = 5

        cs.reset_metrics()

        assert cs._total_requests == 0
        assert cs._hits == 0
        assert cs._misses == 0
        assert cs._tokens_saved == 0
        assert cs._tokens_spent_on_misses == 0
        assert cs._warming_count == 0

    def test_snapshot_after_reset_shows_zeroes(self):
        from app.runner.cache_strategy import CacheStrategy
        cs = CacheStrategy()
        cs._total_requests = 50
        cs._hits = 40
        cs.reset_metrics()
        snap = cs.snapshot()
        assert snap.total_requests == 0
        assert snap.hits == 0


# ---------------------------------------------------------------------------
# evict_expired()
# ---------------------------------------------------------------------------

class TestEvictExpired:
    def test_returns_int(self):
        from app.runner.cache_strategy import CacheStrategy
        cs = CacheStrategy()
        result = cs.evict_expired()
        assert isinstance(result, int)
        assert result >= 0

    def test_evict_empty_cache_returns_zero(self):
        from app.runner.cache_strategy import CacheStrategy
        cs = CacheStrategy()
        # Fresh instance with no entries, should return 0
        result = cs.evict_expired()
        assert result == 0


# ---------------------------------------------------------------------------
# cache_size()
# ---------------------------------------------------------------------------

class TestCacheSize:
    def test_returns_int(self):
        from app.runner.cache_strategy import CacheStrategy
        cs = CacheStrategy()
        result = cs.cache_size()
        assert isinstance(result, int)

    def test_returns_non_negative(self):
        from app.runner.cache_strategy import CacheStrategy
        cs = CacheStrategy()
        assert cs.cache_size() >= 0


# ---------------------------------------------------------------------------
# Category TTL resolution
# ---------------------------------------------------------------------------

class TestCategoryTTLResolution:
    def test_tokenizer_category_168h(self):
        from app.runner.cache_strategy import CacheStrategy
        cs = CacheStrategy()
        ttl = cs._resolve_ttl("tokenizer")
        assert ttl.long_hours == 168

    def test_timing_category_short_4h(self):
        from app.runner.cache_strategy import CacheStrategy
        cs = CacheStrategy()
        ttl = cs._resolve_ttl("timing")
        assert ttl.short_hours == 4

    def test_unknown_category_returns_default_ttl(self):
        from app.runner.cache_strategy import CacheStrategy
        cs = CacheStrategy()
        default_ttl = cs.DEFAULT_TTL
        unknown_ttl = cs._resolve_ttl("nonexistent_category_xyz")
        assert unknown_ttl.default_hours == default_ttl.default_hours

    def test_empty_category_returns_default_ttl(self):
        from app.runner.cache_strategy import CacheStrategy
        cs = CacheStrategy()
        default_ttl = cs.DEFAULT_TTL
        empty_ttl = cs._resolve_ttl("")
        assert empty_ttl.default_hours == default_ttl.default_hours

    def test_extraction_category_72h(self):
        from app.runner.cache_strategy import CacheStrategy
        cs = CacheStrategy()
        ttl = cs._resolve_ttl("extraction")
        assert ttl.long_hours == 72

    def test_identity_category_72h(self):
        from app.runner.cache_strategy import CacheStrategy
        cs = CacheStrategy()
        ttl = cs._resolve_ttl("identity")
        assert ttl.long_hours == 72

    def test_safety_category_48h(self):
        from app.runner.cache_strategy import CacheStrategy
        cs = CacheStrategy()
        ttl = cs._resolve_ttl("safety")
        assert ttl.default_hours == 48

    def test_category_ttl_dict_has_expected_keys(self):
        from app.runner.cache_strategy import CacheStrategy
        expected = {
            "reasoning", "knowledge", "instruction", "coding", "safety",
            "extraction", "identity", "tokenizer", "timing", "multilingual", "adversarial"
        }
        assert set(CacheStrategy.CATEGORY_TTL.keys()) == expected


# ---------------------------------------------------------------------------
# record_miss_tokens (auxiliary method)
# ---------------------------------------------------------------------------

class TestRecordMissTokens:
    def test_record_miss_tokens_increments(self):
        from app.runner.cache_strategy import CacheStrategy
        cs = CacheStrategy()
        cs.record_miss_tokens(500)
        cs.record_miss_tokens(300)
        assert cs._tokens_spent_on_misses == 800

    def test_snapshot_reflects_miss_tokens(self):
        from app.runner.cache_strategy import CacheStrategy
        cs = CacheStrategy()
        cs.record_miss_tokens(1000)
        snap = cs.snapshot()
        assert snap.tokens_spent_on_misses == 1000
