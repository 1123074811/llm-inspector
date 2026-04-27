"""
test_v16_phase7.py — v16 Phase 7 regression tests.

Validates:
  - compress() prompt compression
  - cache_strategy.build_key() with model_name
  - TokenAuditTracker recording and summary
  - TokenAuditEntry dataclass
"""
import pytest


class TestPromptCompression:
    def test_compress_short_prompt_unchanged(self):
        from app.runner.prompt_optimizer import compress
        assert compress("Hello") == "Hello"

    def test_compress_removes_stop_words(self):
        from app.runner.prompt_optimizer import compress
        prompt = "Please note that the model is very really quite good"
        result = compress(prompt, protect_stem=False)
        assert "please" not in result.lower()
        assert "note" not in result.lower()
        assert "model" in result
        assert "good" in result

    def test_compress_synonym_merge(self):
        from app.runner.prompt_optimizer import compress
        prompt = "In order to solve this, due to the fact that it is important"
        result = compress(prompt, protect_stem=False)
        assert "in order to" not in result.lower()
        assert "due to the fact that" not in result.lower()

    def test_compress_protects_stem(self):
        from app.runner.prompt_optimizer import compress
        prompt = "Please note that this is a long instruction.\nWhat is 2+2?"
        result = compress(prompt, protect_stem=True)
        assert "2+2" in result  # Stem preserved

    def test_compress_target_tokens(self):
        from app.runner.prompt_optimizer import compress
        prompt = "A " * 200  # ~400 chars
        result = compress(prompt, target_tokens=10, protect_stem=False)
        assert len(result) <= 50  # ~10 tokens * 4 chars + margin

    def test_compress_empty(self):
        from app.runner.prompt_optimizer import compress
        assert compress("") == ""
        assert compress(None) is None


class TestCacheStrategyV16:
    def test_build_key_with_model_name(self):
        from app.runner.cache_strategy import CacheStrategy
        cs = CacheStrategy()
        key1 = cs.build_key("https://api.openai.com", {"model": "gpt-4o"}, model_name="gpt-4o")
        key2 = cs.build_key("https://api.openai.com", {"model": "gpt-4o"}, model_name="claude-3")
        assert key1 != key2  # Different model_name → different key

    def test_build_key_without_model_name(self):
        from app.runner.cache_strategy import CacheStrategy
        cs = CacheStrategy()
        key = cs.build_key("https://api.openai.com", {"model": "gpt-4o"})
        assert len(key) == 64  # SHA-256 hex digest

    def test_build_key_backward_compat(self):
        from app.runner.cache_strategy import CacheStrategy
        cs = CacheStrategy()
        key_old = cs.build_key("https://api.openai.com", {"model": "gpt-4o"})
        key_new = cs.build_key("https://api.openai.com", {"model": "gpt-4o"}, model_name="")
        assert key_old == key_new  # Empty model_name = backward compatible


class TestTokenAudit:
    def test_token_audit_entry(self):
        from app.runner.token_audit import TokenAuditEntry
        entry = TokenAuditEntry(
            case_id="test_001",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cache_hit=True,
            finish_reason="stop",
            layer="capability",
        )
        d = entry.to_dict()
        assert d["case_id"] == "test_001"
        assert d["total_tokens"] == 150
        assert d["cache_hit"] is True

    def test_token_audit_tracker_record(self):
        from app.runner.token_audit import TokenAuditTracker
        tracker = TokenAuditTracker("test-run-001")
        tracker.record(
            case_id="case_001",
            prompt_tokens=200,
            completion_tokens=100,
            total_tokens=300,
            cache_hit=False,
            finish_reason="stop",
            layer="predetect_L5",
        )
        tracker.record(
            case_id="case_002",
            prompt_tokens=150,
            completion_tokens=80,
            total_tokens=230,
            cache_hit=True,
            finish_reason="stop",
            layer="capability",
        )
        summary = tracker.summary()
        assert summary.entry_count == 2
        assert summary.total_tokens == 530
        assert summary.cache_hits == 1
        assert summary.cache_misses == 1
        assert summary.cache_hit_rate == 0.5

    def test_token_audit_summary_by_layer(self):
        from app.runner.token_audit import TokenAuditTracker
        tracker = TokenAuditTracker("test-run-002")
        tracker.record(case_id="c1", total_tokens=100, layer="predetect")
        tracker.record(case_id="c2", total_tokens=200, layer="predetect")
        tracker.record(case_id="c3", total_tokens=50, layer="capability")
        summary = tracker.summary()
        assert summary.by_layer["predetect"]["total"] == 300
        assert summary.by_layer["predetect"]["count"] == 2
        assert summary.by_layer["capability"]["total"] == 50

    def test_token_audit_tracker_empty(self):
        from app.runner.token_audit import TokenAuditTracker
        tracker = TokenAuditTracker("empty-run")
        summary = tracker.summary()
        assert summary.entry_count == 0
        assert summary.total_tokens == 0
        assert summary.cache_hit_rate == 0.0
