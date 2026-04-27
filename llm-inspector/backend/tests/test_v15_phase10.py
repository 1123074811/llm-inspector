"""
Tests for v15 Phase 10: Reporting, Cache Strategy, Migrations & Seeder.

Covers:
  - NarrativeBuilder (analysis/reporting.py)
  - ProxyLatencyAnalyzer (analysis/reporting.py)
  - ExtractionAuditBuilder (analysis/reporting.py)
  - CacheStrategy (runner/cache_strategy.py): build_key, get/set, evict, snapshot
  - Migration007 (db_migrations.py): llm_response_cache table
  - Seeder v13/v15 fixtures loadable
  - version.json schema
"""
from __future__ import annotations
import json
import math
import os
import pathlib
import pytest


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_scorecard(total=72.0, capability=70.0, authenticity=68.0, performance=78.0):
    from app.core.schemas import ScoreCard
    sc = ScoreCard()
    sc.total_score = total
    sc.capability_score = capability
    sc.authenticity_score = authenticity
    sc.performance_score = performance
    sc.reasoning_score = 0.65
    sc.coding_score = 0.70
    return sc


def _make_verdict(level="trusted"):
    from app.core.schemas import TrustVerdict
    return TrustVerdict(
        level=level,
        label=level.replace("_", " ").title(),
        total_score=80.0,
        reasons=[f"Test verdict — {level}"],
    )


def _make_similarity(name="GPT-4", score=0.88, rank=1):
    from app.core.schemas import SimilarityResult
    return SimilarityResult(
        benchmark_name=name,
        similarity_score=score,
        rank=rank,
        ci_95_low=0.80,
        ci_95_high=0.94,
    )


def _make_predetect(confidence=0.72, identified_as="gpt-4"):
    from app.core.schemas import PreDetectionResult, LayerResult
    return PreDetectionResult(
        confidence=confidence,
        identified_as=identified_as,
        layer_stopped=None,
        layer_results=[
            LayerResult(
                layer="Layer1/SelfReport",
                confidence=0.72,
                identified_as=identified_as,
                evidence=["Model said: I am GPT-4"],
                tokens_used=20,
            )
        ],
        success=True,
    )


# ---------------------------------------------------------------------------
# NarrativeBuilder tests
# ---------------------------------------------------------------------------

def test_narrative_builder_build_returns_expected_keys():
    from app.analysis.reporting import NarrativeBuilder
    nb = NarrativeBuilder()
    result = nb.build(
        model_name="gpt-4",
        verdict=_make_verdict("trusted"),
        scorecard=_make_scorecard(),
        similarities=[_make_similarity()],
        predetect=_make_predetect(),
        features={"instruction_pass_rate": 0.9},
        case_results=[],
    )
    assert "executive_summary" in result
    assert "detection_process" in result
    assert "dimension_analysis" in result
    assert "similarity_narrative" in result
    assert "risk_narrative" in result
    assert "recommendations" in result
    assert "confidence_statement" in result


def test_narrative_builder_executive_summary_mentions_model():
    from app.analysis.reporting import NarrativeBuilder
    nb = NarrativeBuilder()
    result = nb.build(
        model_name="my-custom-model",
        verdict=_make_verdict("suspicious"),
        scorecard=_make_scorecard(total=55.0),
        similarities=[],
        predetect=None,
        features={},
        case_results=[],
    )
    assert "my-custom-model" in result["executive_summary"]


def test_narrative_builder_no_scorecard_safe():
    from app.analysis.reporting import NarrativeBuilder
    nb = NarrativeBuilder()
    result = nb.build(
        model_name="anon",
        verdict=None,
        scorecard=None,
        similarities=[],
        predetect=None,
        features={},
        case_results=[],
    )
    assert isinstance(result["executive_summary"], str)


def test_narrative_builder_detection_process_early_stop():
    from app.analysis.reporting import NarrativeBuilder
    nb = NarrativeBuilder()
    pre = _make_predetect(confidence=0.92)
    result = nb.build("gpt-4", _make_verdict(), _make_scorecard(), [], pre, {}, [])
    assert "提前终止" in result["detection_process"]


def test_narrative_builder_similarity_narrative_with_top_match():
    from app.analysis.reporting import NarrativeBuilder
    nb = NarrativeBuilder()
    sims = [
        _make_similarity("GPT-4", 0.90, 1),
        _make_similarity("GPT-4o", 0.75, 2),
        _make_similarity("Claude", 0.65, 3),
    ]
    result = nb.build("gpt-4", _make_verdict(), _make_scorecard(), sims, None, {}, [])
    assert "GPT-4" in result["similarity_narrative"]


def test_narrative_builder_similarity_narrative_empty():
    from app.analysis.reporting import NarrativeBuilder
    nb = NarrativeBuilder()
    result = nb.build("gpt-4", _make_verdict(), _make_scorecard(), [], None, {}, [])
    assert "未找到" in result["similarity_narrative"]


def test_narrative_builder_dimension_analysis_with_features():
    from app.analysis.reporting import NarrativeBuilder
    nb = NarrativeBuilder()
    result = nb.build(
        "gpt-4", _make_verdict(), _make_scorecard(), [],
        None, {"instruction_pass_rate": 0.85, "dim_consistency_pass_rate": 0.90}, [],
    )
    assert "85" in result["dimension_analysis"] or "指令" in result["dimension_analysis"]


# ---------------------------------------------------------------------------
# CacheStrategy tests (in-memory / isolated DB)
# ---------------------------------------------------------------------------

@pytest.fixture
def cache():
    """Create a fresh CacheStrategy with a temporary in-memory SQLite DB."""
    from app.runner.cache_strategy import CacheStrategy

    # Create a minimal in-memory DB with the required table
    import sqlite3
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS llm_response_cache (
            cache_key TEXT PRIMARY KEY,
            response_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL
        )
    """)
    conn.commit()

    strategy = CacheStrategy()
    # Patch get_conn to return our in-memory connection
    import app.runner.cache_strategy as cs_module
    original_get_conn = None
    try:
        from app.core.db import get_conn as real_get_conn
        original_get_conn = real_get_conn
    except ImportError:
        pass

    import app.core.db as db_module
    _original = db_module.get_conn
    db_module.get_conn = lambda: conn
    yield strategy
    db_module.get_conn = _original
    conn.close()


def test_cache_build_key_deterministic(cache):
    k1 = cache.build_key("http://api.example.com", {"model": "gpt-4", "messages": []})
    k2 = cache.build_key("http://api.example.com", {"model": "gpt-4", "messages": []})
    assert k1 == k2


def test_cache_build_key_different_for_different_payload(cache):
    k1 = cache.build_key("http://api.example.com", {"model": "gpt-4"})
    k2 = cache.build_key("http://api.example.com", {"model": "gpt-3.5"})
    assert k1 != k2


def test_cache_miss_returns_none(cache):
    result = cache.get("nonexistent-key-xyz")
    assert result is None


def test_cache_set_and_get(cache):
    from app.core.schemas import LLMResponse
    resp = LLMResponse(
        content="Hello, world!",
        finish_reason="stop",
        status_code=200,
        error_type=None,
        error_message=None,
        latency_ms=150,
        usage_total_tokens=20,
    )
    key = cache.build_key("http://api.test", {"q": 1})
    cache.set(key, resp, category="reasoning")
    result = cache.get(key)
    assert result is not None
    assert result.content == "Hello, world!"


def test_cache_error_response_not_stored(cache):
    from app.core.schemas import LLMResponse
    resp = LLMResponse(
        content="",
        finish_reason="error",
        status_code=500,
        error_type="NetworkError",
        error_message="Connection refused",
        latency_ms=0,
        usage_total_tokens=0,
    )
    key = "error-test-key"
    cache.set(key, resp, category="reasoning")
    result = cache.get(key)
    assert result is None  # errors should not be cached


def test_cache_snapshot_fields(cache):
    snap = cache.snapshot()
    assert hasattr(snap, "total_requests")
    assert hasattr(snap, "hits")
    assert hasattr(snap, "misses")
    assert hasattr(snap, "tokens_saved")
    assert hasattr(snap, "hit_rate")
    assert hasattr(snap, "cache_size")


def test_cache_snapshot_to_dict(cache):
    snap = cache.snapshot()
    d = snap.to_dict()
    assert "total_requests" in d
    assert "hit_rate" in d
    assert "cache_size" in d


def test_cache_hit_rate_computed(cache):
    from app.core.schemas import LLMResponse
    key = cache.build_key("http://x", {"m": 1})
    resp = LLMResponse(
        content="test", finish_reason="stop", status_code=200,
        error_type=None, error_message=None, latency_ms=10, usage_total_tokens=5,
    )
    cache.set(key, resp, category="reasoning")

    cache.get("nonexistent")   # miss
    cache.get(key)             # hit

    snap = cache.snapshot()
    assert snap.hits >= 1
    assert snap.misses >= 1
    assert 0.0 <= snap.hit_rate <= 1.0


def test_cache_evict_expired_runs_without_error(cache):
    removed = cache.evict_expired()
    assert removed >= 0


# ---------------------------------------------------------------------------
# Migration007: llm_response_cache table
# ---------------------------------------------------------------------------

def test_migration007_creates_cache_table():
    from app.core.db_migrations import Migration007V15CacheTable
    import sqlite3
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    # Ensure migrations table exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY, description TEXT NOT NULL, applied_at TEXT NOT NULL
        )
    """)
    m = Migration007V15CacheTable()
    m.apply(conn)
    # Verify table exists
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "llm_response_cache" in tables


def test_migration007_idempotent():
    from app.core.db_migrations import Migration007V15CacheTable
    import sqlite3
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    m = Migration007V15CacheTable()
    m.apply(conn)
    m.apply(conn)  # second apply should not raise
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "llm_response_cache" in tables


def test_migration007_columns():
    from app.core.db_migrations import Migration007V15CacheTable
    import sqlite3
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    m = Migration007V15CacheTable()
    m.apply(conn)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(llm_response_cache)")}
    assert "cache_key" in cols
    assert "response_json" in cols
    assert "created_at" in cols
    assert "expires_at" in cols


# ---------------------------------------------------------------------------
# version.json schema
# ---------------------------------------------------------------------------

def test_version_json_is_valid():
    version_path = pathlib.Path(__file__).parent.parent / "app" / "_data" / "version.json"
    assert version_path.exists(), "version.json not found"
    data = json.loads(version_path.read_text(encoding="utf-8"))
    assert "version" in data
    assert "phases_complete" in data
    assert isinstance(data["phases_complete"], list)


def test_version_json_version_string():
    version_path = pathlib.Path(__file__).parent.parent / "app" / "_data" / "version.json"
    data = json.loads(version_path.read_text(encoding="utf-8"))
    assert data["version"].startswith("v16"), f"Expected v16*, got {data['version']}"


def test_version_json_all_phases_complete():
    version_path = pathlib.Path(__file__).parent.parent / "app" / "_data" / "version.json"
    data = json.loads(version_path.read_text(encoding="utf-8"))
    phases = data["phases_complete"]
    # Check all phases 0-13 are listed (phase4 is recorded as "phase4-bugs")
    phases_set = set(phases)
    for i in range(14):
        canonical = f"phase{i}"
        # Accept either exact name or variant (e.g. "phase4-bugs")
        has_phase = any(p == canonical or p.startswith(canonical) for p in phases_set)
        assert has_phase, f"Phase {i} missing from phases_complete (got {sorted(phases_set)})"


# ---------------------------------------------------------------------------
# Seeder: suite_v13 and suite_v15 are in the fixture loop
# ---------------------------------------------------------------------------

def test_seeder_includes_v13_v15():
    import inspect
    from app.tasks import seeder
    source = inspect.getsource(seeder._seed_test_cases)
    assert "suite_v13.json" in source, "suite_v13.json not in seeder loop"
    assert "suite_v15.json" in source, "suite_v15.json not in seeder loop"


def test_suite_v15_json_valid():
    suite_path = pathlib.Path(__file__).parent.parent / "app" / "fixtures" / "suite_v15.json"
    assert suite_path.exists(), "suite_v15.json not found"
    data = json.loads(suite_path.read_text(encoding="utf-8"))
    assert "cases" in data
    assert len(data["cases"]) > 0
    assert data.get("version") == "v15"


def test_suite_v15_cases_have_required_fields():
    suite_path = pathlib.Path(__file__).parent.parent / "app" / "fixtures" / "suite_v15.json"
    data = json.loads(suite_path.read_text(encoding="utf-8"))
    required = {"id", "category", "user_prompt", "judge_method", "max_tokens", "n_samples"}
    for case in data["cases"]:
        missing = required - set(case.keys())
        assert not missing, f"Case {case.get('id')} missing fields: {missing}"


def test_suite_v13_json_valid():
    suite_path = pathlib.Path(__file__).parent.parent / "app" / "fixtures" / "suite_v13.json"
    assert suite_path.exists(), "suite_v13.json not found"
    data = json.loads(suite_path.read_text(encoding="utf-8"))
    assert "cases" in data
    assert len(data["cases"]) > 0
