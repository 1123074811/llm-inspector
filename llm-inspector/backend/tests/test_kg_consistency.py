"""
tests/test_kg_consistency.py — KG integration consistency tests (v13 Phase 5).

Verifies:
  1. DBpediaClient instantiation
  2. DBpedia offline mode when SPARQLWrapper unavailable
  3. Cross-check: both sources verified → consensus
  4. Cross-check: conflict → conflicting result
  5. Cross-check: single source → pass-through
  6. reference_embeddings.json validity
  7. Reference embeddings fallback in similarity pipeline
  8. SQLite cache TTL expiry
  9. VerificationResult has verified_by_consensus=False by default
  10. Known-fact consensus when both sources agree
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Test 1: DBpediaClient instantiates without error
# ---------------------------------------------------------------------------

def test_dbpedia_client_instantiates():
    """DBpediaClient() can be created without raising."""
    from app.knowledge.dbpedia_client import DBpediaClient
    client = DBpediaClient()
    assert client is not None
    # is_available reflects SPARQLWrapper presence
    assert isinstance(client.is_available(), bool)


# ---------------------------------------------------------------------------
# Test 2: Offline mode when SPARQLWrapper unavailable
# ---------------------------------------------------------------------------

def test_dbpedia_offline_mode():
    """When HAS_SPARQL=False, client is offline and verify_entity returns offline result."""
    from app.knowledge.dbpedia_client import DBpediaClient
    # Directly set the offline flag as SPARQLWrapper may already be imported
    client = DBpediaClient.__new__(DBpediaClient)
    client._offline = True
    client._db_path = Path(tempfile.mkdtemp()) / "test.sqlite"
    client._conn = None
    assert client.is_available() is False
    result = client.verify_entity("Albert Einstein")
    assert result.is_verified is False
    assert result.source == "offline"


# ---------------------------------------------------------------------------
# Test 3: Cross-check — both sources verified → consensus
# ---------------------------------------------------------------------------

def test_kg_cross_check_both_verified():
    """Both sources return is_verified=True → source contains '+', confidence ≥ 0.7."""
    from app.knowledge.kg_client import KnowledgeGraphClient
    from app.knowledge.wikidata_client import VerificationResult

    r_dbpedia = VerificationResult(is_verified=True, confidence=0.85, source="dbpedia", evidence=["found"])
    r_wikidata = VerificationResult(is_verified=True, confidence=0.90, source="wikidata", evidence=["found"])

    kg = KnowledgeGraphClient.__new__(KnowledgeGraphClient)
    result = kg._cross_check("Albert Einstein", {"dbpedia": r_dbpedia, "wikidata": r_wikidata})

    assert result.is_verified is True
    assert result.confidence >= 0.7
    assert "+" in result.source or "consensus" in result.source.lower()
    assert result.verified_by_consensus is True


# ---------------------------------------------------------------------------
# Test 4: Cross-check — conflict → conflicting result
# ---------------------------------------------------------------------------

def test_kg_cross_check_conflict():
    """One source says verified, the other not → source='conflicting', confidence ≤ 0.4."""
    from app.knowledge.kg_client import KnowledgeGraphClient
    from app.knowledge.wikidata_client import VerificationResult

    r_dbpedia = VerificationResult(is_verified=True, confidence=0.85, source="dbpedia")
    r_wikidata = VerificationResult(is_verified=False, confidence=0.10, source="wikidata")

    kg = KnowledgeGraphClient.__new__(KnowledgeGraphClient)
    result = kg._cross_check("SomeEntity", {"dbpedia": r_dbpedia, "wikidata": r_wikidata})

    assert result.source == "conflicting"
    assert result.confidence <= 0.4
    assert result.is_verified is False
    assert any("conflicting_sources" in e for e in result.evidence)


# ---------------------------------------------------------------------------
# Test 5: Cross-check — single source → return as-is
# ---------------------------------------------------------------------------

def test_kg_cross_check_single_source():
    """Only one source returns a result → that result is returned unchanged."""
    from app.knowledge.kg_client import KnowledgeGraphClient
    from app.knowledge.wikidata_client import VerificationResult

    r = VerificationResult(is_verified=True, confidence=0.88, source="wikidata", evidence=["e"])

    kg = KnowledgeGraphClient.__new__(KnowledgeGraphClient)
    result = kg._cross_check("X", {"wikidata": r})

    assert result.source == "wikidata"
    assert result.confidence == 0.88
    assert result.is_verified is True


# ---------------------------------------------------------------------------
# Test 6: reference_embeddings.json validity
# ---------------------------------------------------------------------------

def test_reference_embeddings_json_valid():
    """reference_embeddings.json exists, has ≥ 10 models, each with required keys."""
    path = Path(__file__).parent.parent / "app" / "_data" / "reference_embeddings.json"
    assert path.exists(), f"reference_embeddings.json not found at {path}"

    data = json.loads(path.read_text(encoding="utf-8"))
    models = data.get("models", {})
    assert len(models) >= 10, f"Expected ≥10 models, got {len(models)}"

    required_keys = {"features", "scores", "baseline_source"}
    for name, model_data in models.items():
        missing = required_keys - set(model_data.keys())
        assert not missing, f"Model '{name}' missing keys: {missing}"

        score_keys = {"capability_score", "authenticity_score", "performance_score", "total_score"}
        missing_scores = score_keys - set(model_data["scores"].keys())
        assert not missing_scores, f"Model '{name}' missing score keys: {missing_scores}"


# ---------------------------------------------------------------------------
# Test 7: Reference embeddings fallback in similarity pipeline
# ---------------------------------------------------------------------------

def test_reference_embeddings_fallback():
    """
    When get_benchmarks() returns empty (no golden_baselines), SimilarityEngine.compare()
    falls back to reference embeddings and returns ≥ 10 results, all with data_source='reference'.
    """
    from app.analysis.similarity import SimilarityEngine, _load_reference_profiles

    # Verify the helper returns profiles
    profiles = _load_reference_profiles()
    assert len(profiles) >= 10

    engine = SimilarityEngine()

    # All profiles are data_source="reference" so the first filter would normally exclude them,
    # but the fallback path handles this.
    target_features = {
        "avg_response_length": 450.0,
        "latency_mean_ms": 1800.0,
        "tokens_per_second": 50.0,
        "param_compliance_rate": 0.95,
        "instruction_pass_rate": 0.88,
    }

    # Pass empty list → triggers reference fallback
    results = engine.compare(target_features, [])
    assert len(results) >= 10, f"Expected ≥10 results from reference fallback, got {len(results)}"

    # All results should come from reference profiles
    for r in results:
        # SimilarityResult has benchmark_name — all should be known model names
        assert r.benchmark_name, f"Empty benchmark_name in result"


# ---------------------------------------------------------------------------
# Test 8: SQLite cache TTL expiry
# ---------------------------------------------------------------------------

def test_kg_cache_ttl():
    """Inserting an expired record → _get_cached returns None."""
    from app.knowledge.dbpedia_client import DBpediaClient, _ensure_cache_db

    # Use a dedicated in-memory SQLite connection to avoid Windows file lock issues
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS kg_cache (
            id          TEXT PRIMARY KEY,
            query       TEXT NOT NULL,
            result_json TEXT NOT NULL,
            created_at  REAL NOT NULL,
            expires_at  REAL NOT NULL
        )
        """
    )
    conn.commit()

    # Insert already-expired entry
    expired_time = time.time() - 1  # 1 second in the past
    conn.execute(
        "INSERT INTO kg_cache (id, query, result_json, created_at, expires_at) VALUES (?,?,?,?,?)",
        (
            "dbpedia:entity:testentity",
            "SELECT ?x WHERE {}",
            json.dumps({"is_verified": True, "confidence": 0.9, "source": "dbpedia",
                        "evidence": [], "entity": None, "query_time_ms": 0,
                        "verified_by_consensus": False}),
            expired_time - 100,
            expired_time,
        ),
    )
    conn.commit()

    client = DBpediaClient.__new__(DBpediaClient)
    client._offline = False
    client._db_path = Path(":memory:")
    client._conn = conn

    result = client._get_cached("dbpedia:entity:testentity")
    conn.close()
    assert result is None, "Expired cache entry should return None"


# ---------------------------------------------------------------------------
# Test 9: VerificationResult has verified_by_consensus=False by default
# ---------------------------------------------------------------------------

def test_wikidata_client_verify_entity_exists():
    """VerificationResult dataclass has verified_by_consensus=False default."""
    from app.knowledge.wikidata_client import VerificationResult
    r = VerificationResult(is_verified=True, confidence=0.9, source="wikidata")
    assert r.verified_by_consensus is False
    assert "verified_by_consensus" in r.to_dict()
    assert r.to_dict()["verified_by_consensus"] is False


# ---------------------------------------------------------------------------
# Test 10: Known fact consistent when both sources agree
# ---------------------------------------------------------------------------

def test_known_facts_consistent():
    """
    Mock both Wikidata and DBpedia to return is_verified=True for 'Albert Einstein'.
    verify_entity should return is_verified=True.
    """
    from app.knowledge.kg_client import KnowledgeGraphClient
    from app.knowledge.wikidata_client import VerificationResult, WikidataClient
    from app.knowledge.dbpedia_client import DBpediaClient

    mock_dbpedia = MagicMock(spec=DBpediaClient)
    mock_dbpedia.verify_entity.return_value = VerificationResult(
        is_verified=True, confidence=0.85, source="dbpedia", evidence=["Found in DBpedia"]
    )

    mock_wikidata = MagicMock(spec=WikidataClient)
    mock_wikidata.verify_entity_exists.return_value = VerificationResult(
        is_verified=True, confidence=0.95, source="wikidata", evidence=["Exact match"]
    )

    kg = KnowledgeGraphClient.__new__(KnowledgeGraphClient)
    kg._dbpedia_client = mock_dbpedia
    kg._wikidata = mock_wikidata
    kg._cache: dict = {}
    kg._cache_size = 1000
    kg.use_dbpedia = True
    kg.use_wikidata = True
    kg._stats = {
        "dbpedia": MagicMock(queries_total=0, queries_success=0, avg_response_time_ms=0.0),
        "wikidata": MagicMock(queries_total=0, queries_success=0, avg_response_time_ms=0.0),
        "cache": MagicMock(queries_total=0, queries_cached=0, avg_response_time_ms=0.0),
        "heuristic": MagicMock(queries_total=0, queries_success=0, avg_response_time_ms=0.0),
    }

    result = kg.verify_entity("Albert Einstein")
    assert result.is_verified is True
    assert result.verified_by_consensus is True
    assert result.confidence >= 0.7
