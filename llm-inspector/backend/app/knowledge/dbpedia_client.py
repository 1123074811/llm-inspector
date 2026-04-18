"""
knowledge/dbpedia_client.py — DBpedia SPARQL client for entity/fact verification.

Queries the public DBpedia SPARQL endpoint at https://dbpedia.org/sparql.
Falls back gracefully to offline_mode when the endpoint is unreachable.

Reference:
    DBpedia SPARQL endpoint: https://dbpedia.org/sparql
    Registered in SOURCES.yaml as "kg.dbpedia_endpoint"

Usage:
    client = DBpediaClient()
    result = client.verify_entity("Albert Einstein")
    # result.source == "dbpedia"
    # result.is_verified == True
    # result.confidence ~= 0.85
"""
from __future__ import annotations

import sqlite3
import time
import json
from pathlib import Path
from typing import Optional

from app.knowledge.wikidata_client import VerificationResult, WikidataEntity
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

try:
    from SPARQLWrapper import SPARQLWrapper, JSON as SPARQL_JSON
    HAS_SPARQL = True
except ImportError:
    HAS_SPARQL = False

# -- Constants -----------------------------------------------------------------

_ENDPOINT = "https://dbpedia.org/sparql"
_TIMEOUT_S = 3
_CACHE_TTL_DAYS = 30  # SRC["kg.cache_ttl_days"]


# -- SQLite cache helpers ------------------------------------------------------

def _get_cache_db_path() -> Path:
    return Path(settings.DATA_DIR) / "kg_cache.sqlite"


def _ensure_cache_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
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
    return conn


class DBpediaClient:
    """
    Dedicated DBpedia SPARQL client with SQLite caching and graceful offline fallback.

    Registered SOURCES.yaml constants:
      - kg.dbpedia_endpoint : https://dbpedia.org/sparql
      - kg.cache_ttl_days   : 30
    """

    def __init__(self) -> None:
        self._offline: bool = not HAS_SPARQL
        self._db_path: Path = _get_cache_db_path()
        self._conn: Optional[sqlite3.Connection] = None

        if not HAS_SPARQL:
            logger.warning("SPARQLWrapper not installed — DBpediaClient running in offline_mode")
        else:
            try:
                self._conn = _ensure_cache_db(self._db_path)
                logger.info("DBpediaClient initialised", endpoint=_ENDPOINT)
            except Exception as exc:
                logger.warning("DBpedia SQLite cache init failed", error=str(exc))
                self._conn = None

    # -- Cache interface -------------------------------------------------------

    def _cache_key(self, query_type: str, name: str) -> str:
        return f"dbpedia:{query_type}:{name.lower().strip()}"

    def _get_cached(self, cache_id: str) -> Optional[VerificationResult]:
        if self._conn is None:
            return None
        try:
            row = self._conn.execute(
                "SELECT result_json, expires_at FROM kg_cache WHERE id=?", (cache_id,)
            ).fetchone()
            if row is None:
                return None
            result_json, expires_at = row
            if time.time() > expires_at:
                # Expired — remove stale entry
                self._conn.execute("DELETE FROM kg_cache WHERE id=?", (cache_id,))
                self._conn.commit()
                return None
            data = json.loads(result_json)
            entity = None
            if data.get("entity"):
                entity = WikidataEntity(
                    id=data["entity"].get("id", ""),
                    label=data["entity"].get("label", ""),
                    description=data["entity"].get("description", ""),
                )
            return VerificationResult(
                is_verified=data["is_verified"],
                confidence=data["confidence"],
                source=data["source"],
                evidence=data.get("evidence", []),
                entity=entity,
                query_time_ms=data.get("query_time_ms", 0),
            )
        except Exception as exc:
            logger.warning("DBpedia cache read error", error=str(exc))
            return None

    def _set_cached(self, cache_id: str, query: str, result: VerificationResult) -> None:
        if self._conn is None:
            return
        try:
            now = time.time()
            expires = now + _CACHE_TTL_DAYS * 86400
            result_data = result.to_dict()
            self._conn.execute(
                """
                INSERT OR REPLACE INTO kg_cache (id, query, result_json, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (cache_id, query, json.dumps(result_data), now, expires),
            )
            self._conn.commit()
        except Exception as exc:
            logger.warning("DBpedia cache write error", error=str(exc))

    # -- Core verification methods ---------------------------------------------

    def verify_entity(self, entity_name: str) -> VerificationResult:
        """
        Verify entity existence in DBpedia via SPARQL.

        SPARQL query uses rdfs:label with @en language tag and optionally
        retrieves dbo:abstract as evidence snippet.

        Args:
            entity_name: Human-readable name (e.g., "Albert Einstein")

        Returns:
            VerificationResult with source="dbpedia"
        """
        cache_id = self._cache_key("entity", entity_name)
        cached = self._get_cached(cache_id)
        if cached is not None:
            return cached

        if self._offline:
            return self._offline_result(entity_name)

        start = time.time()

        # -- SPARQL query (SRC["kg.dbpedia_endpoint"]) -------------------------
        sparql_query = (
            'PREFIX dbo: <http://dbpedia.org/ontology/> '
            'PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> '
            'SELECT ?label ?abstract WHERE { '
            f'  ?entity rdfs:label "{_sparql_escape(entity_name)}"@en . '
            '   OPTIONAL { ?entity dbo:abstract ?abstract . FILTER(lang(?abstract) = "en") } '
            '} LIMIT 1'
        )

        try:
            sparql = SPARQLWrapper(_ENDPOINT)
            sparql.setQuery(sparql_query)
            sparql.setReturnFormat(SPARQL_JSON)
            sparql.setTimeout(_TIMEOUT_S)
            raw = sparql.query().convert()

            query_time_ms = int((time.time() - start) * 1000)
            bindings = raw.get("results", {}).get("bindings", [])

            if bindings:
                abstract = bindings[0].get("abstract", {}).get("value", "")
                evidence = [f"Found in DBpedia as '{entity_name}'"]
                if abstract:
                    evidence.append(f"Abstract: {abstract[:120]}…")

                entity = WikidataEntity(id="", label=entity_name, description=abstract[:200])
                result = VerificationResult(
                    is_verified=True,
                    confidence=0.85,
                    source="dbpedia",
                    evidence=evidence,
                    entity=entity,
                    query_time_ms=query_time_ms,
                )
            else:
                result = VerificationResult(
                    is_verified=False,
                    confidence=0.0,
                    source="dbpedia",
                    evidence=[f"No DBpedia entry found for '{entity_name}'"],
                    query_time_ms=query_time_ms,
                )

            self._set_cached(cache_id, sparql_query, result)
            return result

        except Exception as exc:
            logger.warning("DBpedia verify_entity failed", entity=entity_name, error=str(exc))
            # Mark offline for subsequent calls to avoid repeated timeouts
            self._offline = True
            return self._offline_result(entity_name)

    def verify_fact(self, subject: str, predicate: str, obj: str) -> VerificationResult:
        """
        Verify a structured triple (subject, predicate, object) in DBpedia.

        Checks that subject and object both exist; relation inference is limited
        to co-existence since SPARQL predicate mapping requires ontology lookup.

        Args:
            subject: Subject entity name
            predicate: Predicate / relation label (used as evidence text only)
            obj: Object entity name

        Returns:
            VerificationResult with source="dbpedia"
        """
        cache_id = self._cache_key("fact", f"{subject}|{predicate}|{obj}")
        cached = self._get_cached(cache_id)
        if cached is not None:
            return cached

        if self._offline:
            return self._offline_result(f"{subject} {predicate} {obj}")

        start = time.time()
        sparql_query = (
            'PREFIX dbo: <http://dbpedia.org/ontology/> '
            'PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> '
            'SELECT ?s ?o WHERE { '
            f'  ?s rdfs:label "{_sparql_escape(subject)}"@en . '
            f'  ?o rdfs:label "{_sparql_escape(obj)}"@en . '
            '} LIMIT 1'
        )

        try:
            sparql = SPARQLWrapper(_ENDPOINT)
            sparql.setQuery(sparql_query)
            sparql.setReturnFormat(SPARQL_JSON)
            sparql.setTimeout(_TIMEOUT_S)
            raw = sparql.query().convert()

            query_time_ms = int((time.time() - start) * 1000)
            bindings = raw.get("results", {}).get("bindings", [])

            if bindings:
                result = VerificationResult(
                    is_verified=True,
                    confidence=0.65,
                    source="dbpedia",
                    evidence=[
                        f"Both '{subject}' and '{obj}' found in DBpedia",
                        f"Predicate '{predicate}' not explicitly verified (entity co-existence only)",
                    ],
                    query_time_ms=query_time_ms,
                )
            else:
                result = VerificationResult(
                    is_verified=False,
                    confidence=0.0,
                    source="dbpedia",
                    evidence=[f"Could not verify triple: {subject} — {predicate} — {obj}"],
                    query_time_ms=query_time_ms,
                )

            self._set_cached(cache_id, sparql_query, result)
            return result

        except Exception as exc:
            logger.warning("DBpedia verify_fact failed", error=str(exc))
            self._offline = True
            return self._offline_result(f"{subject} {predicate} {obj}")

    def is_available(self) -> bool:
        """Return True if the DBpedia endpoint is reachable (uses cached flag)."""
        return not self._offline

    # -- Private helpers -------------------------------------------------------

    def _offline_result(self, name: str) -> VerificationResult:
        return VerificationResult(
            is_verified=False,
            confidence=0.0,
            source="offline",
            evidence=[f"DBpedia unavailable — cannot verify '{name}'"],
            query_time_ms=0,
        )


def _sparql_escape(value: str) -> str:
    """Minimal SPARQL string escaping."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
