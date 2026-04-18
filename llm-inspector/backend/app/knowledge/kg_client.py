"""Knowledge Graph Client - Multi-source fact verification.

Aggregates multiple knowledge sources with fallback:
1. DBpedia SPARQL (dedicated DBpediaClient) + Wikidata — concurrent fan-out
2. Local in-memory + SQLite cache
3. Heuristic fallback
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

from app.knowledge.wikidata_client import WikidataClient, VerificationResult
from app.knowledge.dbpedia_client import DBpediaClient
from app.core.logging import get_logger
import time

try:
    from SPARQLWrapper import SPARQLWrapper, JSON
    HAS_SPARQL = True
except ImportError:
    HAS_SPARQL = False

logger = get_logger(__name__)


@dataclass
class KnowledgeSourceStats:
    """Statistics for a knowledge source."""
    source_name: str
    queries_total: int = 0
    queries_success: int = 0
    queries_cached: int = 0
    avg_response_time_ms: float = 0.0
    
    @property
    def success_rate(self) -> float:
        return self.queries_success / self.queries_total if self.queries_total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source_name,
            "total_queries": self.queries_total,
            "success_rate": round(self.success_rate, 3),
            "cached_queries": self.queries_cached,
            "avg_response_ms": round(self.avg_response_time_ms, 2),
        }


class KnowledgeGraphClient:
    """
    Multi-source knowledge graph client with intelligent fallback.
    
    Priority order:
    1. Wikidata API (if available)
    2. Local cache (for repeated queries)
    3. Heuristic fallback (entity name patterns)
    
    Usage:
        kg = KnowledgeGraphClient(use_wikidata=True)
        result = kg.verify_entity("Albert Einstein")
        if result.is_verified:
            print(f"Verified with confidence {result.confidence}")
    """
    
    # Signal weights for ensemble scoring (from v8 spec)
    SOURCE_WEIGHTS = {
        "dbpedia": 1.0,  # v10: DBpedia has highest priority for factual verification
        "wikidata": 0.9,
        "cache": 0.9,
        "heuristic": 0.3,
    }
    
    def __init__(
        self,
        use_wikidata: bool = True,
        use_dbpedia: bool = True,
        wikidata_rate_limit: float = 0.2,
        cache_size: int = 1000,
    ):
        """
        Initialize KG client.
        
        Args:
            use_wikidata: Enable Wikidata API
            use_dbpedia: Enable DBpedia SPARQL API
            wikidata_rate_limit: Seconds between requests
            cache_size: Max cache entries
        """
        self.use_wikidata = use_wikidata
        self.use_dbpedia = use_dbpedia and HAS_SPARQL
        self._cache: Dict[str, VerificationResult] = {}
        self._cache_size = cache_size
        self._stats = {
            "dbpedia": KnowledgeSourceStats("dbpedia"),
            "wikidata": KnowledgeSourceStats("wikidata"),
            "cache": KnowledgeSourceStats("cache"),
            "heuristic": KnowledgeSourceStats("heuristic"),
        }
        
        # Initialize Wikidata client
        self._wikidata: Optional[WikidataClient] = None
        if use_wikidata:
            try:
                self._wikidata = WikidataClient(rate_limit=wikidata_rate_limit)
                logger.info("Wikidata client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Wikidata: {e}")
                self._wikidata = None

        # Initialize dedicated DBpedia client (v13 Phase 5)
        self._dbpedia_client: DBpediaClient = DBpediaClient()
    
    def is_available(self) -> bool:
        """Check if any knowledge source is available."""
        if self._wikidata and self._wikidata.is_available():
            return True
        return len(self._cache) > 0  # Cache is always available
    
    def _get_cache_key(self, query: str, query_type: str = "entity") -> str:
        """Generate cache key."""
        return f"{query_type}:{query.lower().strip()}"
    
    def _add_to_cache(self, key: str, result: VerificationResult) -> None:
        """Add result to cache with LRU eviction."""
        if len(self._cache) >= self._cache_size:
            # Remove oldest entry (simple LRU)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[key] = result
    
    def _dbpedia_verify(self, entity_name: str) -> Optional[VerificationResult]:
        """
        Delegate to the dedicated DBpediaClient (v13 Phase 5).
        Kept for backward compatibility — internal callers use _dbpedia_client directly.
        """
        if not self.use_dbpedia:
            return None
        self._stats["dbpedia"].queries_total += 1
        start_time = time.time()
        try:
            result = self._dbpedia_client.verify_entity(entity_name)
            query_time_ms = int((time.time() - start_time) * 1000)
            if result.is_verified:
                self._stats["dbpedia"].queries_success += 1
            self._update_avg_time("dbpedia", query_time_ms)
            return result if result.is_verified else None
        except Exception as e:
            logger.error(f"DBpedia verification failed for '{entity_name}': {e}")
            query_time_ms = int((time.time() - start_time) * 1000)
            self._update_avg_time("dbpedia", query_time_ms)
            return None

    def _update_avg_time(self, source: str, query_time_ms: int):
        stats = self._stats[source]
        stats.avg_response_time_ms = (
            (stats.avg_response_time_ms * (stats.queries_total - 1) + query_time_ms) / 
            stats.queries_total
        )

    def _heuristic_verify(self, entity_name: str) -> VerificationResult:
        """
        Heuristic verification based on entity name patterns.
        
        Used as fallback when external sources unavailable.
        """
        self._stats["heuristic"].queries_total += 1
        
        confidence = 0.0
        evidence = []
        
        # Heuristic 1: Length check (very short unlikely to be real entity)
        if len(entity_name) < 3:
            confidence = 0.1
            evidence.append("Name too short for typical entity")
        # Heuristic 2: Capitalization (proper nouns start with capital)
        elif entity_name[0].isupper():
            confidence = 0.4
            evidence.append("Proper noun capitalization pattern")
            
            # Multiple words with caps suggests proper name
            words = entity_name.split()
            if len(words) >= 2 and all(w[0].isupper() for w in words if w):
                confidence = 0.5
                evidence.append("Multi-word proper name pattern")
        else:
            confidence = 0.2
            evidence.append("Not capitalized")
        
        # Heuristic 3: Common word check
        common_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are'
        }
        if entity_name.lower() in common_words:
            confidence = 0.05
            evidence.append("Common word, unlikely to be named entity")
        
        self._stats["heuristic"].queries_success += 1
        
        return VerificationResult(
            is_verified=confidence > 0.3,
            confidence=confidence,
            source="heuristic",
            evidence=evidence,
            query_time_ms=0
        )
    
    def verify_entity(self, entity_name: str) -> VerificationResult:
        """
        Verify entity existence through available sources (concurrent fan-out).

        v13 Phase 5: DBpedia and Wikidata are queried concurrently; results are
        cross-checked for consistency. Conflicts are surfaced with a low
        confidence score so callers can escalate to hallucination_v2.

        Args:
            entity_name: Entity name to verify

        Returns:
            VerificationResult with confidence score
        """
        if not entity_name or not isinstance(entity_name, str):
            return VerificationResult(
                is_verified=False,
                confidence=0.0,
                source="none",
                evidence=["Invalid input"],
                query_time_ms=0,
            )

        cache_key = self._get_cache_key(entity_name, "entity")

        # 1. Check in-memory cache first
        if cache_key in self._cache:
            self._stats["cache"].queries_total += 1
            self._stats["cache"].queries_cached += 1
            cached = self._cache[cache_key]
            return VerificationResult(
                is_verified=cached.is_verified,
                confidence=cached.confidence,
                source=f"{cached.source}_cache",
                evidence=cached.evidence + ["(from cache)"],
                entity=cached.entity,
                query_time_ms=0,
            )

        # 2. Fan-out: query both sources concurrently
        futures_map: Dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=2) as ex:
            if self.use_dbpedia:
                futures_map["dbpedia"] = ex.submit(
                    self._dbpedia_client.verify_entity, entity_name
                )
            if self._wikidata:
                futures_map["wikidata"] = ex.submit(
                    self._wikidata.verify_entity_exists, entity_name
                )

            raw_results: Dict[str, VerificationResult] = {}
            for src, fut in futures_map.items():
                try:
                    raw_results[src] = fut.result(timeout=5.0)
                    if src == "dbpedia":
                        self._stats["dbpedia"].queries_total += 1
                        if raw_results[src].is_verified:
                            self._stats["dbpedia"].queries_success += 1
                    elif src == "wikidata":
                        self._stats["wikidata"].queries_total += 1
                        self._stats["wikidata"].queries_success += 1
                except Exception as e:
                    logger.warning(f"KG source {src} failed", error=str(e))

        # 3. Cross-check consistency
        if raw_results:
            result = self._cross_check(entity_name, raw_results)
        else:
            # Both sources offline/failed — fall through to heuristic
            result = self._heuristic_verify(entity_name)

        self._add_to_cache(cache_key, result)
        return result

    def _cross_check(
        self,
        entity_name: str,
        results: Dict[str, VerificationResult],
    ) -> VerificationResult:
        """
        Cross-check results from multiple KG sources.

        Rules:
        - Both agree (both verified or both not) → higher-confidence result,
          source set to "<src1>+<src2>", verified_by_consensus=True
        - Conflict (one verified, one not) → is_verified=False, confidence=0.3,
          source="conflicting", "conflicting_sources" added to evidence
        - Single source → return as-is
        """
        if not results:
            return VerificationResult(
                is_verified=False,
                confidence=0.0,
                source="none",
                evidence=["No KG sources returned results"],
                query_time_ms=0,
            )

        if len(results) == 1:
            return next(iter(results.values()))

        sources = list(results.keys())
        r0 = results[sources[0]]
        r1 = results[sources[1]]

        agree = r0.is_verified == r1.is_verified
        if agree:
            # Return the higher-confidence result with consensus metadata
            winner = r0 if r0.confidence >= r1.confidence else r1
            combined_source = f"{sources[0]}+{sources[1]}"
            return VerificationResult(
                is_verified=winner.is_verified,
                confidence=winner.confidence,
                source=combined_source,
                evidence=winner.evidence + [f"Consensus: {combined_source}"],
                entity=winner.entity,
                query_time_ms=max(r0.query_time_ms, r1.query_time_ms),
                verified_by_consensus=True,
            )
        else:
            # Conflict — surface with low confidence for hallucination_v2 escalation
            return VerificationResult(
                is_verified=False,
                confidence=0.3,
                source="conflicting",
                evidence=[
                    f"conflicting_sources: {sources[0]} says {r0.is_verified}, "
                    f"{sources[1]} says {r1.is_verified}",
                    "Escalate to hallucination_v2 for manual review",
                ],
                query_time_ms=max(r0.query_time_ms, r1.query_time_ms),
                verified_by_consensus=False,
            )
    
    def verify_fact(
        self, 
        claim_text: str,
        extract_entities: bool = True
    ) -> VerificationResult:
        """
        Verify a factual claim.
        
        Args:
            claim_text: Textual claim to verify
            extract_entities: Whether to extract entities automatically
            
        Returns:
            VerificationResult
        """
        if not claim_text:
            return VerificationResult(
                is_verified=False,
                confidence=0.0,
                source="none",
                evidence=["Empty claim"]
            )
        
        cache_key = self._get_cache_key(claim_text, "claim")
        
        # Check cache
        if cache_key in self._cache:
            self._stats["cache"].queries_total += 1
            return self._cache[cache_key]
        
        # Use Wikidata claim verification
        if self._wikidata:
            try:
                result = self._wikidata.verify_claim_text(claim_text)
                self._add_to_cache(cache_key, result)
                return result
            except Exception as e:
                logger.error(f"Fact verification failed: {e}")
        
        # Fallback: extract and verify individual entities
        if extract_entities and self._wikidata:
            entities = self._wikidata._extract_entities(claim_text)
            if entities:
                # Verify first entity as representative
                return self.verify_entity(entities[0])
        
        # Final fallback
        return self._heuristic_verify(claim_text)
    
    def verify_entities_batch(
        self, 
        entity_names: List[str]
    ) -> Dict[str, VerificationResult]:
        """
        Verify multiple entities in batch.
        
        Args:
            entity_names: List of entity names
            
        Returns:
            Dict mapping entity names to results
        """
        results = {}
        for name in entity_names:
            results[name] = self.verify_entity(name)
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "sources": {
                name: stats.to_dict() 
                for name, stats in self._stats.items()
            },
            "cache_size": len(self._cache),
            "wikidata_available": self._wikidata is not None and self._wikidata.is_available(),
        }
    
    def clear_cache(self) -> None:
        """Clear verification cache."""
        self._cache.clear()
        logger.info("Knowledge graph cache cleared")
