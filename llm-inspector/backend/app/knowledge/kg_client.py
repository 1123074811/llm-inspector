"""Knowledge Graph Client - Multi-source fact verification.

Aggregates multiple knowledge sources with fallback:
1. Wikidata (primary)
2. Local cache
3. Heuristic fallback
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

from app.knowledge.wikidata_client import WikidataClient, VerificationResult
from app.core.logging import get_logger

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
        "wikidata": 1.0,
        "cache": 0.9,
        "heuristic": 0.3,
    }
    
    def __init__(
        self,
        use_wikidata: bool = True,
        wikidata_rate_limit: float = 0.2,
        cache_size: int = 1000,
    ):
        """
        Initialize KG client.
        
        Args:
            use_wikidata: Enable Wikidata API
            wikidata_rate_limit: Seconds between requests
            cache_size: Max cache entries
        """
        self.use_wikidata = use_wikidata
        self._cache: Dict[str, VerificationResult] = {}
        self._cache_size = cache_size
        self._stats = {
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
        Verify entity existence through available sources.
        
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
                query_time_ms=0
            )
        
        cache_key = self._get_cache_key(entity_name, "entity")
        
        # Check cache first
        if cache_key in self._cache:
            self._stats["cache"].queries_total += 1
            self._stats["cache"].queries_cached += 1
            cached = self._cache[cache_key]
            # Update source to indicate cache hit
            return VerificationResult(
                is_verified=cached.is_verified,
                confidence=cached.confidence,
                source=f"{cached.source}_cache",
                evidence=cached.evidence + ["(from cache)"],
                entity=cached.entity,
                query_time_ms=0
            )
        
        # Try Wikidata
        if self._wikidata:
            try:
                self._stats["wikidata"].queries_total += 1
                result = self._wikidata.verify_entity_exists(entity_name)
                self._stats["wikidata"].queries_success += 1
                self._stats["wikidata"].avg_response_time_ms = (
                    (self._stats["wikidata"].avg_response_time_ms * 
                     (self._stats["wikidata"].queries_total - 1) + 
                     result.query_time_ms) / 
                    self._stats["wikidata"].queries_total
                )
                
                # Cache successful results
                self._add_to_cache(cache_key, result)
                return result
                
            except Exception as e:
                logger.error(f"Wikidata verification failed: {e}")
        
        # Fallback to heuristic
        result = self._heuristic_verify(entity_name)
        self._add_to_cache(cache_key, result)
        return result
    
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
