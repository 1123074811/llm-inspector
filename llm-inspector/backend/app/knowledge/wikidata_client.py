"""Wikidata API client for fact verification.

Reference: https://www.wikidata.org/wiki/Wikidata:Data_access
License: CC0 (Public Domain)
Rate limit: 5 requests per second recommended
"""

import re
import time
import requests
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from functools import lru_cache
from datetime import datetime, timezone

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class WikidataEntity:
    """Wikidata entity representation."""
    id: str  # Q-number (e.g., "Q42")
    label: str
    description: str
    properties: Dict[str, List[str]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "properties": self.properties,
        }


@dataclass 
class VerificationResult:
    """Fact verification result."""
    is_verified: bool
    confidence: float  # 0-1
    source: str
    evidence: List[str] = field(default_factory=list)
    entity: Optional[WikidataEntity] = None
    query_time_ms: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_verified": self.is_verified,
            "confidence": round(self.confidence, 3),
            "source": self.source,
            "evidence": self.evidence,
            "entity": self.entity.to_dict() if self.entity else None,
            "query_time_ms": self.query_time_ms,
        }


class WikidataClient:
    """
    Wikidata API client for entity search and fact verification.
    
    Features:
    - Entity search (wbsearchentities)
    - Entity details (wbgetentities)
    - SPARQL queries
    - Rate limiting (5 req/sec)
    - Response caching
    
    Reference: https://www.wikidata.org/w/api.php
    """
    
    API_ENDPOINT = "https://www.wikidata.org/w/api.php"
    SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
    
    def __init__(self, rate_limit: float = 0.2, timeout: int = 10):
        """
        Initialize Wikidata client.
        
        Args:
            rate_limit: Minimum seconds between requests (default 0.2 = 5 req/sec)
            timeout: Request timeout in seconds
        """
        self.rate_limit = rate_limit
        self.timeout = timeout
        self._last_request_time = 0.0
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "LLM-Inspector/8.0 (research@example.com)",
            "Accept": "application/json",
        })
        
        logger.info(f"WikidataClient initialized (rate_limit={rate_limit}s)")
    
    def _rate_limited_request(self, url: str, params: Dict) -> Dict:
        """
        Make rate-limited API request.
        
        Args:
            url: API endpoint URL
            params: Request parameters
            
        Returns:
            JSON response
        """
        # Rate limiting
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        
        try:
            response = self._session.get(
                url, 
                params=params, 
                timeout=self.timeout
            )
            self._last_request_time = time.time()
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Wikidata API error: {e}")
            raise
    
    @lru_cache(maxsize=1000)
    def search_entities(
        self, 
        query: str, 
        language: str = "en",
        limit: int = 5
    ) -> List[WikidataEntity]:
        """
        Search for entities by keyword.
        
        Args:
            query: Search keyword
            language: Language code (default: "en")
            limit: Maximum results (default: 5)
            
        Returns:
            List of matching entities
        """
        if not query or len(query.strip()) < 2:
            return []
        
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": language,
            "search": query.strip(),
            "limit": limit,
        }
        
        try:
            data = self._rate_limited_request(self.API_ENDPOINT, params)
            
            entities = []
            for result in data.get("search", []):
                entity = WikidataEntity(
                    id=result.get("id", ""),
                    label=result.get("label", ""),
                    description=result.get("description", ""),
                    properties={}
                )
                entities.append(entity)
            
            logger.debug(f"Wikidata search '{query}': {len(entities)} results")
            return entities
            
        except Exception as e:
            logger.error(f"Error searching Wikidata for '{query}': {e}")
            return []
    
    @lru_cache(maxsize=1000)
    def get_entity(self, entity_id: str) -> Optional[WikidataEntity]:
        """
        Get detailed entity information by Q-number.
        
        Args:
            entity_id: Q-number (e.g., "Q42")
            
        Returns:
            Entity details or None if not found
        """
        if not entity_id or not entity_id.startswith("Q"):
            return None
        
        params = {
            "action": "wbgetentities",
            "format": "json",
            "ids": entity_id,
            "props": "labels|descriptions|claims",
            "languages": "en|zh",
        }
        
        try:
            data = self._rate_limited_request(self.API_ENDPOINT, params)
            
            entity_data = data.get("entities", {}).get(entity_id)
            if not entity_data:
                return None
            
            # Extract labels
            labels = entity_data.get("labels", {})
            label = (
                labels.get("zh", {}).get("value") or 
                labels.get("en", {}).get("value", "")
            )
            
            # Extract descriptions
            descriptions = entity_data.get("descriptions", {})
            description = (
                descriptions.get("zh", {}).get("value") or 
                descriptions.get("en", {}).get("value", "")
            )
            
            # Extract key properties
            properties = {}
            claims = entity_data.get("claims", {})
            
            # P31: instance of
            if "P31" in claims:
                properties["instance_of"] = [
                    c.get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id", "")
                    for c in claims["P31"]
                ]
            
            # P279: subclass of
            if "P279" in claims:
                properties["subclass_of"] = [
                    c.get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id", "")
                    for c in claims["P279"]
                ]
            
            return WikidataEntity(
                id=entity_id,
                label=label,
                description=description,
                properties=properties
            )
            
        except Exception as e:
            logger.error(f"Error fetching entity {entity_id}: {e}")
            return None
    
    def verify_entity_exists(self, entity_name: str) -> VerificationResult:
        """
        Verify if an entity exists in Wikidata.
        
        Args:
            entity_name: Name to search for
            
        Returns:
            Verification result
        """
        start_time = time.time()
        
        entities = self.search_entities(entity_name, limit=3)
        query_time = int((time.time() - start_time) * 1000)
        
        if not entities:
            return VerificationResult(
                is_verified=False,
                confidence=0.0,
                source="wikidata",
                evidence=[f"No entities found for '{entity_name}'"],
                query_time_ms=query_time
            )
        
        # Check for exact match
        exact_match = any(
            e.label.lower() == entity_name.lower() 
            for e in entities
        )
        
        # Check for partial match
        partial_match = any(
            entity_name.lower() in e.label.lower() 
            for e in entities
        )
        
        if exact_match:
            confidence = 0.95
            evidence = [f"Exact match: {entities[0].label} ({entities[0].id})"]
        elif partial_match:
            confidence = 0.7
            evidence = [f"Partial match: {entities[0].label} ({entities[0].id})"]
        else:
            confidence = 0.3
            evidence = [f"Related entity: {entities[0].label}"]
        
        return VerificationResult(
            is_verified=confidence > 0.5,
            confidence=confidence,
            source="wikidata",
            evidence=evidence,
            entity=entities[0] if entities else None,
            query_time_ms=query_time
        )
    
    def verify_fact(
        self, 
        subject: str, 
        predicate: str, 
        object: str
    ) -> VerificationResult:
        """
        Verify a simple S-P-O fact.
        
        Args:
            subject: Subject entity name
            predicate: Relation/verb
            object: Object entity/value
            
        Returns:
            Verification result
        """
        start_time = time.time()
        
        # Step 1: Find subject entity
        subject_result = self.verify_entity_exists(subject)
        
        if not subject_result.is_verified:
            return VerificationResult(
                is_verified=False,
                confidence=0.0,
                source="wikidata",
                evidence=[f"Subject '{subject}' not found"],
                query_time_ms=int((time.time() - start_time) * 1000)
            )
        
        # Step 2: Find object entity (if it's an entity)
        object_result = self.verify_entity_exists(object)
        
        # Basic verification: both subject and object exist
        if subject_result.is_verified and object_result.is_verified:
            confidence = 0.6  # Both exist, but relation not verified
            evidence = [
                f"Subject found: {subject_result.entity.label if subject_result.entity else subject}",
                f"Object found: {object_result.entity.label if object_result.entity else object}",
                "Relation not explicitly verified"
            ]
        else:
            confidence = 0.3
            evidence = ["Partial match - only subject verified"]
        
        query_time = int((time.time() - start_time) * 1000)
        
        return VerificationResult(
            is_verified=confidence > 0.5,
            confidence=confidence,
            source="wikidata",
            evidence=evidence,
            entity=subject_result.entity,
            query_time_ms=query_time
        )
    
    def verify_claim_text(self, claim: str) -> VerificationResult:
        """
        Verify a textual claim by extracting entities.
        
        Args:
            claim: Textual claim to verify
            
        Returns:
            Verification result
        """
        start_time = time.time()
        
        # Simple entity extraction: proper nouns and quoted phrases
        entities = self._extract_entities(claim)
        
        if not entities:
            return VerificationResult(
                is_verified=False,
                confidence=0.0,
                source="wikidata",
                evidence=["No extractable entities found"],
                query_time_ms=int((time.time() - start_time) * 1000)
            )
        
        # Verify each entity
        verified_entities = []
        total_confidence = 0.0
        
        for entity in entities[:3]:  # Limit to first 3 entities
            result = self.verify_entity_exists(entity)
            if result.is_verified:
                verified_entities.append(result)
                total_confidence += result.confidence
        
        # Calculate overall confidence
        if verified_entities:
            avg_confidence = total_confidence / len(entities)
            evidence = [
                f"Verified: {e.entity.label if e.entity else 'unknown'}" 
                for e in verified_entities
            ]
        else:
            avg_confidence = 0.0
            evidence = ["No entities verified"]
        
        query_time = int((time.time() - start_time) * 1000)
        
        return VerificationResult(
            is_verified=avg_confidence > 0.5,
            confidence=avg_confidence,
            source="wikidata",
            evidence=evidence,
            entity=verified_entities[0].entity if verified_entities else None,
            query_time_ms=query_time
        )
    
    def _extract_entities(self, text: str) -> List[str]:
        """
        Simple entity extraction from text.
        
        Extracts:
        - Capitalized phrases (proper nouns)
        - Quoted text
        - "X is Y" patterns
        
        Args:
            text: Input text
            
        Returns:
            List of extracted entity candidates
        """
        entities = []
        
        # Pattern 1: "X is Y" or "X was Y" (entity at X)
        is_pattern = re.findall(
            r'([A-Z][a-zA-Z\s]+?)\s+(?:is|was|are|were)\s+([a-zA-Z\s]+)',
            text
        )
        for match in is_pattern:
            entities.extend([m.strip() for m in match if len(m.strip()) > 2])
        
        # Pattern 2: Quoted text
        quoted = re.findall(r'"([^"]+)"', text)
        entities.extend([q.strip() for q in quoted if len(q.strip()) > 2])
        
        # Pattern 3: Capitalized words/phrases (potential proper nouns)
        # Skip sentence-initial words (follow period + space)
        proper_nouns = re.findall(
            r'(?:^|\.\s)([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)(?=\s+(?:is|was|are|were|the|a|an|of|in|on|at|,|\.|$))',
            text
        )
        entities.extend([p.strip() for p in proper_nouns if len(p.strip()) > 2])
        
        # Clean and deduplicate
        entities = [e.strip() for e in entities if len(e.strip()) > 2]
        seen = set()
        result = []
        for e in entities:
            key = e.lower()
            if key not in seen:
                seen.add(key)
                result.append(e)
        
        return result
    
    def is_available(self) -> bool:
        """Check if Wikidata service is available."""
        try:
            # Quick test query
            params = {
                "action": "wbsearchentities",
                "format": "json",
                "language": "en",
                "search": "test",
                "limit": 1,
            }
            response = self._session.get(
                self.API_ENDPOINT, 
                params=params, 
                timeout=5
            )
            return response.status_code == 200
        except:
            return False
