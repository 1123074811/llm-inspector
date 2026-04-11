"""Knowledge graph integration for LLM Inspector v8.0.

Provides fact verification through external knowledge sources:
- Wikidata API (primary)
- DBpedia (future)
- Local cache
"""
from app.knowledge.wikidata_client import WikidataClient, VerificationResult, WikidataEntity
from app.knowledge.kg_client import KnowledgeGraphClient

__all__ = [
    "WikidataClient",
    "VerificationResult", 
    "WikidataEntity",
    "KnowledgeGraphClient",
]
