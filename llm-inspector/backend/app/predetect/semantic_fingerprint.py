"""
Semantic Fingerprinting for Model Identification.

Uses contrastive learning-inspired semantic pattern analysis to encode 
model-specific semantic patterns.

Reference: 
- Reimers & Gurevych (2019) for sentence embeddings
- Contrastive learning: Chen et al. (2020) "A Simple Framework for Contrastive Learning"
"""
from __future__ import annotations

import hashlib
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from app.core.schemas import LayerResult, LLMRequest, Message
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SemanticFingerprint:
    """Semantic fingerprint for model identification."""
    centroid: List[float]  # Mean embedding vector
    covariance: List[List[float]]  # Covariance matrix
    consistency: float  # Intra-model consistency score
    n_samples: int
    fingerprint_id: str
    response_patterns: Dict[str, str] = field(default_factory=dict)
    topic_characteristics: Dict[str, float] = field(default_factory=dict)


class SemanticFingerprinter:
    """
    Generates semantic fingerprints for model identification.
    
    Uses embedding-based analysis to identify model-specific patterns
    without requiring external API calls.
    """
    
    # Diverse probe prompts covering different topics
    FINGERPRINT_PROMPTS = [
        ("tech", "Explain the concept of {topic} in simple terms."),
        ("comparison", "Compare {topic_a} and {topic_b} in 2-3 sentences."),
        ("example", "Give a real-world example of {topic}."),
        ("pros_cons", "What are the main advantages and disadvantages of {topic}?"),
        ("process", "Describe the process of {topic} step by step."),
        ("definition", "Define {topic} in one clear sentence."),
    ]
    
    # Topics for diverse coverage
    TOPICS = [
        "machine learning", "blockchain", "climate change", "democracy",
        "quantum computing", "photosynthesis", "economic inflation",
        "neural networks", "renewable energy", "artificial intelligence",
        "globalization", "cryptography", "evolution", "supply chain",
    ]
    
    # Topic pairs for comparison prompts
    TOPIC_PAIRS = [
        ("Python", "JavaScript"),
        ("SQL", "NoSQL"),
        ("centralization", "decentralization"),
        ("solar power", "nuclear power"),
        ("capitalism", "socialism"),
    ]
    
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        # Simplified embedding using hash-based feature extraction
        # In production, this would use sentence-transformers
        
    def _simple_hash_embedding(self, text: str) -> np.ndarray:
        """
        Create a simple embedding from text using hash-based features.
        
        This is a lightweight alternative to full sentence embeddings
        that doesn't require external model downloads.
        """
        # Normalize text
        text = text.lower().strip()
        
        # Create feature vector based on character n-grams and word patterns
        features = np.zeros(self.embedding_dim)
        
        # Character n-grams (2-4 grams)
        for n in range(2, 5):
            for i in range(len(text) - n + 1):
                ngram = text[i:i+n]
                # Hash to index
                idx = int(hashlib.md5(ngram.encode()).hexdigest(), 16) % (self.embedding_dim // 3)
                features[idx] += 1
                
        # Word-level features
        words = text.split()
        for word in words:
            # Word length distribution
            word_len_idx = min(len(word), 20) % (self.embedding_dim // 3) + (self.embedding_dim // 3)
            features[word_len_idx] += 1
            
            # Word hash for vocabulary pattern
            word_idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % (self.embedding_dim // 3)
            features[word_idx + 2 * (self.embedding_dim // 3)] += 1
        
        # Normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
            
        return features
    
    def _fill_template(self, template: str, topic_type: str) -> str:
        """Fill template with random topics."""
        import random
        
        if topic_type == "comparison":
            pair = random.choice(self.TOPIC_PAIRS)
            return template.format(topic_a=pair[0], topic_b=pair[1])
        else:
            topic = random.choice(self.TOPICS)
            return template.format(topic=topic)
    
    def generate_fingerprint(
        self,
        adapter,
        model_name: str,
        n_samples: int = 6
    ) -> SemanticFingerprint:
        """
        Generate semantic fingerprint through multiple probe prompts.
        
        Returns:
            SemanticFingerprint with centroid, covariance, and patterns.
        """
        responses = []
        embeddings = []
        patterns: Dict[str, str] = {}
        
        # Select diverse prompts
        selected_prompts = self.FINGERPRINT_PROMPTS[:n_samples]
        
        tokens_used = 0
        for topic_type, template in selected_prompts:
            prompt = self._fill_template(template, topic_type)
            
            # Get response
            resp = adapter.chat(LLMRequest(
                model=model_name,
                messages=[Message("user", prompt)],
                max_tokens=150,
                temperature=0.0,
            ))
            
            if resp.usage_total_tokens:
                tokens_used += resp.usage_total_tokens
            
            if resp.content:
                responses.append(resp.content)
                embedding = self._simple_hash_embedding(resp.content)
                embeddings.append(embedding)
                patterns[topic_type] = resp.content[:200]  # Store truncated
        
        if not embeddings:
            # Return empty fingerprint if all failed
            return SemanticFingerprint(
                centroid=[0.0] * self.embedding_dim,
                covariance=[[0.0] * self.embedding_dim for _ in range(self.embedding_dim)],
                consistency=0.0,
                n_samples=0,
                fingerprint_id="empty",
                response_patterns={},
                topic_characteristics={}
            )
        
        embeddings = np.array(embeddings)
        
        # Calculate fingerprint statistics
        centroid = np.mean(embeddings, axis=0)
        
        # Calculate covariance (simplified diagonal only for efficiency)
        if len(embeddings) > 1:
            covariance = np.cov(embeddings.T)
            # Ensure 2D array
            if covariance.ndim == 0:
                covariance = np.array([[float(covariance)]])
            elif covariance.ndim == 1:
                covariance = np.diag(covariance)
        else:
            covariance = np.eye(self.embedding_dim) * 0.01
        
        # Calculate intra-model consistency
        if len(embeddings) > 1:
            pairwise_dists = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    dist = np.linalg.norm(embeddings[i] - embeddings[j])
                    pairwise_dists.append(dist)
            consistency = 1.0 / (1.0 + np.std(pairwise_dists)) if pairwise_dists else 0.5
        else:
            consistency = 0.5
        
        # Calculate topic characteristics (response length, complexity)
        topic_chars = {}
        for topic_type, response in patterns.items():
            words = len(response.split())
            sentences = response.count('.') + response.count('!') + response.count('?')
            complexity = len(set(response.lower().split())) / max(words, 1)
            topic_chars[topic_type] = {
                'length': words,
                'sentences': max(sentences, 1),
                'complexity': complexity
            }
        
        # Generate fingerprint ID
        centroid_hash = hashlib.sha256(centroid.tobytes()).hexdigest()[:16]
        
        return SemanticFingerprint(
            centroid=centroid.tolist(),
            covariance=covariance.tolist() if isinstance(covariance, np.ndarray) else covariance,
            consistency=float(consistency),
            n_samples=len(responses),
            fingerprint_id=f"fp_{centroid_hash}",
            response_patterns=patterns,
            topic_characteristics={k: v['complexity'] for k, v in topic_chars.items()}
        )
    
    def compare_fingerprints(
        self,
        fp1: SemanticFingerprint,
        fp2: SemanticFingerprint
    ) -> float:
        """
        Compare two semantic fingerprints.
        
        Returns similarity score (0-1) using simplified comparison.
        """
        c1 = np.array(fp1.centroid)
        c2 = np.array(fp2.centroid)
        
        # Cosine similarity
        dot_product = np.dot(c1, c2)
        norm1 = np.linalg.norm(c1)
        norm2 = np.linalg.norm(c2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cosine_sim = dot_product / (norm1 * norm2)
        
        # Convert to [0, 1] range
        similarity = (cosine_sim + 1) / 2
        
        # Adjust by consistency
        consistency_factor = min(fp1.consistency, fp2.consistency)
        adjusted_similarity = similarity * (0.7 + 0.3 * consistency_factor)
        
        return float(min(1.0, adjusted_similarity))
    
    def detect_wrapper_by_fingerprint(
        self,
        claimed_fp: SemanticFingerprint,
        suspected_base_fp: SemanticFingerprint,
        claimed_model: str,
        suspected_model: str
    ) -> Tuple[bool, float, List[str]]:
        """
        Detect if a claimed model is actually a wrapper around suspected base model.
        
        Returns:
            (is_wrapper, confidence, evidence)
        """
        similarity = self.compare_fingerprints(claimed_fp, suspected_base_fp)
        
        evidence = []
        is_wrapper = False
        confidence = 0.0
        
        # High similarity suggests same underlying model
        if similarity > 0.85:
            is_wrapper = True
            confidence = min(0.95, similarity)
            evidence.append(
                f"Semantic fingerprint highly similar ({similarity:.2f}) to {suspected_model}"
            )
        elif similarity > 0.70:
            is_wrapper = True
            confidence = min(0.75, similarity)
            evidence.append(
                f"Semantic fingerprint moderately similar ({similarity:.2f}) to {suspected_model}"
            )
        
        # Compare consistency (wrappers may have lower consistency due to filtering)
        if claimed_fp.consistency < 0.6 and suspected_base_fp.consistency > 0.7:
            is_wrapper = True
            confidence = max(confidence, 0.60)
            evidence.append(
                f"Claimed model consistency ({claimed_fp.consistency:.2f}) "
                f"significantly lower than base model ({suspected_base_fp.consistency:.2f})"
            )
        
        return is_wrapper, confidence, evidence


class Layer8SemanticFingerprint:
    """Layer 8: Semantic fingerprint detection."""
    
    def run(self, adapter, model_name: str) -> LayerResult:
        """Run semantic fingerprinting detection."""
        evidence = []
        tokens_used = 0
        confidence = 0.0
        identified = None
        
        fingerprinter = SemanticFingerprinter()
        
        try:
            # Generate fingerprint for current model
            fp = fingerprinter.generate_fingerprint(adapter, model_name, n_samples=6)
            tokens_used += 600  # Approximate token usage
            
            evidence.append(
                f"Generated semantic fingerprint: {fp.fingerprint_id} "
                f"(consistency={fp.consistency:.2f})"
            )
            
            # Check fingerprint quality
            if fp.consistency > 0.8:
                evidence.append(f"High consistency fingerprint (>{fp.consistency:.2f}) indicates stable model behavior")
                confidence = max(confidence, 0.40)
            elif fp.consistency < 0.5:
                evidence.append(f"Low consistency ({fp.consistency:.2f}) suggests potential wrapper/modification layer")
                confidence = max(confidence, 0.55)
                identified = "Possible wrapper (inconsistent semantic patterns)"
            
            # If we have a claimed model, check for known patterns
            if model_name:
                # Check response complexity characteristics
                avg_complexity = np.mean(list(fp.topic_characteristics.values())) if fp.topic_characteristics else 0
                evidence.append(f"Average response complexity: {avg_complexity:.3f}")
            
            # Store fingerprint for later comparison
            if fp.n_samples > 0:
                confidence = max(confidence, 0.30)
            
        except Exception as e:
            logger.warning("Semantic fingerprint generation failed", error=str(e))
            evidence.append(f"Fingerprint generation failed: {str(e)[:100]}")
        
        return LayerResult(
            layer="semantic_fingerprint",
            confidence=min(confidence, 0.70),
            identified_as=identified,
            evidence=evidence,
            tokens_used=tokens_used,
        )
