"""
Differential Consistency Testing for Wrapper Detection.

Detects routing-based wrappers by:
1. Sending isomorphic prompts with semantic equivalence
2. Comparing response embeddings
3. Detecting distribution shifts

Reference:
- Shin et al. (2020) "Autoprompt: Eliciting Knowledge from Language Models"
- Wallace et al. (2019) "Universal Adversarial Triggers"
"""
from __future__ import annotations

import hashlib
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

from app.core.schemas import LayerResult, LLMRequest, Message
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ConsistencyReport:
    """Report from differential consistency testing."""
    mean_consistency: float  # 0-1
    std_consistency: float
    min_consistency: float
    routing_suspected: bool
    confidence: float  # 0-1
    details: List[Dict]
    isomorphic_pairs_tested: int


class DifferentialConsistencyTester:
    """
    Detects model routing by testing response consistency.
    
    Theory: If a wrapper routes to different backend models,
    semantically equivalent prompts may produce measurably 
    different response distributions.
    """
    
    # Isomorphic prompt pairs (same meaning, different phrasing)
    ISOMORPHIC_PAIRS: List[Tuple[str, str]] = [
        # Technology
        (
            "Explain quantum computing.",
            "Provide an explanation of how quantum computation works."
        ),
        # Simple facts
        (
            "What is 2+2?",
            "Calculate the sum of two and two."
        ),
        # Creative
        (
            "Write a haiku about nature.",
            "Compose a three-line Japanese-style poem about the natural world."
        ),
        # Comparison
        (
            "Compare Python and JavaScript.",
            "What are the differences between Python and JavaScript programming languages?"
        ),
        # Definition
        (
            "Define machine learning.",
            "What is the meaning of machine learning?"
        ),
        # Process
        (
            "How do you bake bread?",
            "Explain the process of baking bread step by step."
        ),
        # Opinion
        (
            "What are the benefits of exercise?",
            "List the advantages of regular physical activity."
        ),
        # Factual
        (
            "Who wrote Romeo and Juliet?",
            "What is the name of the author of Romeo and Juliet?"
        ),
    ]
    
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
    
    def _simple_hash_embedding(self, text: str) -> np.ndarray:
        """Create a simple embedding from text."""
        text = text.lower().strip()
        features = np.zeros(self.embedding_dim)
        
        # Character n-grams
        for n in range(2, 5):
            for i in range(len(text) - n + 1):
                ngram = text[i:i+n]
                idx = int(hashlib.md5(ngram.encode()).hexdigest(), 16) % (self.embedding_dim // 3)
                features[idx] += 1
        
        # Word features
        words = text.split()
        for word in words:
            word_idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % (self.embedding_dim // 3)
            features[word_idx + 2 * (self.embedding_dim // 3)] += 1
        
        # Normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        emb1 = self._simple_hash_embedding(text1)
        emb2 = self._simple_hash_embedding(text2)
        
        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cosine_sim = dot_product / (norm1 * norm2)
        # Convert to [0, 1]
        return (cosine_sim + 1) / 2
    
    def test_consistency(
        self,
        adapter,
        model_name: str,
        n_rounds: int = 5
    ) -> ConsistencyReport:
        """
        Test response consistency across isomorphic prompts.
        
        Args:
            adapter: Model adapter
            model_name: Name of the model to test
            n_rounds: Number of isomorphic pairs to test
        
        Returns:
            ConsistencyReport with analysis results
        """
        if n_rounds > len(self.ISOMORPHIC_PAIRS):
            n_rounds = len(self.ISOMORPHIC_PAIRS)
        
        consistency_scores = []
        details = []
        tokens_used = 0
        
        for i, (original, variant) in enumerate(self.ISOMORPHIC_PAIRS[:n_rounds]):
            # Get multiple responses for each variant
            original_responses = []
            variant_responses = []
            
            # Get 2 responses per prompt for stability
            for _ in range(2):
                resp_orig = adapter.chat(LLMRequest(
                    model=model_name,
                    messages=[Message("user", original)],
                    max_tokens=100,
                    temperature=0.7,  # Use higher temp to test consistency
                ))
                if resp_orig.content:
                    original_responses.append(resp_orig.content)
                    tokens_used += resp_orig.usage_total_tokens or 50
                
                resp_var = adapter.chat(LLMRequest(
                    model=model_name,
                    messages=[Message("user", variant)],
                    max_tokens=100,
                    temperature=0.7,
                ))
                if resp_var.content:
                    variant_responses.append(resp_var.content)
                    tokens_used += resp_var.usage_total_tokens or 50
            
            # Calculate cross-consistency
            if original_responses and variant_responses:
                cross_sims = []
                for orig in original_responses:
                    for var in variant_responses:
                        sim = self._calculate_similarity(orig, var)
                        cross_sims.append(sim)
                
                mean_sim = np.mean(cross_sims) if cross_sims else 0.5
                consistency_scores.append(mean_sim)
                
                # Intra-prompt consistency
                if len(original_responses) >= 2:
                    intra_orig = self._calculate_similarity(
                        original_responses[0], original_responses[1]
                    )
                else:
                    intra_orig = 1.0
                
                if len(variant_responses) >= 2:
                    intra_var = self._calculate_similarity(
                        variant_responses[0], variant_responses[1]
                    )
                else:
                    intra_var = 1.0
                
                details.append({
                    "pair_id": i,
                    "prompt_original": original[:50],
                    "prompt_variant": variant[:50],
                    "cross_consistency": float(mean_sim),
                    "intra_original": float(intra_orig),
                    "intra_variant": float(intra_var),
                    "num_responses_orig": len(original_responses),
                    "num_responses_var": len(variant_responses),
                })
        
        # Analyze consistency distribution
        if consistency_scores:
            mean_consistency = float(np.mean(consistency_scores))
            std_consistency = float(np.std(consistency_scores))
            min_consistency = float(np.min(consistency_scores))
        else:
            mean_consistency = 0.5
            std_consistency = 0.0
            min_consistency = 0.5
        
        # Detection logic
        # Low consistency suggests routing
        routing_suspected = (
            mean_consistency < 0.65 or  # Low average similarity
            std_consistency > 0.25 or  # High variance
            min_consistency < 0.40     # Some pairs very different
        )
        
        # Calculate confidence
        if routing_suspected:
            # More deviation = higher confidence
            confidence = min(0.90, max(0.50, 
                (0.70 - mean_consistency) * 2.0 + 
                std_consistency * 1.5
            ))
        else:
            confidence = max(0.0, (mean_consistency - 0.70) * 1.5)
        
        return ConsistencyReport(
            mean_consistency=mean_consistency,
            std_consistency=std_consistency,
            min_consistency=min_consistency,
            routing_suspected=routing_suspected,
            confidence=confidence,
            details=details,
            isomorphic_pairs_tested=len(consistency_scores)
        )


class Layer8DifferentialTesting:
    """
    Layer 8: Differential consistency testing.
    Detects model routing through semantic equivalence testing.
    """
    
    def __init__(self):
        self.tester = DifferentialConsistencyTester()
    
    def run(self, adapter, model_name: str) -> LayerResult:
        """Run differential consistency testing."""
        evidence = []
        tokens_used = 0
        confidence = 0.0
        identified = None
        
        try:
            report = self.tester.test_consistency(adapter, model_name, n_rounds=5)
            
            # Approximate token usage
            tokens_used = 5 * 2 * 2 * 75  # 5 pairs, 2 prompts each, 2 reps, ~75 tokens
            
            evidence.append(
                f"Tested {report.isomorphic_pairs_tested} isomorphic prompt pairs"
            )
            evidence.append(
                f"Mean consistency: {report.mean_consistency:.3f} "
                f"(std: {report.std_consistency:.3f})"
            )
            evidence.append(
                f"Min consistency: {report.min_consistency:.3f}"
            )
            
            if report.routing_suspected:
                evidence.append(
                    f"ROUTING SUSPECTED: Low consistency across semantically "
                    f"equivalent prompts suggests model routing"
                )
                confidence = min(0.85, report.confidence)
                identified = "Possible model routing detected"
            else:
                evidence.append(
                    f"Consistency normal: Responses to equivalent prompts are similar"
                )
                confidence = max(0.0, 0.30 - report.confidence)
            
            # Add detailed breakdown
            for detail in report.details[:3]:  # Show first 3
                evidence.append(
                    f"  Pair {detail['pair_id']}: "
                    f"cross={detail['cross_consistency']:.2f}, "
                    f"intra_orig={detail['intra_original']:.2f}, "
                    f"intra_var={detail['intra_variant']:.2f}"
                )
        
        except Exception as e:
            logger.warning("Differential testing failed", error=str(e))
            evidence.append(f"Differential testing failed: {str(e)[:100]}")
        
        return LayerResult(
            layer="differential_testing",
            confidence=confidence,
            identified_as=identified,
            evidence=evidence,
            tokens_used=tokens_used,
        )
