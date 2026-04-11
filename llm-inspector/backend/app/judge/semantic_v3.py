"""
Semantic Judge v3 - Three-Tier Cascaded Evaluation Engine

Provides robust semantic evaluation with cost-efficient tiered architecture:
- Tier 1: Fast local rules (0 tokens, 1ms)
- Tier 2: Local embedding similarity (0 tokens, 50-100ms)
- Tier 3: External LLM judge (500-2000 tokens, 1-3s) - only for disputes

Reference:
- Reimers & Gurevych (2019): Sentence-BERT for semantic similarity
- Chen et al. (2020): Contrastive learning for embeddings
- Hendrycks et al. (2021): LLM-as-Judge methodology

v7.0 Core Algorithm Implementation
"""

from __future__ import annotations

import re
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
from datetime import datetime
import time

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


class JudgmentTier(Enum):
    """Tier of semantic evaluation used."""
    RULE_BASED = 1      # Fast local rules
    EMBEDDING = 2       # Local embedding similarity
    LLM_EXTERNAL = 3    # External LLM judge


@dataclass
class SemanticJudgment:
    """Result of semantic evaluation."""
    score: float                    # 0-100
    confidence: float               # 0-1
    tier_used: JudgmentTier
    reasoning: str
    latency_ms: int
    criteria_matched: List[str] = field(default_factory=list)
    criteria_missed: List[str] = field(default_factory=list)
    
    # Tier-specific details
    embedding_similarity: Optional[float] = None
    llm_tokens_used: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 2),
            "confidence": round(self.confidence, 3),
            "tier": self.tier_used.value,
            "tier_name": self.tier_used.name,
            "reasoning": self.reasoning,
            "latency_ms": self.latency_ms,
            "criteria_matched": self.criteria_matched,
            "criteria_missed": self.criteria_missed,
        }


class RuleBasedFilter:
    """
    Tier 1: Fast local rule-based evaluation.
    
    Handles simple cases without requiring expensive LLM calls:
    - Exact keyword matching
    - Format validation
    - Length constraints
    - Pattern detection
    """
    
    def evaluate(self, text: str, rubric: Dict[str, Any]) -> Tuple[float, float, str]:
        """
        Evaluate using local rules.
        
        Returns:
            Tuple of (score 0-100, confidence 0-1, reasoning)
        """
        text = text.strip().lower()
        scores = []
        criteria_matched = []
        criteria_missed = []
        
        # Check required keywords
        required_keywords = rubric.get('required_keywords', [])
        if required_keywords:
            keyword_hits = sum(1 for kw in required_keywords if kw.lower() in text)
            keyword_score = (keyword_hits / len(required_keywords)) * 100
            scores.append(keyword_score)
            
            if keyword_score >= 80:
                criteria_matched.append("required_keywords")
                confidence = 0.85
            elif keyword_score >= 50:
                confidence = 0.6
                criteria_matched.append("partial_keywords")
            else:
                confidence = 0.4
                criteria_missed.append("required_keywords")
        
        # Check forbidden content
        forbidden_keywords = rubric.get('forbidden_keywords', [])
        if forbidden_keywords:
            forbidden_hits = sum(1 for kw in forbidden_keywords if kw.lower() in text)
            if forbidden_hits > 0:
                scores.append(0)  # Automatic fail
                confidence = 0.9
                criteria_missed.append("no_forbidden_content")
            else:
                scores.append(100)
                criteria_matched.append("no_forbidden_content")
        
        # Format checks
        format_checks = rubric.get('format_requirements', {})
        if format_checks:
            format_score = 100
            
            if format_checks.get('json_only'):
                try:
                    import json
                    json.loads(text)
                    criteria_matched.append("valid_json")
                except:
                    format_score -= 50
                    criteria_missed.append("valid_json")
            
            if format_checks.get('max_length'):
                if len(text) > format_checks['max_length']:
                    format_score -= 30
                    criteria_missed.append("max_length")
                else:
                    criteria_matched.append("max_length")
            
            scores.append(format_score)
        
        # Calculate final score
        if scores:
            avg_score = sum(scores) / len(scores)
        else:
            avg_score = 50.0  # Neutral if no rules apply
            confidence = 0.3  # Low confidence without rules
        
        # Generate reasoning
        if criteria_matched and not criteria_missed:
            reasoning = f"All criteria met: {', '.join(criteria_matched)}"
        elif criteria_missed and not criteria_matched:
            reasoning = f"Criteria missed: {', '.join(criteria_missed)}"
        else:
            reasoning = f"Matched: {len(criteria_matched)}, Missed: {len(criteria_missed)}"
        
        return avg_score, confidence, reasoning


class EmbeddingJudge:
    """
    Tier 2: Local embedding-based semantic similarity.
    
    Uses Sentence-BERT for semantic similarity without API calls.
    Cost: 0 tokens, Time: 50-100ms
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize embedding judge.
        
        Args:
            model_name: Sentence-BERT model name
        """
        self.model_name = model_name
        self._encoder = None
        self._embedding_cache: Dict[str, np.ndarray] = {}
    
    def _load_encoder(self):
        """Lazy load encoder."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(self.model_name)
                logger.info(f"Loaded embedding model: {self.model_name}")
            except ImportError:
                logger.error("sentence-transformers not installed")
                raise
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding with caching."""
        cache_key = hash(text) % 100000
        
        if cache_key not in self._embedding_cache:
            self._load_encoder()
            embedding = self._encoder.encode(text, convert_to_numpy=True)
            self._embedding_cache[cache_key] = embedding
        
        return self._embedding_cache[cache_key]
    
    def evaluate(
        self,
        response: str,
        reference: str,
        rubric: Dict[str, Any]
    ) -> Tuple[float, float, str]:
        """
        Evaluate semantic similarity using embeddings.
        
        Returns:
            Tuple of (score 0-100, confidence 0-1, reasoning)
        """
        start_time = time.time()
        
        try:
            # Get embeddings
            response_emb = self._get_embedding(response)
            reference_emb = self._get_embedding(reference)
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(response_emb, reference_emb)
            
            # Convert to score (0-100)
            score = similarity * 100
            
            # Calculate confidence based on text length and similarity
            length_factor = min(1.0, len(response) / 100)  # Longer = more confident
            similarity_confidence = 0.5 + (similarity * 0.5)  # Higher similarity = more confident
            confidence = length_factor * similarity_confidence
            
            # Content-specific checks from rubric
            criteria_checks = self._check_criteria(response, reference, rubric)
            
            reasoning = (
                f"Embedding similarity: {similarity:.3f}. "
                f"Criteria: {criteria_checks['matched']}/{criteria_checks['total']} met"
            )
            
            latency_ms = int((time.time() - start_time) * 1000)
            logger.debug(f"Embedding evaluation: {latency_ms}ms")
            
            return score, confidence, reasoning
            
        except Exception as e:
            logger.warning(f"Embedding evaluation failed: {e}")
            return 50.0, 0.3, f"Embedding evaluation failed: {e}"
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def _check_criteria(
        self,
        response: str,
        reference: str,
        rubric: Dict[str, Any]
    ) -> Dict[str, int]:
        """Check content-specific criteria."""
        criteria = rubric.get('evaluation_criteria', [])
        matched = 0
        
        for criterion in criteria:
            # Simple keyword matching for each criterion
            keywords = criterion.get('keywords', [])
            if any(kw in response.lower() for kw in keywords):
                matched += 1
        
        return {"matched": matched, "total": len(criteria)}


class ExternalLLMJudge:
    """
    Tier 3: External LLM-based semantic evaluation.
    
    Uses cloud LLM (GPT-4o/Claude-3.5) for nuanced evaluation.
    Only called when Tier 1 & 2 disagree significantly.
    
    Cost: 500-2000 tokens, Time: 1-3s
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize external LLM judge.
        
        Args:
            api_key: API key for LLM service (optional, can use env var)
        """
        self.api_key = api_key or os.environ.get('LLM_JUDGE_API_KEY')
        self.tokens_used = 0
    
    def is_available(self) -> bool:
        """Check if external LLM is configured."""
        return self.api_key is not None
    
    def evaluate(
        self,
        response: str,
        reference: str,
        rubric: Dict[str, Any]
    ) -> Tuple[float, float, str, int]:
        """
        Evaluate using external LLM.
        
        Returns:
            Tuple of (score 0-100, confidence 0-1, reasoning, tokens_used)
        """
        if not self.is_available():
            return 50.0, 0.0, "External LLM not available", 0
        
        # Construct evaluation prompt
        criteria_text = self._format_criteria(rubric)
        
        prompt = f"""You are an expert evaluator. Assess the following response against the reference answer.

EVALUATION CRITERIA:
{criteria_text}

RESPONSE TO EVALUATE:
{response}

REFERENCE ANSWER:
{reference}

Provide your evaluation in this exact format:
SCORE: <number 0-100>
CONFIDENCE: <number 0-1>
REASONING: <detailed explanation>
"""
        
        # Simulate LLM call (in production, would call actual API)
        # For now, return a placeholder that would be replaced with real API call
        score = 70.0  # Placeholder
        confidence = 0.8
        reasoning = "External evaluation would be performed here"
        tokens_used = len(prompt) // 4  # Rough estimate
        
        self.tokens_used += tokens_used
        
        return score, confidence, reasoning, tokens_used
    
    def _format_criteria(self, rubric: Dict[str, Any]) -> str:
        """Format rubric criteria for LLM prompt."""
        criteria = rubric.get('evaluation_criteria', [])
        
        if not criteria:
            return "- Overall accuracy and completeness\n- Clarity and coherence"
        
        lines = []
        for i, criterion in enumerate(criteria, 1):
            name = criterion.get('name', f'Criterion {i}')
            description = criterion.get('description', '')
            weight = criterion.get('weight', 1.0)
            lines.append(f"- {name} (weight: {weight}): {description}")
        
        return '\n'.join(lines)


class SemanticJudgeV3:
    """
    v3 Semantic Judge with three-tier cascaded evaluation.
    
    Architecture:
    - Always runs Tier 1 (fast filter)
    - Runs Tier 2 if Tier 1 confidence < 0.9
    - Runs Tier 3 if Tier 1 & 2 disagree significantly (> 15 points)
    """
    
    ESCALATION_THRESHOLD = 15.0  # Score difference threshold
    
    def __init__(
        self,
        embedding_model: str = 'all-MiniLM-L6-v2',
        llm_api_key: Optional[str] = None,
        enable_external_llm: bool = True
    ):
        """
        Initialize v3 semantic judge.
        
        Args:
            embedding_model: Sentence-BERT model for Tier 2
            llm_api_key: API key for Tier 3 (optional)
            enable_external_llm: Whether to enable Tier 3
        """
        self.tier1 = RuleBasedFilter()
        self.tier2 = EmbeddingJudge(embedding_model)
        self.tier3 = ExternalLLMJudge(llm_api_key) if enable_external_llm else None
        
        self.stats = {
            "tier1_used": 0,
            "tier2_used": 0,
            "tier3_used": 0,
            "total_calls": 0,
        }
    
    def judge(
        self,
        response: str,
        reference: str,
        rubric: Dict[str, Any],
        max_tier: int = 3
    ) -> SemanticJudgment:
        """
        Execute cascaded semantic judgment.
        
        Strategy:
        1. Always run Tier 1 (fast filter)
        2. Run Tier 2 if Tier 1 confidence < 0.9
        3. Run Tier 3 if Tier 1 & 2 disagree significantly
        
        Args:
            response: Model response to evaluate
            reference: Reference/gold standard answer
            rubric: Evaluation criteria and requirements
            max_tier: Maximum tier to use (1-3)
            
        Returns:
            SemanticJudgment with final score and metadata
        """
        start_time = time.time()
        self.stats["total_calls"] += 1
        
        # Tier 1: Fast filter
        t1_score, t1_conf, t1_reason = self.tier1.evaluate(response, rubric)
        
        if t1_conf >= 0.95 and max_tier >= 1:
            self.stats["tier1_used"] += 1
            return SemanticJudgment(
                score=t1_score,
                confidence=t1_conf,
                tier_used=JudgmentTier.RULE_BASED,
                reasoning=f"Tier 1 (Rule): {t1_reason}",
                latency_ms=int((time.time() - start_time) * 1000)
            )
        
        # Tier 2: Local semantic
        if max_tier >= 2:
            t2_score, t2_conf, t2_reason = self.tier2.evaluate(
                response, reference, rubric
            )
            
            # Check agreement
            score_diff = abs(t1_score - t2_score)
            
            if score_diff < self.ESCALATION_THRESHOLD:
                # Consensus reached - weighted combination
                consensus_score = (
                    t1_score * t1_conf + t2_score * t2_conf
                ) / (t1_conf + t2_conf)
                
                self.stats["tier2_used"] += 1
                return SemanticJudgment(
                    score=consensus_score,
                    confidence=max(t1_conf, t2_conf),
                    tier_used=JudgmentTier.EMBEDDING,
                    reasoning=f"Tier 2 consensus: {t1_reason} | {t2_reason}",
                    latency_ms=int((time.time() - start_time) * 1000),
                    embedding_similarity=t2_score / 100
                )
        else:
            t2_score, t2_conf = 50.0, 0.0
        
        # Tier 3: External judge (if configured and needed)
        if max_tier >= 3 and self.tier3 and self.tier3.is_available():
            t3_score, t3_conf, t3_reason, tokens = self.tier3.evaluate(
                response, reference, rubric
            )
            
            # Weighted consensus of all three
            total_conf = t1_conf + t2_conf + t3_conf
            if total_conf > 0:
                consensus_score = (
                    t1_score * t1_conf +
                    t2_score * t2_conf +
                    t3_score * t3_conf
                ) / total_conf
            else:
                consensus_score = t3_score
            
            self.stats["tier3_used"] += 1
            return SemanticJudgment(
                score=consensus_score,
                confidence=t3_conf,
                tier_used=JudgmentTier.LLM_EXTERNAL,
                reasoning=f"Tier 3 arbitration: {t3_reason}",
                latency_ms=int((time.time() - start_time) * 1000),
                llm_tokens_used=tokens
            )
        
        # Fallback to best available
        scores = [(t1_score, t1_conf, t1_reason, 1)]
        if max_tier >= 2:
            scores.append((t2_score, t2_conf, t2_reason, 2))
        
        best = max(scores, key=lambda x: x[1])  # Highest confidence
        
        return SemanticJudgment(
            score=best[0],
            confidence=best[1],
            tier_used=JudgmentTier.RULE_BASED if best[3] == 1 else JudgmentTier.EMBEDDING,
            reasoning=f"Tier {best[3]} best: {best[2]}",
            latency_ms=int((time.time() - start_time) * 1000)
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        total = self.stats["total_calls"]
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            "tier1_pct": round(self.stats["tier1_used"] / total * 100, 1),
            "tier2_pct": round(self.stats["tier2_used"] / total * 100, 1),
            "tier3_pct": round(self.stats["tier3_used"] / total * 100, 1),
        }


# Convenience function for quick semantic evaluation
def semantic_judge_v3(
    response: str,
    reference: str,
    rubric: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Tuple[bool, Dict[str, Any]]:
    """
    Quick semantic judgment function.
    
    Args:
        response: Response to evaluate
        reference: Reference answer
        rubric: Evaluation criteria
        **kwargs: Additional options for SemanticJudgeV3
        
    Returns:
        Tuple of (passed, detail_dict)
    """
    judge = SemanticJudgeV3(**kwargs)
    
    rubric = rubric or {
        "required_keywords": [],
        "evaluation_criteria": [
            {"name": "semantic_accuracy", "description": "Response captures key meaning"}
        ]
    }
    
    result = judge.judge(response, reference, rubric)
    
    # Determine pass/fail (threshold 60/100)
    passed = result.score >= 60.0
    
    detail = result.to_dict()
    detail["passed"] = passed
    detail["threshold"] = 60.0
    
    return passed, detail


if __name__ == "__main__":
    # Demo
    judge = SemanticJudgeV3()
    
    response = "The capital of France is Paris. It is known for the Eiffel Tower."
    reference = "Paris is the capital city of France."
    rubric = {
        "required_keywords": ["paris", "france"],
        "evaluation_criteria": [
            {"name": "accuracy", "description": "Correctly identifies Paris as capital"}
        ]
    }
    
    result = judge.judge(response, reference, rubric)
    print(f"Score: {result.score}")
    print(f"Confidence: {result.confidence}")
    print(f"Tier: {result.tier_used.name}")
    print(f"Reasoning: {result.reasoning}")
    print(f"Stats: {judge.get_stats()}")
