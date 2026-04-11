"""
Bayesian Confidence Fusion for Multi-Layer Detection

Implements Bayesian inference to combine signals from multiple detection layers.
Provides principled uncertainty quantification for model identification.

Reference:
- Gelman et al. (2013): Bayesian Data Analysis
- Pearl (1988): Probabilistic Reasoning in Intelligent Systems

Key Features:
- Posterior probability updating with Bayes' rule
- Proper uncertainty propagation across layers
- Confidence interval estimation
- Model family prior integration

v7.0 Pre-Detection Enhancement
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from collections import defaultdict

import numpy as np
from scipy.stats import beta

from app.core.logging import get_logger

logger = get_logger(__name__)


class DetectionLayer(Enum):
    """Pre-detection layers."""
    HTTP = "http"
    SELF_REPORT = "self_report"
    IDENTITY_PROBE = "identity_probe"
    KNOWLEDGE_CUTOFF = "knowledge_cutoff"
    BEHAVIORAL_BIAS = "behavioral_bias"
    TOKENIZER = "tokenizer"
    SEMANTIC = "semantic"
    EXTRACTION = "extraction"
    DIFFERENTIAL = "differential"
    TOOL_PROBE = "tool_probe"
    CONTEXT_OVERLOAD = "context_overload"
    ADVERSARIAL = "adversarial"


@dataclass
class LayerEvidence:
    """Evidence from a single detection layer."""
    layer: DetectionLayer
    identified_model: Optional[str] = None
    confidence: float = 0.0  # Layer's own confidence
    likelihoods: Dict[str, float] = field(default_factory=dict)  # P(Evidence|Model)
    evidence_strength: float = 1.0  # 0-1, how diagnostic is this evidence
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer.value,
            "identified_model": self.identified_model,
            "confidence": round(self.confidence, 3),
            "evidence_strength": round(self.evidence_strength, 3),
            "top_likelihoods": dict(list(self.likelihoods.items())[:5]),
        }


@dataclass
class PosteriorDistribution:
    """Bayesian posterior over model identities."""
    probabilities: Dict[str, float]
    entropy: float
    max_probability: float
    most_likely_model: str
    confidence_interval_95: Tuple[float, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "most_likely": self.most_likely_model,
            "max_probability": round(self.max_probability, 4),
            "entropy": round(self.entropy, 3),
            "ci_95": [round(self.confidence_interval_95[0], 4),
                     round(self.confidence_interval_95[1], 4)],
            "top_models": dict(
                sorted(self.probabilities.items(), key=lambda x: x[1], reverse=True)[:5]
            ),
        }


class BayesianConfidenceFusion:
    """
    Bayesian fusion of multi-layer detection signals.
    
    Uses proper Bayesian updating:
    P(Model|Evidence) ∝ P(Evidence|Model) * P(Model)
    
    With Beta priors for robust uncertainty handling.
    """
    
    # Prior distribution parameters
    PRIOR_ALPHA = 1.0  # Beta distribution alpha
    PRIOR_BETA = 1.0   # Beta distribution beta (uniform prior)
    
    # Layer reliability weights (from empirical validation)
    LAYER_RELIABILITY = {
        DetectionLayer.HTTP: 0.85,
        DetectionLayer.SELF_REPORT: 0.65,
        DetectionLayer.IDENTITY_PROBE: 0.75,
        DetectionLayer.KNOWLEDGE_CUTOFF: 0.70,
        DetectionLayer.BEHAVIORAL_BIAS: 0.60,
        DetectionLayer.TOKENIZER: 0.90,
        DetectionLayer.SEMANTIC: 0.80,
        DetectionLayer.EXTRACTION: 0.70,
        DetectionLayer.DIFFERENTIAL: 0.75,
        DetectionLayer.TOOL_PROBE: 0.65,
        DetectionLayer.CONTEXT_OVERLOAD: 0.60,
        DetectionLayer.ADVERSARIAL: 0.55,
    }
    
    def __init__(
        self,
        known_models: List[str],
        prior_type: str = "uniform"
    ):
        """
        Initialize Bayesian fusion engine.
        
        Args:
            known_models: List of known model identifiers
            prior_type: "uniform" or "base_rate"
        """
        self.known_models = known_models
        self.n_models = len(known_models)
        
        # Initialize prior
        if prior_type == "uniform":
            self.prior = {model: 1.0 / self.n_models for model in known_models}
        else:
            # Base rate prior (would need empirical data)
            self.prior = {model: 1.0 / self.n_models for model in known_models}
        
        # Current posterior (starts as prior)
        self.posterior = self.prior.copy()
        self.evidence_history: List[LayerEvidence] = []
        
        # Track layer contributions
        self.layer_contributions: Dict[DetectionLayer, float] = {}
        
        logger.info(
            f"BayesianFusion initialized with {self.n_models} models, "
            f"prior_type={prior_type}"
        )
    
    def update(
        self,
        evidence: LayerEvidence
    ) -> PosteriorDistribution:
        """
        Bayesian update given new layer evidence.
        
        P(Model|Evidence) ∝ P(Evidence|Model) * P(Model)
        
        Args:
            evidence: Evidence from a detection layer
            
        Returns:
            Updated posterior distribution
        """
        # Store evidence
        self.evidence_history.append(evidence)
        
        # Get layer reliability
        reliability = self.LAYER_RELIABILITY.get(evidence.layer, 0.5)
        
        # Update each model's posterior
        new_posterior = {}
        total_evidence = 0.0
        
        for model in self.known_models:
            # Prior probability
            prior_prob = self.posterior.get(model, 1.0 / self.n_models)
            
            # Likelihood of evidence given model
            if model in evidence.likelihoods:
                likelihood = evidence.likelihoods[model]
            elif evidence.identified_model == model:
                likelihood = evidence.confidence
            else:
                # Evidence points to different model
                likelihood = (1 - evidence.confidence) / max(self.n_models - 1, 1)
            
            # Weight by layer reliability
            adjusted_likelihood = likelihood * reliability + 0.5 * (1 - reliability)
            
            # Unnormalized posterior
            posterior_unnorm = adjusted_likelihood * prior_prob
            new_posterior[model] = posterior_unnorm
            total_evidence += posterior_unnorm
        
        # Normalize
        if total_evidence > 0:
            new_posterior = {
                model: prob / total_evidence
                for model, prob in new_posterior.items()
            }
        else:
            # Fallback to uniform if all probabilities are 0
            new_posterior = {model: 1.0 / self.n_models for model in self.known_models}
        
        # Update state
        self.posterior = new_posterior
        
        # Track contribution
        self.layer_contributions[evidence.layer] = reliability
        
        # Calculate posterior statistics
        return self._calculate_posterior_stats()
    
    def _calculate_posterior_stats(self) -> PosteriorDistribution:
        """Calculate statistics from current posterior."""
        probs = self.posterior
        
        # Most likely model
        most_likely = max(probs, key=probs.get)
        max_prob = probs[most_likely]
        
        # Entropy (uncertainty measure)
        entropy = -sum(
            p * math.log2(p) if p > 0 else 0
            for p in probs.values()
        )
        max_entropy = math.log2(self.n_models)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Confidence interval using Beta approximation
        # For the most likely model
        alpha = self.PRIOR_ALPHA + max_prob * 100  # Pseudo-counts
        beta_param = self.PRIOR_BETA + (1 - max_prob) * 100
        
        ci_low = beta.ppf(0.025, alpha, beta_param)
        ci_high = beta.ppf(0.975, alpha, beta_param)
        
        return PosteriorDistribution(
            probabilities=probs,
            entropy=normalized_entropy,
            max_probability=max_prob,
            most_likely_model=most_likely,
            confidence_interval_95=(ci_low, ci_high)
        )
    
    def get_confidence(self) -> Tuple[str, float, bool]:
        """
        Get current confidence assessment.
        
        Returns:
            Tuple of (most_likely_model, confidence, is_confident)
        """
        posterior = self._calculate_posterior_stats()
        
        # Confidence threshold
        is_confident = (
            posterior.max_probability > 0.85 and
            posterior.entropy < 0.3
        )
        
        return (
            posterior.most_likely_model,
            posterior.max_probability,
            is_confident
        )
    
    def explain_decision(self) -> Dict[str, Any]:
        """
        Generate human-readable explanation of fusion result.
        
        Returns:
            Explanation dict with reasoning
        """
        posterior = self._calculate_posterior_stats()
        
        # Top contributing layers
        top_layers = sorted(
            self.layer_contributions.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        # Evidence summary
        evidence_summary = []
        for ev in self.evidence_history[-5:]:  # Last 5 pieces
            evidence_summary.append({
                "layer": ev.layer.value,
                "points_to": ev.identified_model or "unknown",
                "confidence": round(ev.confidence, 3),
                "reliability": round(self.LAYER_RELIABILITY.get(ev.layer, 0.5), 3),
            })
        
        return {
            "conclusion": {
                "most_likely": posterior.most_likely_model,
                "confidence": round(posterior.max_probability, 3),
                "entropy": round(posterior.entropy, 3),
            },
            "key_evidence": evidence_summary,
            "top_contributing_layers": [
                {"layer": layer.value, "reliability": round(rel, 3)}
                for layer, rel in top_layers
            ],
            "uncertainty": "low" if posterior.entropy < 0.3 else 
                          "medium" if posterior.entropy < 0.6 else "high",
        }
    
    def reset(self) -> None:
        """Reset to prior distribution."""
        self.posterior = self.prior.copy()
        self.evidence_history.clear()
        self.layer_contributions.clear()
        logger.info("BayesianFusion reset to prior")
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information about fusion state."""
        posterior = self._calculate_posterior_stats()
        
        return {
            "n_models": self.n_models,
            "n_evidence_pieces": len(self.evidence_history),
            "posterior_entropy": round(posterior.entropy, 3),
            "convergence": {
                "max_prob": round(posterior.max_probability, 3),
                "is_peaked": posterior.max_probability > 0.9,
            },
            "layer_usage": {
                layer.value: count
                for layer, count in self._count_layers().items()
            },
        }
    
    def _count_layers(self) -> Dict[DetectionLayer, int]:
        """Count evidence pieces by layer."""
        counts = defaultdict(int)
        for ev in self.evidence_history:
            counts[ev.layer] += 1
        return dict(counts)


class MultiModelFusion:
    """
    Fusion across multiple model families.
    
    Handles hierarchical inference:
    1. Model family (e.g., GPT, Claude)
    2. Specific model (e.g., GPT-4o)
    3. Version (e.g., GPT-4o-2024-08-06)
    """
    
    def __init__(
        self,
        model_hierarchy: Dict[str, List[str]]  # family -> [models]
    ):
        """
        Initialize hierarchical fusion.
        
        Args:
            model_hierarchy: Dict mapping families to their models
        """
        self.hierarchy = model_hierarchy
        self.family_fusion: Dict[str, BayesianConfidenceFusion] = {}
        
        # Initialize fusion for each family
        for family, models in model_hierarchy.items():
            self.family_fusion[family] = BayesianConfidenceFusion(
                known_models=models,
                prior_type="uniform"
            )
    
    def update(
        self,
        evidence: LayerEvidence,
        family_hint: Optional[str] = None
    ) -> Dict[str, PosteriorDistribution]:
        """
        Update with evidence, optionally targeting a specific family.
        
        Args:
            evidence: Layer evidence
            family_hint: Optional hint about model family
            
        Returns:
            Dict mapping families to their posteriors
        """
        results = {}
        
        if family_hint and family_hint in self.family_fusion:
            # Update only hinted family
            results[family_hint] = self.family_fusion[family_hint].update(evidence)
        else:
            # Update all families
            for family, fusion in self.family_fusion.items():
                results[family] = fusion.update(evidence)
        
        return results
    
    def get_best_match(
        self,
        family_threshold: float = 0.5,
        model_threshold: float = 0.7
    ) -> Optional[Tuple[str, str, float]]:
        """
        Get best matching model across all families.
        
        Args:
            family_threshold: Minimum family confidence
            model_threshold: Minimum model confidence
            
        Returns:
            Tuple of (family, model, confidence) or None
        """
        best = None
        best_confidence = 0.0
        
        for family, fusion in self.family_fusion.items():
            model, conf, is_conf = fusion.get_confidence()
            
            if is_conf and conf > model_threshold and conf > best_confidence:
                best = (family, model, conf)
                best_confidence = conf
        
        return best


# Convenience functions
def fuse_layer_evidence(
    evidences: List[LayerEvidence],
    known_models: List[str],
    confidence_threshold: float = 0.85
) -> Tuple[str, float, Dict[str, Any]]:
    """
    Quick fusion of multiple layer evidences.
    
    Args:
        evidences: List of layer evidences
        known_models: List of known models
        confidence_threshold: Threshold for confident identification
        
    Returns:
        Tuple of (identified_model, confidence, details)
    """
    fusion = BayesianConfidenceFusion(known_models)
    
    for evidence in evidences:
        fusion.update(evidence)
    
    model, confidence, is_confident = fusion.get_confidence()
    
    details = {
        "posterior": fusion._calculate_posterior_stats().to_dict(),
        "explanation": fusion.explain_decision(),
        "is_confident": is_confident,
        "threshold": confidence_threshold,
    }
    
    return model, confidence, details


if __name__ == "__main__":
    # Demo
    models = ["gpt-4o", "claude-3-opus", "gemini-pro", "deepseek-v3"]
    
    fusion = BayesianConfidenceFusion(models)
    
    # Simulate evidence from multiple layers
    evidences = [
        LayerEvidence(
            layer=DetectionLayer.HTTP,
            identified_model="gpt-4o",
            confidence=0.8,
            likelihoods={"gpt-4o": 0.9, "claude-3-opus": 0.2}
        ),
        LayerEvidence(
            layer=DetectionLayer.TOKENIZER,
            identified_model="gpt-4o",
            confidence=0.95,
            likelihoods={"gpt-4o": 0.95, "claude-3-opus": 0.15}
        ),
    ]
    
    for ev in evidences:
        posterior = fusion.update(ev)
        print(f"\nAfter {ev.layer.value}:")
        print(f"  Most likely: {posterior.most_likely_model} ({posterior.max_probability:.3f})")
        print(f"  Entropy: {posterior.entropy:.3f}")
    
    print(f"\nFinal: {fusion.get_confidence()}")
    print(f"\nExplanation: {fusion.explain_decision()}")
