"""Test suite for LLM Inspector v8.0 Phase 3: Scoring System Refactoring.

Tests the three core components:
1. Theta Scale Scoring (with confidence intervals)
2. Adaptive Testing 2.0 (multi-dimensional CAT)
3. Factor Analysis (dimension validation)
"""

import pytest
import math
import tempfile
import os
import numpy as np

# Phase 3 imports
from app.analysis.theta_scoring import (
    ThetaScore,
    CompositeScore,
    ThetaScoringEngine,
    get_theta_engine,
)
from app.analysis.adaptive_test_v2 import (
    AdaptiveTestConfig,
    TestSessionState,
    AdaptiveTestSelector,
    AdaptiveTestEngine,
    StoppingRule,
    get_adaptive_engine,
)
from app.analysis.factor_analysis import (
    DimensionValidator,
    CFAResult,
    ValidityStatus,
)
from app.analysis.irt_params import IRTParameters


class TestThetaScore:
    """Test ThetaScore dataclass."""
    
    def test_theta_score_creation(self):
        """Test basic ThetaScore creation."""
        score = ThetaScore(
            theta=0.5,
            standard_error=0.2,
            dimension="reasoning"
        )
        
        assert score.theta == 0.5
        assert score.standard_error == 0.2
        assert score.dimension == "reasoning"
    
    def test_confidence_interval_calculation(self):
        """Test confidence interval auto-calculation."""
        score = ThetaScore(theta=0.0, standard_error=0.3)
        
        # CI should be approximately +/- 1.96 * SE
        expected_margin = 1.96 * 0.3
        assert abs(score.ci_lower - (-expected_margin)) < 0.01
        assert abs(score.ci_upper - expected_margin) < 0.01
    
    def test_percentile_conversion(self):
        """Test theta to percentile conversion."""
        # Theta = 0 should give ~50th percentile
        score = ThetaScore(theta=0.0, standard_error=0.1)
        assert abs(score.percentile - 50.0) < 1.0
        
        # Theta = 1 should give ~84th percentile
        score = ThetaScore(theta=1.0, standard_error=0.1)
        assert 80 < score.percentile < 90
        
        # Theta = -1 should give ~16th percentile
        score = ThetaScore(theta=-1.0, standard_error=0.1)
        assert 10 < score.percentile < 20
    
    def test_precision_levels(self):
        """Test precision level classification."""
        # High precision
        score = ThetaScore(theta=0.0, standard_error=0.15)
        assert score.precision == "high"
        
        # Medium precision
        score = ThetaScore(theta=0.0, standard_error=0.3)
        assert score.precision == "medium"
        
        # Low precision
        score = ThetaScore(theta=0.0, standard_error=0.5)
        assert score.precision == "low"
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        score = ThetaScore(
            theta=0.5,
            standard_error=0.2,
            dimension="reasoning",
            reliability=0.85
        )
        
        d = score.to_dict()
        assert d["theta"] == 0.5
        assert d["precision"] == "medium"
        assert "confidence_interval" in d
        assert d["reliability"] == 0.85


class TestThetaScoringEngine:
    """Test ThetaScoringEngine functionality."""
    
    @pytest.fixture
    def sample_items(self):
        """Create sample IRT parameters."""
        return [
            IRTParameters(case_id=f"item_{i}", a=1.0, b=0.0)
            for i in range(5)
        ]
    
    def test_mle_estimation(self, sample_items):
        """Test MLE theta estimation."""
        engine = ThetaScoringEngine("mle")
        
        # Mixed responses
        responses = [True, True, True, False, False]
        result = engine.calculate_theta(responses, sample_items)
        
        # Should return valid theta estimate
        assert -4.0 <= result.theta <= 4.0
        assert result.standard_error > 0
        assert result.reliability >= 0
    
    def test_wle_estimation_bias_correction(self, sample_items):
        """Test WLE bias correction for extreme scores."""
        engine = ThetaScoringEngine("wle")
        
        # All correct (extreme score)
        responses = [True] * 5
        result = engine.calculate_theta(responses, sample_items)
        
        # WLE should not be infinity (bias corrected)
        assert result.theta < 4.0
        assert result.theta > 0
    
    def test_eap_estimation_with_prior(self, sample_items):
        """Test EAP estimation with normal prior."""
        engine = ThetaScoringEngine("eap")
        
        responses = [True, True, False, False, True]
        result = engine.calculate_theta(responses, sample_items)
        
        assert -4.0 < result.theta < 4.0
        assert result.standard_error > 0
    
    def test_dimension_synthesis_information_weighted(self):
        """Test information-weighted dimension synthesis."""
        engine = ThetaScoringEngine()
        
        # Create dimension scores with different information
        dim_scores = {
            "reasoning": ThetaScore(theta=0.5, standard_error=0.2, information=10.0),
            "coding": ThetaScore(theta=0.3, standard_error=0.4, information=5.0),
        }
        
        composite = engine.synthesize_dimensions(dim_scores, "information_weighted")
        
        # Should weight by information (reasoning should have higher weight)
        assert composite.dimension_weights["reasoning"] > composite.dimension_weights["coding"]
        assert composite.score > 0  # Weighted average
    
    def test_equal_weight_synthesis(self):
        """Test equal weight synthesis."""
        engine = ThetaScoringEngine()
        
        dim_scores = {
            "a": ThetaScore(theta=1.0, standard_error=0.3),
            "b": ThetaScore(theta=-1.0, standard_error=0.3),
        }
        
        composite = engine.synthesize_dimensions(dim_scores, "equal")
        
        # Equal weights: both should be 0.5
        assert composite.dimension_weights["a"] == 0.5
        assert composite.dimension_weights["b"] == 0.5
        assert abs(composite.score - 0.0) < 0.01  # Average of 1.0 and -1.0
    
    def test_percent_to_theta_conversion(self, sample_items):
        """Test backward-compatible percentile conversion."""
        engine = ThetaScoringEngine()
        
        # Test that conversion produces valid theta values
        # 50% should give moderate theta
        theta_50, se_50 = engine.percent_to_theta(50.0, sample_items)
        assert -4.0 < theta_50 < 4.0
        assert se_50 > 0
        
        # 100% should give higher theta than 50%
        theta_100, se_100 = engine.percent_to_theta(100.0, sample_items)
        assert theta_100 > theta_50
        
        # 0% should give lower theta than 50%
        theta_0, se_0 = engine.percent_to_theta(0.0, sample_items)
        assert theta_0 < theta_50


class TestAdaptiveTestConfig:
    """Test adaptive testing configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = AdaptiveTestConfig()
        
        assert config.min_items == 5
        assert config.max_items == 20
        assert config.target_se == 0.3
        assert StoppingRule.FIXED_LENGTH in config.stopping_rules
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = AdaptiveTestConfig(
            dimensions=["reasoning", "coding"],
            min_items=3,
            max_items=15,
            target_se=0.25,
            token_budget=5000
        )
        
        assert config.dimensions == ["reasoning", "coding"]
        assert config.token_budget == 5000


class TestAdaptiveTestSession:
    """Test adaptive test session management."""
    
    def test_session_creation(self):
        """Test session initialization."""
        config = AdaptiveTestConfig(dimensions=["reasoning"])
        
        engine = AdaptiveTestEngine()
        session = engine.start_session(config, "test_001")
        
        assert session.session_id == "test_001"
        assert not session.is_complete
        assert len(session.administered_items) == 0
    
    def test_session_response_recording(self):
        """Test recording responses."""
        config = AdaptiveTestConfig(dimensions=["reasoning"])
        engine = AdaptiveTestEngine()
        session = engine.start_session(config, "test_002")
        
        # Record a response
        updated = engine.record_response("test_002", "item_001", True, 500)
        
        assert "item_001" in updated.administered_items
        assert updated.responses["item_001"] is True
        assert updated.tokens_used == 500
    
    def test_stopping_rule_fixed_length(self):
        """Test fixed length stopping rule."""
        config = AdaptiveTestConfig(
            min_items=3,
            max_items=5,
            stopping_rules=[StoppingRule.FIXED_LENGTH]
        )
        
        engine = AdaptiveTestEngine()
        session = engine.start_session(config, "test_003")
        
        # Add items up to max
        for i in range(5):
            engine.record_response("test_003", f"item_{i}", True, 100)
        
        # Should be complete
        assert engine.sessions["test_003"].is_complete


class TestDimensionValidator:
    """Test factor analysis validation."""
    
    def test_cfa_fit_indices(self):
        """Test CFA fit index calculation."""
        validator = DimensionValidator()
        
        # Create synthetic dimension scores
        np.random.seed(42)
        n = 100  # 100 models
        
        dimension_scores = {
            "reasoning": np.random.normal(0.5, 0.2, n).tolist(),
            "coding": np.random.normal(0.4, 0.25, n).tolist(),
            "safety": np.random.normal(0.6, 0.18, n).tolist(),
        }
        
        result = validator.validate_dimensions(dimension_scores)
        
        # Should return valid result structure
        assert isinstance(result, CFAResult)
        assert hasattr(result, 'rmsea')
        assert hasattr(result, 'cfi')
        assert hasattr(result, 'factor_loadings')
    
    def test_fit_criteria_cutoffs(self):
        """Test fit index cutoffs."""
        validator = DimensionValidator()
        
        # Check cutoff values
        assert validator.RMSEA_CUTOFF == 0.06
        assert validator.CFI_CUTOFF == 0.95
        assert validator.SRMR_CUTOFF == 0.08
    
    def test_convergent_validity_threshold(self):
        """Test AVE threshold."""
        validator = DimensionValidator()
        
        # AVE should be > 0.5 for convergent validity
        assert validator.AVE_CUTOFF == 0.50


class TestIntegration:
    """Integration tests for Phase 3 components."""
    
    def test_full_adaptive_test_flow(self):
        """Test complete adaptive testing flow."""
        # Setup
        config = AdaptiveTestConfig(
            dimensions=["reasoning"],
            min_items=3,
            max_items=5,
            target_se=0.4,
            stopping_rules=[StoppingRule.FIXED_LENGTH, StoppingRule.STANDARD_ERROR]
        )
        
        engine = get_adaptive_engine()
        session = engine.start_session(config, "integration_test")
        
        # Simulate test (would normally get items from pool)
        for i in range(4):
            engine.record_response("integration_test", f"item_{i}", True, 200)
        
        # Get results
        results = engine.get_session_results("integration_test")
        
        assert results["items_administered"] == 4
        assert "ability_estimates" in results
        
        # End session
        final = engine.end_session("integration_test")
        assert final["session_id"] == "integration_test"
    
    def test_theta_to_adaptive_integration(self):
        """Test integration between theta scoring and adaptive testing."""
        # Create items with known parameters
        items = [
            IRTParameters(case_id=f"item_{i}", a=1.2, b=float(i-2)/2)
            for i in range(5)
        ]
        
        # Score some responses
        scoring = ThetaScoringEngine()
        responses = [True, True, False, True, False]
        
        theta_score = scoring.calculate_theta(responses, items, "reasoning")
        
        # Theta should be reasonable
        assert -4.0 < theta_score.theta < 4.0
        assert theta_score.standard_error > 0
        assert theta_score.reliability > 0
    
    def test_composite_scoring_with_reliability(self):
        """Test composite score reliability calculation."""
        engine = ThetaScoringEngine()
        
        # Create dimension scores with varying precision
        dim_scores = {
            "high_precision": ThetaScore(theta=0.5, standard_error=0.15, information=20.0),
            "medium_precision": ThetaScore(theta=0.3, standard_error=0.3, information=5.0),
            "low_precision": ThetaScore(theta=0.4, standard_error=0.5, information=2.0),
        }
        
        composite = engine.synthesize_dimensions(dim_scores)
        
        # Higher precision dimensions should have more weight
        weights = composite.dimension_weights
        assert weights["high_precision"] > weights["low_precision"]
        
        # Composite should have reliability
        assert 0 < composite.reliability <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
