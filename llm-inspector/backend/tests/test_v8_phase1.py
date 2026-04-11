"""Test suite for LLM Inspector v8.0 Phase 1: Data Layer Enhancement.

Tests the three core components:
1. Data provenance tracking system
2. Reference database
3. IRT parameter database
"""

import pytest
import tempfile
import os
from datetime import datetime

# Import v8.0 components
from app.core.provenance import (
    DataProvenance,
    ProvenanceTracker,
    get_provenance_tracker,
    reset_provenance_tracker,
)
from app.core.references import (
    Reference,
    ReferenceType,
    ReferenceDatabase,
    get_reference_database,
    validate_formula_source,
)
from app.analysis.irt_params import (
    IRTParameters,
    IRTParameterDB,
    ThetaScoreConverter,
    get_irt_db,
)
from app.core.schemas import TestCaseV8


class TestDataProvenance:
    """Test data provenance tracking system."""
    
    def test_create_fallback(self):
        """Test fallback provenance creation."""
        provenance = DataProvenance.create_fallback("test_weight")
        
        assert provenance.source_type == "fallback"
        assert provenance.confidence == 0.0
        assert not provenance.verified
        assert "test_weight" in provenance.source_id
    
    def test_from_irt_calibration(self):
        """Test IRT calibration provenance creation."""
        provenance = DataProvenance.from_irt_calibration(
            case_id="reason_001",
            calibration_version="v2026q1",
            sample_size=1500,
            confidence=0.85
        )
        
        assert provenance.source_type == "irt_calibration"
        assert provenance.sample_size == 1500
        assert provenance.confidence == 0.85
        assert provenance.verified
        assert "v2026q1" in provenance.source_id
    
    def test_from_literature(self):
        """Test literature-based provenance."""
        provenance = DataProvenance.from_literature(
            title="Item Response Theory for Psychologists",
            authors="Embretson & Reise",
            year=2000,
            doi="10.1234/example"
        )
        
        assert provenance.source_type == "literature"
        assert provenance.sample_size == 0  # Literature has no sample size
        assert "Item Response Theory" in provenance.notes
    
    def test_compute_hash(self):
        """Test data fingerprint calculation."""
        p1 = DataProvenance.create_fallback("test")
        p2 = DataProvenance.create_fallback("test")
        
        # Same data should have same hash
        assert p1.compute_hash() == p2.compute_hash()
        
        # Different data should have different hash
        p3 = DataProvenance.create_fallback("other")
        assert p1.compute_hash() != p3.compute_hash()
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        provenance = DataProvenance.from_irt_calibration(
            case_id="test_001",
            calibration_version="v2026q1",
            sample_size=100,
            confidence=0.9
        )
        
        d = provenance.to_dict()
        assert d["source_type"] == "irt_calibration"
        assert d["sample_size"] == 100
        assert d["confidence"] == 0.9


class TestProvenanceTracker:
    """Test provenance tracker functionality."""
    
    def setup_method(self):
        """Reset tracker before each test."""
        reset_provenance_tracker()
    
    def test_register_and_get(self):
        """Test registration and retrieval."""
        tracker = get_provenance_tracker()
        
        provenance = DataProvenance.create_fallback("test")
        tracker.register("case_001", provenance)
        
        retrieved = tracker.get("case_001")
        assert retrieved is not None
        assert retrieved.source_id == provenance.source_id
    
    def test_validate_chain_valid(self):
        """Test validation of valid provenance."""
        tracker = get_provenance_tracker()
        
        valid_provenance = DataProvenance(
            source_type="irt_calibration",
            source_id="irt_v2026q1_test",
            collected_at="2026-04-11T00:00:00Z",
            sample_size=100,
            confidence=0.85,
            verified=True
        )
        
        tracker.register("valid_case", valid_provenance)
        result = tracker.validate_chain("valid_case")
        
        assert result["valid"]
        assert len(result["issues"]) == 0
    
    def test_validate_chain_low_confidence(self):
        """Test validation of low confidence provenance."""
        tracker = get_provenance_tracker()
        
        low_conf = DataProvenance(
            source_type="experiment",
            source_id="test_exp",
            collected_at="2026-04-11T00:00:00Z",
            sample_size=100,
            confidence=0.3,  # Low confidence
            verified=True
        )
        
        tracker.register("low_conf", low_conf)
        result = tracker.validate_chain("low_conf")
        
        assert not result["valid"]
        assert any("Low confidence" in issue for issue in result["issues"])
    
    def test_validate_chain_small_sample(self):
        """Test validation of small sample provenance."""
        tracker = get_provenance_tracker()
        
        small_sample = DataProvenance(
            source_type="experiment",
            source_id="test_exp",
            collected_at="2026-04-11T00:00:00Z",
            sample_size=10,  # Small sample
            confidence=0.8,
            verified=True
        )
        
        tracker.register("small_sample", small_sample)
        result = tracker.validate_chain("small_sample")
        
        assert not result["valid"]
        assert any("Small sample size" in issue for issue in result["issues"])
    
    def test_get_statistics(self):
        """Test statistics calculation."""
        tracker = get_provenance_tracker()
        
        # Add mixed provenance entries
        tracker.register("valid", DataProvenance.from_irt_calibration(
            "case1", "v2026q1", 100, 0.9
        ))
        tracker.register("fallback", DataProvenance.create_fallback("test"))
        
        stats = tracker.get_statistics()
        
        assert stats["total"] == 2
        assert stats["verified_count"] == 1
        assert stats["fallback_count"] == 1
        assert stats["average_confidence"] == 0.45  # (0.9 + 0.0) / 2


class TestReferenceDatabase:
    """Test reference database functionality."""
    
    def test_get_reference(self):
        """Test reference retrieval."""
        ref = ReferenceDatabase.get_reference("irt_2pl_model")
        
        assert ref is not None
        assert ref.title == "Item Response Theory for Psychologists"
        assert ref.year == 2000
        assert ref.reference_type == ReferenceType.BOOK
    
    def test_search_by_keyword(self):
        """Test keyword search."""
        results = ReferenceDatabase.search_by_keyword("hallucination")
        
        assert len(results) > 0
        assert any("Hallucination" in r.title for r in results)
    
    def test_get_formula_source(self):
        """Test formula source lookup."""
        ref = ReferenceDatabase.get_formula_source("irt_2pl")
        
        assert ref is not None
        assert "Embretson" in ref.authors
    
    def test_validate_citation(self):
        """Test citation validation."""
        result = ReferenceDatabase.validate_citation(
            "irt_2pl_model",
            "IRT 2PL model implementation"
        )
        
        assert result["valid"]
        assert "citation" in result
    
    def test_validate_citation_not_found(self):
        """Test validation of non-existent citation."""
        result = ReferenceDatabase.validate_citation(
            "non_existent_ref",
            "Some context"
        )
        
        assert not result["valid"]
        assert "not found" in result["error"]
    
    def test_format_citation_apa(self):
        """Test APA citation formatting."""
        ref = ReferenceDatabase.get_reference("irt_2pl_model")
        citation = ref.format_citation("apa")
        
        assert "Embretson" in citation
        assert "2000" in citation
    
    def test_get_all_references(self):
        """Test retrieving all references."""
        all_refs = ReferenceDatabase.get_all_references()
        
        assert len(all_refs) > 0
        assert "irt_2pl_model" in all_refs
        assert "hallucination_survey" in all_refs


class TestIRTParameters:
    """Test IRT parameter functionality."""
    
    def test_probability_correct(self):
        """Test probability calculation."""
        params = IRTParameters(
            case_id="test_001",
            a=1.0,  # Discrimination
            b=0.0,  # Difficulty
            c=0.25  # Guessing
        )
        
        # At theta = b, probability = (1 + c) / 2
        prob = params.probability_correct(0.0)
        assert abs(prob - 0.625) < 0.001  # (1 + 0.25) / 2 = 0.625
    
    def test_calculate_information(self):
        """Test Fisher information calculation."""
        params = IRTParameters(
            case_id="test_001",
            a=1.0,
            b=0.0,
            c=0.25
        )
        
        # Information should be maximum at theta = b
        info_at_b = params.calculate_information(0.0)
        info_at_offset = params.calculate_information(1.0)
        
        assert info_at_b > info_at_offset
    
    def test_validate_parameters(self):
        """Test parameter validation."""
        # Good parameters
        good_params = IRTParameters(
            case_id="good",
            a=1.2,
            b=0.5,
            fit_rmse=0.05,
            info_max=0.3
        )
        validation = good_params.validate()
        assert validation["valid"]
        
        # Bad parameters
        bad_params = IRTParameters(
            case_id="bad",
            a=0.3,  # Too low
            b=4.0,  # Too high
            fit_rmse=0.2,  # Too high
            info_max=0.05  # Too low
        )
        validation = bad_params.validate()
        assert not validation["valid"]
        assert len(validation["issues"]) >= 3


class TestIRTParameterDB:
    """Test IRT parameter database."""
    
    def setup_method(self):
        """Create temp directory for each test."""
        self.tmpdir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up temp directory."""
        import shutil
        try:
            shutil.rmtree(self.tmpdir, ignore_errors=True)
        except:
            pass
    
    def test_store_and_retrieve(self):
        """Test storing and retrieving parameters."""
        db_path = os.path.join(self.tmpdir, "test_irt.db")
        db = IRTParameterDB(db_path)
        
        params = IRTParameters(
            case_id="test_case",
            a=1.5,
            b=-0.5,
            c=0.25,
            fit_rmse=0.03,
            info_max=0.4
        )
        
        provenance = DataProvenance.from_irt_calibration(
            "test_case", "v2026q1", 1000, 0.9
        )
        
        # Store
        assert db.store_parameters(params, provenance)
        
        # Retrieve
        result = db.get_parameters("test_case")
        assert result is not None
        
        retrieved_params, retrieved_prov = result
        assert retrieved_params.a == 1.5
        assert retrieved_params.b == -0.5
        assert retrieved_prov.source_type == "irt_calibration"
    
    def test_select_optimal_items(self):
        """Test optimal item selection."""
        db_path = os.path.join(self.tmpdir, "test_irt.db")
        db = IRTParameterDB(db_path)
        
        # Store multiple items
        for i in range(5):
            params = IRTParameters(
                case_id=f"reason_{i:03d}",
                a=1.0,
                b=float(i - 2),  # Range from -2 to 2
            )
            provenance = DataProvenance.from_irt_calibration(
                f"reason_{i:03d}", "v2026q1", 500, 0.85
            )
            db.store_parameters(params, provenance)
        
        # Select items at theta = 0
        selected = db.select_optimal_items("reason", 0.0, n_items=3)
        
        assert len(selected) == 3
        # Items with b close to 0 should have highest information
        assert any("reason_002" in case_id for case_id, _ in selected)
    
    def test_calculate_dimension_information(self):
        """Test dimension information calculation."""
        db_path = os.path.join(self.tmpdir, "test_irt.db")
        db = IRTParameterDB(db_path)
        
        # Store items
        for i in range(3):
            params = IRTParameters(
                case_id=f"test_{i:03d}",
                a=1.0,
                b=0.0,
            )
            provenance = DataProvenance.from_irt_calibration(
                f"test_{i:03d}", "v2026q1", 500, 0.85
            )
            db.store_parameters(params, provenance)
        
        # Calculate information
        info = db.calculate_dimension_information("test", 0.0)
        
        assert info["item_count"] == 3
        assert info["total_information"] > 0
        assert info["standard_error"] > 0
        assert info["reliability"] > 0
    
    def test_export_import_json(self):
        """Test JSON export and import."""
        db_path = os.path.join(self.tmpdir, "test_irt.db")
        export_path = os.path.join(self.tmpdir, "export.json")
        
        db = IRTParameterDB(db_path)
        
        # Store test data
        params = IRTParameters(
            case_id="export_test",
            a=1.2,
            b=0.5,
        )
        provenance = DataProvenance.from_irt_calibration(
            "export_test", "v2026q1", 100, 0.9
        )
        db.store_parameters(params, provenance)
        
        # Export
        assert db.export_to_json(export_path)
        assert os.path.exists(export_path)
        
        # Create new database and import
        db2_path = os.path.join(self.tmpdir, "test_irt2.db")
        db2 = IRTParameterDB(db2_path)
        
        imported = db2.import_from_json(export_path)
        assert imported >= 1
        
        # Verify import
        result = db2.get_parameters("export_test")
        assert result is not None


class TestThetaScoreConverter:
    """Test theta score conversion."""
    
    def test_percent_to_theta(self):
        """Test percent to theta conversion."""
        # Create test parameters
        params = [
            IRTParameters(case_id=f"item_{i}", a=1.0, b=0.0)
            for i in range(5)
        ]
        
        # Test conversion runs without error and returns valid values
        theta_50, se_50 = ThetaScoreConverter.percent_to_theta(50.0, params)
        assert -4.0 <= theta_50 <= 4.0  # Theta in valid range
        assert se_50 >= 0  # Standard error is non-negative
        
        # Higher percent should generally give higher theta
        theta_75, se_75 = ThetaScoreConverter.percent_to_theta(75.0, params)
        theta_25, se_25 = ThetaScoreConverter.percent_to_theta(25.0, params)
        
        # 75% should generally be higher than 25%
        assert theta_75 >= theta_25
    
    def test_theta_to_percentile(self):
        """Test theta to percentile conversion."""
        # Theta = 0 should give 50th percentile
        p50 = ThetaScoreConverter.theta_to_percentile(0.0)
        assert abs(p50 - 50.0) < 1.0
        
        # Theta = 1 should give ~84th percentile
        p84 = ThetaScoreConverter.theta_to_percentile(1.0)
        assert 80 < p84 < 90
        
        # Theta = -1 should give ~16th percentile
        p16 = ThetaScoreConverter.theta_to_percentile(-1.0)
        assert 10 < p16 < 20


class TestTestCaseV8:
    """Test v8.0 TestCase with provenance."""
    
    def test_has_valid_provenance(self):
        """Test provenance validation."""
        case = TestCaseV8(
            id="test_001",
            category="reasoning",
            name="Test Case",
            user_prompt="Test prompt",
            expected_type="text",
            judge_method="exact_match",
        )
        
        # No provenance initially
        assert not case.has_valid_provenance
        
        # Add provenance
        case.weight_provenance = DataProvenance.from_irt_calibration(
            "test_001", "v2026q1", 500, 0.85
        )
        assert case.has_valid_provenance
    
    def test_get_weight_with_fallback(self):
        """Test weight retrieval with fallback."""
        case = TestCaseV8(
            id="test_001",
            category="reasoning",
            name="Test Case",
            user_prompt="Test prompt",
            expected_type="text",
            judge_method="exact_match",
            weight=2.5,
        )
        
        # Without provenance, should return fallback
        weight, prov = case.get_weight_with_fallback()
        assert weight == 1.0  # Fallback weight
        assert prov.source_type == "fallback"
        
        # With provenance, should return actual weight
        case.weight_provenance = DataProvenance.from_irt_calibration(
            "test_001", "v2026q1", 500, 0.85
        )
        weight, prov = case.get_weight_with_fallback()
        assert weight == 2.5
        assert prov.source_type == "irt_calibration"
    
    def test_has_irt_params(self):
        """Test IRT parameter detection."""
        case = TestCaseV8(
            id="test_001",
            category="reasoning",
            name="Test Case",
            user_prompt="Test prompt",
            expected_type="text",
            judge_method="exact_match",
        )
        
        assert not case.has_irt_params
        
        case.irt_a = 1.0
        case.irt_b = 0.0
        assert case.has_irt_params
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        case = TestCaseV8(
            id="test_001",
            category="reasoning",
            name="Test Case",
            user_prompt="Test prompt",
            expected_type="text",
            judge_method="exact_match",
            weight=2.0,
            irt_a=1.2,
            irt_b=-0.5,
        )
        case.weight_provenance = DataProvenance.from_irt_calibration(
            "test_001", "v2026q1", 500, 0.85
        )
        
        d = case.to_dict()
        assert d["id"] == "test_001"
        assert d["weight"] == 2.0
        assert d["has_provenance"]
        assert d["has_irt"]
        assert "irt_params" in d
        assert d["irt_params"]["a"] == 1.2


class TestIntegration:
    """Integration tests for Phase 1 components."""
    
    def setup_method(self):
        """Create temp directory for each test."""
        self.tmpdir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up temp directory."""
        import shutil
        try:
            shutil.rmtree(self.tmpdir, ignore_errors=True)
        except:
            pass
    
    def test_full_data_flow(self):
        """Test complete data flow from provenance to IRT."""
        # 1. Create provenance
        provenance = DataProvenance.from_irt_calibration(
            case_id="integration_test",
            calibration_version="v2026q1",
            sample_size=1500,
            confidence=0.87
        )
        
        # 2. Create IRT parameters
        params = IRTParameters(
            case_id="integration_test",
            a=1.3,
            b=0.2,
            c=0.25,
            fit_rmse=0.04,
            info_max=0.35
        )
        
        # 3. Store in database
        db_path = os.path.join(self.tmpdir, "integration.db")
        db = IRTParameterDB(db_path)
        assert db.store_parameters(params, provenance)
        
        # 4. Create test case
        case = TestCaseV8(
            id="integration_test",
            category="reasoning",
            name="Integration Test",
            user_prompt="Test",
            expected_type="text",
            judge_method="exact_match",
            weight=2.5,
            irt_a=1.3,
            irt_b=0.2,
            weight_provenance=provenance
        )
        
        # 5. Verify integration
        assert case.has_valid_provenance
        assert case.has_irt_params
        
        weight, prov = case.get_weight_with_fallback()
        assert weight == 2.5
        assert prov.source_type == "irt_calibration"
        
        # 6. Retrieve from database and verify
        result = db.get_parameters("integration_test")
        assert result is not None
        
        retrieved_params, retrieved_prov = result
        assert retrieved_params.a == 1.3
        assert retrieved_prov.sample_size == 1500
        
        # 7. Validate provenance chain
        tracker = get_provenance_tracker()
        tracker.register("integration_test", retrieved_prov)
        validation = tracker.validate_chain("integration_test")
        assert validation["valid"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
