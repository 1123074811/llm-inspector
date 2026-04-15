import unittest
from datetime import datetime
from app.validation.lineage_guard import LineageGuard
from app.core.provenance import DataProvenance, get_provenance_tracker, reset_provenance_tracker

class TestLineageGuard(unittest.TestCase):
    def setUp(self):
        reset_provenance_tracker()
        self.tracker = get_provenance_tracker()
        self.guard = LineageGuard(strict_mode=True)

    def test_missing_provenance_blocks_report(self):
        report = {
            "reasoning_score": 85.0,
            "coding_score": 90.0
        }
        # No provenance registered in tracker
        results = self.guard.validate_report_data(report)
        self.assertFalse(results["is_valid"])
        self.assertIn("Metric reasoning_score blocked: Missing provenance record", results["issues"])

    def test_fallback_provenance_is_insufficient(self):
        # Register fallback
        p = DataProvenance.create_fallback("reasoning_score", "No data")
        self.tracker.register("reasoning_score", p)
        
        report = {"reasoning_score": 50.0}
        results = self.guard.validate_report_data(report)
        
        # In strict mode, fallback might not block the whole report but should be flagged
        # The current implementation flags it as 'insufficient_evidence'
        self.assertEqual(results["annotations"]["reasoning_score"]["status"], "insufficient_evidence")

    def test_valid_provenance(self):
        p = DataProvenance(
            source_type="irt_calibration",
            source_id="irt_v12_reasoning",
            collected_at=datetime.utcnow().isoformat(),
            sample_size=100,
            confidence=0.9,
            verified=True
        )
        self.tracker.register("reasoning_score", p)
        
        report = {"reasoning_score": 85.0}
        results = self.guard.validate_report_data(report)
        
        self.assertTrue(results["is_valid"])
        self.assertEqual(results["annotations"]["reasoning_score"]["status"], "measured")
        self.assertTrue(results["annotations"]["reasoning_score"]["lineage_valid"])

    def test_insufficient_sample_size(self):
        p = DataProvenance(
            source_type="irt_calibration",
            source_id="irt_v12_reasoning",
            collected_at=datetime.utcnow().isoformat(),
            sample_size=10, # Below MIN_SAMPLE_SIZE (30)
            confidence=0.9,
            verified=True
        )
        self.tracker.register("reasoning_score", p)
        
        report = {"reasoning_score": 85.0}
        results = self.guard.validate_report_data(report)
        
        self.assertEqual(results["annotations"]["reasoning_score"]["status"], "insufficient_evidence")
        self.assertTrue(results["annotations"]["reasoning_score"]["lineage_valid"])

if __name__ == "__main__":
    unittest.main()
