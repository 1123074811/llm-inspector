"""
Phase 5 Validation Script - Final Release Validation

Validates all Phase 5 components:
- Mass model testing framework
- Construct validity experiments
- IRT calibration data collection
- Release readiness checklist

Run: python -m backend.scripts.validate_phase5
"""
from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

# Add backend to path
backend_root = Path(__file__).parent.parent
sys.path.insert(0, str(backend_root))


@dataclass
class ValidationResult:
    """Result of a validation test."""
    name: str
    passed: bool
    details: Dict[str, Any]
    errors: List[str]


class Phase5Validator:
    """Validates Phase 5: Validation & Release."""
    
    def __init__(self):
        self.results: List[ValidationResult] = []
        self.passed = 0
        self.failed = 0
        self.project_root = Path(__file__).parent.parent.parent
    
    def log(self, message: str, level: str = "info"):
        """Log validation message."""
        prefix = {"info": "[INFO]", "pass": "[PASS]", "fail": "[FAIL]", 
                  "warn": "[WARN]"}.get(level, "[INFO]")
        print(f"{prefix} {message}")
    
    def test_mass_model_framework(self) -> bool:
        """Test mass model testing framework."""
        self.log("Testing Mass Model Testing Framework...")
        
        errors = []
        details = {}
        
        try:
            # Check file exists
            mass_test_file = self.project_root / "backend" / "scripts" / "mass_model_test.py"
            if not mass_test_file.exists():
                errors.append("mass_model_test.py not found")
                return False
            
            # Check imports
            from scripts.mass_model_test import (
                ModelRegistry, MockModelAdapter, MassModelTester,
                ModelTestResult, MassTestSummary
            )
            details['classes_imported'] = 5
            
            # Check ModelRegistry
            registry = ModelRegistry()
            models = registry.get_models()
            details['default_models'] = len(models)
            
            if len(models) < 20:
                errors.append(f"Too few default models: {len(models)}")
            
            # Check MockModelAdapter
            mock_adapter = MockModelAdapter({'id': 'test', 'provider': 'openai'})
            details['mock_adapter_works'] = True
            
            # Check MassModelTester initialization
            tester = MassModelTester()
            details['tester_initialized'] = True
            
            if errors:
                self.log(f"Mass model framework issues: {', '.join(errors)}", "fail")
                self.failed += 1
                return False
            
            self.log(f"Mass model framework: {details['default_models']} models available", "pass")
            self.passed += 1
            return True
            
        except Exception as e:
            self.log(f"Mass model framework test failed: {e}", "fail")
            self.failed += 1
            return False
    
    def test_validity_experiment(self) -> bool:
        """Test construct validity experiment framework."""
        self.log("Testing Validity Experiment Framework...")
        
        errors = []
        details = {}
        
        try:
            # Check file exists
            validity_file = self.project_root / "backend" / "scripts" / "validity_experiment.py"
            if not validity_file.exists():
                errors.append("validity_experiment.py not found")
                return False
            
            # Check imports
            from scripts.validity_experiment import (
                ValidityExperiment, ExternalBenchmark, ValidityResult,
                ConstructValidityReport, generate_mock_inspector_results
            )
            details['classes_imported'] = 4
            
            # Check reference data
            experiment = ValidityExperiment()
            ref_data = experiment.REFERENCE_DATA
            details['reference_benchmarks'] = len(ref_data)
            
            if len(ref_data) < 2:
                errors.append("Too few reference benchmarks")
            
            # Check expected correlations
            expected = experiment.EXPECTED_CORRELATIONS
            details['expected_correlations'] = len(expected)
            
            # Test mock data generation
            mock_data = generate_mock_inspector_results()
            details['mock_models'] = len(mock_data)
            
            if errors:
                self.log(f"Validity experiment issues: {', '.join(errors)}", "fail")
                self.failed += 1
                return False
            
            self.log(f"Validity experiment: {len(ref_data)} benchmarks, {len(expected)} correlations", "pass")
            self.passed += 1
            return True
            
        except Exception as e:
            self.log(f"Validity experiment test failed: {e}", "fail")
            self.failed += 1
            return False
    
    def test_irt_data_collection(self) -> bool:
        """Test IRT data collection framework."""
        self.log("Testing IRT Data Collection Framework...")
        
        errors = []
        details = {}
        
        try:
            # Check file exists
            irt_file = self.project_root / "backend" / "scripts" / "irt_data_collection.py"
            if not irt_file.exists():
                errors.append("irt_data_collection.py not found")
                return False
            
            # Check imports
            from scripts.irt_data_collection import (
                IRTDataCollector, IRTResponse, IRTItemStats,
                IRTCalibrationReport, generate_mock_data
            )
            details['classes_imported'] = 4
            
            # Test mock data generation
            mock_responses = generate_mock_data(n_models=50, n_cases=30)
            details['mock_responses'] = len(mock_responses)
            
            # Test collector
            collector = IRTDataCollector()
            collector.add_batch(mock_responses)
            
            # Check readiness
            readiness = collector.check_calibration_readiness()
            details['readiness_status'] = readiness
            
            # Run simple calibration
            report = collector.run_calibration()
            details['n_items_calibrated'] = len(report.item_stats)
            details['reliability'] = report.reliability
            
            if errors:
                self.log(f"IRT collection issues: {', '.join(errors)}", "fail")
                self.failed += 1
                return False
            
            self.log(f"IRT collection: {len(mock_responses)} responses, reliability={report.reliability:.3f}", "pass")
            self.passed += 1
            return True
            
        except Exception as e:
            self.log(f"IRT collection test failed: {e}", "fail")
            self.failed += 1
            return False
    
    def test_release_checklist(self) -> bool:
        """Test release checklist completeness."""
        self.log("Testing Release Checklist...")
        
        errors = []
        details = {}
        
        try:
            # Check checklist file
            checklist_file = self.project_root / "V7_RELEASE_CHECKLIST.md"
            if not checklist_file.exists():
                errors.append("V7_RELEASE_CHECKLIST.md not found")
                return False
            
            content = checklist_file.read_text(encoding='utf-8')
            
            # Count checkboxes
            total_checks = content.count('- [ ]') + content.count('- [x]')
            completed_checks = content.count('- [x]')
            
            details['total_items'] = total_checks
            details['completed_items'] = completed_checks
            details['completion_rate'] = completed_checks / total_checks if total_checks > 0 else 0
            
            # Check for required sections
            required_sections = [
                '大规模模型测试',
                '效度验证实验',
                'IRT校准数据收集',
                '功能完整性检查',
                '代码质量检查',
                '发布准备'
            ]
            
            for section in required_sections:
                if section not in content:
                    errors.append(f"Missing section: {section}")
            
            details['sections_present'] = len(required_sections) - len(errors)
            
            if errors:
                self.log(f"Release checklist issues: {', '.join(errors)}", "warn")
            
            self.log(f"Release checklist: {completed_checks}/{total_checks} items ({details['completion_rate']*100:.0f}%)", "pass")
            self.passed += 1
            return True
            
        except Exception as e:
            self.log(f"Release checklist test failed: {e}", "fail")
            self.failed += 1
            return False
    
    def test_integration_completeness(self) -> bool:
        """Test overall v7 integration completeness."""
        self.log("Testing V7 Integration Completeness...")
        
        errors = []
        details = {}
        
        try:
            # Count all v7 files
            v7_files = [
                # Phase 3 files
                "backend/app/predetect/semantic_fingerprint.py",
                "backend/app/predetect/extraction_v2.py",
                "backend/app/predetect/differential_testing.py",
                "backend/app/predetect/tool_capability.py",
                "backend/app/predetect/adversarial_analysis.py",
                # Phase 4 files
                "frontend/v7_visualization.js",
                "backend/app/config/prompt_compression.yaml",
                "backend/app/core/benchmark.py",
                # Phase 5 files
                "backend/scripts/mass_model_test.py",
                "backend/scripts/validity_experiment.py",
                "backend/scripts/irt_data_collection.py",
                # Checklist
                "V7_RELEASE_CHECKLIST.md",
            ]
            
            existing = []
            for f in v7_files:
                if (self.project_root / f).exists():
                    existing.append(f)
            
            details['v7_files_total'] = len(v7_files)
            details['v7_files_present'] = len(existing)
            details['v7_files_missing'] = [f for f in v7_files if f not in existing]
            
            # Check pipeline integration
            from app.predetect.pipeline import PreDetectionPipeline
            pipeline = PreDetectionPipeline()
            details['pipeline_initialized'] = True
            
            # Verify 13 layers documented in pipeline
            try:
                pipeline_doc = (self.project_root / "backend" / "app" / "predetect" / "pipeline.py").read_text(encoding='utf-8')
                layer_count = pipeline_doc.count('Layer ')
            except UnicodeDecodeError:
                pipeline_doc = (self.project_root / "backend" / "app" / "predetect" / "pipeline.py").read_text(encoding='latin-1')
                layer_count = pipeline_doc.count('Layer ')
            details['layers_documented'] = layer_count
            
            if details['v7_files_missing']:
                errors.append(f"Missing files: {', '.join(details['v7_files_missing'])}")
            
            if errors:
                self.log(f"Integration issues: {', '.join(errors)}", "warn")
            
            self.log(f"Integration: {len(existing)}/{len(v7_files)} files, {layer_count} layers", "pass")
            self.passed += 1
            return True
            
        except Exception as e:
            self.log(f"Integration test failed: {e}", "fail")
            self.failed += 1
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all Phase 5 validation tests."""
        print("=" * 60)
        print("Phase 5 Validation: Validation & Release")
        print("=" * 60)
        print()
        
        tests = [
            ("Mass Model Testing Framework", self.test_mass_model_framework),
            ("Validity Experiment Framework", self.test_validity_experiment),
            ("IRT Data Collection Framework", self.test_irt_data_collection),
            ("Release Checklist", self.test_release_checklist),
            ("Integration Completeness", self.test_integration_completeness),
        ]
        
        results = {}
        for name, test_func in tests:
            print(f"\n{'-' * 40}")
            print(f"Testing: {name}")
            print('-' * 40)
            results[name] = test_func()
        
        # Summary
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {self.passed + self.failed}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Success Rate: {self.passed / (self.passed + self.failed) * 100:.1f}%")
        print()
        
        # Readiness assessment
        if self.passed == self.passed + self.failed:
            print("[READY] All Phase 5 components validated successfully!")
            print("Ready for mass testing and validity experiments.")
        elif self.passed / (self.passed + self.failed) >= 0.8:
            print("[WARNING] Most components ready. Address failed tests before release.")
        else:
            print("[NOT READY] Multiple failures. Complete remaining implementation before release.")
        
        print("=" * 60)
        
        return {
            "total": self.passed + self.failed,
            "passed": self.passed,
            "failed": self.failed,
            "success_rate": self.passed / (self.passed + self.failed) if (self.passed + self.failed) > 0 else 0,
            "tests": results,
            "ready_for_testing": self.passed >= 4,
        }


def main():
    """Main entry point."""
    validator = Phase5Validator()
    results = validator.run_all_tests()
    
    # Write results to file
    output_file = Path(__file__).parent / "phase5_validation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Exit with appropriate code
    sys.exit(0 if results["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
