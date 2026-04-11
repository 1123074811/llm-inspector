"""
Phase 4 Validation Script - Optimization & Integration

Validates:
- Frontend visualization components
- Token optimization configuration
- Performance benchmark framework
- Documentation completeness

Run: python -m backend.scripts.validate_phase4
"""
from __future__ import annotations

import sys
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add backend to path
backend_root = Path(__file__).parent.parent
sys.path.insert(0, str(backend_root))


class Phase4Validator:
    """Validates Phase 4 optimization & integration components."""
    
    def __init__(self):
        self.results: Dict[str, Any] = {}
        self.passed = 0
        self.failed = 0
        self.project_root = Path(__file__).parent.parent.parent
    
    def log(self, message: str, level: str = "info"):
        """Log validation message."""
        prefix = {"info": "[INFO]", "pass": "[PASS]", "fail": "[FAIL]", 
                  "warn": "[WARN]"}.get(level, "[INFO]")
        print(f"{prefix} {message}")
    
    def test_frontend_visualization(self) -> bool:
        """Test frontend visualization module exists and is valid."""
        self.log("Testing Frontend Visualization Module...")
        
        try:
            viz_file = self.project_root / "frontend" / "v7_visualization.js"
            
            if not viz_file.exists():
                self.log("v7_visualization.js not found", "fail")
                return False
            
            # Read and validate content
            content = viz_file.read_text(encoding='utf-8')
            
            # Check for required functions
            required_functions = [
                'renderIRTParameters',
                'renderSimilarityWithCI',
                'renderScoreTrace',
                'TestProgressMonitor',
                'renderRealtimeProgress',
                'calculateICC',
                'calculateInformation'
            ]
            
            missing = []
            for func in required_functions:
                if func not in content:
                    missing.append(func)
            
            if missing:
                self.log(f"Missing functions: {', '.join(missing)}", "fail")
                return False
            
            # Check for CSS styles
            if 'V7_STYLES' not in content:
                self.log("Missing V7_STYLES CSS export", "warn")
            
            # Count lines
            lines = len(content.split('\n'))
            self.log(f"v7_visualization.js: {lines} lines")
            
            self.log("Frontend visualization module valid", "pass")
            self.passed += 1
            return True
            
        except Exception as e:
            self.log(f"Frontend visualization test failed: {e}", "fail")
            self.failed += 1
            return False
    
    def test_token_optimization(self) -> bool:
        """Test token optimization configuration."""
        self.log("Testing Token Optimization Config...")
        
        try:
            config_file = self.project_root / "backend" / "app" / "config" / "prompt_compression.yaml"
            
            if not config_file.exists():
                self.log("prompt_compression.yaml not found", "fail")
                return False
            
            # Parse YAML
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Validate structure
            required_keys = ['version', 'templates', 'rules', 'dimension_targets']
            for key in required_keys:
                if key not in config:
                    self.log(f"Missing config key: {key}", "fail")
                    return False
            
            # Check version
            if config.get('version') != '7.0':
                self.log(f"Unexpected version: {config.get('version')}", "warn")
            
            # Check templates
            templates = config.get('templates', {})
            if 'system' not in templates or 'user' not in templates:
                self.log("Missing template sections", "fail")
                return False
            
            # Check dimension targets
            targets = config.get('dimension_targets', {})
            expected_dims = ['reasoning', 'coding', 'instruction', 'knowledge', 'safety']
            for dim in expected_dims:
                if dim not in targets:
                    self.log(f"Missing dimension target: {dim}", "warn")
            
            # Check compression targets
            for dim, settings in targets.items():
                if 'system_prompt_max' not in settings:
                    self.log(f"Missing system_prompt_max for {dim}", "warn")
            
            self.log(f"Config valid: {len(templates)} template sections, {len(targets)} dimensions")
            self.log("Token optimization config valid", "pass")
            self.passed += 1
            return True
            
        except Exception as e:
            self.log(f"Token optimization test failed: {e}", "fail")
            self.failed += 1
            return False
    
    def test_benchmark_framework(self) -> bool:
        """Test performance benchmark framework."""
        self.log("Testing Benchmark Framework...")
        
        try:
            from app.core.benchmark import (
                PerformanceBenchmark,
                BenchmarkResult,
                BenchmarkStats,
                DetectionBenchmarks,
                TokenEfficiencyBenchmarks,
                benchmark_timer
            )
            
            # Test basic benchmark
            benchmark = PerformanceBenchmark()
            
            # Test simple function benchmark
            def test_func():
                return {"result": "success", "tokens_used": 100}
            
            stats = benchmark.run_benchmark('test', test_func, n_runs=3)
            
            if stats.n_runs != 3:
                self.log(f"Wrong run count: {stats.n_runs}", "fail")
                return False
            
            if stats.mean_duration_ms < 0:
                self.log("Invalid duration", "fail")
                return False
            
            self.log(f"Benchmark stats: {stats.mean_duration_ms:.2f}ms, {stats.success_rate*100:.0f}% success")
            
            # Test DetectionBenchmarks class
            detection = DetectionBenchmarks()
            if not hasattr(detection, 'benchmark'):
                self.log("DetectionBenchmarks missing benchmark attribute", "warn")
            
            # Test TokenEfficiencyBenchmarks
            token_bench = TokenEfficiencyBenchmarks()
            test_prompts = {
                "short": "Hello world",
                "medium": "This is a test prompt with more content",
            }
            results = token_bench.run_compression_benchmark(test_prompts)
            
            if len(results) != len(test_prompts):
                self.log("Wrong number of compression results", "fail")
                return False
            
            self.log(f"Compression benchmark: {len(results)} prompts analyzed")
            self.log("Benchmark framework valid", "pass")
            self.passed += 1
            return True
            
        except Exception as e:
            self.log(f"Benchmark framework test failed: {e}", "fail")
            self.failed += 1
            return False
    
    def test_documentation(self) -> bool:
        """Test documentation completeness."""
        self.log("Testing Documentation...")
        
        try:
            docs = []
            
            # Check for upgrade plan
            upgrade_plan = self.project_root / "V7_UPGRADE_PLAN.md"
            if upgrade_plan.exists():
                docs.append("V7_UPGRADE_PLAN.md")
            
            # Check for phase 3 completion doc
            phase3_doc = self.project_root / "V7_PHASE3_COMPLETE.md"
            if phase3_doc.exists():
                docs.append("V7_PHASE3_COMPLETE.md")
            
            # Check README
            readme = self.project_root / "README.md"
            if readme.exists():
                try:
                    content = readme.read_text(encoding='utf-8')
                    if 'v7' in content.lower() or '7.0' in content:
                        docs.append("README.md (v7 mentioned)")
                except UnicodeDecodeError:
                    docs.append("README.md (encoding issue)")
            
            self.log(f"Documentation files: {', '.join(docs)}")
            
            # Check for inline documentation
            viz_file = self.project_root / "frontend" / "v7_visualization.js"
            if viz_file.exists():
                try:
                    content = viz_file.read_text(encoding='utf-8')
                    doc_comments = content.count('/**')
                    self.log(f"JSDoc comments in v7_visualization.js: {doc_comments}")
                except UnicodeDecodeError:
                    self.log("JSDoc check skipped (encoding issue)")
            
            self.log("Documentation check complete", "pass")
            self.passed += 1
            return True
            
        except Exception as e:
            self.log(f"Documentation test failed: {e}", "fail")
            self.failed += 1
            return False
    
    def test_integration(self) -> bool:
        """Test overall integration."""
        self.log("Testing Overall Integration...")
        
        try:
            # Check all new files exist
            required_files = [
                "frontend/v7_visualization.js",
                "backend/app/config/prompt_compression.yaml",
                "backend/app/core/benchmark.py",
                "backend/app/predetect/semantic_fingerprint.py",
                "backend/app/predetect/extraction_v2.py",
                "backend/app/predetect/differential_testing.py",
                "backend/app/predetect/tool_capability.py",
                "backend/app/predetect/adversarial_analysis.py",
            ]
            
            missing = []
            for file_path in required_files:
                full_path = self.project_root / file_path
                if not full_path.exists():
                    missing.append(file_path)
            
            if missing:
                self.log(f"Missing files: {', '.join(missing)}", "fail")
                return False
            
            self.log(f"All {len(required_files)} required files present")
            
            # Check pipeline imports
            from app.predetect.pipeline import PreDetectionPipeline
            from app.predetect.semantic_fingerprint import Layer8SemanticFingerprint
            from app.predetect.extraction_v2 import Layer7AdvancedExtraction
            from app.predetect.differential_testing import Layer8DifferentialTesting
            from app.predetect.tool_capability import Layer9ToolCapability
            from app.predetect.adversarial_analysis import Layer11AdversarialAnalysis
            
            self.log("All pipeline imports successful")
            
            # Count total lines of new code
            total_lines = 0
            code_files = [
                "backend/app/predetect/semantic_fingerprint.py",
                "backend/app/predetect/extraction_v2.py",
                "backend/app/predetect/differential_testing.py",
                "backend/app/predetect/tool_capability.py",
                "backend/app/predetect/adversarial_analysis.py",
                "backend/app/core/benchmark.py",
                "frontend/v7_visualization.js",
            ]
            
            for file_path in code_files:
                full_path = self.project_root / file_path
                if full_path.exists():
                    lines = len(full_path.read_text(encoding='utf-8').split('\n'))
                    total_lines += lines
            
            self.log(f"Total new code: ~{total_lines} lines")
            
            self.log("Integration test complete", "pass")
            self.passed += 1
            return True
            
        except Exception as e:
            self.log(f"Integration test failed: {e}", "fail")
            self.failed += 1
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all Phase 4 validation tests."""
        print("=" * 60)
        print("Phase 4 Validation: Optimization & Integration")
        print("=" * 60)
        print()
        
        tests = [
            ("Frontend Visualization", self.test_frontend_visualization),
            ("Token Optimization", self.test_token_optimization),
            ("Benchmark Framework", self.test_benchmark_framework),
            ("Documentation", self.test_documentation),
            ("Integration", self.test_integration),
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
        print("=" * 60)
        
        return {
            "total": self.passed + self.failed,
            "passed": self.passed,
            "failed": self.failed,
            "success_rate": self.passed / (self.passed + self.failed) if (self.passed + self.failed) > 0 else 0,
            "tests": results,
        }


def main():
    """Main entry point."""
    validator = Phase4Validator()
    results = validator.run_all_tests()
    
    # Write results to file
    output_file = Path(__file__).parent / "phase4_validation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Exit with appropriate code
    sys.exit(0 if results["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
