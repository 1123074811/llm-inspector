"""
Phase 3 Validation Script - Detection Enhancement

Validates the 12-layer detection pipeline and new detection modules:
- Semantic Fingerprinting (Layer 8)
- Advanced Extraction v2 (Layer 9)
- Differential Consistency Testing (Layer 10)
- Tool Use Capability Probe (Layer 11)
- Adversarial Response Analysis (Layer 13)

Run: python -m backend.scripts.validate_phase3
"""
from __future__ import annotations

import sys
import json
from pathlib import Path
from dataclasses import asdict
from typing import Dict, List, Any

# Add backend to path for imports
backend_root = Path(__file__).parent.parent
sys.path.insert(0, str(backend_root))

from app.predetect.semantic_fingerprint import (
    SemanticFingerprinter, 
    SemanticFingerprint,
    Layer8SemanticFingerprint
)
from app.predetect.extraction_v2 import (
    AdvancedExtractionAttacks,
    Layer7AdvancedExtraction
)
from app.predetect.differential_testing import (
    DifferentialConsistencyTester,
    Layer8DifferentialTesting
)
from app.predetect.tool_capability import (
    ToolCapabilityProbe,
    Layer9ToolCapability
)
from app.predetect.adversarial_analysis import (
    AdversarialAnalyzer,
    Layer11AdversarialAnalysis
)
from app.core.schemas import LayerResult


class MockAdapter:
    """Mock adapter for testing without actual API calls."""
    
    def __init__(self, model_family: str = "openai"):
        self.model_family = model_family
        self.call_count = 0
    
    def chat(self, request):
        """Mock chat response."""
        self.call_count += 1
        
        # Simple mock response based on model family
        responses = {
            "openai": f"I'm GPT-4, an AI assistant by OpenAI. Here's my response to: {request.messages[0].content[:30]}...",
            "claude": f"I'm Claude, created by Anthropic. Here's my response to: {request.messages[0].content[:30]}...",
            "gemini": f"I'm Gemini, from Google. Here's my response to: {request.messages[0].content[:30]}...",
            "wrapper": f"I am a helpful assistant. [Filtered response to: {request.messages[0].content[:30]}...]",
        }
        
        # Create a mock response object
        class MockResponse:
            def __init__(self, content, tokens=50):
                self.content = content
                self.usage_total_tokens = tokens
                self.usage_prompt_tokens = tokens // 2
                self.raw_json = {"model": "mock-model"}
                self.ok = True
                self.error_type = None
                self.error_message = None
                self.status_code = 200
                self.latency_ms = 100
                self.logprobs = None
        
        content = responses.get(self.model_family, responses["openai"])
        return MockResponse(content, tokens=50 + len(content) // 4)
    
    def head_request(self):
        return {"headers": {}}
    
    def bad_request(self):
        return {"status_code": 400, "body": {}}
    
    def list_models(self):
        return {"status_code": 200, "body": {"data": []}}


class Phase3Validator:
    """Validates Phase 3 detection enhancement components."""
    
    def __init__(self):
        self.results: Dict[str, Any] = {}
        self.passed = 0
        self.failed = 0
    
    def log(self, message: str, level: str = "info"):
        """Log validation message."""
        prefix = {"info": "[INFO]", "pass": "[PASS]", "fail": "[FAIL]", 
                  "warn": "[WARN]"}.get(level, "[INFO]")
        print(f"{prefix} {message}")
    
    def test_semantic_fingerprint(self) -> bool:
        """Test semantic fingerprinting module."""
        self.log("Testing Semantic Fingerprinting...")
        
        try:
            fingerprinter = SemanticFingerprinter()
            
            # Test embedding generation
            text = "This is a test sentence for embedding."
            embedding = fingerprinter._simple_hash_embedding(text)
            
            if len(embedding) != 128:
                self.log(f"Embedding dimension wrong: {len(embedding)} != 128", "fail")
                return False
            
            # Test normalization
            norm = sum(e**2 for e in embedding) ** 0.5
            if abs(norm - 1.0) > 0.01 and norm > 0:
                self.log(f"Embedding not normalized: norm={norm:.3f}", "warn")
            
            # Test fingerprint generation with mock adapter
            mock_adapter = MockAdapter("openai")
            fp = fingerprinter.generate_fingerprint(mock_adapter, "test-model", n_samples=3)
            
            if fp.n_samples == 0:
                self.log("No samples collected in fingerprint", "fail")
                return False
            
            if not fp.fingerprint_id:
                self.log("Missing fingerprint ID", "fail")
                return False
            
            # Test fingerprint comparison
            fp2 = fingerprinter.generate_fingerprint(mock_adapter, "test-model-2", n_samples=3)
            similarity = fingerprinter.compare_fingerprints(fp, fp2)
            
            if not (0 <= similarity <= 1):
                self.log(f"Similarity out of range: {similarity}", "fail")
                return False
            
            # Test Layer8
            layer = Layer8SemanticFingerprint()
            result = layer.run(mock_adapter, "test-model")
            
            if not isinstance(result, LayerResult):
                self.log("Layer8 did not return LayerResult", "fail")
                return False
            
            self.log(f"Layer8: confidence={result.confidence:.2f}, tokens={result.tokens_used}")
            self.log("Semantic Fingerprinting: PASSED", "pass")
            self.passed += 1
            return True
            
        except Exception as e:
            self.log(f"Semantic Fingerprinting failed: {e}", "fail")
            self.failed += 1
            return False
    
    def test_advanced_extraction(self) -> bool:
        """Test advanced extraction v2 module."""
        self.log("Testing Advanced Extraction v2...")
        
        try:
            attacks = AdvancedExtractionAttacks()
            
            # Check attack templates
            if len(attacks.ATTACK_TEMPLATES) < 5:
                self.log(f"Too few attack templates: {len(attacks.ATTACK_TEMPLATES)}", "fail")
                return False
            
            # Test base64 payload
            b64_payload = attacks._B64_PAYLOADS.get("ignore_system")
            if not b64_payload:
                self.log("Missing base64 payloads", "fail")
                return False
            
            # Test attack execution with mock adapter
            mock_adapter = MockAdapter("wrapper")
            result = attacks.execute_attack(
                mock_adapter, 
                "claimed-model", 
                "few_shot_disclosure",
                run_id="test-123"
            )
            
            if not result.attack_type:
                self.log("Missing attack type in result", "fail")
                return False
            
            self.log(f"Attack result: success={result.success}, confidence={result.confidence:.2f}")
            
            # Test Layer7 (Advanced Extraction v2)
            layer = Layer7AdvancedExtraction()
            result = layer.run(mock_adapter, "claimed-model", run_id="test-123")
            
            if not isinstance(result, LayerResult):
                self.log("Layer9 did not return LayerResult", "fail")
                return False
            
            self.log(f"Layer9: confidence={result.confidence:.2f}, evidence={len(result.evidence)}")
            self.log("Advanced Extraction v2: PASSED", "pass")
            self.passed += 1
            return True
            
        except Exception as e:
            self.log(f"Advanced Extraction v2 failed: {e}", "fail")
            self.failed += 1
            return False
    
    def test_differential_testing(self) -> bool:
        """Test differential consistency testing module."""
        self.log("Testing Differential Consistency Testing...")
        
        try:
            tester = DifferentialConsistencyTester()
            
            # Check isomorphic pairs
            if len(tester.ISOMORPHIC_PAIRS) < 3:
                self.log(f"Too few isomorphic pairs: {len(tester.ISOMORPHIC_PAIRS)}", "fail")
                return False
            
            # Test embedding
            emb1 = tester._simple_hash_embedding("Test sentence one.")
            emb2 = tester._simple_hash_embedding("Test sentence two.")
            
            if len(emb1) != 128 or len(emb2) != 128:
                self.log("Embedding dimension incorrect", "fail")
                return False
            
            # Test similarity calculation
            sim = tester._calculate_similarity("Hello world", "Hello world")
            if sim < 0.9:  # Same text should have high similarity
                self.log(f"Self-similarity too low: {sim}", "fail")
                return False
            
            # Test with mock adapter
            mock_adapter = MockAdapter("openai")
            report = tester.test_consistency(mock_adapter, "test-model", n_rounds=3)
            
            if report.isomorphic_pairs_tested == 0:
                self.log("No pairs tested", "fail")
                return False
            
            if not (0 <= report.mean_consistency <= 1):
                self.log(f"Mean consistency out of range: {report.mean_consistency}", "fail")
                return False
            
            self.log(f"Consistency report: mean={report.mean_consistency:.3f}, routing={report.routing_suspected}")
            
            # Test Layer8 (Differential Testing)
            layer = Layer8DifferentialTesting()
            result = layer.run(mock_adapter, "test-model")
            
            if not isinstance(result, LayerResult):
                self.log("Layer10 did not return LayerResult", "fail")
                return False
            
            self.log(f"Layer10: confidence={result.confidence:.2f}")
            self.log("Differential Consistency Testing: PASSED", "pass")
            self.passed += 1
            return True
            
        except Exception as e:
            self.log(f"Differential Consistency Testing failed: {e}", "fail")
            self.failed += 1
            return False
    
    def test_tool_capability(self) -> bool:
        """Test tool capability probe module."""
        self.log("Testing Tool Capability Probe...")
        
        try:
            probe = ToolCapabilityProbe()
            
            # Check test tools
            if len(probe.TEST_TOOLS) < 2:
                self.log(f"Too few test tools: {len(probe.TEST_TOOLS)}", "fail")
                return False
            
            # Check tool patterns
            if "openai" not in probe.TOOL_PATTERNS:
                self.log("Missing OpenAI tool patterns", "fail")
                return False
            
            # Test with mock adapter (won't detect tools in mock)
            mock_adapter = MockAdapter("openai")
            result = probe.probe_tool_capability(mock_adapter, "test-model")
            
            if result.confidence < 0:
                self.log("Negative confidence", "fail")
                return False
            
            self.log(f"Tool capability: supports={result.supports_tools}, format={result.tool_format}")
            
            # Test Layer9 (Tool Capability)
            layer = Layer9ToolCapability()
            result = layer.run(mock_adapter, "test-model")
            
            if not isinstance(result, LayerResult):
                self.log("Layer11 did not return LayerResult", "fail")
                return False
            
            self.log(f"Layer11: confidence={result.confidence:.2f}")
            self.log("Tool Capability Probe: PASSED", "pass")
            self.passed += 1
            return True
            
        except Exception as e:
            self.log(f"Tool Capability Probe failed: {e}", "fail")
            self.failed += 1
            return False
    
    def test_adversarial_analysis(self) -> bool:
        """Test adversarial response analysis module."""
        self.log("Testing Adversarial Response Analysis...")
        
        try:
            analyzer = AdversarialAnalyzer()
            
            # Check adversarial prompts
            if len(analyzer.ADVERSARIAL_PROMPTS) < 3:
                self.log(f"Too few adversarial prompts: {len(analyzer.ADVERSARIAL_PROMPTS)}", "fail")
                return False
            
            # Test response analysis
            test_response = "I cannot assist with that request. It violates my safety guidelines."
            analysis = analyzer.analyze_response("boundary_test", test_response)
            
            if not analysis.refusal_detected:
                self.log("Failed to detect refusal", "warn")
            
            self.log(f"Adversarial analysis: refusal={analysis.refusal_detected}, safety={analysis.safety_triggered}")
            
            # Test with mock adapter
            mock_adapter = MockAdapter("openai")
            layer = Layer11AdversarialAnalysis()
            result = layer.run(mock_adapter, "test-model")
            
            if not isinstance(result, LayerResult):
                self.log("Layer13 did not return LayerResult", "fail")
                return False
            
            self.log(f"Layer13: confidence={result.confidence:.2f}, evidence={len(result.evidence)}")
            self.log("Adversarial Response Analysis: PASSED", "pass")
            self.passed += 1
            return True
            
        except Exception as e:
            self.log(f"Adversarial Response Analysis failed: {e}", "fail")
            self.failed += 1
            return False
    
    def test_pipeline_integration(self) -> bool:
        """Test that all new layers can be imported and used in pipeline."""
        self.log("Testing Pipeline Integration...")
        
        try:
            # Test pipeline import
            from app.predetect.pipeline import PreDetectionPipeline
            
            # Check that all new layer classes are importable
            from app.predetect.pipeline import (
                Layer8SemanticFingerprint,
                Layer7AdvancedExtraction,
                Layer8DifferentialTesting,
                Layer9ToolCapability,
                Layer11AdversarialAnalysis,
            )
            
            self.log("All new layer classes importable from pipeline")
            
            # Create pipeline instance
            pipeline = PreDetectionPipeline()
            
            self.log("Pipeline instantiation successful")
            self.log("Pipeline Integration: PASSED", "pass")
            self.passed += 1
            return True
            
        except Exception as e:
            self.log(f"Pipeline Integration failed: {e}", "fail")
            self.failed += 1
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all Phase 3 validation tests."""
        print("=" * 60)
        print("Phase 3 Validation: Detection Enhancement (12-Layer Pipeline)")
        print("=" * 60)
        print()
        
        tests = [
            ("Semantic Fingerprinting", self.test_semantic_fingerprint),
            ("Advanced Extraction v2", self.test_advanced_extraction),
            ("Differential Consistency Testing", self.test_differential_testing),
            ("Tool Capability Probe", self.test_tool_capability),
            ("Adversarial Response Analysis", self.test_adversarial_analysis),
            ("Pipeline Integration", self.test_pipeline_integration),
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
    validator = Phase3Validator()
    results = validator.run_all_tests()
    
    # Write results to file
    output_file = Path(__file__).parent / "phase3_validation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Exit with appropriate code
    sys.exit(0 if results["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
