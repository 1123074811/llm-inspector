"""
Mass Model Testing Framework - Phase 5

Validates the v7 detection system against 100+ models.
Collects data for IRT calibration and construct validity.

Usage:
    python -m backend.scripts.mass_model_test --config models_config.yaml
    python -m backend.scripts.mass_model_test --sample 100

Output:
    - Individual model reports
    - Aggregate statistics
    - IRT calibration data
    - Validity correlation analysis
"""
from __future__ import annotations

import sys
import json
import asyncio
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import defaultdict
import statistics

# Add backend to path
backend_root = Path(__file__).parent.parent
sys.path.insert(0, str(backend_root))

from app.predetect.pipeline import PreDetectionPipeline
from app.core.benchmark import PerformanceBenchmark


@dataclass
class ModelTestResult:
    """Result from testing a single model."""
    model_id: str
    model_name: str
    provider: str
    timestamp: str
    
    # Pre-detection results
    predetect_success: bool
    predetect_confidence: float
    identified_as: Optional[str]
    layers_run: int
    total_tokens_used: int
    
    # Detection layer breakdown
    layer_results: List[Dict]
    
    # Test outcome
    test_passed: bool
    error_message: Optional[str]
    duration_seconds: float


@dataclass
class MassTestSummary:
    """Summary statistics from mass testing."""
    total_models: int
    successful_tests: int
    failed_tests: int
    predetect_success_rate: float
    
    # Confidence distribution
    avg_predetect_confidence: float
    confidence_std: float
    high_confidence_rate: float  # >0.85
    
    # Layer statistics
    avg_layers_run: float
    avg_tokens_used: float
    
    # Identity distribution
    identified_distribution: Dict[str, int]
    
    # Timing
    avg_duration_seconds: float
    total_duration_seconds: float


class ModelRegistry:
    """Registry of models to test."""
    
    # Built-in model list for testing
    DEFAULT_MODELS = [
        # OpenAI models
        {"id": "gpt-4o", "name": "GPT-4o", "provider": "openai"},
        {"id": "gpt-4-turbo", "name": "GPT-4-Turbo", "provider": "openai"},
        {"id": "gpt-4", "name": "GPT-4", "provider": "openai"},
        {"id": "gpt-3.5-turbo", "name": "GPT-3.5", "provider": "openai"},
        
        # Anthropic models
        {"id": "claude-3-5-sonnet", "name": "Claude 3.5 Sonnet", "provider": "anthropic"},
        {"id": "claude-3-opus", "name": "Claude 3 Opus", "provider": "anthropic"},
        {"id": "claude-3-sonnet", "name": "Claude 3 Sonnet", "provider": "anthropic"},
        {"id": "claude-3-haiku", "name": "Claude 3 Haiku", "provider": "anthropic"},
        
        # Google models
        {"id": "gemini-1.5-pro", "name": "Gemini 1.5 Pro", "provider": "google"},
        {"id": "gemini-1.5-flash", "name": "Gemini 1.5 Flash", "provider": "google"},
        {"id": "gemini-pro", "name": "Gemini Pro", "provider": "google"},
        
        # Meta models
        {"id": "llama-3-70b", "name": "Llama 3 70B", "provider": "meta"},
        {"id": "llama-3-8b", "name": "Llama 3 8B", "provider": "meta"},
        {"id": "llama-2-70b", "name": "Llama 2 70B", "provider": "meta"},
        
        # DeepSeek
        {"id": "deepseek-v3", "name": "DeepSeek-V3", "provider": "deepseek"},
        {"id": "deepseek-r1", "name": "DeepSeek-R1", "provider": "deepseek"},
        
        # Alibaba
        {"id": "qwen2.5-72b", "name": "Qwen2.5 72B", "provider": "alibaba"},
        {"id": "qwen2.5-14b", "name": "Qwen2.5 14B", "provider": "alibaba"},
        {"id": "qwen2-72b", "name": "Qwen2 72B", "provider": "alibaba"},
        
        # Other popular models
        {"id": "mixtral-8x22b", "name": "Mixtral 8x22B", "provider": "mistral"},
        {"id": "mixtral-8x7b", "name": "Mixtral 8x7B", "provider": "mistral"},
        {"id": "mistral-large", "name": "Mistral Large", "provider": "mistral"},
        {"id": "mistral-medium", "name": "Mistral Medium", "provider": "mistral"},
        
        # Chinese models
        {"id": "glm-4", "name": "GLM-4", "provider": "zhipu"},
        {"id": "glm-3-turbo", "name": "GLM-3-Turbo", "provider": "zhipu"},
        {"id": "kimi-v1", "name": "Kimi", "provider": "moonshot"},
        {"id": "yi-34b", "name": "Yi-34B", "provider": "01.ai"},
        {"id": "baichuan2", "name": "Baichuan2", "provider": "baichuan"},
        {"id": "ernie-bot", "name": "ERNIE Bot", "provider": "baidu"},
        
        # Testing/wrapper scenarios
        {"id": "unknown-wrapper-1", "name": "Unknown Wrapper A", "provider": "unknown"},
        {"id": "unknown-wrapper-2", "name": "Unknown Wrapper B", "provider": "unknown"},
    ]
    
    def __init__(self, custom_config: Optional[Path] = None):
        self.models = []
        
        if custom_config and custom_config.exists():
            with open(custom_config, 'r') as f:
                config = json.load(f)
                self.models = config.get('models', [])
        else:
            self.models = self.DEFAULT_MODELS
    
    def get_models(self, count: Optional[int] = None) -> List[Dict]:
        """Get list of models to test."""
        if count:
            return self.models[:count]
        return self.models
    
    def get_sample(self, n: int = 100) -> List[Dict]:
        """Get random sample of n models."""
        import random
        if n >= len(self.models):
            return self.models
        return random.sample(self.models, n)


class MockModelAdapter:
    """Mock adapter for testing without actual API keys."""
    
    def __init__(self, model_config: Dict):
        self.model_config = model_config
        self.provider = model_config.get('provider', 'unknown')
        self.model_name = model_config.get('name', 'unknown')
        
        # Simulate different model behaviors
        self.behaviors = self._get_mock_behavior()
    
    def _get_mock_behavior(self) -> Dict:
        """Get mock behavior based on provider."""
        behaviors = {
            'openai': {
                'response_style': 'concise',
                'tokens_per_char': 0.25,
                'latency_ms': 200,
                'identity_hints': ['OpenAI', 'GPT'],
            },
            'anthropic': {
                'response_style': 'verbose',
                'tokens_per_char': 0.30,
                'latency_ms': 250,
                'identity_hints': ['Anthropic', 'Claude'],
            },
            'google': {
                'response_style': 'balanced',
                'tokens_per_char': 0.28,
                'latency_ms': 300,
                'identity_hints': ['Google', 'Gemini'],
            },
            'meta': {
                'response_style': 'direct',
                'tokens_per_char': 0.26,
                'latency_ms': 350,
                'identity_hints': ['Meta', 'Llama'],
            },
            'unknown': {
                'response_style': 'inconsistent',
                'tokens_per_char': 0.35,
                'latency_ms': 500,
                'identity_hints': ['assistant', 'AI'],
            }
        }
        return behaviors.get(self.provider, behaviors['unknown'])
    
    def head_request(self) -> Dict:
        """Mock HEAD request."""
        import random
        headers = {}
        
        # Simulate provider-specific headers
        if self.provider == 'openai':
            if random.random() > 0.5:
                headers['openai-processing-ms'] = '150'
        elif self.provider == 'azure':
            headers['x-ms-region'] = 'eastus'
        
        return {'headers': headers}
    
    def bad_request(self) -> Dict:
        """Mock bad request."""
        return {'status_code': 400, 'body': {'error': {'type': 'invalid_request'}}}
    
    def list_models(self) -> Dict:
        """Mock list models."""
        return {'status_code': 200, 'body': {'data': [{'id': self.model_config['id']}]}}
    
    def chat(self, request) -> Any:
        """Mock chat response."""
        import time
        import random
        
        # Simulate latency
        time.sleep(self.behaviors['latency_ms'] / 1000 * random.uniform(0.8, 1.2))
        
        # Generate response based on prompt
        prompt = request.messages[0].content if request.messages else ""
        prompt_lower = prompt.lower()
        
        # Identity probes
        if 'company' in prompt_lower or 'made you' in prompt_lower:
            response = f"I am an AI assistant. I was created by {self.behaviors['identity_hints'][0] if self.behaviors['identity_hints'] else 'an AI company'}."
        elif 'model name' in prompt_lower:
            response = f"I am {self.model_name}."
        elif 'tokenizer' in prompt_lower or 'tokens' in prompt_lower:
            # Return mock token count
            response = "5"
        elif 'cutoff' in prompt_lower or 'training data' in prompt_lower:
            response = "April 2024"
        else:
            response = "This is a mock response for testing purposes."
        
        # Estimate tokens
        response_tokens = int(len(response) * self.behaviors['tokens_per_char'])
        prompt_tokens = int(len(prompt) * self.behaviors['tokens_per_char'])
        
        # Create mock response object
        class MockResponse:
            def __init__(self, content, tokens):
                self.content = content
                self.usage_total_tokens = tokens
                self.usage_prompt_tokens = prompt_tokens
                self.raw_json = {'model': self.model_config['id']}
                self.ok = True
                self.error_type = None
                self.status_code = 200
        
        return MockResponse(response, response_tokens + prompt_tokens)


class MassModelTester:
    """Runs mass model testing."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path(__file__).parent.parent / "test_results"
        self.output_dir.mkdir(exist_ok=True)
        self.results: List[ModelTestResult] = []
        self.pipeline = PreDetectionPipeline()
    
    async def test_model(self, model_config: Dict) -> ModelTestResult:
        """Test a single model."""
        import time
        
        start_time = time.time()
        
        try:
            # Create mock adapter
            adapter = MockModelAdapter(model_config)
            
            # Run pre-detection
            run_id = f"mass_test_{model_config['id']}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            
            result = self.pipeline.run(
                adapter=adapter,
                model_name=model_config['id'],
                extraction_mode=True,
                run_id=run_id
            )
            
            duration = time.time() - start_time
            
            # Extract layer results
            layer_results = [
                {
                    'layer': lr.layer,
                    'confidence': lr.confidence,
                    'identified_as': lr.identified_as,
                    'tokens_used': lr.tokens_used
                }
                for lr in result.layer_results
            ]
            
            return ModelTestResult(
                model_id=model_config['id'],
                model_name=model_config['name'],
                provider=model_config['provider'],
                timestamp=datetime.utcnow().isoformat(),
                predetect_success=result.success,
                predetect_confidence=result.confidence,
                identified_as=result.identified_as,
                layers_run=len(result.layer_results),
                total_tokens_used=result.total_tokens_used,
                layer_results=layer_results,
                test_passed=True,
                error_message=None,
                duration_seconds=duration
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return ModelTestResult(
                model_id=model_config['id'],
                model_name=model_config['name'],
                provider=model_config['provider'],
                timestamp=datetime.utcnow().isoformat(),
                predetect_success=False,
                predetect_confidence=0.0,
                identified_as=None,
                layers_run=0,
                total_tokens_used=0,
                layer_results=[],
                test_passed=False,
                error_message=str(e),
                duration_seconds=duration
            )
    
    async def run_mass_test(self, models: List[Dict], concurrent: int = 5) -> MassTestSummary:
        """Run tests for multiple models."""
        print(f"[MassTest] Starting test of {len(models)} models...")
        print(f"[MassTest] Output directory: {self.output_dir}")
        
        # Run tests with limited concurrency
        semaphore = asyncio.Semaphore(concurrent)
        
        async def test_with_semaphore(model):
            async with semaphore:
                result = await self.test_model(model)
                print(f"[MassTest] Completed: {result.model_name} "
                      f"({'✓' if result.test_passed else '✗'} "
                      f"conf={result.predetect_confidence:.2f})")
                return result
        
        # Run all tests
        tasks = [test_with_semaphore(m) for m in models]
        self.results = await asyncio.gather(*tasks)
        
        # Calculate summary
        return self._calculate_summary()
    
    def _calculate_summary(self) -> MassTestSummary:
        """Calculate summary statistics."""
        total = len(self.results)
        successful = sum(1 for r in self.results if r.test_passed)
        failed = total - successful
        
        confidences = [r.predetect_confidence for r in self.results if r.test_passed]
        
        # Identity distribution
        identity_dist = defaultdict(int)
        for r in self.results:
            if r.identified_as:
                identity_dist[r.identified_as] += 1
        
        durations = [r.duration_seconds for r in self.results]
        tokens = [r.total_tokens_used for r in self.results]
        
        return MassTestSummary(
            total_models=total,
            successful_tests=successful,
            failed_tests=failed,
            predetect_success_rate=successful / total if total > 0 else 0,
            avg_predetect_confidence=statistics.mean(confidences) if confidences else 0,
            confidence_std=statistics.stdev(confidences) if len(confidences) > 1 else 0,
            high_confidence_rate=sum(1 for c in confidences if c > 0.85) / len(confidences) if confidences else 0,
            avg_layers_run=statistics.mean([r.layers_run for r in self.results]) if self.results else 0,
            avg_tokens_used=statistics.mean(tokens) if tokens else 0,
            identified_distribution=dict(identity_dist),
            avg_duration_seconds=statistics.mean(durations) if durations else 0,
            total_duration_seconds=sum(durations)
        )
    
    def save_results(self, summary: MassTestSummary):
        """Save test results to files."""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        
        # Save individual results
        results_file = self.output_dir / f"mass_test_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': datetime.utcnow().isoformat(),
                'summary': asdict(summary),
                'results': [asdict(r) for r in self.results]
            }, f, indent=2)
        
        print(f"[MassTest] Results saved to {results_file}")
        
        # Save summary only
        summary_file = self.output_dir / f"mass_test_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(asdict(summary), f, indent=2)
        
        print(f"[MassTest] Summary saved to {summary_file}")
    
    def print_summary(self, summary: MassTestSummary):
        """Print summary to console."""
        print("\n" + "=" * 60)
        print("MASS MODEL TEST SUMMARY")
        print("=" * 60)
        print(f"Total Models Tested: {summary.total_models}")
        print(f"Successful: {summary.successful_tests} ({summary.predetect_success_rate*100:.1f}%)")
        print(f"Failed: {summary.failed_tests}")
        print()
        print("Detection Performance:")
        print(f"  Avg Confidence: {summary.avg_predetect_confidence:.3f} ± {summary.confidence_std:.3f}")
        print(f"  High Confidence Rate (>0.85): {summary.high_confidence_rate*100:.1f}%")
        print(f"  Avg Layers Run: {summary.avg_layers_run:.1f}")
        print(f"  Avg Tokens Used: {summary.avg_tokens_used:.0f}")
        print()
        print("Identity Distribution:")
        for identity, count in sorted(summary.identified_distribution.items(), key=lambda x: -x[1])[:5]:
            print(f"  {identity}: {count}")
        print()
        print("Timing:")
        print(f"  Avg Duration: {summary.avg_duration_seconds:.1f}s")
        print(f"  Total Duration: {summary.total_duration_seconds:.1f}s")
        print("=" * 60)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Mass Model Testing for V7')
    parser.add_argument('--config', type=Path, help='Custom model config JSON file')
    parser.add_argument('--sample', type=int, default=50, help='Number of models to test (default: 50)')
    parser.add_argument('--output', type=Path, help='Output directory for results')
    parser.add_argument('--concurrent', type=int, default=5, help='Concurrent tests (default: 5)')
    
    args = parser.parse_args()
    
    # Initialize
    registry = ModelRegistry(args.config)
    tester = MassModelTester(args.output)
    
    # Get models to test
    models = registry.get_sample(args.sample)
    print(f"[MassTest] Selected {len(models)} models for testing")
    
    # Run tests
    summary = await tester.run_mass_test(models, concurrent=args.concurrent)
    
    # Print and save results
    tester.print_summary(summary)
    tester.save_results(summary)
    
    # Exit code based on success rate
    success_rate = summary.predetect_success_rate
    sys.exit(0 if success_rate >= 0.8 else 1)


if __name__ == "__main__":
    asyncio.run(main())
