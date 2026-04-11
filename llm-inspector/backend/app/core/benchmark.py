"""
Performance Benchmarking Framework for V7.

Tracks performance metrics and detects regressions.
Based on V7_UPGRADE_PLAN.md Section 7.3 (Token Efficiency) and Section 8.2.
"""
from __future__ import annotations

import time
import json
import statistics
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
from pathlib import Path
import functools


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    name: str
    timestamp: str
    duration_ms: float
    tokens_used: int
    success: bool
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class BenchmarkStats:
    """Statistics for a benchmark over multiple runs."""
    name: str
    n_runs: int
    mean_duration_ms: float
    std_duration_ms: float
    min_duration_ms: float
    max_duration_ms: float
    mean_tokens: float
    std_tokens: float
    total_tokens: int
    success_rate: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


class PerformanceBenchmark:
    """
    Performance benchmarking suite for LLM Inspector.
    
    Tracks:
    - Response latency (TTFT, total time)
    - Token efficiency
    - Detection accuracy
    - System resource usage
    """
    
    def __init__(self, results_dir: Optional[Path] = None):
        self.results_dir = results_dir or Path(__file__).parent.parent / "benchmarks"
        self.results_dir.mkdir(exist_ok=True)
        self.results: List[BenchmarkResult] = []
        self._baseline: Optional[Dict[str, BenchmarkStats]] = None
    
    def run_benchmark(
        self,
        name: str,
        func: Callable,
        *args,
        n_runs: int = 5,
        warmup_runs: int = 1,
        **kwargs
    ) -> BenchmarkStats:
        """
        Run a benchmark function multiple times and collect statistics.
        
        Args:
            name: Benchmark name
            func: Function to benchmark
            args: Positional arguments for function
            n_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs (not counted)
            kwargs: Keyword arguments for function
        
        Returns:
            BenchmarkStats with aggregated results
        """
        print(f"[Benchmark] Running {name} ({n_runs} runs, {warmup_runs} warmup)...")
        
        # Warmup runs
        for _ in range(warmup_runs):
            try:
                func(*args, **kwargs)
            except Exception:
                pass
        
        # Actual benchmark runs
        durations = []
        tokens_list = []
        successes = []
        
        for i in range(n_runs):
            start = time.perf_counter()
            tokens = 0
            success = False
            metadata = {}
            
            try:
                result = func(*args, **kwargs)
                success = True
                
                # Try to extract token usage from result
                if isinstance(result, dict):
                    tokens = result.get('tokens_used', 0) or result.get('total_tokens', 0)
                    metadata = {k: v for k, v in result.items() if k not in ['tokens_used', 'total_tokens']}
            except Exception as e:
                metadata['error'] = str(e)
            
            duration = (time.perf_counter() - start) * 1000
            
            durations.append(duration)
            tokens_list.append(tokens)
            successes.append(success)
            
            # Store individual result
            benchmark_result = BenchmarkResult(
                name=name,
                timestamp=datetime.utcnow().isoformat(),
                duration_ms=duration,
                tokens_used=tokens,
                success=success,
                metadata=metadata
            )
            self.results.append(benchmark_result)
        
        # Calculate statistics
        stats = BenchmarkStats(
            name=name,
            n_runs=n_runs,
            mean_duration_ms=statistics.mean(durations),
            std_duration_ms=statistics.stdev(durations) if len(durations) > 1 else 0,
            min_duration_ms=min(durations),
            max_duration_ms=max(durations),
            mean_tokens=statistics.mean(tokens_list) if any(tokens_list) else 0,
            std_tokens=statistics.stdev([t for t in tokens_list if t]) if len([t for t in tokens_list if t]) > 1 else 0,
            total_tokens=sum(tokens_list),
            success_rate=sum(successes) / len(successes)
        )
        
        self._print_stats(stats)
        return stats
    
    def _print_stats(self, stats: BenchmarkStats):
        """Print benchmark statistics."""
        print(f"\n[Benchmark] {stats.name} Results:")
        print(f"  Duration: {stats.mean_duration_ms:.1f} ± {stats.std_duration_ms:.1f} ms")
        print(f"  Range: [{stats.min_duration_ms:.1f}, {stats.max_duration_ms:.1f}] ms")
        print(f"  Tokens: {stats.mean_tokens:.0f} ± {stats.std_tokens:.0f} (total: {stats.total_tokens})")
        print(f"  Success: {stats.success_rate*100:.1f}%")
    
    def load_baseline(self, baseline_file: Optional[Path] = None) -> Dict[str, BenchmarkStats]:
        """Load baseline statistics from file."""
        baseline_file = baseline_file or self.results_dir / "baseline.json"
        
        if not baseline_file.exists():
            print(f"[Benchmark] No baseline found at {baseline_file}")
            return {}
        
        with open(baseline_file, 'r') as f:
            data = json.load(f)
        
        self._baseline = {
            name: BenchmarkStats(**stats)
            for name, stats in data.items()
        }
        
        print(f"[Benchmark] Loaded baseline with {len(self._baseline)} benchmarks")
        return self._baseline
    
    def save_baseline(self, baseline_file: Optional[Path] = None):
        """Save current results as new baseline."""
        baseline_file = baseline_file or self.results_dir / "baseline.json"
        
        # Aggregate current results by name
        grouped: Dict[str, List[BenchmarkResult]] = {}
        for r in self.results:
            grouped.setdefault(r.name, []).append(r)
        
        baseline = {}
        for name, results in grouped.items():
            durations = [r.duration_ms for r in results]
            tokens = [r.tokens_used for r in results]
            successes = [r.success for r in results]
            
            baseline[name] = BenchmarkStats(
                name=name,
                n_runs=len(results),
                mean_duration_ms=statistics.mean(durations),
                std_duration_ms=statistics.stdev(durations) if len(durations) > 1 else 0,
                min_duration_ms=min(durations),
                max_duration_ms=max(durations),
                mean_tokens=statistics.mean(tokens) if any(tokens) else 0,
                std_tokens=statistics.stdev([t for t in tokens if t]) if len([t for t in tokens if t]) > 1 else 0,
                total_tokens=sum(tokens),
                success_rate=sum(successes) / len(successes)
            ).to_dict()
        
        with open(baseline_file, 'w') as f:
            json.dump(baseline, f, indent=2)
        
        print(f"[Benchmark] Saved baseline to {baseline_file}")
    
    def check_regression(
        self,
        stats: BenchmarkStats,
        threshold: float = 1.2  # 20% regression threshold
    ) -> Dict[str, Any]:
        """
        Check if current results show regression vs baseline.
        
        Returns:
            Dict with regression status and details
        """
        if not self._baseline or stats.name not in self._baseline:
            return {'regression_detected': False, 'reason': 'no_baseline'}
        
        baseline = self._baseline[stats.name]
        issues = []
        
        # Check duration regression
        if stats.mean_duration_ms > baseline.mean_duration_ms * threshold:
            increase = (stats.mean_duration_ms / baseline.mean_duration_ms - 1) * 100
            issues.append(f"Duration increased by {increase:.1f}%")
        
        # Check token regression
        if stats.mean_tokens > baseline.mean_tokens * threshold:
            increase = (stats.mean_tokens / baseline.mean_tokens - 1) * 100
            issues.append(f"Token usage increased by {increase:.1f}%")
        
        # Check success rate regression
        if stats.success_rate < baseline.success_rate * 0.9:  # 10% drop
            issues.append(f"Success rate dropped from {baseline.success_rate*100:.1f}% to {stats.success_rate*100:.1f}%")
        
        return {
            'regression_detected': len(issues) > 0,
            'issues': issues,
            'baseline': baseline.to_dict(),
            'current': stats.to_dict()
        }
    
    def save_results(self, filename: Optional[str] = None):
        """Save benchmark results to file."""
        filename = filename or f"benchmark_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.results_dir / filename
        
        data = {
            'timestamp': datetime.utcnow().isoformat(),
            'results': [r.to_dict() for r in self.results]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[Benchmark] Results saved to {filepath}")


def benchmark_timer(name: str, benchmark: Optional[PerformanceBenchmark] = None):
    """Decorator to benchmark function execution time."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            duration = (time.perf_counter() - start) * 1000
            
            if benchmark:
                benchmark.results.append(BenchmarkResult(
                    name=name,
                    timestamp=datetime.utcnow().isoformat(),
                    duration_ms=duration,
                    tokens_used=0,
                    success=True,
                    metadata={'function': func.__name__}
                ))
            else:
                print(f"[Benchmark] {name}: {duration:.1f} ms")
            
            return result
        return wrapper
    return decorator


# Pre-defined benchmark suites for common operations

class DetectionBenchmarks:
    """Benchmarks for pre-detection pipeline."""
    
    def __init__(self):
        self.benchmark = PerformanceBenchmark()
    
    def run_all(self, adapter, model_name: str = "test-model") -> Dict[str, BenchmarkStats]:
        """Run all detection benchmarks."""
        from app.predetect.pipeline import PreDetectionPipeline
        from app.predetect.semantic_fingerprint import Layer8SemanticFingerprint
        from app.predetect.extraction_v2 import Layer7AdvancedExtraction
        from app.predetect.differential_testing import Layer8DifferentialTesting
        
        results = {}
        
        # Layer 0-5 benchmarks
        results['layer0_http'] = self.benchmark.run_benchmark(
            'layer0_http',
            lambda: self._run_layer0(adapter),
            n_runs=3
        )
        
        # Layer 8-13 benchmarks (new in v7)
        results['layer8_semantic_fp'] = self.benchmark.run_benchmark(
            'layer8_semantic_fp',
            lambda: Layer8SemanticFingerprint().run(adapter, model_name),
            n_runs=3
        )
        
        results['layer9_adv_extraction'] = self.benchmark.run_benchmark(
            'layer9_adv_extraction',
            lambda: Layer7AdvancedExtraction().run(adapter, model_name, run_id="benchmark"),
            n_runs=2  # Fewer runs due to high token usage
        )
        
        results['layer10_differential'] = self.benchmark.run_benchmark(
            'layer10_differential',
            lambda: Layer8DifferentialTesting().run(adapter, model_name),
            n_runs=2
        )
        
        # Full pipeline benchmark
        results['full_pipeline'] = self.benchmark.run_benchmark(
            'full_pipeline',
            lambda: PreDetectionPipeline().run(adapter, model_name),
            n_runs=1
        )
        
        return results
    
    def _run_layer0(self, adapter):
        """Run Layer 0 benchmark."""
        from app.predetect.pipeline import Layer0HTTP
        return Layer0HTTP().run(adapter)


class TokenEfficiencyBenchmarks:
    """Benchmarks for token efficiency optimization."""
    
    def __init__(self):
        self.benchmark = PerformanceBenchmark()
    
    def run_compression_benchmark(self, prompts: Dict[str, str]) -> Dict[str, BenchmarkStats]:
        """
        Benchmark token compression effectiveness.
        
        Args:
            prompts: Dict of {name: prompt_text}
        
        Returns:
            Statistics on token usage
        """
        results = {}
        
        for name, prompt in prompts.items():
            # Estimate tokens
            words = len(prompt.split())
            chars = len(prompt)
            estimated_tokens = int(chars / 4)  # Rough estimate
            
            stats = BenchmarkStats(
                name=f"prompt_{name}",
                n_runs=1,
                mean_duration_ms=0,
                std_duration_ms=0,
                min_duration_ms=0,
                max_duration_ms=0,
                mean_tokens=estimated_tokens,
                std_tokens=0,
                total_tokens=estimated_tokens,
                success_rate=1.0
            )
            results[name] = stats
            
            print(f"[TokenEfficiency] {name}: ~{estimated_tokens} tokens ({chars} chars, {words} words)")
        
        return results


# Convenience function for CI/CD integration
def run_ci_benchmarks(adapter=None, save_results: bool = True) -> bool:
    """
    Run all benchmarks for CI/CD pipeline.
    
    Returns:
        True if no regressions detected, False otherwise
    """
    print("=" * 60)
    print("V7 Performance Benchmarks - CI Mode")
    print("=" * 60)
    
    benchmark = PerformanceBenchmark()
    
    # Load baseline
    baseline = benchmark.load_baseline()
    
    # Run detection benchmarks if adapter provided
    if adapter:
        detection = DetectionBenchmarks()
        detection.benchmark = benchmark
        results = detection.run_all(adapter)
        
        # Check for regressions
        regressions = []
        for name, stats in results.items():
            check = benchmark.check_regression(stats)
            if check['regression_detected']:
                regressions.append((name, check['issues']))
        
        if regressions:
            print("\n" + "=" * 60)
            print("REGRESSIONS DETECTED:")
            for name, issues in regressions:
                print(f"  {name}:")
                for issue in issues:
                    print(f"    - {issue}")
            print("=" * 60)
    
    # Save results
    if save_results:
        benchmark.save_results()
    
    return len(regressions) == 0 if 'regressions' in dir() else True


if __name__ == "__main__":
    # Quick test
    print("PerformanceBenchmark module ready")
