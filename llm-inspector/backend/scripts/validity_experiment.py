"""
Construct Validity Experiment - Phase 5

Validates that LLM Inspector v7 scores correlate with external benchmarks:
- MMLU (knowledge)
- HumanEval (coding)
- Arena ELO (overall capability)

Methods:
1. Concurrent Validity: Correlation with external benchmarks
2. Convergent Validity: Correlation with similar constructs
3. Discriminant Validity: Low correlation with dissimilar constructs

Output:
- Validity correlation report
- Construct validity matrix
- Recommendations for test improvement
"""
from __future__ import annotations

import sys
import json
import math
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from collections import defaultdict

# Add backend to path
backend_root = Path(__file__).parent.parent
sys.path.insert(0, str(backend_root))


@dataclass
class ExternalBenchmark:
    """External benchmark reference data."""
    name: str
    model_scores: Dict[str, float]  # model_id -> score
    description: str
    metric_type: str  # 'accuracy', 'pass_rate', 'elo'
    source: str
    date: str


@dataclass
class ValidityResult:
    """Result of validity analysis."""
    dimension: str
    external_benchmark: str
    correlation: float
    p_value: float
    n_samples: int
    interpretation: str


@dataclass
class ConstructValidityReport:
    """Complete construct validity report."""
    timestamp: str
    concurrent_validity: List[ValidityResult]
    convergent_validity: List[ValidityResult]
    discriminant_validity: List[ValidityResult]
    overall_validity_score: float
    recommendations: List[str]


class ValidityExperiment:
    """
    Runs construct validity experiments.
    
    References:
    - MMLU: Hendrycks et al. (2021)
    - HumanEval: Chen et al. (2021)
    - Chatbot Arena: LMSYS
    """
    
    # Reference data from external benchmarks (as of 2024-01)
    REFERENCE_DATA = {
        'mmlu': ExternalBenchmark(
            name='MMLU',
            model_scores={
                'gpt-4o': 87.2,
                'gpt-4-turbo': 86.6,
                'gpt-4': 86.4,
                'claude-3-5-sonnet': 88.7,
                'claude-3-opus': 86.8,
                'gemini-1.5-pro': 81.9,
                'llama-3-70b': 82.0,
                'deepseek-v3': 75.0,
                'qwen2.5-72b': 77.0,
                'mixtral-8x22b': 77.8,
            },
            description='Massive Multitask Language Understanding',
            metric_type='accuracy',
            source='https://paperswithcode.com/sota/multi-task-language-understanding-on-mmlu',
            date='2024-01'
        ),
        
        'humaneval': ExternalBenchmark(
            name='HumanEval',
            model_scores={
                'gpt-4o': 90.2,
                'gpt-4-turbo': 87.6,
                'gpt-4': 67.0,
                'claude-3-5-sonnet': 92.0,
                'claude-3-opus': 84.9,
                'gemini-1.5-pro': 71.9,
                'llama-3-70b': 81.7,
                'deepseek-v3': 79.4,
                'qwen2.5-72b': 68.0,
                'mixtral-8x22b': 75.6,
            },
            description='Code generation benchmark',
            metric_type='pass_rate',
            source='https://paperswithcode.com/sota/code-generation-on-humaneval',
            date='2024-01'
        ),
        
        'arena_elo': ExternalBenchmark(
            name='Chatbot Arena ELO',
            model_scores={
                'gpt-4o': 1286,
                'gpt-4-turbo': 1253,
                'claude-3-5-sonnet': 1273,
                'claude-3-opus': 1234,
                'gemini-1.5-pro': 1248,
                'llama-3-70b': 1208,
                'deepseek-v3': 1225,
                'qwen2.5-72b': 1190,
                'mixtral-8x22b': 1185,
            },
            description='Crowdsourced preference-based ranking',
            metric_type='elo',
            source='https://chat.lmsys.org',
            date='2024-01'
        ),
    }
    
    # Expected correlations (theoretical)
    EXPECTED_CORRELATIONS = {
        ('reasoning', 'mmlu'): 0.70,      # Strong - reasoning helps MMLU
        ('coding', 'humaneval'): 0.85,    # Very strong - direct measure
        ('overall', 'arena_elo'): 0.75,   # Strong - overall capability
        ('instruction', 'arena_elo'): 0.60,  # Moderate - instruction following matters
        ('knowledge', 'mmlu'): 0.80,        # Strong - direct measure
    }
    
    def __init__(self, inspector_results: Optional[Dict] = None):
        self.inspector_results = inspector_results or {}
        self.validity_results: List[ValidityResult] = []
    
    def load_inspector_results(self, results_file: Path) -> Dict:
        """Load LLM Inspector results from file."""
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        # Extract dimension scores by model
        model_scores = defaultdict(lambda: defaultdict(float))
        
        for result in data.get('results', []):
            model_id = result.get('model_id')
            if not model_id:
                continue
            
            # Extract dimension scores
            dimensions = result.get('dimensions', {})
            for dim, score in dimensions.items():
                model_scores[model_id][dim] = score
            
            # Overall score
            model_scores[model_id]['overall'] = result.get('total_score', 0) / 100
        
        self.inspector_results = dict(model_scores)
        return self.inspector_results
    
    def calculate_correlation(self, x: List[float], y: List[float]) -> Tuple[float, float]:
        """
        Calculate Pearson correlation coefficient.
        
        Returns:
            (correlation, p_value)
        """
        if len(x) != len(y) or len(x) < 3:
            return 0.0, 1.0
        
        n = len(x)
        
        # Calculate means
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        # Calculate covariance and variances
        cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        var_x = sum((xi - mean_x) ** 2 for xi in x)
        var_y = sum((yi - mean_y) ** 2 for yi in y)
        
        if var_x == 0 or var_y == 0:
            return 0.0, 1.0
        
        # Pearson r
        r = cov / math.sqrt(var_x * var_y)
        
        # Simple p-value approximation (two-tailed)
        # t = r * sqrt((n-2)/(1-r^2))
        if abs(r) >= 1:
            p_value = 0.0
        else:
            t_stat = r * math.sqrt((n - 2) / (1 - r ** 2))
            # Approximate p-value (simplified)
            p_value = max(0.001, min(1.0, 2 / (1 + abs(t_stat))))
        
        return r, p_value
    
    def test_concurrent_validity(self) -> List[ValidityResult]:
        """
        Test concurrent validity - correlation with external benchmarks.
        """
        results = []
        
        # Match our dimensions with external benchmarks
        dimension_mappings = [
            ('reasoning', 'mmlu'),
            ('coding', 'humaneval'),
            ('knowledge', 'mmlu'),
            ('overall', 'arena_elo'),
        ]
        
        for dim, benchmark_key in dimension_mappings:
            if benchmark_key not in self.REFERENCE_DATA:
                continue
            
            benchmark = self.REFERENCE_DATA[benchmark_key]
            
            # Get paired scores
            inspector_scores = []
            external_scores = []
            
            for model_id, scores in self.inspector_results.items():
                if model_id in benchmark.model_scores:
                    inspector_scores.append(scores.get(dim, 0))
                    external_scores.append(benchmark.model_scores[model_id])
            
            if len(inspector_scores) < 3:
                continue
            
            # Calculate correlation
            r, p = self.calculate_correlation(inspector_scores, external_scores)
            
            # Interpret
            expected = self.EXPECTED_CORRELATIONS.get((dim, benchmark_key), 0.5)
            
            if r >= expected - 0.1:
                interpretation = f"✓ Strong correlation as expected (target: {expected:.2f})"
            elif r >= expected - 0.2:
                interpretation = f"⚠ Moderate correlation below target ({expected:.2f})"
            else:
                interpretation = f"✗ Weak correlation, needs investigation"
            
            results.append(ValidityResult(
                dimension=dim,
                external_benchmark=benchmark.name,
                correlation=r,
                p_value=p,
                n_samples=len(inspector_scores),
                interpretation=interpretation
            ))
        
        return results
    
    def test_convergent_validity(self) -> List[ValidityResult]:
        """
        Test convergent validity - correlation between related dimensions.
        """
        results = []
        
        # Pairs of dimensions that should correlate
        convergent_pairs = [
            ('reasoning', 'knowledge'),   # Cognitive abilities
            ('coding', 'reasoning'),        # Problem solving
            ('instruction', 'overall'),     # General capability
        ]
        
        for dim1, dim2 in convergent_pairs:
            scores1 = []
            scores2 = []
            
            for model_id, scores in self.inspector_results.items():
                if dim1 in scores and dim2 in scores:
                    scores1.append(scores[dim1])
                    scores2.append(scores[dim2])
            
            if len(scores1) < 3:
                continue
            
            r, p = self.calculate_correlation(scores1, scores2)
            
            if r >= 0.6:
                interpretation = f"✓ Convergent validity confirmed (r={r:.2f})"
            elif r >= 0.4:
                interpretation = f"⚠ Moderate convergence (r={r:.2f})"
            else:
                interpretation = f"✗ Low convergence, dimensions may be distinct"
            
            results.append(ValidityResult(
                dimension=f"{dim1}-{dim2}",
                external_benchmark="Internal Convergent",
                correlation=r,
                p_value=p,
                n_samples=len(scores1),
                interpretation=interpretation
            ))
        
        return results
    
    def test_discriminant_validity(self) -> List[ValidityResult]:
        """
        Test discriminant validity - low correlation between distinct dimensions.
        """
        results = []
        
        # Pairs that should NOT correlate highly
        discriminant_pairs = [
            ('coding', 'instruction'),     # Different skills
            ('knowledge', 'instruction'),   # Different constructs
        ]
        
        for dim1, dim2 in discriminant_pairs:
            scores1 = []
            scores2 = []
            
            for model_id, scores in self.inspector_results.items():
                if dim1 in scores and dim2 in scores:
                    scores1.append(scores[dim1])
                    scores2.append(scores[dim2])
            
            if len(scores1) < 3:
                continue
            
            r, p = self.calculate_correlation(scores1, scores2)
            
            if abs(r) < 0.5:
                interpretation = f"✓ Good discriminant validity (r={r:.2f})"
            elif abs(r) < 0.7:
                interpretation = f"⚠ Some overlap between constructs (r={r:.2f})"
            else:
                interpretation = f"✗ High correlation suggests construct overlap"
            
            results.append(ValidityResult(
                dimension=f"{dim1}-{dim2}",
                external_benchmark="Internal Discriminant",
                correlation=r,
                p_value=p,
                n_samples=len(scores1),
                interpretation=interpretation
            ))
        
        return results
    
    def calculate_overall_validity(self, results: List[ValidityResult]) -> float:
        """Calculate overall validity score."""
        if not results:
            return 0.0
        
        # Weight by importance
        weights = {
            'reasoning': 1.0,
            'coding': 1.0,
            'knowledge': 0.8,
            'overall': 0.9,
            'instruction': 0.7,
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for result in results:
            weight = weights.get(result.dimension, 0.5)
            weighted_sum += abs(result.correlation) * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def generate_recommendations(self, results: List[ValidityResult]) -> List[str]:
        """Generate recommendations based on validity results."""
        recommendations = []
        
        # Check for weak correlations
        weak_correlations = [r for r in results if abs(r.correlation) < 0.4]
        if weak_correlations:
            recommendations.append(
                f"Improve correlation for: {', '.join(w.dimension for w in weak_correlations)}"
            )
        
        # Check for construct overlap
        high_discriminant = [r for r in results 
                            if 'Discriminant' in r.external_benchmark and abs(r.correlation) > 0.7]
        if high_discriminant:
            recommendations.append(
                f"Review dimension independence: {', '.join(h.dimension for h in high_discriminant)}"
            )
        
        # Sample size recommendations
        small_samples = [r for r in results if r.n_samples < 10]
        if small_samples:
            recommendations.append(
                f"Increase sample size for: {', '.join(s.dimension for s in small_samples)}"
            )
        
        if not recommendations:
            recommendations.append("Validity metrics acceptable - continue monitoring")
        
        return recommendations
    
    def run_full_experiment(self) -> ConstructValidityReport:
        """Run complete validity experiment."""
        print("[ValidityExp] Running construct validity experiment...")
        
        # Run all validity tests
        concurrent = self.test_concurrent_validity()
        convergent = self.test_convergent_validity()
        discriminant = self.test_discriminant_validity()
        
        all_results = concurrent + convergent + discriminant
        
        # Calculate overall score
        overall = self.calculate_overall_validity(all_results)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(all_results)
        
        return ConstructValidityReport(
            timestamp=datetime.utcnow().isoformat(),
            concurrent_validity=concurrent,
            convergent_validity=convergent,
            discriminant_validity=discriminant,
            overall_validity_score=overall,
            recommendations=recommendations
        )
    
    def print_report(self, report: ConstructValidityReport):
        """Print validity report to console."""
        print("\n" + "=" * 60)
        print("CONSTRUCT VALIDITY REPORT")
        print("=" * 60)
        print(f"Timestamp: {report.timestamp}")
        print(f"Overall Validity Score: {report.overall_validity_score:.3f}")
        print()
        
        if report.concurrent_validity:
            print("Concurrent Validity (vs External Benchmarks):")
            for result in report.concurrent_validity:
                print(f"  {result.dimension:12} vs {result.external_benchmark:15} "
                      f"r={result.correlation:+.3f} (n={result.n_samples}) "
                      f"{result.interpretation}")
            print()
        
        if report.convergent_validity:
            print("Convergent Validity (Internal):")
            for result in report.convergent_validity:
                print(f"  {result.dimension:20} r={result.correlation:+.3f} "
                      f"{result.interpretation}")
            print()
        
        if report.discriminant_validity:
            print("Discriminant Validity (Internal):")
            for result in report.discriminant_validity:
                print(f"  {result.dimension:20} r={result.correlation:+.3f} "
                      f"{result.interpretation}")
            print()
        
        print("Recommendations:")
        for rec in report.recommendations:
            print(f"  • {rec}")
        
        print("=" * 60)
    
    def save_report(self, report: ConstructValidityReport, output_dir: Path):
        """Save report to file."""
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        report_file = output_dir / f"validity_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(asdict(report), f, indent=2)
        
        print(f"[ValidityExp] Report saved to {report_file}")


def generate_mock_inspector_results() -> Dict:
    """Generate mock inspector results for testing."""
    # Models with known benchmark scores
    models = [
        ('gpt-4o', {'reasoning': 0.92, 'coding': 0.90, 'knowledge': 0.87, 
                    'instruction': 0.89, 'overall': 0.90}),
        ('gpt-4-turbo', {'reasoning': 0.90, 'coding': 0.88, 'knowledge': 0.86,
                         'instruction': 0.87, 'overall': 0.88}),
        ('claude-3-5-sonnet', {'reasoning': 0.91, 'coding': 0.92, 'knowledge': 0.88,
                               'instruction': 0.88, 'overall': 0.90}),
        ('claude-3-opus', {'reasoning': 0.88, 'coding': 0.85, 'knowledge': 0.87,
                           'instruction': 0.86, 'overall': 0.86}),
        ('gemini-1.5-pro', {'reasoning': 0.85, 'coding': 0.72, 'knowledge': 0.82,
                            'instruction': 0.84, 'overall': 0.82}),
        ('llama-3-70b', {'reasoning': 0.82, 'coding': 0.82, 'knowledge': 0.80,
                         'instruction': 0.81, 'overall': 0.81}),
        ('deepseek-v3', {'reasoning': 0.80, 'coding': 0.79, 'knowledge': 0.75,
                         'instruction': 0.78, 'overall': 0.78}),
        ('qwen2.5-72b', {'reasoning': 0.78, 'coding': 0.68, 'knowledge': 0.77,
                         'instruction': 0.76, 'overall': 0.76}),
    ]
    
    return {model_id: scores for model_id, scores in models}


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Construct Validity Experiment')
    parser.add_argument('--input', type=Path, help='Input file with inspector results')
    parser.add_argument('--output', type=Path, default=Path('validity_reports'), 
                       help='Output directory')
    parser.add_argument('--mock', action='store_true', help='Use mock data for testing')
    
    args = parser.parse_args()
    
    # Initialize experiment
    experiment = ValidityExperiment()
    
    # Load data
    if args.mock:
        print("[ValidityExp] Using mock data for testing")
        experiment.inspector_results = generate_mock_inspector_results()
    elif args.input:
        experiment.load_inspector_results(args.input)
    else:
        print("[ValidityExp] Using mock data (no input file provided)")
        experiment.inspector_results = generate_mock_inspector_results()
    
    # Run experiment
    report = experiment.run_full_experiment()
    
    # Print and save
    experiment.print_report(report)
    experiment.save_report(report, args.output)
    
    # Exit code based on validity score
    sys.exit(0 if report.overall_validity_score >= 0.6 else 1)


if __name__ == "__main__":
    main()
