"""
IRT Data Collection Script - Phase 5

Collects response data for IRT (Item Response Theory) calibration.

Requirements for IRT 2PL calibration:
- Minimum 100 models tested for stable calibration
- Minimum 20 responses per test case
- Diverse ability range (weak to strong models)

Output:
- Response matrix (models x test cases)
- IRT parameter estimates (a, b, c)
- Item fit statistics
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
class IRTResponse:
    """Single response to a test case."""
    model_id: str
    case_id: str
    passed: bool  # 1 if correct, 0 if incorrect
    ability_theta: float  # Current ability estimate
    response_time_ms: int


@dataclass
class IRTItemStats:
    """Statistics for a single test case (item)."""
    case_id: str
    n_responses: int
    pass_rate: float
    
    # IRT 2PL parameters
    a: float  # Discrimination (0.5-2.0 optimal)
    b: float  # Difficulty (-3 to 3)
    c: float  # Guessing parameter (typically 0.25 for 4-option)
    
    # Fit statistics
    fit_rmse: float
    information_max: float


@dataclass
class IRTCalibrationReport:
    """Complete IRT calibration report."""
    timestamp: str
    n_models: int
    n_items: int
    
    # Global statistics
    mean_discrimination: float
    mean_difficulty: float
    reliability: float
    
    # Per-item statistics
    item_stats: List[IRTItemStats]
    
    # Model ability estimates
    model_abilities: Dict[str, float]
    
    # Quality indicators
    quality_summary: Dict[str, any]


class IRTDataCollector:
    """
    Collects and manages response data for IRT calibration.
    
    References:
    - Embretson & Reise (2000) Item Response Theory for Psychologists
    - Baker & Kim (2004) Item Response Theory: Parameter Estimation Techniques
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path(__file__).parent.parent / "irt_data"
        self.data_dir.mkdir(exist_ok=True)
        
        # Response storage
        self.responses: List[IRTResponse] = []
        self.models: set = set()
        self.cases: set = set()
    
    def add_response(self, response: IRTResponse):
        """Add a response to the dataset."""
        self.responses.append(response)
        self.models.add(response.model_id)
        self.cases.add(response.case_id)
    
    def add_batch(self, responses: List[IRTResponse]):
        """Add multiple responses."""
        for r in responses:
            self.add_response(r)
    
    def get_response_matrix(self) -> Dict[str, Dict[str, bool]]:
        """
        Get response matrix as nested dict.
        matrix[model_id][case_id] = passed
        """
        matrix = defaultdict(lambda: defaultdict(lambda: None))
        
        for r in self.responses:
            matrix[r.model_id][r.case_id] = r.passed
        
        return dict(matrix)
    
    def check_calibration_readiness(self) -> Dict[str, any]:
        """Check if dataset is ready for IRT calibration."""
        n_models = len(self.models)
        n_cases = len(self.cases)
        
        # Count responses per case
        case_counts = defaultdict(int)
        for r in self.responses:
            case_counts[r.case_id] += 1
        
        # Count responses per model
        model_counts = defaultdict(int)
        for r in self.responses:
            model_counts[r.model_id] += 1
        
        # Check requirements
        requirements = {
            'min_models': 100,
            'min_responses_per_case': 20,
            'min_cases_per_model': 10,
        }
        
        status = {
            'ready': False,
            'n_models': n_models,
            'n_cases': n_cases,
            'total_responses': len(self.responses),
            'models_met': n_models >= requirements['min_models'],
            'cases_with_min_responses': sum(1 for c in case_counts.values() 
                                           if c >= requirements['min_responses_per_case']),
            'avg_responses_per_case': sum(case_counts.values()) / len(case_counts) if case_counts else 0,
            'min_responses_per_case': min(case_counts.values()) if case_counts else 0,
            'requirements': requirements,
        }
        
        # Determine readiness
        status['ready'] = (
            status['models_met'] and
            status['cases_with_min_responses'] >= n_cases * 0.8  # 80% of cases have enough data
        )
        
        return status
    
    def estimate_ability_mle(self, model_id: str, item_params: Dict[str, Tuple[float, float]]) -> float:
        """
        Estimate ability (theta) using Maximum Likelihood Estimation.
        
        MLE finds theta that maximizes:
        L(theta) = prod(P_i^correct * (1-P_i)^incorrect)
        
        Args:
            model_id: Model to estimate ability for
            item_params: Dict of case_id -> (a, b)
        
        Returns:
            MLE estimate of theta
        """
        # Get responses for this model
        model_responses = [r for r in self.responses if r.model_id == model_id]
        
        if not model_responses:
            return 0.0
        
        # Simple iterative MLE
        theta = 0.0  # Start at average ability
        learning_rate = 0.1
        
        for _ in range(50):  # Max iterations
            gradient = 0.0
            
            for r in model_responses:
                if r.case_id not in item_params:
                    continue
                
                a, b = item_params[r.case_id]
                
                # 2PL probability
                p = 1 / (1 + math.exp(-a * (theta - b)))
                
                # Gradient of log-likelihood
                if r.passed:
                    gradient += a * (1 - p)
                else:
                    gradient -= a * p
            
            theta += learning_rate * gradient
            
            # Constrain to reasonable range
            theta = max(-4, min(4, theta))
            
            if abs(gradient) < 0.001:
                break
        
        return theta
    
    def calibrate_items_simple(self) -> Dict[str, IRTItemStats]:
        """
        Simple heuristic calibration of item parameters.
        
        This is a simplified version; production should use proper
        marginal maximum likelihood (MML) estimation.
        """
        stats = {}
        
        # Group responses by case
        case_responses = defaultdict(list)
        for r in self.responses:
            case_responses[r.case_id].append(r)
        
        for case_id, responses in case_responses.items():
            n = len(responses)
            if n < 10:
                continue
            
            # Calculate pass rate (proxy for difficulty)
            pass_rate = sum(1 for r in responses if r.passed) / n
            
            # Convert to difficulty (b parameter)
            # Use inverse logit, constrained to reasonable range
            if pass_rate <= 0.01:
                b = 3.0  # Very difficult
            elif pass_rate >= 0.99:
                b = -3.0  # Very easy
            else:
                b = -math.log(pass_rate / (1 - pass_rate))
                b = max(-3, min(3, b))
            
            # Estimate discrimination (simplified)
            # Higher variance in responses -> higher discrimination
            ability_range = max(r.ability_theta for r in responses) - min(r.ability_theta for r in responses)
            
            if ability_range > 2:
                a = min(2.0, 0.5 + ability_range * 0.3)  # Good spread
            else:
                a = 0.8  # Default moderate discrimination
            
            # Fit statistic (simplified RMSE)
            # Compare observed vs predicted
            predictions = []
            for r in responses:
                p = 1 / (1 + math.exp(-a * (r.ability_theta - b)))
                predictions.append(p)
            
            observed = [1 if r.passed else 0 for r in responses]
            rmse = math.sqrt(sum((o - p) ** 2 for o, p in zip(observed, predictions)) / n)
            
            # Information at peak (at theta = b)
            p_peak = 0.5
            info_max = (a ** 2) * p_peak * (1 - p_peak)
            
            stats[case_id] = IRTItemStats(
                case_id=case_id,
                n_responses=n,
                pass_rate=pass_rate,
                a=a,
                b=b,
                c=0.25,  # Fixed guessing parameter
                fit_rmse=rmse,
                information_max=info_max
            )
        
        return stats
    
    def run_calibration(self) -> IRTCalibrationReport:
        """Run full IRT calibration pipeline."""
        print("[IRT] Starting IRT calibration...")
        
        # Check readiness
        readiness = self.check_calibration_readiness()
        print(f"[IRT] Dataset: {readiness['n_models']} models, "
              f"{readiness['n_cases']} cases, "
              f"{readiness['total_responses']} responses")
        
        if not readiness['ready']:
            print(f"[IRT] Warning: Dataset may not meet all IRT requirements")
            print(f"[IRT] Models: {readiness['n_models']}/{readiness['requirements']['min_models']}")
            print(f"[IRT] Cases with min responses: {readiness['cases_with_min_responses']}")
        
        # Step 1: Initial ability estimates (assume all models average)
        model_abilities = {m: 0.0 for m in self.models}
        
        # Step 2: Calibrate items with initial abilities
        item_stats = self.calibrate_items_simple()
        
        # Step 3: Re-estimate abilities with calibrated items
        item_params = {s.case_id: (s.a, s.b) for s in item_stats.values()}
        
        for model_id in self.models:
            model_abilities[model_id] = self.estimate_ability_mle(model_id, item_params)
        
        # Step 4: Calculate global statistics
        discriminations = [s.a for s in item_stats.values()]
        difficulties = [s.b for s in item_stats.values()]
        
        mean_a = sum(discriminations) / len(discriminations) if discriminations else 0
        mean_b = sum(difficulties) / len(difficulties) if difficulties else 0
        
        # Estimate reliability (simplified)
        # Higher average information -> higher reliability
        avg_info = sum(s.information_max for s in item_stats.values()) / len(item_stats) if item_stats else 0
        reliability = min(0.95, avg_info / (1 + avg_info))  # Simplified formula
        
        # Quality summary
        quality = {
            'items_with_good_discrimination': sum(1 for s in item_stats.values() if s.a >= 0.8),
            'items_with_extreme_difficulty': sum(1 for s in item_stats.values() if abs(s.b) > 2.5),
            'items_with_poor_fit': sum(1 for s in item_stats.values() if s.fit_rmse > 0.1),
            'reliability_estimate': reliability,
        }
        
        return IRTCalibrationReport(
            timestamp=datetime.utcnow().isoformat(),
            n_models=len(self.models),
            n_items=len(self.cases),
            mean_discrimination=mean_a,
            mean_difficulty=mean_b,
            reliability=reliability,
            item_stats=list(item_stats.values()),
            model_abilities=model_abilities,
            quality_summary=quality
        )
    
    def print_report(self, report: IRTCalibrationReport):
        """Print calibration report."""
        print("\n" + "=" * 60)
        print("IRT CALIBRATION REPORT")
        print("=" * 60)
        print(f"Timestamp: {report.timestamp}")
        print(f"Models: {report.n_models}, Items: {report.n_items}")
        print()
        print(f"Global Statistics:")
        print(f"  Mean Discrimination (a): {report.mean_discrimination:.3f}")
        print(f"  Mean Difficulty (b): {report.mean_difficulty:.3f}")
        print(f"  Reliability Estimate: {report.reliability:.3f}")
        print()
        
        if report.item_stats:
            print("Item Statistics (sample):")
            for s in report.item_stats[:10]:
                quality = "✓" if s.a >= 0.5 and abs(s.b) <= 3 and s.fit_rmse < 0.1 else "✗"
                print(f"  {s.case_id:20} a={s.a:.2f} b={s.b:.2f} "
                      f"fit={s.fit_rmse:.3f} n={s.n_responses} {quality}")
            if len(report.item_stats) > 10:
                print(f"  ... and {len(report.item_stats) - 10} more items")
            print()
        
        print("Quality Summary:")
        for key, value in report.quality_summary.items():
            print(f"  {key}: {value}")
        print()
        
        print("Model Abilities (sample):")
        sorted_models = sorted(report.model_abilities.items(), key=lambda x: -x[1])
        for model_id, theta in sorted_models[:10]:
            print(f"  {model_id:20} θ = {theta:+.2f}")
        if len(sorted_models) > 10:
            print(f"  ... and {len(sorted_models) - 10} more models")
        
        print("=" * 60)
    
    def save_data(self, report: IRTCalibrationReport, output_dir: Optional[Path] = None):
        """Save calibration data and report."""
        output_dir = output_dir or self.data_dir
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        
        # Save report
        report_file = output_dir / f"irt_calibration_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(asdict(report), f, indent=2)
        print(f"[IRT] Calibration report saved to {report_file}")
        
        # Save raw responses
        responses_file = output_dir / f"irt_responses_{timestamp}.json"
        with open(responses_file, 'w') as f:
            json.dump({
                'timestamp': datetime.utcnow().isoformat(),
                'n_responses': len(self.responses),
                'responses': [asdict(r) for r in self.responses]
            }, f, indent=2)
        print(f"[IRT] Response data saved to {responses_file}")
        
        # Save item parameters for test suite
        item_params = {
            s.case_id: {
                'a': s.a,
                'b': s.b,
                'c': s.c,
                'fit_rmse': s.fit_rmse,
                'information_max': s.information_max,
                'n_calibrated': s.n_responses
            }
            for s in report.item_stats
        }
        
        params_file = output_dir / f"irt_item_params_{timestamp}.json"
        with open(params_file, 'w') as f:
            json.dump({
                'timestamp': datetime.utcnow().isoformat(),
                'calibration_method': 'simple_mle',
                'items': item_params
            }, f, indent=2)
        print(f"[IRT] Item parameters saved to {params_file}")


def generate_mock_data(n_models: int = 50, n_cases: int = 30) -> List[IRTResponse]:
    """Generate mock IRT response data for testing."""
    import random
    
    responses = []
    
    # Generate model abilities (normal distribution)
    model_abilities = {f"model_{i}": random.gauss(0, 1) for i in range(n_models)}
    
    # Generate item parameters
    item_params = {
        f"case_{i}": (random.uniform(0.5, 1.5), random.uniform(-2, 2))
        for i in range(n_cases)
    }
    
    # Generate responses
    for model_id, theta in model_abilities.items():
        for case_id, (a, b) in item_params.items():
            # 2PL probability
            p = 1 / (1 + math.exp(-a * (theta - b)))
            
            # Sample response
            passed = random.random() < p
            
            responses.append(IRTResponse(
                model_id=model_id,
                case_id=case_id,
                passed=passed,
                ability_theta=theta,
                response_time_ms=random.randint(500, 3000)
            ))
    
    return responses


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='IRT Data Collection and Calibration')
    parser.add_argument('--input', type=Path, help='Input file with existing responses')
    parser.add_argument('--output', type=Path, help='Output directory')
    parser.add_argument('--mock', action='store_true', help='Use mock data for testing')
    parser.add_argument('--mock-models', type=int, default=100, help='Number of mock models')
    parser.add_argument('--mock-cases', type=int, default=50, help='Number of mock test cases')
    
    args = parser.parse_args()
    
    # Initialize collector
    collector = IRTDataCollector(args.output)
    
    # Load or generate data
    if args.input:
        print(f"[IRT] Loading data from {args.input}")
        with open(args.input, 'r') as f:
            data = json.load(f)
            responses = [IRTResponse(**r) for r in data.get('responses', [])]
            collector.add_batch(responses)
    elif args.mock:
        print(f"[IRT] Generating mock data: {args.mock_models} models, {args.mock_cases} cases")
        responses = generate_mock_data(args.mock_models, args.mock_cases)
        collector.add_batch(responses)
    else:
        print("[IRT] No input data provided. Use --mock to generate test data.")
        sys.exit(1)
    
    # Run calibration
    report = collector.run_calibration()
    
    # Print and save
    collector.print_report(report)
    collector.save_data(report, args.output)
    
    # Exit code based on quality
    quality_ok = (
        report.quality_summary.get('reliability_estimate', 0) >= 0.7 and
        report.quality_summary.get('items_with_good_discrimination', 0) >= report.n_items * 0.6
    )
    
    sys.exit(0 if quality_ok else 1)


if __name__ == "__main__":
    main()
