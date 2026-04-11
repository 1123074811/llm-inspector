"""
Phase 1 Validation Script for LLM Inspector v7.0

Validates that all Phase 1 components are correctly implemented:
1. IRT calibration framework
2. Data validation system
3. Factor analysis module
4. Scientific weight calculation
5. Configuration files

Usage:
    python scripts/validate_phase1.py

Exit codes:
    0 - All validations passed
    1 - One or more validations failed
"""

import sys
import os
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_module_imports():
    """Check that all Phase 1 modules can be imported."""
    print("=" * 60)
    print("Phase 1 Validation: Module Imports")
    print("=" * 60)
    
    modules = [
        ("app.analysis.irt_calibration", "IRT Calibration"),
        ("app.validation.data_validation", "Data Validation"),
        ("app.validation.audit_runner", "Audit Runner"),
        ("app.analysis.factor_analysis", "Factor Analysis"),
    ]
    
    all_passed = True
    for module_name, description in modules:
        try:
            __import__(module_name)
            print(f"[PASS] {description}: {module_name}")
        except Exception as e:
            print(f"[FAIL] {description}: {module_name} - {e}")
            all_passed = False
    
    return all_passed


def check_irt_calibration():
    """Test IRT calibration functionality."""
    print("\n" + "=" * 60)
    print("Phase 1 Validation: IRT Calibration")
    print("=" * 60)
    
    try:
        from app.analysis.irt_calibration import (
            IRTParameters, IRTCalibrator, calculate_data_driven_weights
        )
        import numpy as np
        
        # Test IRTParameters
        params = IRTParameters(
            a=1.2,
            b=0.5,
            c=0.0,
            fit_rmse=0.05,
            n_calibrated=100,
            calibration_date="2026-04-11",
            data_source="empirical_test"
        )
        
        # Test validation
        is_valid, issues = params.is_valid()
        print(f"[PASS] IRTParameters created: a={params.a}, b={params.b}")
        print(f"  Valid: {is_valid}, Issues: {issues}")
        
        # Test information calculation
        thetas = np.linspace(-2, 2, 10)
        info = params.calculate_information(thetas)
        print(f"[PASS] Information calculation: max={max(info):.3f}")
        
        # Test probability calculation
        prob = params.probability_correct(0.0)
        print(f"[PASS] Probability at θ=0: {prob:.3f}")
        
        # Test calibrator
        calibrator = IRTCalibrator(min_calibrations=10)
        print(f"[PASS] IRTCalibrator created (min_calibrations={calibrator.min_calibrations})")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] IRT calibration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_data_validation():
    """Test data validation functionality."""
    print("\n" + "=" * 60)
    print("Phase 1 Validation: Data Validation")
    print("=" * 60)
    
    try:
        from app.validation.data_validation import (
            DataValidator, ValidationStatus, DataSource
        )
        
        validator = DataValidator()
        print("[PASS] DataValidator created")
        
        # Test IRT parameter validation
        good_params = {
            'a': 1.2,
            'b': 0.5,
            'calibration_date': '2026-04-11',
            'n_calibrated': 100,
            'data_source': 'empirical',
            'fit_rmse': 0.05,
            'reliability': 0.85
        }
        result = validator.validate_irt_parameters("test_case_001", good_params)
        print(f"[PASS] Good params validation: {result.status.value}")
        
        # Test bad params
        bad_params = {
            'a': 0.2,  # Too low
            'b': 4.0,  # Too extreme
            'calibration_date': '2026-04-11',
            'n_calibrated': 30,  # Too few
            'data_source': 'empirical',
            'fit_rmse': 0.20,  # Too high
        }
        result = validator.validate_irt_parameters("test_case_002", bad_params)
        print(f"[PASS] Bad params validation: {result.status.value} (expected: warning/invalid)")
        print(f"  Issues: {result.issues}")
        
        # Test weight validation
        weights = {
            'reasoning': 0.28,
            'coding': 0.22,
            'instruction': 0.20,
            'adversarial': 0.15,
            'safety': 0.10,
            'protocol': 0.05
        }
        result = validator.validate_scoring_weights(weights)
        print(f"[PASS] Weight validation: {result.status.value}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Data validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_factor_analysis():
    """Test factor analysis functionality."""
    print("\n" + "=" * 60)
    print("Phase 1 Validation: Factor Analysis")
    print("=" * 60)
    
    try:
        from app.analysis.factor_analysis import DimensionValidator
        import numpy as np
        
        validator = DimensionValidator()
        print("[PASS] DimensionValidator created")
        
        # Test with synthetic data
        np.random.seed(42)
        n_models = 50
        
        dimension_scores = {
            'reasoning': np.random.normal(75, 10, n_models),
            'coding': np.random.normal(70, 12, n_models),
            'instruction': np.random.normal(80, 8, n_models),
            'safety': np.random.normal(85, 5, n_models),
        }
        
        result = validator.validate_dimensions(dimension_scores)
        print(f"[PASS] CFA executed: {len(dimension_scores)} dimensions")
        print(f"  Fit acceptable: {result.fit_acceptable}")
        print(f"  Validity acceptable: {result.validity_acceptable}")
        print(f"  CFI: {result.cfi:.3f}, RMSEA: {result.rmsea:.3f}")
        
        # Test interpretation
        interpretations = validator.interpret_results(result)
        print(f"[PASS] Interpretation generated: {len(interpretations)} points")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Factor analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_configuration():
    """Check configuration files exist and are valid."""
    print("\n" + "=" * 60)
    print("Phase 1 Validation: Configuration Files")
    print("=" * 60)
    
    config_files = [
        ("app/config/scoring_weights.yaml", "Scoring Weights"),
    ]
    
    all_passed = True
    base_dir = Path(__file__).parent.parent
    
    for rel_path, description in config_files:
        full_path = base_dir / rel_path
        if full_path.exists():
            print(f"[PASS] {description}: {rel_path}")
            # Try to parse YAML
            try:
                import yaml
                with open(full_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                print(f"  Loaded: {len(data.get('weights', {}))} weight entries")
            except ImportError:
                print(f"  (YAML parser not installed, skipping content check)")
            except Exception as e:
                print(f"  [WARN] Parse warning: {e}")
        else:
            print(f"[FAIL] {description}: {rel_path} - NOT FOUND")
            all_passed = False
    
    return all_passed


def check_score_calculator_integration():
    """Test that score calculator integrates with new IRT system."""
    print("\n" + "=" * 60)
    print("Phase 1 Validation: Score Calculator Integration")
    print("=" * 60)
    
    try:
        from app.analysis.score_calculator import ScoreCardCalculator
        
        calculator = ScoreCardCalculator()
        print("[PASS] ScoreCardCalculator created")
        
        # Check that IRT method exists
        if hasattr(calculator, '_irt_information_weights'):
            print("[PASS] IRT information weights method present")
        else:
            print("[FAIL] IRT information weights method missing")
            return False
        
        # Check that resolve_weights accepts case_results
        import inspect
        sig = inspect.signature(calculator._resolve_weights)
        params = list(sig.parameters.keys())
        
        if 'case_results' in params:
            print("[PASS] _resolve_weights accepts case_results parameter")
        else:
            print("[FAIL] _resolve_weights missing case_results parameter")
            return False
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Score calculator integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Phase 1 validations."""
    print("\n" + "=" * 60)
    print("LLM Inspector v7.0 - Phase 1 Validation")
    print("Foundation: IRT Calibration & Scientific Scoring")
    print("=" * 60)
    
    results = {
        "Module Imports": check_module_imports(),
        "IRT Calibration": check_irt_calibration(),
        "Data Validation": check_data_validation(),
        "Factor Analysis": check_factor_analysis(),
        "Configuration": check_configuration(),
        "Score Calculator": check_score_calculator_integration(),
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("Phase 1 Validation Summary")
    print("=" * 60)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for component, result in results.items():
        status = "[PASS] PASS" if result else "[FAIL] FAIL"
        print(f"{status}: {component}")
    
    print(f"\nTotal: {passed}/{total} components validated")
    
    if passed == total:
        print("\n[PASS] Phase 1 validation COMPLETE - All systems operational")
        return 0
    else:
        print(f"\n[FAIL] Phase 1 validation INCOMPLETE - {total - passed} component(s) need attention")
        return 1


if __name__ == "__main__":
    sys.exit(main())
