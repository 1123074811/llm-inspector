"""
test_v16_phase3.py — v16 Phase 3 regression tests.

Validates:
  - weighted_ece function
  - ScoreCard coverage/weight_provenance_trace/weighted_ece/excluded_case_count fields
  - ScoreCard.to_dict() includes v16 fields
  - fit_weights.py Bradley-Terry function
  - total_v16.yaml weight file exists
"""
import pathlib
import pytest

_BACKEND = pathlib.Path(__file__).resolve().parent.parent


class TestWeightedECE:
    def test_weighted_ece_uniform_weights_equals_ece(self):
        from app.analysis.calibration_metrics import ece, weighted_ece
        probs = [0.9, 0.7, 0.3, 0.1, 0.8]
        outcomes = [1, 1, 0, 0, 1]
        ece_val = ece(probs, outcomes, n_bins=5)
        wece_val = weighted_ece(probs, outcomes, weights=None, n_bins=5)
        # With uniform weights, should be approximately equal
        if ece_val is not None and wece_val is not None:
            assert abs(ece_val - wece_val) < 0.01

    def test_weighted_ece_with_custom_weights(self):
        from app.analysis.calibration_metrics import weighted_ece
        probs = [0.9, 0.7, 0.3, 0.1, 0.8]
        outcomes = [1, 1, 0, 0, 1]
        weights = [2.0, 1.0, 1.0, 1.0, 2.0]
        result = weighted_ece(probs, outcomes, weights=weights, n_bins=5)
        assert result is not None
        assert 0.0 <= result <= 1.0

    def test_weighted_ece_empty_input(self):
        from app.analysis.calibration_metrics import weighted_ece
        assert weighted_ece([], [], n_bins=5) is None

    def test_weighted_ece_mismatched_lengths(self):
        from app.analysis.calibration_metrics import weighted_ece
        assert weighted_ece([0.5], [1, 0], n_bins=5) is None

    def test_weighted_ece_perfect_calibration(self):
        from app.analysis.calibration_metrics import weighted_ece
        # Perfectly calibrated: prob always matches outcome
        probs = [1.0, 1.0, 0.0, 0.0]
        outcomes = [1, 1, 0, 0]
        result = weighted_ece(probs, outcomes, n_bins=4)
        assert result is not None
        assert result < 0.01  # Should be near zero


class TestScoreCardV16:
    def test_coverage_field(self):
        from app.core.schemas import ScoreCard
        sc = ScoreCard(coverage=0.85)
        assert sc.coverage == 0.85

    def test_weight_provenance_trace_field(self):
        from app.core.schemas import ScoreCard
        trace = {"reasoning": {"source": "irt", "version": "v16", "method": "nnls"}}
        sc = ScoreCard(weight_provenance_trace=trace)
        assert sc.weight_provenance_trace is not None
        assert "reasoning" in sc.weight_provenance_trace

    def test_weighted_ece_field(self):
        from app.core.schemas import ScoreCard
        sc = ScoreCard(weighted_ece=0.05)
        assert sc.weighted_ece == 0.05

    def test_excluded_case_count_field(self):
        from app.core.schemas import ScoreCard
        sc = ScoreCard(excluded_case_count=3)
        assert sc.excluded_case_count == 3

    def test_to_dict_includes_v16_fields(self):
        from app.core.schemas import ScoreCard
        sc = ScoreCard(
            coverage=0.9,
            weight_provenance_trace={"test": {"source": "default"}},
            weighted_ece=0.03,
            excluded_case_count=1,
        )
        d = sc.to_dict()
        assert "coverage" in d
        # coverage is converted via pct() — 0.9 -> 90
        assert "weight_provenance_trace" in d
        assert "weighted_ece" in d
        assert "excluded_case_count" in d


class TestTotalV16Weights:
    def test_weight_file_exists(self):
        wp = _BACKEND / "app" / "_data" / "weights" / "total_v16.yaml"
        assert wp.exists(), "total_v16.yaml not found"

    def test_weight_file_loads(self):
        import yaml
        wp = _BACKEND / "app" / "_data" / "weights" / "total_v16.yaml"
        with open(wp, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["version"] == "v16"
        assert "composition" in data
        assert "capability" in data
        assert "authenticity" in data
        assert "performance" in data

    def test_capability_weights_sum_to_one(self):
        import yaml
        wp = _BACKEND / "app" / "_data" / "weights" / "total_v16.yaml"
        with open(wp, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        cap = data["capability"]
        total = sum(cap.values())
        assert abs(total - 1.0) < 0.01, f"Capability weights sum to {total}, expected 1.0"

    def test_composition_weights_sum_to_one(self):
        import yaml
        wp = _BACKEND / "app" / "_data" / "weights" / "total_v16.yaml"
        with open(wp, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        comp = data["composition"]
        total = sum(comp.values())
        assert abs(total - 1.0) < 0.01, f"Composition weights sum to {total}, expected 1.0"


class TestBradleyTerry:
    @pytest.fixture(autouse=True)
    def _import_fit_weights(self):
        import importlib.util, sys
        scripts_path = _BACKEND / "scripts" / "fit_weights.py"
        spec = importlib.util.spec_from_file_location("fit_weights", str(scripts_path))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["fit_weights"] = mod
        spec.loader.exec_module(mod)
        self._mod = mod

    def test_fit_bradley_terry_function_exists(self):
        assert callable(self._mod.fit_bradley_terry)

    def test_fit_bradley_terry_simple(self):
        dims = ["reasoning", "coding", "safety"]
        pairwise = [
            {"dim_a": "reasoning", "dim_b": "coding", "wins_a": 8, "wins_b": 2},
            {"dim_a": "reasoning", "dim_b": "safety", "wins_a": 7, "wins_b": 3},
            {"dim_a": "coding", "dim_b": "safety", "wins_a": 6, "wins_b": 4},
        ]
        weights, delta = self._mod.fit_bradley_terry(dims, pairwise)
        assert len(weights) == 3
        assert abs(weights.sum() - 1.0) < 0.01
        assert weights[0] > weights[2]  # reasoning > safety

    def test_fit_bradley_terry_empty(self):
        weights, delta = self._mod.fit_bradley_terry([], [])
        assert len(weights) == 0
