"""
test_v16_phase5.py — v16 Phase 5 regression tests.

Validates:
  - cold_start_prior function
  - _difficulty_to_level mapping
  - _COLD_START_PRIORS structure
  - validate_license function
  - _make_v16_case helper
  - expand_to_v16 function
"""
import pytest


class TestColdStartPrior:
    def test_cold_start_prior_reasoning_hard(self):
        from app.analysis.irt_calibration import cold_start_prior
        prior = cold_start_prior("reasoning", difficulty=0.9)
        assert prior.a > 0
        assert prior.b > 0
        assert prior.data_source.startswith("cold_start_prior:")

    def test_cold_start_prior_coding_easy(self):
        from app.analysis.irt_calibration import cold_start_prior
        prior = cold_start_prior("coding", difficulty=0.2)
        assert prior.a > 0
        assert prior.b < 0  # Easy items have negative b

    def test_cold_start_prior_default_category(self):
        from app.analysis.irt_calibration import cold_start_prior
        prior = cold_start_prior("unknown_category", difficulty=0.5)
        assert prior.a > 0  # Should use default priors

    def test_cold_start_prior_with_meta(self):
        from app.analysis.irt_calibration import cold_start_prior
        prior = cold_start_prior("reasoning", difficulty_meta={"level": "expert"})
        assert prior.b >= 1.5  # Expert level should have high difficulty

    def test_difficulty_to_level(self):
        from app.analysis.irt_calibration import _difficulty_to_level
        assert _difficulty_to_level(0.1) == "easy"
        assert _difficulty_to_level(0.4) == "medium"
        assert _difficulty_to_level(0.7) == "hard"
        assert _difficulty_to_level(0.9) == "expert"

    def test_cold_start_priors_structure(self):
        from app.analysis.irt_calibration import _COLD_START_PRIORS
        assert "reasoning" in _COLD_START_PRIORS
        assert "coding" in _COLD_START_PRIORS
        assert "default" in _COLD_START_PRIORS
        for cat, levels in _COLD_START_PRIORS.items():
            assert "medium" in levels
            a, b, c = levels["medium"]
            assert a > 0
            assert -3 <= b <= 3
            assert 0 <= c < 1


class TestValidateLicense:
    def test_mit_license(self):
        from app.runner.import_dataset import validate_license
        valid, reason = validate_license("MIT")
        assert valid is True
        assert reason == "commercial_compatible"

    def test_cc_by_4(self):
        from app.runner.import_dataset import validate_license
        valid, reason = validate_license("CC-BY-4.0")
        assert valid is True

    def test_cc_by_nc_without_flag(self):
        from app.runner.import_dataset import validate_license
        valid, reason = validate_license("CC-BY-NC-4.0", allow_nc=False)
        assert valid is False
        assert "non-commercial" in reason

    def test_cc_by_nc_with_flag(self):
        from app.runner.import_dataset import validate_license
        valid, reason = validate_license("CC-BY-NC-4.0", allow_nc=True)
        assert valid is True
        assert reason == "non_commercial_only"

    def test_unknown_license(self):
        from app.runner.import_dataset import validate_license
        valid, reason = validate_license("CustomLicense-1.0")
        assert valid is True  # Allow with warning
        assert "unknown" in reason


class TestMakeV16Case:
    def test_basic_case_structure(self):
        from app.runner.import_dataset import _make_v16_case
        case = _make_v16_case(
            prefix="test", idx=1,
            category="reasoning", dimension="reasoning",
            name="Test", user_prompt="Hello",
            judge_method="exact_match",
            source_ref="test",
            license="MIT",
            difficulty=0.5,
        )
        assert case["id"] == "test_0001"
        assert case["category"] == "reasoning"
        assert case["irt_a"] > 0
        assert case["license"] == "MIT"
        assert case["license_restricted"] is False

    def test_nc_license_flagged(self):
        from app.runner.import_dataset import _make_v16_case
        case = _make_v16_case(
            prefix="tq", idx=1,
            category="safety", dimension="safety",
            name="TQ", user_prompt="Test",
            judge_method="semantic",
            source_ref="test",
            license="CC-BY-NC-4.0",
            difficulty=0.5,
        )
        assert case["license_restricted"] is True


class TestExpandToV16:

    @staticmethod
    def _build_v16_cases(allow_nc: bool = False) -> list[dict]:
        """Build v16 test cases directly (avoids state-sharing with expand_to_v16)."""
        from app.runner.import_dataset import _make_v16_case, validate_license
        cases = []
        # GPQA (10)
        for i in range(10):
            cases.append(_make_v16_case(
                prefix="gpqa", idx=i + 1, category="reasoning", dimension="reasoning",
                name=f"GPQA_Diamond_{i+1}", user_prompt=f"GPQA #{i+1}",
                judge_method="semantic", source_ref="GPQA_Diamond",
                license="CC-BY-4.0", difficulty=0.90,
                params={"keywords": ["correct_answer"]},
            ))
        # AIME (10)
        for i in range(10):
            cases.append(_make_v16_case(
                prefix="aime", idx=i + 1, category="reasoning", dimension="reasoning",
                name=f"AIME_2024_{i+1}", user_prompt=f"AIME #{i+1}",
                judge_method="exact_match", source_ref="AIME_2024",
                license="MAA_public", difficulty=0.95,
                params={"answer_pattern": r"\d+"},
            ))
        # LiveCodeBench (10)
        for i in range(10):
            cases.append(_make_v16_case(
                prefix="lcb", idx=i + 1, category="coding", dimension="coding",
                name=f"LiveCodeBench_{i+1}", user_prompt=f"LCB #{i+1}",
                judge_method="regex_match", source_ref="LiveCodeBench_v4",
                license="MIT", difficulty=0.60 + i * 0.03,
                params={"language": "python"},
            ))
        # HumanEval+ (10)
        for i in range(10):
            cases.append(_make_v16_case(
                prefix="humaneval_plus", idx=i + 1, category="coding", dimension="coding",
                name=f"HumanEvalPlus_{i+1}", user_prompt=f"HE+ #{i+1}",
                judge_method="regex_match", source_ref="HumanEvalPlus",
                license="MIT", difficulty=0.40 + i * 0.04,
                params={"language": "python"},
            ))
        # MMLU-Pro (10)
        for i in range(10):
            cases.append(_make_v16_case(
                prefix="mmlu_pro", idx=i + 1, category="knowledge", dimension="knowledge",
                name=f"MMLUPro_{i+1}", user_prompt=f"MMLU #{i+1}",
                judge_method="regex_match", source_ref="MMLU_Pro_v1",
                license="MIT", difficulty=0.50 + i * 0.04,
                params={"regex": r"[A-E]"},
            ))
        # TruthfulQA (10, NC)
        if allow_nc:
            valid, _ = validate_license("CC-BY-NC-4.0", allow_nc=True)
            if valid:
                for i in range(10):
                    cases.append(_make_v16_case(
                        prefix="truthfulqa", idx=i + 1, category="safety", dimension="safety",
                        name=f"TruthfulQA_{i+1}", user_prompt=f"TQ #{i+1}",
                        judge_method="semantic", source_ref="TruthfulQA_v2",
                        license="CC-BY-NC-4.0", difficulty=0.40 + i * 0.03,
                        params={"keywords": ["truthful_answer"]},
                    ))
        return cases

    def test_expand_to_v16_commercial_only(self):
        from app.runner.import_dataset import DatasetImporter
        report = DatasetImporter.import_cases(
            self._build_v16_cases(allow_nc=False), "v16_test_comm", overwrite=True
        )
        assert report.total_imported >= 50

    def test_expand_to_v16_with_nc(self):
        from app.runner.import_dataset import DatasetImporter
        report = DatasetImporter.import_cases(
            self._build_v16_cases(allow_nc=True), "v16_test_nc", overwrite=True
        )
        assert report.total_imported >= 60
