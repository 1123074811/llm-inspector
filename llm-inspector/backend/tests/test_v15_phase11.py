"""
Tests for v15 Phase 11: Dataset Import Pipeline.

Covers runner/import_dataset.py:
  - ImportReport dataclass + to_dict()
  - DatasetImporter.generate_id()
  - DatasetImporter.validate_case()
  - DatasetImporter.load_suite() / save_suite()
  - DatasetImporter.import_cases()
  - expand_to_v15() (dry-run via monkeypatching)
"""
from __future__ import annotations
import json
import pathlib
import tempfile
import pytest


# ---------------------------------------------------------------------------
# Minimal valid case fixture
# ---------------------------------------------------------------------------

def _valid_case(suffix="001") -> dict:
    return {
        "id": f"re_test_{suffix}",
        "category": "reasoning",
        "name": f"TestCase{suffix}",
        "user_prompt": "What is 2+2?",
        "judge_method": "regex_match",
        "max_tokens": 64,
        "n_samples": 2,
        "temperature": 0.0,
        "params": {"regex": "4"},
        "weight": 1.0,
    }


# ---------------------------------------------------------------------------
# ImportReport tests
# ---------------------------------------------------------------------------

def test_import_report_defaults():
    from app.runner.import_dataset import ImportReport
    r = ImportReport()
    assert r.total_imported == 0
    assert r.skipped_duplicates == 0
    assert r.validation_errors == []
    assert r.source_datasets == []
    assert r.new_categories == []


def test_import_report_to_dict():
    from app.runner.import_dataset import ImportReport
    r = ImportReport(total_imported=5, skipped_duplicates=2)
    d = r.to_dict()
    assert d["total_imported"] == 5
    assert d["skipped_duplicates"] == 2
    assert "validation_errors" in d
    assert "source_datasets" in d
    assert "new_categories" in d


# ---------------------------------------------------------------------------
# DatasetImporter.generate_id
# ---------------------------------------------------------------------------

def test_generate_id_format():
    from app.runner.import_dataset import DatasetImporter
    id_ = DatasetImporter.generate_id("reasoning", "MyTest", 1)
    assert id_.startswith("rea_")
    assert "0001" in id_


def test_generate_id_slugifies_name():
    from app.runner.import_dataset import DatasetImporter
    id_ = DatasetImporter.generate_id("coding", "My Complex Test Name!", 42)
    assert " " not in id_
    assert "!" not in id_


def test_generate_id_uses_first_3_chars_of_category():
    from app.runner.import_dataset import DatasetImporter
    id_ = DatasetImporter.generate_id("knowledge", "Nobel", 1)
    assert id_.startswith("kno_")


def test_generate_id_pads_index():
    from app.runner.import_dataset import DatasetImporter
    id_ = DatasetImporter.generate_id("safety", "Test", 7)
    assert id_.endswith("0007")


# ---------------------------------------------------------------------------
# DatasetImporter.validate_case
# ---------------------------------------------------------------------------

def test_validate_case_valid():
    from app.runner.import_dataset import DatasetImporter
    err = DatasetImporter.validate_case(_valid_case())
    assert err is None


def test_validate_case_missing_id():
    from app.runner.import_dataset import DatasetImporter
    case = _valid_case()
    del case["id"]
    err = DatasetImporter.validate_case(case)
    assert err is not None
    assert "id" in err.lower()


def test_validate_case_missing_name():
    from app.runner.import_dataset import DatasetImporter
    case = _valid_case()
    del case["name"]
    err = DatasetImporter.validate_case(case)
    assert err is not None


def test_validate_case_missing_user_prompt():
    from app.runner.import_dataset import DatasetImporter
    case = _valid_case()
    del case["user_prompt"]
    err = DatasetImporter.validate_case(case)
    assert err is not None


def test_validate_case_missing_judge_method():
    from app.runner.import_dataset import DatasetImporter
    case = _valid_case()
    del case["judge_method"]
    err = DatasetImporter.validate_case(case)
    assert err is not None


def test_validate_case_invalid_category():
    from app.runner.import_dataset import DatasetImporter
    case = _valid_case()
    case["category"] = "nonexistent_category"
    err = DatasetImporter.validate_case(case)
    assert err is not None
    assert "category" in err.lower()


def test_validate_case_invalid_judge_method():
    from app.runner.import_dataset import DatasetImporter
    case = _valid_case()
    case["judge_method"] = "nonexistent_method"
    err = DatasetImporter.validate_case(case)
    assert err is not None
    assert "judge_method" in err.lower()


def test_validate_case_all_valid_categories():
    from app.runner.import_dataset import DatasetImporter
    valid_cats = [
        "reasoning", "knowledge", "instruction", "coding",
        "safety", "extraction", "identity", "adversarial",
        "multilingual", "timing",
    ]
    for cat in valid_cats:
        case = _valid_case()
        case["category"] = cat
        err = DatasetImporter.validate_case(case)
        assert err is None, f"Category '{cat}' should be valid but got: {err}"


def test_validate_case_all_valid_judge_methods():
    from app.runner.import_dataset import DatasetImporter
    valid_methods = [
        "exact_match", "regex_match", "semantic", "multi_choice",
        "numeric_tolerance", "semantic_entailment",
    ]
    for method in valid_methods:
        case = _valid_case()
        case["judge_method"] = method
        err = DatasetImporter.validate_case(case)
        assert err is None, f"Judge method '{method}' should be valid but got: {err}"


# ---------------------------------------------------------------------------
# DatasetImporter.load_suite / save_suite (via temp dir)
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_suite_dir(monkeypatch, tmp_path):
    """Override DatasetImporter.SUITE_DIR to use a temp directory."""
    from app.runner import import_dataset
    monkeypatch.setattr(import_dataset.DatasetImporter, "SUITE_DIR", tmp_path)
    monkeypatch.setattr(import_dataset, "_FIXTURES_DIR", tmp_path)
    return tmp_path


def test_load_suite_empty_when_missing(temp_suite_dir):
    from app.runner.import_dataset import DatasetImporter
    cases, meta = DatasetImporter.load_suite("test_empty")
    assert cases == []
    assert "version" in meta


def test_save_and_load_suite_roundtrip(temp_suite_dir):
    from app.runner.import_dataset import DatasetImporter
    cases = [_valid_case("001"), _valid_case("002")]
    meta = {"description": "Test suite"}
    DatasetImporter.save_suite("test_v", cases, meta)

    loaded_cases, loaded_meta = DatasetImporter.load_suite("test_v")
    assert len(loaded_cases) == 2
    assert loaded_cases[0]["id"] == "re_test_001"
    assert loaded_meta["version"] == "test_v"
    assert loaded_meta["case_count"] == 2


def test_save_suite_generates_at(temp_suite_dir):
    from app.runner.import_dataset import DatasetImporter
    DatasetImporter.save_suite("test_ts", [_valid_case()], {})
    _, meta = DatasetImporter.load_suite("test_ts")
    assert "generated_at" in meta


# ---------------------------------------------------------------------------
# DatasetImporter.import_cases
# ---------------------------------------------------------------------------

def test_import_cases_adds_new(temp_suite_dir):
    from app.runner.import_dataset import DatasetImporter
    cases = [_valid_case("x01"), _valid_case("x02")]
    report = DatasetImporter.import_cases(cases, "test_import")
    assert report.total_imported == 2
    assert report.skipped_duplicates == 0
    assert report.validation_errors == []


def test_import_cases_skips_duplicate(temp_suite_dir):
    from app.runner.import_dataset import DatasetImporter
    case = _valid_case("dup")
    DatasetImporter.import_cases([case], "test_dup")  # first import
    report = DatasetImporter.import_cases([case], "test_dup")  # duplicate
    assert report.skipped_duplicates == 1
    assert report.total_imported == 0


def test_import_cases_overwrite(temp_suite_dir):
    from app.runner.import_dataset import DatasetImporter
    case = _valid_case("ow1")
    DatasetImporter.import_cases([case], "test_ow")
    case["user_prompt"] = "Updated prompt"
    report = DatasetImporter.import_cases([case], "test_ow", overwrite=True)
    assert report.total_imported == 1
    # Verify updated
    loaded, _ = DatasetImporter.load_suite("test_ow")
    assert any(c["user_prompt"] == "Updated prompt" for c in loaded)


def test_import_cases_validation_error_skipped(temp_suite_dir):
    from app.runner.import_dataset import DatasetImporter
    bad_case = {"id": "bad", "category": "unknown_cat", "name": "Bad",
                "user_prompt": "x", "judge_method": "regex_match"}
    report = DatasetImporter.import_cases([bad_case], "test_val_err")
    assert len(report.validation_errors) >= 1
    assert report.total_imported == 0


def test_import_cases_applies_defaults(temp_suite_dir):
    from app.runner.import_dataset import DatasetImporter
    case = _valid_case("def1")
    # Remove optional fields to test defaulting
    case.pop("max_tokens", None)
    case.pop("n_samples", None)
    case.pop("weight", None)
    DatasetImporter.import_cases([case], "test_defaults")
    loaded, _ = DatasetImporter.load_suite("test_defaults")
    c = next(c for c in loaded if c["id"] == "re_test_def1")
    assert c["max_tokens"] == 256
    assert c["n_samples"] == 2
    assert c["weight"] == 1.0


def test_import_cases_mixed_valid_invalid(temp_suite_dir):
    from app.runner.import_dataset import DatasetImporter
    good = _valid_case("g1")
    bad = {"id": "bad99", "category": "???", "name": "Bad",
           "user_prompt": "x", "judge_method": "regex_match"}
    report = DatasetImporter.import_cases([good, bad], "test_mix")
    assert report.total_imported == 1
    assert len(report.validation_errors) == 1
