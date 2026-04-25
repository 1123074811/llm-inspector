"""
Tests for v15 Phase 12: Judge Registry.

Covers:
  - analysis/judge_registry.py: list_methods, get_method, methods_by_mode,
    applicable_for, registry_summary
  - _data/judge_registry.yaml: schema correctness
  - v15_handlers.py: handle_judge_registry, handle_judge_registry_method
"""
from __future__ import annotations
import pathlib
import pytest


# ---------------------------------------------------------------------------
# judge_registry.yaml content tests
# ---------------------------------------------------------------------------

def test_judge_registry_yaml_exists():
    path = pathlib.Path(__file__).parent.parent / "app" / "_data" / "judge_registry.yaml"
    assert path.exists(), "judge_registry.yaml not found"


def test_judge_registry_yaml_not_empty():
    path = pathlib.Path(__file__).parent.parent / "app" / "_data" / "judge_registry.yaml"
    content = path.read_text(encoding="utf-8")
    assert len(content) > 100, "judge_registry.yaml appears empty"


def test_judge_registry_yaml_has_key_methods():
    path = pathlib.Path(__file__).parent.parent / "app" / "_data" / "judge_registry.yaml"
    content = path.read_text(encoding="utf-8")
    for method in ("exact_match", "regex", "semantic_v2"):
        assert method in content, f"Expected method '{method}' in judge_registry.yaml"


# ---------------------------------------------------------------------------
# judge_registry module tests
# ---------------------------------------------------------------------------

def test_list_methods_returns_list():
    from app.analysis.judge_registry import list_methods
    methods = list_methods()
    assert isinstance(methods, list)


def test_list_methods_not_empty():
    from app.analysis.judge_registry import list_methods
    methods = list_methods()
    assert len(methods) >= 1, "Registry should have at least one method"


def test_list_methods_contains_known_methods():
    from app.analysis.judge_registry import list_methods
    methods = list_methods()
    known = {"exact_match", "regex", "semantic_v2"}
    found = known & set(methods)
    assert len(found) >= 1, f"Expected some known methods in {methods}"


def test_get_method_returns_dict_for_valid_name():
    from app.analysis.judge_registry import list_methods, get_method
    methods = list_methods()
    if not methods:
        pytest.skip("Registry is empty")
    first = methods[0]
    result = get_method(first)
    assert result is not None
    assert isinstance(result, dict)


def test_get_method_returns_none_for_unknown():
    from app.analysis.judge_registry import get_method
    assert get_method("nonexistent_method_xyz") is None


def test_get_method_has_name_and_mode():
    from app.analysis.judge_registry import list_methods, get_method
    methods = list_methods()
    if not methods:
        pytest.skip("Registry is empty")
    entry = get_method(methods[0])
    assert entry is not None
    # At least one of name/mode should be present
    has_content = "name" in entry or "mode" in entry or "description" in entry
    assert has_content, f"Entry for {methods[0]} appears empty: {entry}"


def test_methods_by_mode_rule():
    from app.analysis.judge_registry import methods_by_mode
    rule_methods = methods_by_mode("rule")
    assert isinstance(rule_methods, list)
    # exact_match and regex should be rule-based
    assert len(rule_methods) >= 1


def test_methods_by_mode_llm():
    from app.analysis.judge_registry import methods_by_mode
    llm_methods = methods_by_mode("llm")
    assert isinstance(llm_methods, list)


def test_methods_by_mode_unknown_returns_empty():
    from app.analysis.judge_registry import methods_by_mode
    result = methods_by_mode("not_a_real_mode_xyz")
    assert result == []


def test_methods_by_mode_all_are_valid_names():
    from app.analysis.judge_registry import methods_by_mode, list_methods
    all_methods = set(list_methods())
    for mode in ("rule", "llm", "nli"):
        for m in methods_by_mode(mode):
            assert m in all_methods, f"methods_by_mode('{mode}') returned unknown method '{m}'"


def test_applicable_for_multiple_choice():
    from app.analysis.judge_registry import applicable_for
    results = applicable_for("multiple_choice")
    assert isinstance(results, list)


def test_applicable_for_math():
    from app.analysis.judge_registry import applicable_for
    results = applicable_for("math")
    assert isinstance(results, list)


def test_applicable_for_unknown_type_returns_empty():
    from app.analysis.judge_registry import applicable_for
    results = applicable_for("not_a_real_question_type_xyz")
    assert results == []


def test_applicable_for_returns_valid_names():
    from app.analysis.judge_registry import applicable_for, list_methods
    all_methods = set(list_methods())
    for qt in ("multiple_choice", "open_ended", "math", "code"):
        for m in applicable_for(qt):
            assert m in all_methods, f"applicable_for('{qt}') returned unknown method '{m}'"


def test_registry_summary_fields():
    from app.analysis.judge_registry import registry_summary
    summary = registry_summary()
    assert "total" in summary
    assert "by_mode" in summary
    assert "methods" in summary


def test_registry_summary_total_matches_list_methods():
    from app.analysis.judge_registry import registry_summary, list_methods
    summary = registry_summary()
    assert summary["total"] == len(list_methods())


def test_registry_summary_by_mode_is_dict():
    from app.analysis.judge_registry import registry_summary
    summary = registry_summary()
    assert isinstance(summary["by_mode"], dict)


def test_registry_summary_methods_is_dict():
    from app.analysis.judge_registry import registry_summary
    summary = registry_summary()
    assert isinstance(summary["methods"], dict)


def test_registry_summary_methods_has_name_mode_description():
    from app.analysis.judge_registry import registry_summary
    summary = registry_summary()
    for name, entry in summary["methods"].items():
        assert "name" in entry, f"Method '{name}' missing 'name' field"
        assert "mode" in entry, f"Method '{name}' missing 'mode' field"
        assert "description" in entry, f"Method '{name}' missing 'description' field"


# ---------------------------------------------------------------------------
# v15_handlers: judge registry handlers
# ---------------------------------------------------------------------------

def test_handle_judge_registry_returns_200():
    from app.handlers.v15_handlers import handle_judge_registry
    result = handle_judge_registry("/api/v15/judge-registry", {}, {})
    assert result["status"] == 200
    body = result["body"]
    assert "total" in body
    assert "methods" in body


def test_handle_judge_registry_method_found():
    from app.handlers.v15_handlers import handle_judge_registry_method
    from app.analysis.judge_registry import list_methods
    methods = list_methods()
    if not methods:
        pytest.skip("Registry is empty")
    method_name = methods[0]
    result = handle_judge_registry_method(
        f"/api/v15/judge-registry/{method_name}", {}, {}
    )
    assert result["status"] == 200
    assert result["body"]["method"] == method_name


def test_handle_judge_registry_method_not_found():
    from app.handlers.v15_handlers import handle_judge_registry_method
    result = handle_judge_registry_method(
        "/api/v15/judge-registry/nonexistent_method_xyz", {}, {}
    )
    assert result["status"] == 404


# ---------------------------------------------------------------------------
# v15_handlers: dataset import handlers
# ---------------------------------------------------------------------------

def test_handle_import_dataset_missing_cases():
    from app.handlers.v15_handlers import handle_import_dataset
    result = handle_import_dataset("/api/v15/dataset/import", {}, {})
    assert result["status"] == 400


def test_handle_import_dataset_empty_cases():
    from app.handlers.v15_handlers import handle_import_dataset
    result = handle_import_dataset("/api/v15/dataset/import", {}, {"cases": []})
    assert result["status"] == 400


def test_handle_validate_case_missing_body():
    from app.handlers.v15_handlers import handle_validate_case
    result = handle_validate_case("/api/v15/dataset/validate", {}, {})
    assert result["status"] == 400


def test_handle_validate_case_valid():
    from app.handlers.v15_handlers import handle_validate_case
    case = {
        "id": "re_test_h01",
        "category": "reasoning",
        "name": "TestH01",
        "user_prompt": "What is 2+2?",
        "judge_method": "regex_match",
    }
    result = handle_validate_case("/api/v15/dataset/validate", {}, {"case": case})
    assert result["status"] == 200
    assert result["body"]["valid"] is True
    assert result["body"]["error"] is None


def test_handle_validate_case_invalid():
    from app.handlers.v15_handlers import handle_validate_case
    case = {
        "id": "bad_case",
        "category": "not_a_category",
        "name": "Bad",
        "user_prompt": "x",
        "judge_method": "regex_match",
    }
    result = handle_validate_case("/api/v15/dataset/validate", {}, {"case": case})
    assert result["status"] == 200
    assert result["body"]["valid"] is False
    assert result["body"]["error"] is not None
