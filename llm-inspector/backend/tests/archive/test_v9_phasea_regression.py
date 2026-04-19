"""v9 Phase A regression tests.

Coverage:
1) Unified ScoreCardCalculator export path
2) Metric provenance registry validation behavior
3) Strict provenance environment check integration
"""

from __future__ import annotations

import importlib

import pytest


def test_scorecardcalculator_export_is_unified():
    """`app.analysis.ScoreCardCalculator` should point to score_calculator module."""
    from app.analysis import ScoreCardCalculator

    assert ScoreCardCalculator.__module__ == "app.analysis.score_calculator"


def test_metric_registry_validation_passes_default():
    """Default registry should pass required checks."""
    from app.analysis.metric_registry import validate_required_metric_sources

    issues = validate_required_metric_sources(strict=False)
    assert issues == []


def test_metric_registry_strict_raises_on_missing(monkeypatch):
    """Strict mode should raise when required metric registration is missing."""
    from app.analysis import metric_registry

    original = dict(metric_registry.METRIC_SOURCES)
    try:
        monkeypatch.setattr(metric_registry, "METRIC_SOURCES", {
            k: v for k, v in original.items() if k != "VERDICT_TRUSTED_THRESHOLD"
        })

        with pytest.raises(ValueError):
            metric_registry.validate_required_metric_sources(strict=True)
    finally:
        monkeypatch.setattr(metric_registry, "METRIC_SOURCES", original)


def test_start_check_environment_strict_provenance(monkeypatch):
    """Startup env check should fail in strict mode if provenance is invalid."""
    start = importlib.import_module("start")
    metric_registry = importlib.import_module("app.analysis.metric_registry")

    original = dict(metric_registry.METRIC_SOURCES)
    try:
        monkeypatch.setattr(metric_registry, "METRIC_SOURCES", {
            k: v for k, v in original.items() if k != "PREDETECT_CONFIDENCE_THRESHOLD"
        })
        ok = start.check_environment(strict_provenance=True)
        assert ok is False
    finally:
        monkeypatch.setattr(metric_registry, "METRIC_SOURCES", original)


def test_start_check_environment_non_strict_allows_warning(monkeypatch):
    """Non-strict mode should continue even when provenance has issues."""
    start = importlib.import_module("start")
    metric_registry = importlib.import_module("app.analysis.metric_registry")

    original = dict(metric_registry.METRIC_SOURCES)
    try:
        monkeypatch.setattr(metric_registry, "METRIC_SOURCES", {
            k: v for k, v in original.items() if k != "PREDETECT_CONFIDENCE_THRESHOLD"
        })
        ok = start.check_environment(strict_provenance=False)
        assert ok is True
    finally:
        monkeypatch.setattr(metric_registry, "METRIC_SOURCES", original)
