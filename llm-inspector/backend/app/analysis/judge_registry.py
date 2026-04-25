"""
analysis/judge_registry.py — v15 Phase 12: Judge method registry loader.

Loads _data/judge_registry.yaml and exposes:
  - list_methods()        → list of method name strings
  - get_method(name)      → dict with schema, biases, references, etc.
  - methods_by_mode(mode) → filter by "rule" | "llm" | "nli"
  - applicable_for(qtype) → methods that list a given question type

References:
  - judge_registry.yaml in _data/
"""
from __future__ import annotations

import pathlib
from functools import lru_cache
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)

_REGISTRY_PATH = pathlib.Path(__file__).resolve().parent.parent / "_data" / "judge_registry.yaml"


@lru_cache(maxsize=1)
def _load_registry() -> dict[str, Any]:
    """Load and cache the judge registry from YAML. Returns {} on error."""
    if not _REGISTRY_PATH.exists():
        logger.warning("judge_registry.yaml not found", path=str(_REGISTRY_PATH))
        return {}
    try:
        import yaml  # type: ignore[import]
        with _REGISTRY_PATH.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return data
    except ImportError:
        # PyYAML not installed — fall back to a minimal JSON-compatible loader
        return _load_registry_fallback()
    except Exception as e:
        logger.error("Failed to load judge_registry.yaml", error=str(e))
        return {}


def _load_registry_fallback() -> dict[str, Any]:
    """Minimal YAML loader for key: value + list entries (no PyYAML dependency)."""
    registry: dict[str, Any] = {}
    if not _REGISTRY_PATH.exists():
        return registry
    try:
        current_key: str | None = None
        current_entry: dict[str, Any] = {}
        with _REGISTRY_PATH.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.rstrip()
                if not stripped or stripped.startswith("#"):
                    continue
                if not stripped.startswith(" ") and stripped.endswith(":"):
                    if current_key:
                        registry[current_key] = current_entry
                    current_key = stripped[:-1].strip()
                    current_entry = {}
                elif stripped.strip().startswith("name:") and current_key:
                    current_entry["name"] = stripped.split("name:", 1)[1].strip().strip('"')
                elif stripped.strip().startswith("mode:") and current_key:
                    current_entry["mode"] = stripped.split("mode:", 1)[1].strip().strip('"')
                elif stripped.strip().startswith("description:") and current_key:
                    current_entry["description"] = stripped.split("description:", 1)[1].strip().strip('"')
        if current_key:
            registry[current_key] = current_entry
    except Exception as e:
        logger.error("Fallback YAML load failed", error=str(e))
    return registry


def _registry() -> dict[str, Any]:
    """Return the registry, re-loading if empty (for test isolation)."""
    return _load_registry()


def list_methods() -> list[str]:
    """Return all registered judge method names."""
    return list(_registry().keys())


def get_method(name: str) -> dict[str, Any] | None:
    """Return the registry entry for a judge method, or None if not found."""
    return _registry().get(name)


def methods_by_mode(mode: str) -> list[str]:
    """Return method names whose 'mode' field equals the given value.

    Args:
        mode: "rule" | "llm" | "nli"

    Returns:
        List of matching method names.
    """
    return [
        name for name, entry in _registry().items()
        if isinstance(entry, dict) and entry.get("mode") == mode
    ]


def applicable_for(question_type: str) -> list[str]:
    """Return method names that list *question_type* in their applicable_types.

    Args:
        question_type: e.g. "multiple_choice", "open_ended", "math"

    Returns:
        List of matching method names.
    """
    result: list[str] = []
    for name, entry in _registry().items():
        if not isinstance(entry, dict):
            continue
        types = entry.get("applicable_types") or []
        if question_type in types:
            result.append(name)
    return result


def registry_summary() -> dict[str, Any]:
    """Return a lightweight summary of the registry for the API."""
    reg = _registry()
    by_mode: dict[str, list[str]] = {}
    for name, entry in reg.items():
        if not isinstance(entry, dict):
            continue
        mode = entry.get("mode", "unknown")
        by_mode.setdefault(mode, []).append(name)

    return {
        "total": len(reg),
        "by_mode": by_mode,
        "methods": {
            name: {
                "name": entry.get("name", name),
                "mode": entry.get("mode", "unknown"),
                "description": entry.get("description", ""),
            }
            for name, entry in reg.items()
            if isinstance(entry, dict)
        },
    }
