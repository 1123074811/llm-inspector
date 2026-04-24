"""
tests/test_v15_phase0.py — v15 Phase 0 + Phase 4 bug-fix validation.

Covers:
  - _data/version.json exists and has required keys
  - /api/v14/health returns valid JSON with dynamic version info
  - /api/v15/health returns valid JSON with api_version=v15
  - list_case_results() is importable and callable
  - save_pairwise_result is defined exactly once in repo module
  - logging.get_logger() shim re-export works
  - frontend/app.js SSE URL uses v10, not v8
"""
from __future__ import annotations

import importlib
import inspect
import json
import pathlib
import sys

import pytest

# Working-dir-agnostic path to project root
_BACKEND = pathlib.Path(__file__).parent.parent   # llm-inspector/backend/
_FRONTEND = _BACKEND.parent / "frontend"
_DATA_DIR = _BACKEND / "app" / "_data"


# ---------------------------------------------------------------------------
# 1. version.json
# ---------------------------------------------------------------------------

class TestVersionJson:

    def test_file_exists(self):
        vpath = _DATA_DIR / "version.json"
        assert vpath.exists(), f"version.json not found at {vpath}"

    def test_required_keys(self):
        vpath = _DATA_DIR / "version.json"
        data = json.loads(vpath.read_text(encoding="utf-8"))
        for key in ("version", "phases_complete", "built_at"):
            assert key in data, f"Missing key: {key}"

    def test_phases_complete_includes_phase0(self):
        vpath = _DATA_DIR / "version.json"
        data = json.loads(vpath.read_text(encoding="utf-8"))
        assert "phase0" in data["phases_complete"]

    def test_phases_complete_includes_phase4_bugs(self):
        vpath = _DATA_DIR / "version.json"
        data = json.loads(vpath.read_text(encoding="utf-8"))
        assert "phase4-bugs" in data["phases_complete"]


# ---------------------------------------------------------------------------
# 2. Health endpoints — dynamic version info
# ---------------------------------------------------------------------------

class TestHealthEndpoints:

    def test_v14_health_returns_dict(self):
        from app.main import _handle_v14_health
        result = _handle_v14_health("", {}, {})
        # result is (status_code, body_bytes, content_type)
        assert isinstance(result, tuple) and len(result) == 3
        status, body, _ = result
        assert status == 200
        data = json.loads(body)
        assert data["status"] == "ok"
        assert data["api_version"] == "v14"

    def test_v14_health_contains_version(self):
        from app.main import _handle_v14_health
        _, body, _ = _handle_v14_health("", {}, {})
        data = json.loads(body)
        assert "version" in data
        assert data["version"].startswith("v15")

    def test_v14_health_contains_phases_complete(self):
        from app.main import _handle_v14_health
        _, body, _ = _handle_v14_health("", {}, {})
        data = json.loads(body)
        assert "phases_complete" in data
        assert isinstance(data["phases_complete"], list)

    def test_v15_health_registered(self):
        """GET /api/v15/health is registered in ROUTES."""
        from app.main import ROUTES
        patterns = [pat for method, pat, _ in ROUTES if method == "GET"]
        assert any("v15/health" in p for p in patterns), \
            "/api/v15/health not found in ROUTES"

    def test_v15_health_returns_api_version_v15(self):
        from app.main import _handle_v15_health
        result = _handle_v15_health("", {}, {})
        status, body, _ = result
        assert status == 200
        data = json.loads(body)
        assert data["api_version"] == "v15"
        assert data["status"] == "ok"


# ---------------------------------------------------------------------------
# 3. list_case_results in repo
# ---------------------------------------------------------------------------

class TestListCaseResults:

    def test_function_exists(self):
        from app.repository import repo
        assert hasattr(repo, "list_case_results"), \
            "list_case_results not found in repo module"

    def test_function_is_callable(self):
        from app.repository.repo import list_case_results
        assert callable(list_case_results)

    def test_signature_accepts_run_id(self):
        from app.repository.repo import list_case_results
        sig = inspect.signature(list_case_results)
        assert "run_id" in sig.parameters

    def test_returns_list_for_nonexistent_run(self):
        """Should return an empty list for an unknown run_id, not raise."""
        from app.repository.repo import list_case_results
        result = list_case_results("nonexistent-run-id-00000000")
        assert isinstance(result, list)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# 4. save_pairwise_result defined exactly once
# ---------------------------------------------------------------------------

class TestNoDuplicateSavePairwise:

    def test_single_definition(self):
        """repo.py must contain exactly one definition of save_pairwise_result."""
        repo_path = _BACKEND / "app" / "repository" / "repo.py"
        source = repo_path.read_text(encoding="utf-8")
        count = source.count("def save_pairwise_result(")
        assert count == 1, \
            f"Expected 1 definition of save_pairwise_result, found {count}"

    def test_importable(self):
        from app.repository.repo import save_pairwise_result
        assert callable(save_pairwise_result)


# ---------------------------------------------------------------------------
# 5. logging shim re-export
# ---------------------------------------------------------------------------

class TestLoggingShim:

    def test_get_logger_importable(self):
        from app.core.logging import get_logger
        assert callable(get_logger)

    def test_get_logger_returns_structured_logger(self):
        from app.core.logging import get_logger, StructuredLogger
        lg = get_logger("test.shim")
        assert isinstance(lg, StructuredLogger)

    def test_set_sse_publisher_importable(self):
        from app.core.logging import set_sse_publisher
        assert callable(set_sse_publisher)

    def test_setup_logging_importable(self):
        from app.core.logging import setup_logging
        assert callable(setup_logging)

    def test_logger_info_does_not_raise(self):
        from app.core.logging import get_logger
        lg = get_logger("test.shim.info")
        lg.info("test message", key="value")  # should not raise


# ---------------------------------------------------------------------------
# 6. SSE URL in frontend/app.js uses v10
# ---------------------------------------------------------------------------

class TestFrontendSseUrl:

    def _read_appjs(self) -> str:
        appjs = _FRONTEND / "app.js"
        assert appjs.exists(), f"frontend/app.js not found at {appjs}"
        return appjs.read_text(encoding="utf-8")

    def test_sse_uses_v10_not_v8(self):
        src = self._read_appjs()
        assert "/api/v10/runs/" in src and "/logs/stream" in src, \
            "SSE URL should reference /api/v10/runs/.../logs/stream"

    def test_sse_does_not_use_v8_stream(self):
        src = self._read_appjs()
        assert "/api/v8/runs/" not in src or "/stream'" not in src, \
            "Old /api/v8/runs/.../stream URL still present in app.js"
