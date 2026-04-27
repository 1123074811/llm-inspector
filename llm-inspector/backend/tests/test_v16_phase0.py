"""
test_v16_phase0.py — v16 Phase 0 regression tests.

Validates code audit & cleanup changes:
  - version.json updated to v16
  - config.py SUITE_VERSION default = v16
  - start.bat / start.sh version strings
  - Dependency check includes certifi/tiktoken
"""
import pathlib
import pytest

_BACKEND = pathlib.Path(__file__).resolve().parent.parent


class TestVersionJson:
    def test_version_is_v16(self):
        import json
        vp = _BACKEND / "app" / "_data" / "version.json"
        data = json.loads(vp.read_text(encoding="utf-8"))
        assert data["version"].startswith("v16"), f"Expected v16*, got {data['version']}"

    def test_phase0_v16_in_phases_complete(self):
        import json
        vp = _BACKEND / "app" / "_data" / "version.json"
        data = json.loads(vp.read_text(encoding="utf-8"))
        assert "phase0_v16" in data["phases_complete"]


class TestConfigV16:
    def test_suite_version_default(self):
        from app.core.config import Settings
        s = Settings()
        assert s.SUITE_VERSION == "v16"

    def test_preflight_config_exists(self):
        from app.core.config import Settings
        s = Settings()
        assert hasattr(s, "PREFLIGHT_TIMEOUT_S")
        assert hasattr(s, "PREFLIGHT_VERIFY_SSL")

    def test_official_endpoint_config_exists(self):
        from app.core.config import Settings
        s = Settings()
        assert hasattr(s, "OFFICIAL_ENDPOINT_ENABLED")
        assert hasattr(s, "OFFICIAL_ENDPOINT_REGISTRY")
        assert s.OFFICIAL_ENDPOINT_MIN_CONFIDENCE > 0

    def test_retry_config_v16(self):
        from app.core.config import Settings
        s = Settings()
        assert hasattr(s, "RETRY_MAX_5XX")
        assert hasattr(s, "RETRY_MAX_TRUNCATION")

    def test_weights_file_config(self):
        from app.core.config import Settings
        s = Settings()
        assert "v16" in s.WEIGHTS_FILE


class TestStartupScripts:
    def test_bat_version_v16(self):
        bp = _BACKEND.parent / "start.bat"
        if bp.exists():
            content = bp.read_text(encoding="utf-8")
            assert "v16.0" in content

    def test_sh_version_v16(self):
        sp = _BACKEND.parent / "start.sh"
        if sp.exists():
            content = sp.read_text(encoding="utf-8")
            assert "v16.0" in content

    def test_bat_dep_check_includes_tiktoken(self):
        bp = _BACKEND.parent / "start.bat"
        if bp.exists():
            content = bp.read_text(encoding="utf-8")
            assert "tiktoken" in content

    def test_sh_dep_check_includes_tiktoken(self):
        sp = _BACKEND.parent / "start.sh"
        if sp.exists():
            content = sp.read_text(encoding="utf-8")
            assert "tiktoken" in content
