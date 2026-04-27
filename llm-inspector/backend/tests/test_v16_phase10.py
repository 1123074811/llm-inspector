"""
test_v16_phase10.py — v16 Phase 10 regression tests.

Validates:
  - version.json has all phase entries
  - CHANGELOG.md exists and has v16 section
  - MIGRATION guide exists
  - version string format
"""
import pytest
import pathlib as _pl

_DATA = _pl.Path(__file__).resolve().parent.parent / "app" / "_data"


class TestVersionJson:
    def test_version_starts_with_v16(self):
        import json
        data = json.loads((_DATA / "version.json").read_text(encoding="utf-8"))
        assert data["version"].startswith("v16")

    def test_all_v16_phases_complete(self):
        import json
        data = json.loads((_DATA / "version.json").read_text(encoding="utf-8"))
        phases = data["phases_complete"]
        required = [
            "phase0_v16", "phase1_v16", "phase1.5_v16",
            "phase2_v16", "phase3_v16", "phase4_v16",
            "phase5_v16", "phase6_v16", "phase7_v16",
            "phase8_v16", "phase9_v16", "phase10_v16",
        ]
        for p in required:
            assert p in phases, f"Missing phase: {p}"

    def test_built_at_is_iso(self):
        import json
        data = json.loads((_DATA / "version.json").read_text(encoding="utf-8"))
        assert "T" in data["built_at"]
        assert data["built_at"].endswith("Z")


class TestChangelog:
    def test_changelog_exists(self):
        assert (_DATA / "CHANGELOG.md").exists()

    def test_changelog_has_v16_section(self):
        content = (_DATA / "CHANGELOG.md").read_text(encoding="utf-8")
        assert "v16.0.0" in content

    def test_changelog_has_added_section(self):
        content = (_DATA / "CHANGELOG.md").read_text(encoding="utf-8")
        assert "### Added" in content

    def test_changelog_has_changed_section(self):
        content = (_DATA / "CHANGELOG.md").read_text(encoding="utf-8")
        assert "### Changed" in content

    def test_changelog_has_security_section(self):
        content = (_DATA / "CHANGELOG.md").read_text(encoding="utf-8")
        assert "### Security" in content

    def test_changelog_mentions_phase7(self):
        content = (_DATA / "CHANGELOG.md").read_text(encoding="utf-8")
        assert "Phase 7" in content

    def test_changelog_mentions_phase8(self):
        content = (_DATA / "CHANGELOG.md").read_text(encoding="utf-8")
        assert "Phase 8" in content


class TestMigrationGuide:
    def test_migration_guide_exists(self):
        assert (_DATA / "MIGRATION_v15_to_v16.md").exists()

    def test_migration_mentions_new_endpoints(self):
        content = (_DATA / "MIGRATION_v15_to_v16.md").read_text(encoding="utf-8")
        assert "token-audit" in content
        assert "real-model-card" in content

    def test_migration_mentions_breaking_changes(self):
        content = (_DATA / "MIGRATION_v15_to_v16.md").read_text(encoding="utf-8")
        assert "Breaking Changes" in content
