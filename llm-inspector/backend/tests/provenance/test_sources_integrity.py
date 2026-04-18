"""
tests/provenance/test_sources_integrity.py

Verifies that SOURCES.yaml is complete, well-formed, and that the
SourcesRegistry singleton serves values correctly.

These tests are fast (pure in-process, no network, no DB) and must pass
before every commit.
"""
from __future__ import annotations

import pathlib
import pytest

# -- Helpers ------------------------------------------------------------------

def _registry():
    from app._data.sources import get_registry, reload_registry
    return reload_registry()  # always fresh from file


# -- Tests --------------------------------------------------------------------

class TestSourcesYamlExists:
    def test_file_exists(self):
        path = pathlib.Path(__file__).parent.parent.parent / "app" / "_data" / "SOURCES.yaml"
        assert path.exists(), f"SOURCES.yaml not found at {path}"

    def test_file_not_empty(self):
        path = pathlib.Path(__file__).parent.parent.parent / "app" / "_data" / "SOURCES.yaml"
        assert path.stat().st_size > 500, "SOURCES.yaml appears to be empty or nearly empty"


class TestRegistryLoads:
    def test_registry_loads_without_error(self):
        reg = _registry()
        assert reg is not None

    def test_minimum_entry_count(self):
        reg = _registry()
        ids = reg.all_ids()
        assert len(ids) >= 20, f"Only {len(ids)} entries; expected >= 20"

    def test_sha256_is_hex_string(self):
        reg = _registry()
        assert len(reg.sha256) == 64
        assert all(c in "0123456789abcdef" for c in reg.sha256)


class TestRequiredIds:
    REQUIRED = [
        "irt.model",
        "irt.theta_mean",
        "irt.theta_sd",
        "cat.sem_stop_threshold",
        "stanine.boundaries",
        "verdict.adv_spoof_cap",
        "verdict.trusted_threshold",
        "verdict.suspicious_threshold",
        "verdict.high_risk_threshold",
        "similarity.match_cosine_threshold",
        "judge.kappa_upgrade_threshold",
        "predetect.confidence_early_stop",
        "scorecard.weight.capability",
        "scorecard.weight.authenticity",
        "scorecard.weight.performance",
    ]

    @pytest.mark.parametrize("src_id", REQUIRED)
    def test_required_id_present(self, src_id: str):
        reg = _registry()
        assert src_id in reg, (
            f"Required id '{src_id}' missing from SOURCES.yaml.\n"
            f"Available ids: {reg.all_ids()}"
        )

    @pytest.mark.parametrize("src_id", REQUIRED)
    def test_required_id_has_source_url(self, src_id: str):
        reg = _registry()
        if src_id not in reg:
            pytest.skip(f"{src_id} missing — covered by test_required_id_present")
        entry = reg[src_id]
        assert entry.source_url.startswith("http"), (
            f"'{src_id}' source_url='{entry.source_url}' doesn't look like a URL"
        )

    @pytest.mark.parametrize("src_id", REQUIRED)
    def test_required_id_has_retrieved_at(self, src_id: str):
        reg = _registry()
        if src_id not in reg:
            pytest.skip(f"{src_id} missing")
        entry = reg[src_id]
        assert entry.retrieved_at, f"'{src_id}' missing retrieved_at"
        # basic ISO date format check
        assert len(entry.retrieved_at) >= 10, (
            f"'{src_id}' retrieved_at='{entry.retrieved_at}' doesn't look like ISO-8601"
        )


class TestValues:
    def test_theta_mean_is_zero(self):
        """v13 uses native logit scale centred at 0."""
        reg = _registry()
        assert reg["irt.theta_mean"].value == 0.0

    def test_theta_sd_is_one(self):
        reg = _registry()
        assert reg["irt.theta_sd"].value == 1.0

    def test_stanine_boundaries_has_eight_values(self):
        """Stanine-9 needs 8 boundary points."""
        reg = _registry()
        bounds = reg["stanine.boundaries"].value
        assert isinstance(bounds, list), "stanine.boundaries must be a list"
        assert len(bounds) == 8, f"Expected 8 boundaries, got {len(bounds)}"

    def test_stanine_boundaries_sorted(self):
        reg = _registry()
        bounds = [float(b) for b in reg["stanine.boundaries"].value]
        assert bounds == sorted(bounds), "stanine.boundaries must be in ascending order"

    def test_confidence_early_stop_in_range(self):
        reg = _registry()
        val = reg["predetect.confidence_early_stop"].value
        assert 0.5 <= float(val) <= 1.0, f"confidence_early_stop={val} out of [0.5, 1.0]"

    def test_kappa_threshold_in_range(self):
        reg = _registry()
        val = reg["judge.kappa_upgrade_threshold"].value
        assert 0.0 <= float(val) <= 1.0, f"kappa={val} not in [0,1]"

    def test_capability_weights_sum_to_one(self):
        reg = _registry()
        keys = [k for k in reg.all_ids()
                if k.startswith("capability.weight.") and k.endswith(".default")]
        assert len(keys) >= 7, f"Expected >= 7 capability weight keys, got {len(keys)}"
        total = sum(float(reg[k].value) for k in keys)
        assert abs(total - 1.0) <= 0.011, (
            f"Capability weights sum to {total:.4f}, not 1.0 (drift={total-1.0:+.4f})"
        )

    def test_scorecard_weights_sum_to_one(self):
        reg = _registry()
        keys = ["scorecard.weight.capability",
                "scorecard.weight.authenticity",
                "scorecard.weight.performance"]
        total = sum(float(reg[k].value) for k in keys)
        assert abs(total - 1.0) <= 0.011, (
            f"Scorecard weights sum to {total:.4f}, not 1.0"
        )

    def test_similarity_thresholds_ordered(self):
        reg = _registry()
        match = float(reg["similarity.match_cosine_threshold"].value)
        suspicious = float(reg["similarity.suspicious_cosine_threshold"].value)
        assert suspicious < match, (
            f"suspicious_threshold ({suspicious}) should be < match_threshold ({match})"
        )


class TestSourceTypes:
    VALID_TYPES = {"paper", "official_doc", "dataset", "empirical", "derived"}

    def test_all_entries_have_valid_source_type(self):
        reg = _registry()
        invalid = [
            eid for eid in reg.all_ids()
            if reg[eid].source_type not in self.VALID_TYPES
        ]
        assert not invalid, (
            f"Entries with invalid source_type: {invalid}\n"
            f"Valid types: {self.VALID_TYPES}"
        )


class TestProvenanceGuard:
    def test_guard_passes_in_warn_mode(self):
        from app._data.provenance_guard import ProvenanceGuard
        guard = ProvenanceGuard()
        report = guard.verify(strict=False)
        # Print for CI logs
        report.print_summary()
        # At this point we expect all checks to pass (fresh registry)
        failures = [c.name for c in report.failures]
        assert not failures, f"ProvenanceGuard failed checks: {failures}"

    def test_guard_passes_in_strict_mode(self):
        from app._data.provenance_guard import ProvenanceGuard
        guard = ProvenanceGuard()
        # strict=True would raise if anything fails — must not raise
        report = guard.verify(strict=True)
        assert report.passed

    def test_missing_id_raises_keyerror(self):
        reg = _registry()
        with pytest.raises(KeyError, match="not found in SOURCES.yaml"):
            _ = reg["this.key.does.not.exist.v13"]

    def test_src_proxy_access(self):
        from app._data import SRC
        entry = SRC["irt.model"]
        assert entry.value == "2PL"
        assert "irt.model" in SRC

    def test_placeholders_are_tagged(self):
        """Phase-2 placeholder entries must be tagged phase2_replace=True."""
        reg = _registry()
        ph = reg.placeholders()
        # We have multiple placeholders registered
        assert len(ph) >= 5, f"Expected >= 5 placeholders, got {len(ph)}: {ph}"
        # All verdict thresholds are placeholders until Phase 2
        assert "verdict.adv_spoof_cap" in ph
        assert "verdict.trusted_threshold" in ph
