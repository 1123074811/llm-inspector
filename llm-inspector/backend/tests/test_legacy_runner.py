"""Compatibility legacy test runner entry.

Phase B introduces layered test execution via pytest markers/files,
while preserving a single legacy-compatible entry point.
"""

from __future__ import annotations

import pytest


def test_legacy_runner_entrypoint_exists():
    """Sanity marker test for compatibility entry."""
    assert True


if __name__ == "__main__":
    raise SystemExit(
        pytest.main([
            "-q",
            "backend/tests/test_all.py",
            "backend/tests/test_v8_regression.py",
            "backend/tests/test_v9_phasea_regression.py",
            "backend/tests/test_v9_phaseb_regression.py",
        ])
    )
