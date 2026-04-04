"""
Seed the database with fixture data on first startup.
- Test cases from suite_v1.json
- Benchmark profiles from benchmarks/default_profiles.json
"""
from __future__ import annotations

import json
import pathlib
from app.core.logging import get_logger
from app.repository import repo

logger = get_logger(__name__)

_FIXTURES = pathlib.Path(__file__).parent.parent / "fixtures"


def seed_all() -> None:
    _seed_test_cases()
    _seed_benchmarks()


def _seed_test_cases() -> None:
    suite_path = _FIXTURES / "suite_v1.json"
    if not suite_path.exists():
        logger.warning("suite_v1.json not found, skipping seed")
        return
    data = json.loads(suite_path.read_text(encoding='utf-8'))
    cases = data.get("cases", [])
    for case in cases:
        try:
            repo.upsert_test_case(case)
        except Exception as e:
            logger.warning(f"Failed to seed case {case.get('id')}: {e}")
    logger.info(f"Test cases seeded: {len(cases)}")


def _seed_benchmarks() -> None:
    bp_path = _FIXTURES / "benchmarks" / "default_profiles.json"
    if not bp_path.exists():
        logger.warning("default_profiles.json not found, skipping seed")
        return
    data = json.loads(bp_path.read_text(encoding='utf-8'))
    benchmarks = data.get("benchmarks", [])
    for b in benchmarks:
        try:
            repo.upsert_benchmark(
                name=b["name"],
                suite_version=b["suite_version"],
                feature_vector=b["feature_vector"],
                sample_count=b.get("sample_count", 3),
            )
        except Exception as e:
            logger.warning(f"Failed to seed benchmark {b.get('name')}: {e}")
    logger.info(f"Benchmarks seeded: {len(benchmarks)}")
