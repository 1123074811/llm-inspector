"""
Seed the database with fixture data on first startup.
- Test cases from suite_v1.json, suite_v2.json, suite_extraction.json
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


def _difficulty_to_irt_b(difficulty: float | None) -> float:
    """
    Convert case difficulty (0-1 probability of failure) to IRT b parameter.
    
    difficulty=0.5 → b=0.0   (50% of models fail = medium difficulty)
    difficulty=0.9 → b=2.20  (90% fail = very hard)
    difficulty=0.3 → b=-0.85 (30% fail = relatively easy)
    
    Uses logit transform: b = log(d / (1-d)), clamped to [-3, 3].
    """
    if difficulty is None:
        return 0.0
    import math as _math
    d = max(0.05, min(0.95, float(difficulty)))  # avoid log(0)
    return round(max(-3.0, min(3.0, _math.log(d / (1.0 - d)))), 4)


def _seed_test_cases() -> None:
    total = 0
    for suite_file in ("suite_v3.json", "suite_v2.json", "suite_v1.json", "suite_extraction.json", "suite_v1.js"):
        suite_path = _FIXTURES / suite_file
        if not suite_path.exists():
            continue
        data = json.loads(suite_path.read_text(encoding='utf-8'))
        cases = data.get("cases", [])
        version = data.get("version", "v1")
        seeded_in_file = 0
        for case in cases:
            case.setdefault("suite_version", version)
            case.setdefault("dimension", case.get("category", "general"))
            case.setdefault("tags", [case.get("category", "general")])
            case.setdefault("judge_rubric", {
                "name": case.get("judge_method", "judge"),
                "criteria": [
                    f"method={case.get('judge_method', 'unknown')}",
                    f"expected_type={case.get('expected_type', 'any')}",
                ],
                "fail_condition": "does not satisfy expected output constraints",
            })
            meta = (case.get("params", {}) or {}).get("_meta", {})
            dimension = meta.get("dimension") or case.get("dimension") or case.get("category", "general")
            anchor = bool(meta.get("anchor", False))
            info_gain_prior = float(meta.get("info_gain_prior", case.get("weight", 1.0)))
            try:
                repo.upsert_test_case(case)
                repo.upsert_item_bank(
                    item_id=case["id"],
                    dimension=dimension,
                    anchor_flag=anchor,
                    active=bool(case.get("enabled", True)),
                    version=version,
                )
                if not repo.get_item_stat(case["id"]):
                    repo.upsert_item_stat(
                        item_id=case["id"],
                        dimension=dimension,
                        a=1.0,
                        b=_difficulty_to_irt_b(case.get("difficulty")),  # Use difficulty from JSON
                        c=None,
                        info_score=info_gain_prior,
                        sample_size=0,
                    )
                seeded_in_file += 1
            except Exception as e:
                logger.warning(f"Failed to seed case {case.get('id')}: {e}")
        total += seeded_in_file
        logger.info(f"Test cases seeded from {suite_file}: {seeded_in_file}/{len(cases)}")
    logger.info(f"Total test cases successfully seeded: {total}")
