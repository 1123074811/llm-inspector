"""Judge consensus arbitration utilities (Phase B).

Combines rule-based judge result with semantic judge signal and records
transparent disagreement evidence for auditability.
"""

from __future__ import annotations

from typing import Any

from app.judge.semantic_v2 import semantic_judge_v2


def should_run_consensus(method: str, params: dict[str, Any] | None = None) -> bool:
    """Return whether this case should apply rule-vs-semantic arbitration."""
    p = params or {}
    if p.get("disable_consensus"):
        return False
    if p.get("force_consensus"):
        return True

    # Apply on open-ended / complex methods where rule-only may be brittle
    return method in {
        "constraint_reasoning",
        "multi_step_verify",
        "semantic_judge",
        "semantic_judge_v2",
        "hallucination_detect",
        "hallucination_detect_v2",
    }


def arbitrate_with_semantic(
    *,
    method: str,
    text: str,
    params: dict[str, Any],
    rule_passed: bool | None,
    rule_detail: dict[str, Any] | None,
) -> tuple[bool | None, dict[str, Any]]:
    """Fuse rule and semantic judgments and retain disagreement evidence.

    Policy:
    - If either side is None, keep the available side as final.
    - If both agree, keep that decision.
    - If disagreement:
      - semantic confidence >= 0.80 -> semantic decision wins
      - otherwise rule decision wins
    """
    base_detail = dict(rule_detail or {})

    semantic_passed, semantic_detail = semantic_judge_v2(text, params)
    semantic_conf = float((semantic_detail or {}).get("confidence", 0.5) or 0.5)

    conflict = (
        rule_passed is not None
        and semantic_passed is not None
        and bool(rule_passed) != bool(semantic_passed)
    )

    if rule_passed is None and semantic_passed is None:
        final_passed = None
        arbitration_mode = "no_signal"
        winner = "none"
    elif rule_passed is None:
        final_passed = semantic_passed
        arbitration_mode = "semantic_only"
        winner = "semantic"
    elif semantic_passed is None:
        final_passed = rule_passed
        arbitration_mode = "rule_only"
        winner = "rule"
    elif not conflict:
        final_passed = rule_passed
        arbitration_mode = "agreement"
        winner = "both"
    else:
        if semantic_conf >= 0.80:
            final_passed = semantic_passed
            winner = "semantic"
        else:
            final_passed = rule_passed
            winner = "rule"
        arbitration_mode = "disagreement"

    arbitration = {
        "enabled": True,
        "method": method,
        "mode": arbitration_mode,
        "winner": winner,
        "conflict": conflict,
        "rule": {
            "passed": rule_passed,
            "detail": base_detail,
        },
        "semantic": {
            "passed": semantic_passed,
            "confidence": semantic_conf,
            "detail": semantic_detail,
        },
    }

    merged_detail = dict(base_detail)
    merged_detail["judge_consensus"] = arbitration

    return final_passed, merged_detail
