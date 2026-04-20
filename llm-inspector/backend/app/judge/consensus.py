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


def fleiss_kappa(ratings: list[list[int]], n_categories: int = 2) -> float:
    """
    Compute Fleiss's kappa for ≥3 raters applied to N items.

    Reference:
        Fleiss, J.L. (1971). "Measuring nominal scale agreement among many raters."
        Psychological Bulletin, 76(5), 378-382.
        DOI: https://doi.org/10.1037/h0031619

    Args:
        ratings:      List of N items; each element is a list of R binary ratings
                      (0 or 1) from each rater for that item.
        n_categories: Number of rating categories (default 2: pass/fail).

    Returns:
        kappa: float in [-1, 1].
               1  = perfect agreement
               0  = chance-level agreement
              <0  = worse than chance

    Note:
        Requires at least 2 items and 2 raters. Returns 0.0 for degenerate inputs.
    """
    if not ratings or len(ratings) < 2:
        return 0.0

    n_items = len(ratings)
    n_raters = len(ratings[0])

    if n_raters < 2:
        return 0.0

    # Build n_ij matrix: [item i][category j] = count of raters assigning category j to item i
    n_ij: list[list[float]] = []
    for item_ratings in ratings:
        counts = [0.0] * n_categories
        for r in item_ratings:
            idx = int(r)
            if 0 <= idx < n_categories:
                counts[idx] += 1
        n_ij.append(counts)

    # P_bar_j: proportion of all assignments to category j
    p_j: list[float] = [0.0] * n_categories
    for j in range(n_categories):
        p_j[j] = sum(n_ij[i][j] for i in range(n_items)) / (n_items * n_raters)

    # P_i: extent of agreement for item i
    p_i_list: list[float] = []
    for i in range(n_items):
        if n_raters <= 1:
            p_i_list.append(0.0)
            continue
        s = sum(n_ij[i][j] * (n_ij[i][j] - 1) for j in range(n_categories))
        p_i = s / (n_raters * (n_raters - 1))
        p_i_list.append(p_i)

    # P_bar: mean observed agreement
    p_bar = sum(p_i_list) / n_items if n_items > 0 else 0.0

    # P_e_bar: expected agreement by chance
    p_e_bar = sum(p_j[j] ** 2 for j in range(n_categories))

    if abs(1.0 - p_e_bar) < 1e-12:
        # All raters agree with chance — kappa undefined, return 1.0 if p_bar == 1
        return 1.0 if abs(p_bar - 1.0) < 1e-12 else 0.0

    kappa = (p_bar - p_e_bar) / (1.0 - p_e_bar)
    return float(kappa)
