"""
analysis/judge_calibration.py — v15 Phase 9: Judge reliability monitoring.

Provides:
  - compute_fleiss_kappa(ratings_matrix) — Multi-rater agreement
  - compute_cohen_kappa(rater1, rater2)  — Pairwise agreement
  - judge_bias_detection(semantic_v2_results, rule_results) — Bias monitoring

References:
  Fleiss (1971) "Measuring nominal scale agreement among many raters"
  Cohen (1960) "A coefficient of agreement for nominal scales"
"""
from __future__ import annotations


def compute_fleiss_kappa(ratings_matrix: list[list[int]]) -> float | None:
    """Fleiss' kappa for multi-rater agreement.

    Args:
        ratings_matrix: A 2D list where each row represents a subject/item
            and each column represents a category.  ratings_matrix[i][j]
            is the number of raters who assigned subject i to category j.

    Returns:
        Kappa value (-1 to 1), or None if computation fails.

    Reference: Fleiss (1971) "Measuring nominal scale agreement
               among many raters", Psychometrika.
    """
    if not ratings_matrix or not ratings_matrix[0]:
        return None

    n_subjects = len(ratings_matrix)
    n_categories = len(ratings_matrix[0])

    # Total ratings per subject
    N = sum(ratings_matrix[0])  # assume equal across rows
    if N <= 1:
        return None

    # Proportion of all assignments to each category
    p_j = [0.0] * n_categories
    total_ratings = 0
    for row in ratings_matrix:
        for j, val in enumerate(row):
            p_j[j] += val
            total_ratings += val
    if total_ratings > 0:
        p_j = [p / total_ratings for p in p_j]

    # P_i: proportion of agreeing pairs for each subject
    P_i = []
    for row in ratings_matrix:
        n_ij_sum = sum(v * v for v in row)
        P_i.append((n_ij_sum - N) / (N * (N - 1)) if N > 1 else 0.0)

    P_bar = sum(P_i) / n_subjects if n_subjects > 0 else 0.0
    P_e = sum(p * p for p in p_j)

    if P_e >= 1.0:
        return 1.0  # perfect agreement by chance

    kappa = (P_bar - P_e) / (1.0 - P_e)
    return kappa


def compute_cohen_kappa(rater1: list[int], rater2: list[int]) -> float | None:
    """Cohen's kappa for two raters.

    Args:
        rater1: Binary or categorical ratings from rater 1.
        rater2: Corresponding ratings from rater 2.

    Returns:
        Kappa value, or None if inputs are empty/invalid.

    Reference: Cohen (1960) "A coefficient of agreement for nominal scales".
    """
    if not rater1 or not rater2 or len(rater1) != len(rater2):
        return None

    n = len(rater1)
    # For binary ratings, construct contingency table
    categories = sorted(set(rater1) | set(rater2))
    size = len(categories)
    if size < 2:
        return None

    # Build contingency table
    table = [[0] * size for _ in range(size)]
    for a, b in zip(rater1, rater2):
        i = categories.index(a)
        j = categories.index(b)
        table[i][j] += 1

    # Observed agreement
    po = sum(table[i][i] for i in range(size)) / n

    # Expected agreement by chance
    row_marg = [sum(row) for row in table]
    col_marg = [sum(table[i][j] for i in range(size)) for j in range(size)]
    pe = sum(row_marg[i] * col_marg[i] for i in range(size)) / (n * n)

    if pe >= 1.0:
        return 1.0

    return (po - pe) / (1.0 - pe)


def judge_bias_detection(
    primary_results: list[dict],
    rule_results: list[dict],
    threshold: float = 0.20,
) -> dict:
    """Compare primary judge vs rule-based judge, detect systematic bias.

    Args:
        primary_results: List of {case_id, passed, confidence} from primary judge.
        rule_results: List of {case_id, passed, confidence} from rule-based judge.
        threshold: Maximum allowed disagreement rate before flagging bias.

    Returns:
        dict with bias analysis results.
    """
    if not primary_results or not rule_results:
        return {"bias_detected": False, "disagreement_rate": 0.0, "warnings": []}

    # Build lookup for rule results
    rule_map = {r["case_id"]: r for r in rule_results if "case_id" in r}
    disagreements = []
    total = 0

    for p in primary_results:
        cid = p.get("case_id")
        if cid not in rule_map:
            continue
        r = rule_map[cid]
        if p.get("passed") != r.get("passed"):
            disagreements.append({
                "case_id": cid,
                "primary_passed": p.get("passed"),
                "rule_passed": r.get("passed"),
                "primary_conf": p.get("confidence"),
                "rule_conf": r.get("confidence"),
            })
        total += 1

    d_rate = len(disagreements) / max(total, 1)
    bias_detected = d_rate > threshold

    warnings = []
    if bias_detected:
        warnings.append(
            f"Judge bias detected: {d_rate:.1%} disagreement with rule-based judge "
            f"(threshold={threshold:.0%})"
        )

    # Check direction of bias
    primary_liberal = sum(
        1 for d in disagreements if d["primary_passed"] and not d["rule_passed"]
    )
    primary_conservative = sum(
        1 for d in disagreements if not d["primary_passed"] and d["rule_passed"]
    )

    if primary_liberal > primary_conservative * 2:
        warnings.append("Primary judge appears overly lenient vs rule-based judge")
    elif primary_conservative > primary_liberal * 2:
        warnings.append("Primary judge appears overly strict vs rule-based judge")

    return {
        "bias_detected": bias_detected,
        "disagreement_rate": round(d_rate, 4),
        "total_comparisons": total,
        "disagreements": disagreements[:20],  # cap for transport
        "primary_liberal_count": primary_liberal,
        "primary_conservative_count": primary_conservative,
        "warnings": warnings,
    }
