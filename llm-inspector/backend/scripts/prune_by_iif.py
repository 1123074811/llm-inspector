"""
scripts/prune_by_iif.py — Mark low-discrimination cases in a test suite.

Usage:
    python prune_by_iif.py [--suite suite_v13.json] [--iif-threshold 0.6]
                           [--pass-rate-min 0.05] [--pass-rate-max 0.95]
                           [--theta 0.0] [--dry-run] [--output OUTPUT]

What it does:
- For each case: compute I_max using the 2PL IIF formula at theta* ≈ b (c=0 approx)
    I(θ) = a² · P(θ) · (1 - P(θ))
    P(θ) = c + (1-c) / (1 + exp(-a·(θ-b)))    [3PL]
- Mark cases where: IIF < threshold OR pass_rate outside (pass_rate_min, pass_rate_max)
- Output: pruning report (JSON) + modified suite JSON with
    discriminative_valid=False on weak cases inside params._meta
- Never deletes cases — only marks them

Peak theta for maximum information:
    θ* ≈ b + ln((1 + sqrt(1 + 8c²)) / (4c)) / a   (3PL exact)
    θ* = b                                           (c=0 special case, 2PL)

Exit codes:
    0 — success (even in dry-run mode)
    1 — input file not found or invalid JSON
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path defaults
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_FIXTURES_DIR = _SCRIPT_DIR.parent / "app" / "fixtures"
_DEFAULT_SUITE = _FIXTURES_DIR / "suite_v13.json"


# ---------------------------------------------------------------------------
# IRT helpers
# ---------------------------------------------------------------------------

def _p_3pl(theta: float, a: float, b: float, c: float) -> float:
    """3PL item response function: P(theta | a, b, c)."""
    # clamp exponent to avoid overflow
    exponent = max(-500.0, min(500.0, -a * (theta - b)))
    return c + (1.0 - c) / (1.0 + math.exp(exponent))


def _iif_3pl(theta: float, a: float, b: float, c: float) -> float:
    """Item Information Function at theta for 3PL model.
    I(theta) = a^2 * [P(theta) - c]^2 / [(1-c)^2 * P(theta) * (1-P(theta))]
    which simplifies for the non-trivial case. Using the common formula:
    I(theta) = a^2 * (1 - P) * (P - c)^2 / ((1-c)^2 * P)
    """
    p = _p_3pl(theta, a, b, c)
    q = 1.0 - p
    if p <= 0.0 or q <= 0.0:
        return 0.0
    if c >= 1.0:
        return 0.0
    numerator = (p - c) ** 2
    denominator = (1.0 - c) ** 2 * p * q
    if denominator <= 0.0:
        return 0.0
    return a * a * numerator / denominator


def _peak_theta(a: float, b: float, c: float) -> float:
    """Approximate theta at which IIF is maximised (3PL peak).

    For c > 0:
        theta* = b + ln((1 + sqrt(1 + 8*c^2)) / (4*c)) / a
    For c == 0 (2PL):
        theta* = b
    """
    if c <= 0.0 or a <= 0.0:
        return b
    inner = (1.0 + math.sqrt(1.0 + 8.0 * c * c)) / (4.0 * c)
    if inner <= 0.0:
        return b
    return b + math.log(inner) / a


def _iif_at_peak(a: float, b: float, c: float) -> float:
    """Maximum IIF value (at peak theta)."""
    theta_star = _peak_theta(a, b, c)
    return _iif_3pl(theta_star, a, b, c)


# ---------------------------------------------------------------------------
# Pruning logic
# ---------------------------------------------------------------------------

_INVALID_REASONS = {
    "low_discrimination": "IRT a-parameter < 0.5 — item does not differentiate ability levels",
    "near_zero_information": "Peak IIF below threshold — item contributes little information",
    "ceiling_effect": "Estimated pass rate > 95% — item is too easy to discriminate",
    "floor_effect": "Estimated pass rate < 5% — item is too hard to discriminate",
}


def _estimate_pass_rate(a: float, b: float, c: float, ability_mean: float = 0.0) -> float:
    """Estimate population pass rate using P at the ability mean."""
    return _p_3pl(ability_mean, a, b, c)


def _assess_case(
    case: dict[str, Any],
    iif_threshold: float,
    pass_rate_min: float,
    pass_rate_max: float,
    theta: float,
) -> tuple[bool, list[str]]:
    """Return (is_valid, reasons_for_invalidity)."""
    a = float(case.get("irt_a", 1.0))
    b = float(case.get("irt_b", 0.0))
    c = float(case.get("irt_c", 0.25))

    reasons: list[str] = []

    # 1. Low discrimination
    if a < 0.5:
        reasons.append("low_discrimination")

    # 2. Near-zero information
    peak_iif = _iif_at_peak(a, b, c)
    if peak_iif < iif_threshold:
        reasons.append("near_zero_information")

    # 3. Ceiling / floor effects
    pass_rate = _estimate_pass_rate(a, b, c, ability_mean=theta)
    if pass_rate > pass_rate_max:
        reasons.append("ceiling_effect")
    elif pass_rate < pass_rate_min:
        reasons.append("floor_effect")

    return len(reasons) == 0, reasons


def prune_suite(
    suite: dict[str, Any],
    iif_threshold: float = 0.6,
    pass_rate_min: float = 0.05,
    pass_rate_max: float = 0.95,
    theta: float = 0.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (modified_suite, pruning_report).

    The suite is deep-copied; cases are never deleted, only marked.
    """
    suite_out = copy.deepcopy(suite)
    cases = suite_out.get("cases", [])

    report_cases: list[dict[str, Any]] = []
    n_marked = 0
    n_valid = 0
    total_irt_a = 0.0

    for case in cases:
        a = float(case.get("irt_a", 1.0))
        b = float(case.get("irt_b", 0.0))
        c = float(case.get("irt_c", 0.25))

        is_valid, reasons = _assess_case(
            case, iif_threshold, pass_rate_min, pass_rate_max, theta
        )

        peak_iif = _iif_at_peak(a, b, c)
        pass_rate = _estimate_pass_rate(a, b, c, ability_mean=theta)

        # Write marker into params._meta
        meta = case.setdefault("params", {}).setdefault("_meta", {})
        meta["discriminative_valid"] = is_valid
        if not is_valid:
            meta["discriminative_invalid_reasons"] = reasons

        total_irt_a += a
        if is_valid:
            n_valid += 1
        else:
            n_marked += 1

        report_cases.append({
            "id": case.get("id", "?"),
            "category": case.get("category", "?"),
            "irt_a": a,
            "irt_b": b,
            "irt_c": c,
            "peak_iif": round(peak_iif, 4),
            "pass_rate_at_theta": round(pass_rate, 4),
            "discriminative_valid": is_valid,
            "reasons": reasons,
        })

    n_total = len(cases)
    mean_irt_a = total_irt_a / n_total if n_total > 0 else 0.0

    # Estimate token savings: marked cases would be skipped in pruned mode
    # Assume average ~100 tokens per case (prompt + response overhead)
    avg_tokens_per_case = 100
    estimated_token_saving_pct = (
        round(100.0 * n_marked / n_total, 1) if n_total > 0 else 0.0
    )

    pruning_report = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "suite_version": suite.get("version", "?"),
        "parameters": {
            "iif_threshold": iif_threshold,
            "pass_rate_min": pass_rate_min,
            "pass_rate_max": pass_rate_max,
            "theta": theta,
        },
        "summary": {
            "total_cases": n_total,
            "valid_cases": n_valid,
            "marked_invalid": n_marked,
            "mean_irt_a": round(mean_irt_a, 3),
            "estimated_token_saving_pct": estimated_token_saving_pct,
        },
        "invalid_case_ids": [r["id"] for r in report_cases if not r["discriminative_valid"]],
        "cases": report_cases,
    }

    return suite_out, pruning_report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_suite(path: Path) -> dict[str, Any]:
    if not path.exists():
        print(f"[prune_by_iif] ERROR: Suite file not found: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        print(f"[prune_by_iif] ERROR: Invalid JSON in {path}: {exc}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mark low-discrimination test cases using IIF analysis."
    )
    parser.add_argument(
        "--suite",
        type=Path,
        default=_DEFAULT_SUITE,
        help=f"Input suite JSON (default: {_DEFAULT_SUITE})",
    )
    parser.add_argument(
        "--iif-threshold",
        type=float,
        default=0.6,
        metavar="FLOAT",
        help="Minimum acceptable peak IIF value (default: 0.6)",
    )
    parser.add_argument(
        "--pass-rate-min",
        type=float,
        default=0.05,
        metavar="FLOAT",
        help="Minimum acceptable pass rate — items below this are floor-effect (default: 0.05)",
    )
    parser.add_argument(
        "--pass-rate-max",
        type=float,
        default=0.95,
        metavar="FLOAT",
        help="Maximum acceptable pass rate — items above this are ceiling-effect (default: 0.95)",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=0.0,
        metavar="FLOAT",
        help="Target ability level theta for pass-rate estimation (default: 0.0 = population mean)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print the report without writing any files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for the pruned suite JSON (default: overwrites --suite input).",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=None,
        help="Output path for the pruning report JSON (default: <suite_stem>_pruning_report.json).",
    )
    args = parser.parse_args()

    suite = _load_suite(args.suite)

    suite_out, report = prune_suite(
        suite,
        iif_threshold=args.iif_threshold,
        pass_rate_min=args.pass_rate_min,
        pass_rate_max=args.pass_rate_max,
        theta=args.theta,
    )

    summary = report["summary"]
    print(
        f"[prune_by_iif] Suite: {args.suite.name}  |  "
        f"Total: {summary['total_cases']}  |  "
        f"Valid: {summary['valid_cases']}  |  "
        f"Marked: {summary['marked_invalid']}  |  "
        f"Est. token saving: {summary['estimated_token_saving_pct']}%",
        flush=True,
    )

    if report["invalid_case_ids"]:
        print("[prune_by_iif] Marked cases:")
        for cid in report["invalid_case_ids"]:
            reasons = next(
                (r["reasons"] for r in report["cases"] if r["id"] == cid), []
            )
            reason_strs = ", ".join(_INVALID_REASONS.get(r, r) for r in reasons)
            print(f"  - {cid}: {reason_strs}")
    else:
        print("[prune_by_iif] No cases marked — all pass IIF and pass-rate checks.")

    if args.dry_run:
        print("[prune_by_iif] Dry-run mode: no files written.")
        return

    # Determine output paths
    suite_output = args.output if args.output else args.suite
    if args.report_output:
        report_output = args.report_output
    else:
        report_output = args.suite.parent / (args.suite.stem + "_pruning_report.json")

    # Write modified suite
    suite_output.parent.mkdir(parents=True, exist_ok=True)
    with open(suite_output, "w", encoding="utf-8") as fh:
        json.dump(suite_out, fh, ensure_ascii=False, indent=2)
    print(f"[prune_by_iif] Wrote modified suite to: {suite_output}", flush=True)

    # Write report
    with open(report_output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f"[prune_by_iif] Wrote pruning report to: {report_output}", flush=True)


if __name__ == "__main__":
    main()
