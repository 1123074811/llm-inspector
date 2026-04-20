"""
judge/numeric_tolerance.py — Numeric answer tolerance judge.

Parses numeric values from model responses and checks whether they fall
within a configurable relative (or absolute) tolerance of the expected answer.

Reference:
    NIST Special Publication 330 (2019) — SI units and relative uncertainty.
    URL: https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.330-2019.pdf
    Registered as SRC["judge.numeric_tolerance.relative_threshold"]
"""
from __future__ import annotations

import re

from app.core.logging import get_logger

logger = get_logger(__name__)

# -- Constants (NIST SP 330-2019 relative uncertainty guidance) ----------------

# Default relative tolerance: 5% (SRC["judge.numeric_tolerance.relative_threshold"])
_DEFAULT_TOLERANCE = 0.05

# Default absolute tolerance for near-zero expected values
# (SRC["judge.numeric_tolerance.absolute_threshold"])
_DEFAULT_ABSOLUTE_TOLERANCE = 1e-6

# Regex to capture the first numeric value in a response.
# Handles: scientific notation (3.14e-5), comma thousands-sep (1,000), % suffix, units.
_NUM_PATTERN = re.compile(
    r"""
    (?<![a-zA-Z])               # not preceded by a letter (avoid word-embedded numbers)
    (-?)                         # optional sign
    (\d{1,3}(?:,\d{3})*         # integer with optional comma thousands separators
        |\d+)                    # OR plain integer
    (?:\.(\d+))?                 # optional decimal part
    (?:[eE]([+-]?\d+))?         # optional exponent
    \s*(%)?                      # optional percent symbol
    """,
    re.VERBOSE,
)


def _parse_number(text: str) -> float | None:
    """
    Extract the first numeric value from *text*.

    Handles:
    - Scientific notation: ``3.14e-5`` → 3.14e-5
    - Comma separators: ``1,234.56`` → 1234.56
    - Percentage: ``42.5%`` → 0.425
    - Trailing units: ``9.8 m/s²`` → 9.8 (units stripped before matching)

    Returns:
        Parsed float or None if no numeric value could be extracted.
    """
    if not text:
        return None

    text = text.strip()
    m = _NUM_PATTERN.search(text)
    if m is None:
        return None

    sign_str, integer_part, frac_part, exp_part, pct = m.groups()

    # Remove comma separators
    integer_part = integer_part.replace(",", "")

    # Reconstruct numeric string
    num_str = integer_part
    if frac_part:
        num_str += "." + frac_part
    if exp_part:
        num_str += "e" + exp_part

    sign = -1.0 if sign_str == "-" else 1.0

    try:
        value = sign * float(num_str)
    except ValueError:
        return None

    if pct:
        value /= 100.0  # percentage → decimal

    return value


def numeric_tolerance_judge(
    response: str,
    params: dict,
) -> tuple[bool, dict]:
    """
    Numeric answer tolerance judge.

    Checks whether the first numeric value parsed from *response* is within
    the specified tolerance of *expected*.

    Args:
        response: Model response text.
        params:
            expected         — expected numeric value (float or str, required)
            tolerance        — relative tolerance (default 0.05 = 5%)
                               (SRC["judge.numeric_tolerance.relative_threshold"])
            absolute_tolerance — absolute tolerance for near-zero values
                               (default 1e-6)
                               (SRC["judge.numeric_tolerance.absolute_threshold"])

    Returns:
        (passed, detail_dict)
    """
    raw_expected = params.get("expected")
    if raw_expected is None:
        return False, {
            "method": "numeric_tolerance",
            "error": "missing_expected",
            "note": "params['expected'] is required",
        }

    try:
        expected = float(raw_expected)
    except (TypeError, ValueError):
        return False, {
            "method": "numeric_tolerance",
            "error": "invalid_expected",
            "expected_raw": str(raw_expected),
        }

    tolerance = float(params.get("tolerance", _DEFAULT_TOLERANCE))
    absolute_tolerance = float(
        params.get("absolute_tolerance", _DEFAULT_ABSOLUTE_TOLERANCE)
    )

    parsed = _parse_number(response)
    if parsed is None:
        return False, {
            "method": "numeric_tolerance",
            "error": "parse_failed",
            "response_excerpt": response[:80],
            "expected": expected,
            "tolerance": tolerance,
        }

    # Choose tolerance mode based on magnitude of expected value
    # (NIST SP 330-2019: absolute uncertainty for near-zero quantities)
    if abs(expected) < 1e-9:
        tolerance_type = "absolute"
        error_val = abs(parsed - expected)
        passed = error_val <= absolute_tolerance
    else:
        tolerance_type = "relative"
        error_val = abs(parsed - expected) / abs(expected)
        passed = error_val <= tolerance

    return passed, {
        "method": "numeric_tolerance",
        "parsed": parsed,
        "expected": expected,
        "error": error_val,
        "tolerance_type": tolerance_type,
        "tolerance": absolute_tolerance if tolerance_type == "absolute" else tolerance,
        "passed": passed,
    }
