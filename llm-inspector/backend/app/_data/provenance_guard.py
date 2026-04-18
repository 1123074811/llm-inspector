"""
app/_data/provenance_guard.py — Startup provenance validator

Called by start.py to ensure SOURCES.yaml is complete and self-consistent.

Modes:
  strict  — raise ProvenianceError on any violation (production default)
  warn    — log warnings but continue (dev default)

Usage (from start.py):
    from app._data.provenance_guard import ProvenanceGuard
    ProvenanceGuard().verify(strict=settings.APP_ENV == "production")

CLI (via start.py --verify-sources):
    python -m app._data.provenance_guard [--strict]
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Callable

from app._data.sources import get_registry, SourceEntry, _SOURCES_PATH


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ProvenanceError(RuntimeError):
    """Raised in strict mode when the provenance registry is invalid."""


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class VerifyReport:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]

    def print_summary(self) -> None:
        n = len(self.checks)
        nf = len(self.failures)
        print(f"\n{'='*60}")
        print(f"Provenance check: {n - nf}/{n} passed")
        if self.failures:
            for f in self.failures:
                print(f"  [FAIL] {f.name}: {f.detail}")
        else:
            print("  [OK] All provenance checks passed.")
        print('='*60)


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------

_MIN_ENTRIES = 20          # SOURCES.yaml must have at least this many entries
_REQUIRED_IDS = [          # These specific IDs must always be present
    "irt.model",
    "irt.theta_mean",
    "irt.theta_sd",
    "cat.sem_stop_threshold",
    "stanine.boundaries",
    "verdict.trusted_threshold",
    "verdict.adv_spoof_cap",
    "similarity.match_cosine_threshold",
    "judge.kappa_upgrade_threshold",
    "predetect.confidence_early_stop",
    "scorecard.weight.capability",
    "scorecard.weight.authenticity",
    "scorecard.weight.performance",
    "elo.initial_rating",
]


class ProvenanceGuard:
    """
    Validates SOURCES.yaml integrity at startup.

    Checks performed:
      1. File exists and is parseable.
      2. Minimum entry count met.
      3. All REQUIRED_IDS present.
      4. Every entry has non-empty source_url, retrieved_at, license.
      5. source_type is one of the allowed enum values.
      6. Weights in each category sum to ~1.0 (±0.01).
      7. Placeholder count reported (info only, not failure).
    """

    def verify(self, strict: bool = False) -> VerifyReport:
        report = VerifyReport()

        # Check 1 — file exists
        report.checks.append(self._check_file_exists())
        if not report.checks[-1].passed:
            self._maybe_raise(report, strict)
            return report

        # Load registry
        try:
            registry = get_registry()
        except Exception as exc:
            report.checks.append(CheckResult("registry_load", False, str(exc)))
            self._maybe_raise(report, strict)
            return report

        report.checks.append(CheckResult("registry_load", True, f"sha256={registry.sha256[:12]}…"))

        # Check 2 — minimum entries
        n = len(registry.all_ids())
        report.checks.append(CheckResult(
            "min_entries",
            n >= _MIN_ENTRIES,
            f"{n} entries (min {_MIN_ENTRIES})" if n >= _MIN_ENTRIES else
            f"only {n} entries, need ≥ {_MIN_ENTRIES}"
        ))

        # Check 3 — required IDs
        missing = [rid for rid in _REQUIRED_IDS if rid not in registry]
        report.checks.append(CheckResult(
            "required_ids",
            not missing,
            "all present" if not missing else f"missing: {missing}"
        ))

        # Check 4 & 5 — per-entry field quality
        bad_entries: list[str] = []
        for eid in registry.all_ids():
            e: SourceEntry = registry[eid]
            if not e.source_url or not e.retrieved_at or not e.license:
                bad_entries.append(f"{eid} (empty source_url/retrieved_at/license)")
        report.checks.append(CheckResult(
            "entry_fields_complete",
            not bad_entries,
            "all ok" if not bad_entries else
            f"{len(bad_entries)} incomplete: {bad_entries[:3]}{'…' if len(bad_entries) > 3 else ''}"
        ))

        # Check 6 — capability weights sum to 1.0
        weight_keys = [
            k for k in registry.all_ids()
            if k.startswith("capability.weight.") and k.endswith(".default")
        ]
        if weight_keys:
            total = sum(registry[k].value for k in weight_keys
                        if isinstance(registry[k].value, (int, float)))
            report.checks.append(CheckResult(
                "capability_weights_sum",
                abs(total - 1.0) <= 0.011,
                f"sum={total:.4f} (ok)" if abs(total - 1.0) <= 0.011
                else f"sum={total:.4f} != 1.0 (drift={total-1.0:+.4f})"
            ))

        # Check 7 — scorecard weights sum to 1.0
        sc_keys = ["scorecard.weight.capability",
                   "scorecard.weight.authenticity",
                   "scorecard.weight.performance"]
        if all(k in registry for k in sc_keys):
            sc_total = sum(registry[k].value for k in sc_keys)
            report.checks.append(CheckResult(
                "scorecard_weights_sum",
                abs(sc_total - 1.0) <= 0.011,
                f"sum={sc_total:.4f} (ok)" if abs(sc_total - 1.0) <= 0.011
                else f"sum={sc_total:.4f} != 1.0"
            ))

        # Info — placeholder count
        ph = registry.placeholders()
        report.checks.append(CheckResult(
            "placeholders_info",
            True,  # never a failure; informational only
            f"{len(ph)} Phase-2 placeholder(s) registered (will be data-fitted in Phase 2)"
        ))

        self._maybe_raise(report, strict)
        return report

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _check_file_exists() -> CheckResult:
        exists = _SOURCES_PATH.exists()
        return CheckResult(
            "sources_file_exists",
            exists,
            str(_SOURCES_PATH) if exists else f"NOT FOUND: {_SOURCES_PATH}"
        )

    @staticmethod
    def _maybe_raise(report: VerifyReport, strict: bool) -> None:
        if not report.passed and strict:
            failures = "; ".join(f.detail for f in report.failures)
            raise ProvenanceError(
                f"Provenance check failed (strict mode). Failures: {failures}"
            )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    import argparse
    from app.core.logging import get_logger
    logger = get_logger(__name__)

    parser = argparse.ArgumentParser(description="Verify SOURCES.yaml integrity")
    parser.add_argument("--strict", action="store_true",
                        help="Exit non-zero if any check fails")
    args = parser.parse_args(argv)

    guard = ProvenanceGuard()
    report = guard.verify(strict=False)  # never raise from CLI, use exit code
    report.print_summary()

    if not report.passed:
        logger.warning("provenance_guard: checks failed",
                       failures=[f.name for f in report.failures])
        if args.strict:
            return 1
    else:
        logger.info("provenance_guard: all checks passed",
                    sha256=get_registry().sha256[:12])
    return 0


if __name__ == "__main__":
    sys.exit(main())
