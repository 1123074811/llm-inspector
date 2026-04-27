"""
tasks/pruner_job.py — v17 Phase 10: automatic suite pruning + persistence.

What it does
------------

  1. Compute observed pass-rate for every case in ``test_cases`` using
     responses from baseline-eligible models only (Phase 11 query).
  2. Feed those metrics into ``analysis.suite_pruner.SuitePruner`` to
     get IRT-aware quality flags (ceiling / floor / low_discrimination /
     near_zero_information).
  3. Persist results to ``case_quality_flags`` so that CAT-style
     selection (and any future selector) can skip non-discriminative
     items without rerunning the analysis.
  4. Emit a "suite_exhaustion_warning" SSE event whenever a model
     family's strongest member has driven pass-rate above
     ``EXHAUSTION_PASS_RATE_THRESHOLD`` over the recent window.

Public API
----------

  run_pruner_job(now=None) -> PrunerJobReport

The function is best-effort and never raises; database access lives on
the same thread-local connection used by the rest of the repository.

CLI
---
  python -m app.tasks.pruner_job --once
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from typing import Any

from app.core.logging import get_logger
from app.core.db import get_conn
from app.analysis.suite_pruner import SuitePruner, CaseQualityMetrics
from app.repository import registry_repo

logger = get_logger(__name__)


# ── Tunables ────────────────────────────────────────────────────────────────


EXHAUSTION_PASS_RATE_THRESHOLD = 0.95
EXHAUSTION_MIN_SAMPLES = 20
RECENT_WINDOW_DAYS = 30


# ── Aggregation helpers ─────────────────────────────────────────────────────


def _eligible_baseline_model_ids(now: int) -> set[str]:
    """Return model_ids that count as authoritative for pass-rate stats.

    Uses Phase 5's registry filter (active ∧ aged ∧ fresh ∧ official-source);
    falls back to "all observed models" when the registry is empty so the
    job still produces useful flags during bootstrap.
    """
    try:
        rows = registry_repo.list_eligible_baselines(now=now)
        ids = {row["model_id"] for row in rows if row and row.get("model_id")}
        if ids:
            return ids
    except Exception as e:
        logger.warning("pruner_job: registry lookup failed", error=str(e))
    # Fallback: observe every model that has produced any response.
    rows = get_conn().execute(
        "SELECT DISTINCT model_name FROM test_runs WHERE model_name IS NOT NULL"
    ).fetchall()
    return {r["model_name"] for r in rows if r["model_name"]}


def _compute_case_pass_rates(eligible_models: set[str]) -> dict[str, dict[str, float]]:
    """Return ``{case_id: {pass_rate, n_responses}}`` for *eligible* baselines.

    A response counts as "passed" when ``judge_passed=1``; ``judge_passed
    IS NULL`` rows are ignored (judge failure / not yet judged).
    """
    if not eligible_models:
        return {}
    placeholders = ",".join("?" for _ in eligible_models)
    sql = (
        "SELECT r.case_id AS case_id, "
        "       SUM(CASE WHEN r.judge_passed = 1 THEN 1 ELSE 0 END) AS passes, "
        "       SUM(CASE WHEN r.judge_passed IS NOT NULL THEN 1 ELSE 0 END) AS judged "
        "FROM test_responses r "
        "JOIN test_runs t ON t.id = r.run_id "
        f"WHERE t.model_name IN ({placeholders}) "
        "GROUP BY r.case_id"
    )
    rows = get_conn().execute(sql, list(eligible_models)).fetchall()
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        judged = int(row["judged"] or 0)
        passes = int(row["passes"] or 0)
        if judged == 0:
            continue
        out[row["case_id"]] = {
            "pass_rate": passes / judged,
            "n_responses": judged,
        }
    return out


def _load_test_cases() -> list[dict]:
    rows = get_conn().execute(
        "SELECT id, params FROM test_cases WHERE enabled = 1"
    ).fetchall()
    out = []
    for row in rows:
        try:
            params = json.loads(row["params"]) if row["params"] else {}
        except Exception:
            params = {}
        out.append({"id": row["id"], "params": params})
    return out


# ── Persistence ─────────────────────────────────────────────────────────────


def _upsert_quality_flag(metric: CaseQualityMetrics, now: int) -> None:
    flags = list(metric.flags or [])
    ceiling = "ceiling_effect" in flags
    floor = "floor_effect" in flags
    low_disc = "low_discrimination" in flags
    valid = bool(metric.is_discriminative)
    get_conn().execute(
        """
        INSERT OR REPLACE INTO case_quality_flags
            (case_id, discriminative_valid, ceiling_effect, floor_effect,
             low_discrimination, pass_rate, discrimination_a, n_responses,
             flags_json, last_evaluated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            metric.case_id,
            1 if valid else 0,
            1 if ceiling else 0,
            1 if floor else 0,
            1 if low_disc else 0,
            metric.pass_rate,
            metric.discrimination_a,
            metric.n_responses,
            json.dumps(flags, ensure_ascii=False),
            now,
        ),
    )
    get_conn().commit()


def get_quality_flag(case_id: str) -> dict[str, Any] | None:
    row = get_conn().execute(
        "SELECT * FROM case_quality_flags WHERE case_id = ?", (case_id,)
    ).fetchone()
    if not row:
        return None
    out = dict(row)
    if out.get("flags_json"):
        try:
            out["flags"] = json.loads(out["flags_json"])
        except Exception:
            out["flags"] = []
    else:
        out["flags"] = []
    return out


def list_discriminative_case_ids() -> set[str]:
    """Return case_ids with ``discriminative_valid=1``.

    Cases never evaluated are *implicitly* discriminative — callers
    should treat absence as "no quality decision yet".
    """
    rows = get_conn().execute(
        "SELECT case_id FROM case_quality_flags WHERE discriminative_valid = 1"
    ).fetchall()
    return {r["case_id"] for r in rows}


def list_non_discriminative_case_ids() -> set[str]:
    rows = get_conn().execute(
        "SELECT case_id FROM case_quality_flags WHERE discriminative_valid = 0"
    ).fetchall()
    return {r["case_id"] for r in rows}


# ── SSE warning event ──────────────────────────────────────────────────────


def _maybe_emit_exhaustion_warning(
    metrics: list[CaseQualityMetrics],
) -> dict[str, Any] | None:
    """Build an exhaustion warning payload when most cases are saturated.

    Phase 10 spec: when family-level pass_rate > 0.95, recommend
    isomorphic generation or new dataset import.  This implementation
    operates suite-wide; per-family attribution is left to v17.1.
    """
    high = [m for m in metrics if m.n_responses >= EXHAUSTION_MIN_SAMPLES
            and m.pass_rate > EXHAUSTION_PASS_RATE_THRESHOLD]
    if not high:
        return None
    remaining = sum(1 for m in metrics if m.is_discriminative)
    payload = {
        "event": "suite_exhaustion_warning",
        "high_pass_rate_cases": len(high),
        "remaining_discriminative_items": remaining,
        "pass_rate_threshold": EXHAUSTION_PASS_RATE_THRESHOLD,
        "recommendation": "trigger_isomorphic_generation_or_pull_new_dataset",
    }
    try:
        from app.core import sse  # type: ignore[import-not-found]
        if hasattr(sse, "publish"):
            sse.publish("suite_exhaustion_warning", payload)
    except Exception as e:
        logger.debug("pruner_job: SSE publish skipped", error=str(e))
    logger.warning(
        "suite_exhaustion_warning",
        high_pass_rate_cases=len(high),
        remaining_discriminative_items=remaining,
    )
    return payload


# ── Job entrypoint ─────────────────────────────────────────────────────────


@dataclass
class PrunerJobReport:
    started_at: int
    finished_at: int
    cases_seen: int
    cases_evaluated: int
    cases_marked_invalid: int
    pass_rate_window_models: int
    exhaustion_warning: dict[str, Any] | None = None
    flag_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_sec": self.finished_at - self.started_at,
            "cases_seen": self.cases_seen,
            "cases_evaluated": self.cases_evaluated,
            "cases_marked_invalid": self.cases_marked_invalid,
            "pass_rate_window_models": self.pass_rate_window_models,
            "exhaustion_warning": self.exhaustion_warning,
            "flag_counts": self.flag_counts,
        }


def run_pruner_job(now: int | None = None) -> PrunerJobReport:
    """Recompute quality flags for every enabled case.  Best-effort; never raises."""
    started = int(now or time.time())
    cases = _load_test_cases()
    eligible = _eligible_baseline_model_ids(started)
    pass_rates = _compute_case_pass_rates(eligible)

    pruner = SuitePruner()
    metrics_list: list[CaseQualityMetrics] = []
    flag_counts: dict[str, int] = {}
    invalid = 0

    for case in cases:
        meta = (case.get("params") or {}).get("_meta") or {}
        irt_a = meta.get("irt_a")
        irt_b = meta.get("irt_b")
        irt_c = meta.get("irt_c")
        # Coerce numeric params if present
        try:
            irt_a = float(irt_a) if irt_a is not None else None
        except (TypeError, ValueError):
            irt_a = None
        try:
            irt_b = float(irt_b) if irt_b is not None else 0.0
        except (TypeError, ValueError):
            irt_b = 0.0
        # SuitePruner._fisher_information assumes a numeric c; default to 0
        # for non-MCQ cases (no guessing parameter).  For MCQ items the
        # caller can stash a different value in params._meta.irt_c.
        try:
            irt_c = float(irt_c) if irt_c is not None else 0.0
        except (TypeError, ValueError):
            irt_c = 0.0

        stat = pass_rates.get(case["id"]) or {}
        pr = stat.get("pass_rate")
        n_resp = int(stat.get("n_responses") or 0)

        try:
            metric = pruner.analyze_case(
                case_id=case["id"],
                irt_a=irt_a,
                irt_b=irt_b or 0.0,
                irt_c=irt_c,
                pass_rate=pr,
                n_responses=n_resp,
            )
        except Exception as e:
            logger.warning("pruner.analyze_case failed", case_id=case["id"], error=str(e))
            continue

        metrics_list.append(metric)
        for flag in metric.flags or ():
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
        try:
            _upsert_quality_flag(metric, started)
        except Exception as e:
            logger.warning("upsert quality flag failed", case_id=case["id"], error=str(e))
        if not metric.is_discriminative:
            invalid += 1

    warning = _maybe_emit_exhaustion_warning(metrics_list)
    finished = int(time.time())

    return PrunerJobReport(
        started_at=started,
        finished_at=finished,
        cases_seen=len(cases),
        cases_evaluated=len(metrics_list),
        cases_marked_invalid=invalid,
        pass_rate_window_models=len(eligible),
        exhaustion_warning=warning,
        flag_counts=flag_counts,
    )


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", default=True)
    parser.parse_args()
    from app.core.db import init_db, get_conn
    from app.core.db_migrations import migrate
    init_db()
    migrate(get_conn())
    print(json.dumps(run_pruner_job().to_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())


__all__ = [
    "run_pruner_job",
    "PrunerJobReport",
    "get_quality_flag",
    "list_discriminative_case_ids",
    "list_non_discriminative_case_ids",
    "EXHAUSTION_PASS_RATE_THRESHOLD",
    "EXHAUSTION_MIN_SAMPLES",
]
