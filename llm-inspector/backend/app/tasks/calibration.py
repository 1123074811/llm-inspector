"""
Offline calibration tasks for IRT-ish item statistics.
Can be triggered manually or scheduled.
"""
from __future__ import annotations

import math
from app.core.logging import get_logger
from app.core.config import settings
from app.repository import repo

logger = get_logger(__name__)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def recalibrate_item_stats() -> dict:
    """
    Lightweight periodic calibration:
    - estimates empirical pass rate per item
    - maps pass rate to pseudo Rasch difficulty b = -logit(pass_rate)
    - updates info_score using p(1-p) weighted by prior
    """
    conn = repo.get_conn()
    rows = conn.execute(
        """
        SELECT case_id,
               AVG(CASE
                    WHEN judge_passed = 1 THEN 1.0
                    WHEN judge_passed = 0 THEN 0.0
                    ELSE NULL
               END) AS pass_rate,
               COUNT(CASE WHEN judge_passed IN (0,1) THEN 1 END) AS n
        FROM test_responses
        GROUP BY case_id
        """
    ).fetchall()

    updated = 0
    for r in rows:
        case_id = r["case_id"]
        n = int(r["n"] or 0)
        if n < 3:
            continue

        pass_rate = float(r["pass_rate"] or 0.5)
        eps = 1e-4
        pass_rate = min(1 - eps, max(eps, pass_rate))

        b = -math.log(pass_rate / (1.0 - pass_rate))
        fisher = pass_rate * (1.0 - pass_rate)

        case_row = conn.execute(
            "SELECT category, params FROM test_cases WHERE id=?",
            (case_id,),
        ).fetchone()
        if not case_row:
            continue

        params = repo.from_json_col(case_row["params"]) or {}
        meta = (params.get("_meta") or {})
        dimension = meta.get("dimension") or case_row["category"] or "general"
        prior = float(meta.get("info_gain_prior", 1.0) or 1.0)
        info_score = fisher * prior * math.log1p(n)

        repo.upsert_item_stat(
            item_id=case_id,
            dimension=dimension,
            a=1.0,
            b=float(b),
            c=None,
            info_score=float(info_score),
            sample_size=n,
        )
        updated += 1

    logger.info("Item stats recalibrated", updated=updated)
    return {"updated_items": updated, "observed_items": len(rows)}


def snapshot_calibration(version: str | None = None, notes: str = "") -> dict:
    if not version:
        version = settings.CALIBRATION_VERSION

    item_stats = repo.list_item_stats()
    payload = {
        "version": version,
        "items": [
            {
                "item_id": r.get("item_id"),
                "dimension": r.get("dimension"),
                "a": r.get("irt_a"),
                "b": r.get("irt_b"),
                "c": r.get("irt_c"),
                "info_score": r.get("info_score"),
                "sample_size": r.get("sample_size"),
                "last_calibrated_at": r.get("last_calibrated_at"),
            }
            for r in item_stats
        ],
    }
    repo.save_calibration_snapshot(version=version, item_params_json=payload, notes=notes)
    logger.info("Calibration snapshot saved", version=version, items=len(item_stats))
    return {"version": version, "items": len(item_stats)}


def recalibrate_and_snapshot(version: str | None = None) -> dict:
    recalib = recalibrate_item_stats()
    snap = snapshot_calibration(version=version, notes="auto recalibrate")
    return {"recalibration": recalib, "snapshot": snap}
