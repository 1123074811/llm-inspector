"""
Calibration replay task runner.
Executes asynchronously from worker thread pool.
"""
from __future__ import annotations

from app.core.logging import get_logger
from app.repository import repo

logger = get_logger(__name__)


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _compute_metrics(expected: list[str], predicted: list[str]) -> dict:
    labels = sorted(set(expected) | set(predicted))

    correct = sum(1 for e, p in zip(expected, predicted) if e == p)
    accuracy = _safe_div(correct, len(expected))

    per_class = {}
    p_sum = r_sum = f_sum = 0.0

    for label in labels:
        tp = sum(1 for e, p in zip(expected, predicted) if e == label and p == label)
        fp = sum(1 for e, p in zip(expected, predicted) if e != label and p == label)
        fn = sum(1 for e, p in zip(expected, predicted) if e == label and p != label)

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)

        per_class[label] = {
            "support": sum(1 for e in expected if e == label),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }

        p_sum += precision
        r_sum += recall
        f_sum += f1

    macro_precision = _safe_div(p_sum, len(labels))
    macro_recall = _safe_div(r_sum, len(labels))
    macro_f1 = _safe_div(f_sum, len(labels))

    confusion = {}
    for e, p in zip(expected, predicted):
        confusion.setdefault(e, {})
        confusion[e][p] = confusion[e].get(p, 0) + 1

    return {
        "sample_count": len(expected),
        "labels": labels,
        "accuracy": round(accuracy, 6),
        "macro_precision": round(macro_precision, 6),
        "macro_recall": round(macro_recall, 6),
        "macro_f1": round(macro_f1, 6),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def _extract_predicted_level_from_run(run_id: str) -> tuple[str, str]:
    run = repo.get_run(run_id)
    status = (run or {}).get("status") or "unknown"

    report_row = repo.get_report(run_id)
    if not report_row:
        return status, "missing_report"

    details = report_row.get("details") or {}
    verdict = details.get("verdict") or {}
    level = verdict.get("level")
    if level:
        return status, str(level)

    risk = details.get("risk") or {}
    if risk.get("level"):
        return status, f"risk:{risk['level']}"

    return status, "unknown"


def run_calibration_replay(replay_id: str) -> None:
    row = repo.get_calibration_replay(replay_id)
    if not row:
        logger.error("Calibration replay not found", replay_id=replay_id)
        return

    repo.update_calibration_replay(replay_id, status="running")

    try:
        payload = row.get("cases_json") or {}
        cases = payload.get("cases") or []
        if not isinstance(cases, list) or not cases:
            raise ValueError("cases_json.cases must be a non-empty list")

        rows = []
        expected = []
        predicted = []

        for item in cases:
            case_id = str(item.get("case_id") or item.get("id") or "")
            run_id = str(item.get("run_id") or "")
            expected_level = str(item.get("expected_level") or item.get("label") or "")

            if not run_id:
                rows.append({
                    "case_id": case_id,
                    "run_id": None,
                    "status": "invalid",
                    "expected_level": expected_level,
                    "predicted_level": "missing_run_id",
                })
                continue

            status, pred_level = _extract_predicted_level_from_run(run_id)
            rows.append({
                "case_id": case_id,
                "run_id": run_id,
                "status": status,
                "expected_level": expected_level,
                "predicted_level": pred_level,
            })

            if expected_level:
                expected.append(expected_level)
                predicted.append(pred_level)

        metrics = _compute_metrics(expected, predicted) if expected else {
            "sample_count": 0,
            "labels": [],
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "per_class": {},
            "confusion_matrix": {},
        }

        result = {
            "replay_id": replay_id,
            "run_count": len(rows),
            "metrics": metrics,
            "rows": rows,
        }

        repo.update_calibration_replay(
            replay_id,
            status="completed",
            result_json=result,
            error_message=None,
        )
        logger.info("Calibration replay completed", replay_id=replay_id, runs=len(rows))
    except Exception as e:
        repo.update_calibration_replay(
            replay_id,
            status="failed",
            error_message=str(e)[:500],
        )
        logger.error("Calibration replay failed", replay_id=replay_id, error=str(e))
