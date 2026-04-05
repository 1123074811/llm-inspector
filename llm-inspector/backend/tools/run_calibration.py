"""
Run calibration replay and export evaluation metrics.

This tool can:
1) Submit calibration runs from a case file (optional)
2) Wait for completion
3) Evaluate predicted verdicts against ground truth labels
4) Export JSON + CSV results

Usage examples:
  # Replay all calibration cases and export metrics
  python backend/tools/run_calibration.py \
    --cases backend/app/fixtures/calibration/cases.json \
    --out-json backend/output/calibration-result.json \
    --out-csv backend/output/calibration-result.csv

  # Evaluate existing run IDs only (no new execution)
  python backend/tools/run_calibration.py \
    --cases backend/app/fixtures/calibration/cases.json \
    --skip-submit
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import time
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).resolve().parents[2]
BACKEND_ROOT = ROOT / "backend"


@dataclass
class CalibrationCase:
    case_id: str
    expected_level: str
    run_id: str | None
    base_url: str | None
    api_key: str | None
    model: str | None
    test_mode: str
    evaluation_mode: str
    scoring_profile_version: str | None


def _load_cases(path: pathlib.Path) -> list[CalibrationCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        items = raw.get("cases", [])
    elif isinstance(raw, list):
        items = raw
    else:
        raise SystemExit("Invalid cases file format: expected list or {\"cases\": [...]}")

    out: list[CalibrationCase] = []
    for i, item in enumerate(items):
        case_id = str(item.get("case_id") or item.get("id") or f"case_{i + 1}")
        expected = str(item.get("expected_level") or item.get("label") or "").strip()
        if not expected:
            raise SystemExit(f"Case {case_id}: missing expected_level")

        out.append(
            CalibrationCase(
                case_id=case_id,
                expected_level=expected,
                run_id=item.get("run_id"),
                base_url=item.get("base_url"),
                api_key=item.get("api_key"),
                model=item.get("model"),
                test_mode=str(item.get("test_mode") or "standard"),
                evaluation_mode=str(item.get("evaluation_mode") or "calibration"),
                scoring_profile_version=item.get("scoring_profile_version"),
            )
        )
    return out


def _submit_case_run(case: CalibrationCase) -> str:
    from app.core.db import init_db
    from app.core.security import get_key_manager, validate_and_sanitize_url
    from app.repository import repo
    from app.tasks.worker import submit_run

    if not (case.base_url and case.api_key and case.model):
        raise SystemExit(
            f"Case {case.case_id}: missing base_url/api_key/model and no run_id provided"
        )

    init_db()

    clean_url = validate_and_sanitize_url(case.base_url)
    km = get_key_manager()
    encrypted, key_hash = km.encrypt(case.api_key)

    run_metadata = {
        "evaluation_mode": case.evaluation_mode,
        "calibration_case_id": case.case_id,
        "scoring_profile_version": case.scoring_profile_version,
        "calibration_tag": "baseline-v1.0",
    }

    run_id = repo.create_run(
        base_url=clean_url,
        api_key_encrypted=encrypted,
        api_key_hash=key_hash,
        model_name=case.model,
        test_mode=case.test_mode,
        suite_version="v2",
        metadata=run_metadata,
    )
    submit_run(run_id)
    return run_id


def _wait_for_runs(run_ids: list[str], timeout_sec: int, poll_sec: int) -> dict[str, str]:
    from app.repository import repo

    terminal = {"completed", "partial_failed", "failed"}
    end_time = time.time() + timeout_sec
    status_map: dict[str, str] = {}

    while time.time() < end_time:
        pending = []
        for run_id in run_ids:
            run = repo.get_run(run_id)
            status = (run or {}).get("status") or "unknown"
            status_map[run_id] = status
            if status not in terminal:
                pending.append(run_id)

        if not pending:
            return status_map
        time.sleep(poll_sec)

    return status_map


def _extract_predicted_level(run_id: str) -> str:
    from app.repository import repo

    report_row = repo.get_report(run_id)
    if not report_row:
        return "missing_report"

    details = report_row.get("details") or {}
    verdict = details.get("verdict") or {}
    level = verdict.get("level")
    if level:
        return str(level)

    risk = details.get("risk") or {}
    if risk.get("level"):
        return f"risk:{risk['level']}"

    return "unknown"


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _compute_metrics(expected: list[str], predicted: list[str]) -> dict:
    labels = sorted(set(expected) | set(predicted))

    # Accuracy
    correct = sum(1 for e, p in zip(expected, predicted) if e == p)
    accuracy = _safe_div(correct, len(expected))

    # Per-class precision/recall/f1 + macro average
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


def _export_csv(path: pathlib.Path, rows: list[dict], metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "run_id", "status", "expected_level", "predicted_level", "matched"])
        for r in rows:
            writer.writerow([
                r["case_id"],
                r["run_id"],
                r["status"],
                r["expected_level"],
                r["predicted_level"],
                1 if r["expected_level"] == r["predicted_level"] else 0,
            ])

        writer.writerow([])
        writer.writerow(["metric", "value"])
        writer.writerow(["sample_count", metrics["sample_count"]])
        writer.writerow(["accuracy", metrics["accuracy"]])
        writer.writerow(["macro_precision", metrics["macro_precision"]])
        writer.writerow(["macro_recall", metrics["macro_recall"]])
        writer.writerow(["macro_f1", metrics["macro_f1"]])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run calibration replay and export metrics")
    parser.add_argument("--cases", required=True, help="Path to calibration cases JSON")
    parser.add_argument("--out-json", default="backend/output/calibration-result.json")
    parser.add_argument("--out-csv", default="backend/output/calibration-result.csv")
    parser.add_argument("--skip-submit", action="store_true", help="Use existing run_id from cases file")
    parser.add_argument("--timeout-sec", type=int, default=7200)
    parser.add_argument("--poll-sec", type=int, default=3)
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(BACKEND_ROOT))

    cases_path = pathlib.Path(args.cases)
    if not cases_path.is_absolute():
        cases_path = ROOT / cases_path

    cases = _load_cases(cases_path)

    # Submit runs if needed
    run_ids: list[str] = []
    case_run_map: dict[str, str] = {}
    for case in cases:
        if args.skip_submit:
            if not case.run_id:
                raise SystemExit(f"Case {case.case_id}: skip-submit requires run_id")
            rid = case.run_id
        else:
            rid = _submit_case_run(case)
        run_ids.append(rid)
        case_run_map[case.case_id] = rid

    status_map = _wait_for_runs(run_ids, timeout_sec=args.timeout_sec, poll_sec=args.poll_sec)

    rows = []
    expected = []
    predicted = []

    for case in cases:
        run_id = case_run_map[case.case_id]
        status = status_map.get(run_id, "unknown")
        pred = _extract_predicted_level(run_id)

        rows.append(
            {
                "case_id": case.case_id,
                "run_id": run_id,
                "status": status,
                "expected_level": case.expected_level,
                "predicted_level": pred,
            }
        )
        expected.append(case.expected_level)
        predicted.append(pred)

    metrics = _compute_metrics(expected, predicted)

    result = {
        "cases_file": str(cases_path),
        "run_count": len(rows),
        "metrics": metrics,
        "rows": rows,
    }

    out_json = pathlib.Path(args.out_json)
    if not out_json.is_absolute():
        out_json = ROOT / out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    out_csv = pathlib.Path(args.out_csv)
    if not out_csv.is_absolute():
        out_csv = ROOT / out_csv
    _export_csv(out_csv, rows, metrics)

    print(f"Calibration completed: {len(rows)} runs")
    print(f"Accuracy: {metrics['accuracy']:.4f} | Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"JSON: {out_json}")
    print(f"CSV:  {out_csv}")


if __name__ == "__main__":
    main()
