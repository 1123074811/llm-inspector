"""
Export run report into CSV summary.

Usage:
  python backend/tools/export_run_report.py --run-id <RUN_ID> --out backend/output/<RUN_ID>.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "llm_inspector.db"


def _load_report(run_id: str) -> dict:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT details FROM reports WHERE run_id=?", (run_id,)).fetchone()
    conn.close()
    if not row:
        raise SystemExit(f"report not found for run_id={run_id}")
    return json.loads(row["details"])


def export_csv(report: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scorecard = report.get("scorecard", {})
    verdict = report.get("verdict", {})
    risk = report.get("risk", {})

    rows = [
        ("run_id", report.get("run_id")),
        ("model", (report.get("target") or {}).get("model")),
        ("base_url", (report.get("target") or {}).get("base_url")),
        ("test_mode", (report.get("target") or {}).get("test_mode")),
        ("total_score", scorecard.get("total_score")),
        ("capability_score", scorecard.get("capability_score")),
        ("authenticity_score", scorecard.get("authenticity_score")),
        ("performance_score", scorecard.get("performance_score")),
        ("reasoning_score", scorecard.get("reasoning_score")),
        ("instruction_score", scorecard.get("instruction_score")),
        ("coding_score", scorecard.get("coding_score")),
        ("safety_score", scorecard.get("safety_score")),
        ("protocol_score", scorecard.get("protocol_score")),
        ("consistency_score", scorecard.get("consistency_score")),
        ("speed_score", scorecard.get("speed_score")),
        ("stability_score", scorecard.get("stability_score")),
        ("cost_efficiency", scorecard.get("cost_efficiency")),
        ("verdict_level", verdict.get("level")),
        ("risk_level", risk.get("level")),
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    report = _load_report(args.run_id)
    export_csv(report, Path(args.out))
    print(f"CSV exported: {args.out}")
