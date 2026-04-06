"""Calibration handlers."""
from __future__ import annotations

from app.handlers.helpers import _json, _error, _extract_id
from app.repository import repo
from app.tasks.calibration import recalibrate_and_snapshot, snapshot_calibration
from app.tasks.worker import submit_calibration_replay

__all__ = [
    "handle_calibration_rebuild",
    "handle_calibration_snapshot_only",
    "handle_create_calibration_replay",
    "handle_get_calibration_replay",
    "handle_list_calibration_replays",
]


def handle_calibration_rebuild(_path, qs, _body) -> tuple:
    version = qs.get("version", [None])[0]
    try:
        result = recalibrate_and_snapshot(version=version)
        return _json({"status": "ok", **result})
    except Exception as e:
        return _error(f"Calibration rebuild failed: {e}", 500)


def handle_calibration_snapshot_only(_path, qs, _body) -> tuple:
    version = qs.get("version", [None])[0]
    notes = qs.get("notes", [""])[0]
    try:
        result = snapshot_calibration(version=version, notes=notes)
        return _json({"status": "ok", "snapshot": result})
    except Exception as e:
        return _error(f"Calibration snapshot failed: {e}", 500)


def handle_create_calibration_replay(_path, _qs, body: dict) -> tuple:
    cases = body.get("cases")
    if not isinstance(cases, list) or not cases:
        return _error("Missing required field: cases (non-empty list)", 400)
    replay_id = repo.create_calibration_replay({"cases": cases})
    submit_calibration_replay(replay_id)
    return _json({"replay_id": replay_id, "status": "queued"}, 201)


def handle_get_calibration_replay(path, _qs, _body) -> tuple:
    replay_id = _extract_id(path, r"/api/v1/calibration/replay/([^/]+)$")
    if not replay_id:
        return _error("Invalid replay ID", 400)
    row = repo.get_calibration_replay(replay_id)
    if not row:
        return _error("Calibration replay not found", 404)
    return _json({
        "replay_id": row["id"],
        "status": row["status"],
        "created_at": row.get("created_at"),
        "started_at": row.get("started_at"),
        "completed_at": row.get("completed_at"),
        "error_message": row.get("error_message"),
        "cases": (row.get("cases_json") or {}).get("cases", []),
        "result": row.get("result_json"),
    })


def handle_list_calibration_replays(_path, qs, _body) -> tuple:
    limit = int(qs.get("limit", ["20"])[0])
    rows = repo.list_calibration_replays(min(limit, 100))
    return _json([
        {
            "replay_id": r["id"],
            "status": r["status"],
            "created_at": r.get("created_at"),
            "started_at": r.get("started_at"),
            "completed_at": r.get("completed_at"),
            "error_message": r.get("error_message"),
            "case_count": len((r.get("cases_json") or {}).get("cases", [])),
            "has_result": bool(r.get("result_json")),
        }
        for r in rows
    ])
