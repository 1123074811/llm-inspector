"""Compare run handlers."""
from __future__ import annotations

from app.handlers.helpers import _json, _error, _extract_id
from app.repository import repo
from app.tasks.worker import submit_compare

__all__ = [
    "handle_create_compare_run",
    "handle_get_compare_run",
    "handle_list_compare_runs",
]


def handle_create_compare_run(_path, _qs, body: dict) -> tuple:
    golden_run_id = body.get("golden_run_id")
    candidate_run_id = body.get("candidate_run_id")
    if not golden_run_id or not candidate_run_id:
        return _error("Missing required field: golden_run_id/candidate_run_id")

    golden = repo.get_run(golden_run_id)
    candidate = repo.get_run(candidate_run_id)
    if not golden or not candidate:
        return _error("Run not found", 404)

    if golden.get("status") not in ("completed", "partial_failed"):
        return _error("golden_run is not completed", 400)
    if candidate.get("status") not in ("completed", "partial_failed"):
        return _error("candidate_run is not completed", 400)

    compare_id = repo.create_compare_run(golden_run_id, candidate_run_id)
    submit_compare(compare_id)
    return _json({"compare_id": compare_id, "status": "queued"}, 201)


def handle_get_compare_run(path, _qs, _body) -> tuple:
    compare_id = _extract_id(path, r"/api/v1/compare-runs/([^/]+)$")
    if not compare_id:
        return _error("Invalid compare ID", 400)

    row = repo.get_compare_run(compare_id)
    if not row:
        return _error("Compare run not found", 404)

    return _json({
        "compare_id": row["id"],
        "golden_run_id": row["golden_run_id"],
        "candidate_run_id": row["candidate_run_id"],
        "status": row["status"],
        "created_at": row.get("created_at"),
        "completed_at": row.get("completed_at"),
        "details": row.get("details"),
    })


def handle_list_compare_runs(_path, qs, _body) -> tuple:
    limit = int(qs.get("limit", ["20"])[0])
    rows = repo.list_compare_runs(min(limit, 100))
    return _json(rows)
