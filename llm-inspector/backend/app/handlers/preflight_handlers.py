"""
handlers/preflight_handlers.py — v15 Preflight API endpoints.
"""
from __future__ import annotations
import json
from app.core.logging import get_logger
from app.repository import repo

logger = get_logger(__name__)


def handle_get_preflight_result(path: str, qs: dict, body: dict) -> dict:
    """GET /api/v15/runs/{id}/preflight — Return preflight result for a run."""
    run_id = path.split("/")[5]  # /api/v15/runs/{id}/preflight
    run = repo.get_run(run_id)
    if not run:
        return {"status": 404, "body": {"error": "Run not found"}}

    preflight_raw = run.get("preflight_report")
    if preflight_raw:
        try:
            data = json.loads(preflight_raw)
        except Exception:
            data = {"error": "Could not parse preflight report"}
    else:
        data = {"passed": None, "note": "No preflight data (run predates v15)"}

    return {"status": 200, "body": data}
