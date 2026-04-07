"""Run management handlers."""
from __future__ import annotations

from app.core.config import settings
from app.core.security import validate_and_sanitize_url, get_key_manager
from app.handlers.helpers import _json, _error, _extract_id
from app.repository import repo
from app.tasks.worker import submit_run, submit_continue, submit_skip_testing
from app.core.logging import get_logger

logger = get_logger(__name__)


def handle_list_runs(_path, qs, _body) -> tuple:
    limit_values = qs.get("limit", ["50"])
    offset_values = qs.get("offset", ["0"])
    limit = int(limit_values[0]) if limit_values else 50
    offset = int(offset_values[0]) if offset_values else 0
    runs = repo.list_runs(limit=limit, offset=offset)
    # Add frontend-compatible aliases (run_id, model)
    result = []
    for r in runs:
        d = dict(r)
        d["run_id"] = d.get("id", "")
        d["model"] = d.get("model_name", "")
        result.append(d)
    return _json({"runs": result, "limit": limit, "offset": offset})


def handle_create_run(_path, _qs, body: dict) -> tuple:
    for field in ("base_url", "api_key", "model"):
        if not body.get(field):
            return _error(f"Missing required field: {field}")

    try:
        clean_url = validate_and_sanitize_url(body["base_url"])
    except ValueError as e:
        return _error(str(e))

    api_key: str = str(body["api_key"]).strip()
    if api_key.lower().startswith("bearer "):
        api_key = api_key[7:].strip()
    if len(api_key) < 4:
        return _error("api_key too short")

    km = get_key_manager()
    encrypted, key_hash = km.encrypt(api_key)

    test_mode = body.get("test_mode", "standard")
    # Backward compatibility: map legacy mode names
    if test_mode == "full":
        test_mode = "deep"
    elif test_mode == "extraction":
        test_mode = "deep"
    if test_mode not in ("quick", "standard", "deep"):
        test_mode = "standard"

    evaluation_mode = str(body.get("evaluation_mode", "normal") or "normal").strip().lower()
    if evaluation_mode not in ("normal", "calibration"):
        evaluation_mode = "normal"

    scoring_profile_version = str(body.get("scoring_profile_version", settings.CALIBRATION_VERSION) or settings.CALIBRATION_VERSION).strip()
    if not scoring_profile_version:
        scoring_profile_version = settings.CALIBRATION_VERSION

    calibration_case_id = body.get("calibration_case_id")
    suite_version = settings.SUITE_VERSION

    run_metadata = {
        "evaluation_mode": evaluation_mode,
        "calibration_case_id": calibration_case_id,
        "scoring_profile_version": scoring_profile_version,
        "calibration_tag": "baseline-v1.0" if evaluation_mode == "calibration" else None,
    }

    run_id = repo.create_run(
        base_url=clean_url,
        api_key_encrypted=encrypted,
        api_key_hash=key_hash,
        model_name=body["model"],
        test_mode=test_mode,
        suite_version=suite_version,
        metadata=run_metadata,
    )

    submit_run(run_id)
    logger.info("Run created", run_id=run_id, model=body["model"])

    return _json({
        "run_id": run_id,
        "status": "queued",
        "evaluation_mode": evaluation_mode,
        "scoring_profile_version": scoring_profile_version,
    }, 201)


def handle_get_run(path, _qs, _body) -> tuple:
    run_id = _extract_id(path, r"/api/v1/runs/([^/]+)$")
    if not run_id:
        return _error("Invalid run ID", 400)

    run = repo.get_run(run_id)
    if not run:
        return _error("Run not found", 404)

    responses = repo.get_responses(run_id)
    cases = repo.load_cases(run.get("suite_version", "v1"), run.get("test_mode", "standard"))
    completed = len(set(r["case_id"] for r in responses))
    total = len(cases)

    baseline = repo.get_baseline_by_run_id(run_id)
    return _json({
        "run_id": run_id,
        "status": run["status"],
        "model": run["model_name"],
        "base_url": run["base_url"],
        "test_mode": run.get("test_mode"),
        "created_at": run["created_at"],
        "started_at": run.get("started_at"),
        "completed_at": run.get("completed_at"),
        "error_message": run.get("error_message"),
        "progress": {
            "completed": completed,
            "total": total,
            "phase": run["status"],
        },
        "predetect_result": run.get("predetect_result"),
        "predetect_confidence": run.get("predetect_confidence"),
        "predetect_identified": bool(run.get("predetect_identified")),
        "evaluation_mode": run.get("evaluation_mode", "normal"),
        "calibration_case_id": run.get("calibration_case_id"),
        "scoring_profile_version": run.get("scoring_profile_version", settings.CALIBRATION_VERSION),
        "calibration_tag": run.get("calibration_tag"),
        "baseline_id": baseline["id"] if baseline else None,
    })


def handle_delete_run(path, _qs, _body) -> tuple:
    run_id = _extract_id(path, r"/api/v1/runs/([^/]+)$")
    if not run_id:
        return _error("Invalid run ID", 400)
    try:
        repo.delete_run(run_id)
    except Exception as e:
        return _error(f"Delete failed: {str(e)}", 500)
    return _json({"deleted": run_id})


def handle_batch_delete_runs(_path, _qs, body: dict) -> tuple:
    run_ids = body.get("run_ids")
    if not run_ids or not isinstance(run_ids, list):
        return _error("Invalid or empty run_ids list", 400)

    deleted_count = 0
    errors = []
    for rid in run_ids:
        try:
            repo.delete_run(rid)
            deleted_count += 1
        except Exception as e:
            errors.append({"run_id": rid, "error": str(e)})

    return _json({
        "deleted_count": deleted_count,
        "total_requested": len(run_ids),
        "errors": errors
    })


def handle_cancel_run(path, _qs, _body) -> tuple:
    run_id = _extract_id(path, r"/api/v1/runs/([^/]+)/cancel$")
    if not run_id:
        return _error("Invalid run ID", 400)
    repo.set_run_cancel_requested(run_id)
    return _json({"run_id": run_id, "cancel_requested": True})


def handle_retry_run(path, _qs, _body) -> tuple:
    run_id = _extract_id(path, r"/api/v1/runs/([^/]+)/retry$")
    if not run_id:
        return _error("Invalid run ID", 400)
    run = repo.get_run(run_id)
    if not run:
        return _error("Run not found", 404)
    repo.mark_run_retry(run_id)
    submit_run(run_id)
    return _json({"run_id": run_id, "status": "queued"})


def handle_continue_run(path, _qs, _body) -> tuple:
    run_id = _extract_id(path, r"/api/v1/runs/([^/]+)/continue$")
    if not run_id:
        return _error("Invalid run ID", 400)
    run = repo.get_run(run_id)
    if not run:
        return _error("Run not found", 404)
    if run["status"] not in ("pre_detected", "cancelled"):
        return _error("Run cannot be continued", 400)
    submit_continue(run_id)
    return _json({"run_id": run_id, "status": "running"})


def handle_skip_testing(path, _qs, _body) -> tuple:
    run_id = _extract_id(path, r"/api/v1/runs/([^/]+)/skip-testing$")
    if not run_id:
        return _error("Invalid run ID", 400)
    run = repo.get_run(run_id)
    if not run:
        return _error("Run not found", 404)
    if run["status"] not in ("pre_detected",):
        return _error("Run cannot skip testing", 400)
    submit_skip_testing(run_id)
    return _json({"run_id": run_id, "status": "running"})
