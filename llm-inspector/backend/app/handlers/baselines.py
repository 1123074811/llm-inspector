"""Baseline and benchmark handlers."""
from __future__ import annotations

from app.handlers.helpers import _json, _error, _extract_id
from app.repository import repo
from app.tasks.worker import submit_run
from app.core.logging import get_logger

logger = get_logger(__name__)


def handle_benchmarks(_path, qs, _body) -> tuple:
    suite_version = qs.get("suite_version", ["v1"])[0]
    benchmarks = repo.get_benchmarks(suite_version)
    return _json([
        {
            "name": b.get("benchmark_name") or b.get("name", "unknown"),
            "suite_version": b.get("suite_version", ""),
            "data_source": b.get("data_source", "estimated"),
            "sample_count": b.get("sample_count", 3),
        }
        for b in benchmarks
    ])


def handle_create_baseline(_path, _qs, body: dict) -> tuple:
    # Mode 1: Mark an existing run as baseline (from frontend "标记为基准模型")
    if body.get("run_id") and not body.get("base_url"):
        run_id = body["run_id"]
        run = repo.get_run(run_id)
        if not run:
            return _error("Run not found", 404)
        if run["status"] not in ("completed", "partial_failed"):
            return _error("Run not completed yet", 400)

        model_name = body.get("model_name") or run["model_name"]
        display_name = body.get("display_name") or model_name

        try:
            result = repo.create_baseline(
                run_id=run_id,
                model_name=model_name,
                display_name=display_name,
            )
        except ValueError as e:
            return _error(str(e), 400)
        logger.info("Baseline created from run", display_name=display_name, run_id=run_id)
        return _json({"baseline_id": result["id"], "status": "created"}, 201)

    # Mode 2: Create a new run and mark as baseline (full params)
    for field in ("name", "base_url", "api_key", "model"):
        if not body.get(field):
            return _error(f"Missing required field: {field}")

    from app.core.security import validate_and_sanitize_url, get_key_manager
    try:
        clean_url = validate_and_sanitize_url(body["base_url"])
    except ValueError as e:
        return _error(str(e))

    api_key = str(body["api_key"]).strip()
    if api_key.lower().startswith("bearer "):
        api_key = api_key[7:].strip()

    km = get_key_manager()
    encrypted, key_hash = km.encrypt(api_key)

    test_mode = body.get("test_mode", "standard")
    suite_version = body.get("suite_version", "v3")

    run_metadata = {
        "evaluation_mode": "normal",
        "scoring_profile_version": "v1",
        "calibration_tag": "baseline-v1.0",
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

    repo.create_baseline(run_id=run_id, model_name=body["model"], display_name=body["name"])
    submit_run(run_id)
    logger.info("Baseline created", name=body["name"], run_id=run_id)
    return _json({"baseline_id": run_id, "status": "queued"}, 201)


def handle_list_baselines(_path, qs, _body) -> tuple:
    limit = int(qs.get("limit", ["50"])[0])
    baselines = repo.list_baselines(limit=min(limit, 100))
    return _json({"baselines": baselines})


def handle_compare_baseline(_path, _qs, body: dict) -> tuple:
    baseline_id = body.get("baseline_id")
    run_id = body.get("run_id")
    if not baseline_id or not run_id:
        return _error("baseline_id and run_id required")

    baseline = repo.get_baseline(baseline_id)
    if not baseline:
        return _error("Baseline not found", 404)

    baseline_run = repo.get_run(baseline["run_id"])
    target_run = repo.get_run(run_id)
    if not target_run:
        return _error("Target run not found", 404)

    if target_run["status"] not in ("completed", "partial_failed"):
        return _error("Target run not completed", 400)

    report_row = repo.get_report(run_id)
    if not report_row:
        return _error("Target report not found", 404)

    baseline_report = repo.get_report(baseline["run_id"])
    if not baseline_report:
        return _error("Baseline report not found", 404)

    from app.analysis.pipeline import AnalysisPipeline
    pipeline = AnalysisPipeline()
    comparison = pipeline.compare_with_baseline(
        target_report=report_row["details"],
        baseline_report=baseline_report["details"],
        baseline_name=baseline["name"],
    )
    return _json(comparison)


def handle_delete_baseline(path, _qs, _body) -> tuple:
    baseline_id = _extract_id(path, r"/api/v1/baselines/([^/]+)$")
    if not baseline_id:
        return _error("Invalid baseline ID", 400)
    repo.delete_baseline(baseline_id)
    return _json({"deleted": baseline_id})


def handle_get_baseline(path, _qs, _body) -> tuple:
    baseline_id = _extract_id(path, r"/api/v1/baselines/([^/]+)$")
    if not baseline_id:
        return _error("Invalid baseline ID", 400)
    baseline = repo.get_baseline(baseline_id)
    if not baseline:
        return _error("Baseline not found", 404)
    return _json(baseline)
