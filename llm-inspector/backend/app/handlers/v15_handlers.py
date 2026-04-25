"""
handlers/v15_handlers.py — v15 API handlers.
"""
from __future__ import annotations
import json
from app.core.logging import get_logger
from app.repository import repo

logger = get_logger(__name__)


def handle_get_evidence_ledger(path: str, qs: dict, body: dict) -> dict:
    """GET /api/v15/runs/{id}/evidence-ledger"""
    parts = path.split("/")
    run_id = parts[5] if len(parts) > 5 else None
    if not run_id:
        return {"status": 400, "body": {"error": "run_id required"}}

    run = repo.get_run(run_id)
    if not run:
        return {"status": 404, "body": {"error": "Run not found"}}

    # Build evidence ledger from available data
    try:
        from app.authenticity.evidence_ledger import extract_evidence_from_predetect
        predetect_raw = run.get("predetect_result") or {}
        if isinstance(predetect_raw, str):
            predetect_raw = json.loads(predetect_raw)

        identity_exposure = repo.get_identity_exposure(run_id)
        claimed_model = run.get("model_name", "")

        ledger = extract_evidence_from_predetect(
            run_id=run_id,
            claimed_model=claimed_model,
            predetect_result=predetect_raw,
            identity_exposure=identity_exposure,
        )
        return {"status": 200, "body": ledger.to_dict()}
    except Exception as e:
        logger.error("Error building evidence ledger", run_id=run_id, error=str(e))
        return {"status": 500, "body": {"error": str(e)}}


def handle_get_model_card_diff(path: str, qs: dict, body: dict) -> dict:
    """GET /api/v15/runs/{id}/model-card-diff"""
    parts = path.split("/")
    run_id = parts[5] if len(parts) > 5 else None
    if not run_id:
        return {"status": 400, "body": {"error": "run_id required"}}

    run = repo.get_run(run_id)
    if not run:
        return {"status": 404, "body": {"error": "Run not found"}}

    try:
        from app.authenticity.evidence_ledger import extract_evidence_from_predetect
        from app.authenticity.model_card_diff import build_model_card_diff

        predetect_raw = run.get("predetect_result") or {}
        if isinstance(predetect_raw, str):
            predetect_raw = json.loads(predetect_raw)

        identity_exposure = repo.get_identity_exposure(run_id)
        claimed_model = run.get("model_name", "")

        ledger = extract_evidence_from_predetect(
            run_id=run_id,
            claimed_model=claimed_model,
            predetect_result=predetect_raw,
            identity_exposure=identity_exposure,
        )

        diff = build_model_card_diff(
            claimed_model=claimed_model,
            suspected_model=ledger.suspected_actual_model(),
            wrapper_probability=ledger.wrapper_probability(),
            risk_level=ledger.risk_level(),
            evidence_list=[e.to_dict() for e in ledger.evidence],
        )
        return {"status": 200, "body": diff.to_dict()}
    except Exception as e:
        logger.error("Error building model card diff", run_id=run_id, error=str(e))
        return {"status": 500, "body": {"error": str(e)}}


def handle_token_audit(path: str, qs: dict, body: dict) -> dict:
    """GET /api/v15/runs/{id}/token-audit — v15 Phase 10: Token efficiency audit."""
    parts = path.split("/")
    run_id = parts[5] if len(parts) > 5 else None
    if not run_id:
        return {"status": 400, "body": {"error": "run_id required"}}

    run = repo.get_run(run_id)
    if not run:
        return {"status": 404, "body": {"error": "Run not found"}}

    try:
        report = repo.get_report(run_id)
        if not report:
            return {"status": 200, "body": {"run_id": run_id, "token_audit": None,
                                            "note": "Report not yet generated"}}

        details = report.get("details", {}) if isinstance(report, dict) else {}
        token_audit = None
        if isinstance(details, dict):
            token_audit = details.get("token_audit")

        # Also fetch cache metrics from the global cache strategy
        cache_metrics = None
        try:
            from app.runner.cache_strategy import cache_strategy
            cache_metrics = cache_strategy.snapshot().to_dict()
        except Exception:
            pass

        return {"status": 200, "body": {
            "run_id": run_id,
            "token_audit": token_audit,
            "cache_metrics": cache_metrics,
        }}
    except Exception as e:
        logger.error("Error fetching token audit", run_id=run_id, error=str(e))
        return {"status": 500, "body": {"error": str(e)}}


def handle_cache_stats(path: str, qs: dict, body: dict) -> dict:
    """GET /api/v15/cache-stats — v15 Phase 10: Global cache strategy metrics."""
    try:
        from app.runner.cache_strategy import cache_strategy
        metrics = cache_strategy.snapshot()
        return {"status": 200, "body": metrics.to_dict()}
    except Exception as e:
        logger.error("Error fetching cache stats", error=str(e))
        return {"status": 500, "body": {"error": str(e)}}


def handle_evict_expired_cache(path: str, qs: dict, body: dict) -> dict:
    """POST /api/v15/cache/evict — Evict expired cache entries."""
    try:
        from app.runner.cache_strategy import cache_strategy
        removed = cache_strategy.evict_expired()
        return {"status": 200, "body": {"evicted": removed}}
    except Exception as e:
        logger.error("Error evicting cache", error=str(e))
        return {"status": 500, "body": {"error": str(e)}}


# ---------------------------------------------------------------------------
# v15 Phase 11: Dataset import
# ---------------------------------------------------------------------------

def handle_import_dataset(path: str, qs: dict, body: dict) -> dict:
    """POST /api/v15/dataset/import — Import test cases into the v15 suite."""
    cases = body.get("cases")
    if not cases or not isinstance(cases, list):
        return {"status": 400, "body": {"error": "Request body must contain 'cases' list"}}

    target_version = body.get("target_version", "v15")
    overwrite = bool(body.get("overwrite", False))

    try:
        from app.runner.import_dataset import DatasetImporter
        report = DatasetImporter.import_cases(cases, target_version, overwrite)
        return {"status": 200, "body": report.to_dict()}
    except Exception as e:
        logger.error("Error importing dataset", error=str(e))
        return {"status": 500, "body": {"error": str(e)}}


def handle_validate_case(path: str, qs: dict, body: dict) -> dict:
    """POST /api/v15/dataset/validate — Validate a single test case dict."""
    case = body.get("case")
    if not case or not isinstance(case, dict):
        return {"status": 400, "body": {"error": "Request body must contain 'case' dict"}}

    try:
        from app.runner.import_dataset import DatasetImporter
        error = DatasetImporter.validate_case(case)
        if error:
            return {"status": 200, "body": {"valid": False, "error": error}}
        return {"status": 200, "body": {"valid": True, "error": None}}
    except Exception as e:
        logger.error("Error validating case", error=str(e))
        return {"status": 500, "body": {"error": str(e)}}


# ---------------------------------------------------------------------------
# v15 Phase 12: Judge registry
# ---------------------------------------------------------------------------

def handle_judge_registry(path: str, qs: dict, body: dict) -> dict:
    """GET /api/v15/judge-registry — List all registered judge methods."""
    try:
        from app.analysis.judge_registry import registry_summary
        return {"status": 200, "body": registry_summary()}
    except Exception as e:
        logger.error("Error fetching judge registry", error=str(e))
        return {"status": 500, "body": {"error": str(e)}}


def handle_judge_registry_method(path: str, qs: dict, body: dict) -> dict:
    """GET /api/v15/judge-registry/{method} — Detail for a single judge method."""
    parts = path.split("/")
    method_name = parts[-1] if parts else None
    if not method_name:
        return {"status": 400, "body": {"error": "method name required"}}

    try:
        from app.analysis.judge_registry import get_method
        entry = get_method(method_name)
        if entry is None:
            return {"status": 404, "body": {"error": f"Method '{method_name}' not found"}}
        return {"status": 200, "body": {"method": method_name, "detail": entry}}
    except Exception as e:
        logger.error("Error fetching judge method", method=method_name, error=str(e))
        return {"status": 500, "body": {"error": str(e)}}
