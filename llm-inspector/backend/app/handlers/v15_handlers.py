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
