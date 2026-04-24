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
