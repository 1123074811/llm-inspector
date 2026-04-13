"""
v11 API handlers — Circuit Breaker + Pipeline Tracing + CDM + Attribution endpoints.

Phase 1:
- GET /api/v1/circuit-breaker          → circuit breaker state for all endpoints
- GET /api/v1/circuit-breaker?url=<u>  → circuit breaker state for specific endpoint
- POST /api/v1/circuit-breaker/reset   → reset circuit breaker (admin)
- GET /api/v1/runs/{id}/trace          → pipeline trace for a completed/active run
- GET /api/v1/tracers/progress         → progress info for all active tracers

Phase 2:
- GET /api/v1/runs/{id}/cdm            → CDM skill mastery diagnosis
- GET /api/v1/runs/{id}/attribution    → Shapley Value score attribution
- GET /api/v1/cdm/skills               → list all CDM skills (taxonomy)
"""
from __future__ import annotations

import urllib.parse
from app.core.circuit_breaker import circuit_breaker
from app.core.tracer import get_tracer, remove_tracer, get_all_tracer_progress
from app.handlers.helpers import _json, _error


def handle_circuit_breaker_status(path: str, qs: dict, body: dict):
    """
    GET /api/v1/circuit-breaker
    GET /api/v1/circuit-breaker?url=<encoded_base_url>

    Returns circuit breaker state for all endpoints, or a specific one.
    """
    url_param = qs.get("url", [None])[0]
    if url_param:
        decoded_url = urllib.parse.unquote(url_param)
        metrics = circuit_breaker.get_metrics(decoded_url)
        if not metrics:
            return _json({"base_url": decoded_url, "state": "unknown", "message": "No circuit state for this endpoint"})
        return _json(metrics)
    return _json({
        "stats": circuit_breaker.stats,
        "endpoints": circuit_breaker.get_metrics(),
    })


def handle_circuit_breaker_reset(path: str, qs: dict, body: dict):
    """
    POST /api/v1/circuit-breaker/reset

    Reset circuit breaker state. Optionally specify a base_url to reset only
    that endpoint.

    Body (optional):
        {"base_url": "https://api.example.com/v1"}
    """
    base_url = body.get("base_url") if body else None
    circuit_breaker.reset(base_url)
    scope = base_url or "all endpoints"
    return _json({"status": "ok", "message": f"Circuit breaker reset for {scope}"})


def handle_run_trace(path: str, qs: dict, body: dict):
    """
    GET /api/v1/runs/{id}/trace

    Returns the pipeline trace for a run:
    - If run is active: returns live progress from the tracer
    - If run is completed: returns the persisted trace (from tracer registry
      or DB)

    Includes per-span timing, token usage, and event stream.
    """
    parts = path.strip("/").split("/")
    # /api/v1/runs/{run_id}/trace
    if len(parts) < 4:
        return _error("Invalid trace path", 400)
    run_id = parts[3]

    tracer = get_tracer(run_id)
    if tracer:
        # Active run — return live progress
        return _json({
            "run_id": run_id,
            "status": "active",
            "progress": tracer.get_progress(),
        })

    # Not active — check if we have a completed trace
    # For now, return a not-found. In Phase 2 we'll persist traces to DB.
    return _json({
        "run_id": run_id,
        "status": "not_found",
        "message": "No active trace for this run. Traces are only available during pipeline execution.",
    })


def handle_tracer_progress_all(path: str, qs: dict, body: dict):
    """
    GET /api/v1/tracers/progress

    Admin endpoint: returns progress info for all active tracers.
    """
    progress = get_all_tracer_progress()
    return _json({
        "active_tracers": len(progress),
        "tracers": progress,
    })


# ── Phase 2: CDM + Attribution ──────────────────────────────────────────────

def handle_run_cdm(path: str, qs: dict, body: dict):
    """
    GET /api/v1/runs/{id}/cdm

    Returns CDM (Cognitive Diagnostic Model) skill mastery diagnosis
    for a completed run. Includes per-skill mastery probabilities,
    attribute pattern, and strongest/weakest skills.
    """
    parts = path.strip("/").split("/")
    # /api/v1/runs/{run_id}/cdm
    if len(parts) < 4:
        return _error("Invalid CDM path", 400)
    run_id = parts[3]

    from app.repository import repo
    report = repo.get_report(run_id)
    if not report:
        return _error(f"Run {run_id} not found", 404)

    cdm_data = report.get("cdm")
    if not cdm_data:
        return _json({
            "run_id": run_id,
            "status": "unavailable",
            "message": "CDM diagnosis not available for this run. Re-run with v11 pipeline to enable CDM.",
        })

    return _json({
        "run_id": run_id,
        "status": "available",
        **cdm_data,
    })


def handle_run_attribution(path: str, qs: dict, body: dict):
    """
    GET /api/v1/runs/{id}/attribution

    Returns Shapley Value score attribution for a completed run.
    Shows which features/dimensions contributed most to the final score,
    and how much each contributed.
    """
    parts = path.strip("/").split("/")
    # /api/v1/runs/{run_id}/attribution
    if len(parts) < 4:
        return _error("Invalid attribution path", 400)
    run_id = parts[3]

    from app.repository import repo
    report = repo.get_report(run_id)
    if not report:
        return _error(f"Run {run_id} not found", 404)

    attr_data = report.get("attribution")
    if not attr_data:
        return _json({
            "run_id": run_id,
            "status": "unavailable",
            "message": "Score attribution not available for this run. Re-run with v11 pipeline to enable attribution.",
        })

    return _json({
        "run_id": run_id,
        "status": "available",
        **attr_data,
    })


def handle_cdm_skills(path: str, qs: dict, body: dict):
    """
    GET /api/v1/cdm/skills

    Returns the CDM skill taxonomy — all micro-skills and their
    dimension mappings. Useful for frontend skill radar chart.
    """
    from app.analysis.cdm_engine import SKILL_TAXONOMY, ALL_SKILLS
    return _json({
        "total_skills": len(ALL_SKILLS),
        "skills": ALL_SKILLS,
        "taxonomy": SKILL_TAXONOMY,
    })
