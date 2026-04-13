"""
v11 API handlers — Circuit Breaker + Pipeline Tracing endpoints.

Phase 1 of the v11 upgrade plan:
- GET /api/v1/circuit-breaker          → circuit breaker state for all endpoints
- GET /api/v1/circuit-breaker/{url}    → circuit breaker state for specific endpoint
- POST /api/v1/circuit-breaker/reset   → reset circuit breaker (admin)
- GET /api/v1/runs/{id}/trace          → pipeline trace for a completed/active run
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
        return _error(400, "Invalid trace path")
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
