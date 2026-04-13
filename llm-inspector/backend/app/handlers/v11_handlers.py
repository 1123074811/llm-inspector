"""
v11 API handlers — Circuit Breaker + Pipeline Tracing + CDM + Attribution + Phase 3 endpoints.

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

Phase 3:
- POST /api/v1/suite/prune             → IIF-based suite pruning analysis
- GET  /api/v1/suite/pruning-report    → latest pruning report
- GET  /api/v1/prompt-optimizer/report → prompt optimizer statistics
- GET  /api/v1/gpqa/questions          → GPQA question bank
- GET  /api/v1/attacks/multilingual    → multilingual attack templates
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


# ── Phase 3: Suite Pruning + Prompt Optimizer + GPQA + Multilingual ──────────

# In-memory cache for latest pruning report (cleared on server restart)
_latest_pruning_report = None


def handle_suite_prune(path: str, qs: dict, body: dict):
    """
    POST /api/v1/suite/prune

    Run IIF-based suite pruning analysis. Reads IRT parameters for all
    test cases and flags non-discriminative items.

    This is a dry-run: it marks cases but does NOT delete them.
    The pruning report is cached for later retrieval.

    Body (optional):
        {
            "discrimination_threshold": 0.5,
            "min_information": 0.01,
            "floor_pass_rate": 0.05,
            "ceiling_pass_rate": 0.95
        }
    """
    global _latest_pruning_report

    from app.analysis.suite_pruner import SuitePruner
    from app.repository import repo

    # Read custom thresholds from body
    disc_threshold = body.get("discrimination_threshold", 0.5) if body else 0.5
    min_info = body.get("min_information", 0.01) if body else 0.01
    floor_rate = body.get("floor_pass_rate", 0.05) if body else 0.05
    ceiling_rate = body.get("ceiling_pass_rate", 0.95) if body else 0.95

    pruner = SuitePruner(
        discrimination_threshold=disc_threshold,
        min_information=min_info,
        floor_pass_rate=floor_rate,
        ceiling_pass_rate=ceiling_rate,
    )

    # Load all cases from repository
    try:
        cases = repo.load_cases()
    except Exception:
        cases = []

    if not cases:
        return _json({
            "status": "no_data",
            "message": "No test cases found in the database to analyze.",
        })

    # Convert to pruner format
    case_dicts = []
    for c in cases:
        case_dicts.append({
            "id": c.get("id", ""),
            "irt_a": c.get("irt_a"),
            "irt_b": c.get("irt_b"),
            "irt_c": c.get("irt_c", 0.25),
            "pass_rate": c.get("pass_rate"),
            "n_responses": c.get("n_responses", 0),
            "weight": c.get("weight", 1.0),
            "max_tokens": c.get("max_tokens", 100),
        })

    report = pruner.analyze_suite(case_dicts)
    _latest_pruning_report = report

    return _json({
        "status": "completed",
        **report.to_dict(),
    })


def handle_suite_pruning_report(path: str, qs: dict, body: dict):
    """
    GET /api/v1/suite/pruning-report

    Returns the latest suite pruning report (from the most recent
    POST /api/v1/suite/prune call).
    """
    if _latest_pruning_report is None:
        return _json({
            "status": "no_report",
            "message": "No pruning report available. Run POST /api/v1/suite/prune first.",
        })

    return _json({
        "status": "available",
        **_latest_pruning_report.to_dict(),
    })


def handle_prompt_optimizer_report(path: str, qs: dict, body: dict):
    """
    GET /api/v1/prompt-optimizer/report

    Returns statistics from the dynamic Few-Shot prompt optimizer:
    total candidates, compilations, average examples selected,
    tokens saved, and methods used.
    """
    from app.runner.prompt_optimizer import prompt_optimizer
    report = prompt_optimizer.get_report()
    return _json(report.to_dict())


def handle_gpqa_questions(path: str, qs: dict, body: dict):
    """
    GET /api/v1/gpqa/questions

    Returns the GPQA question bank — graduate-level science questions
    that can be added to the test suite for high-difficulty coverage.
    """
    from app.analysis.suite_pruner import gpqa_adapter
    questions = gpqa_adapter.to_eval_cases()
    return _json({
        "total_questions": gpqa_adapter.n_questions,
        "questions": questions,
        "source": "GPQA Diamond (sample)",
    })


def handle_multilingual_attacks(path: str, qs: dict, body: dict):
    """
    GET /api/v1/attacks/multilingual

    Returns the multilingual attack templates and their metadata.
    Useful for understanding available attack vectors.
    """
    from app.predetect.multilingual_attack import MULTILINGUAL_TEMPLATES, B64_MULTILINGUAL_PAYLOADS
    return _json({
        "total_templates": len(MULTILINGUAL_TEMPLATES),
        "languages": list(set(l for l, _, _ in MULTILINGUAL_TEMPLATES)),
        "templates": [
            {"language": lang, "code": code, "prompt_preview": prompt[:80] + "..."}
            for lang, code, prompt in MULTILINGUAL_TEMPLATES
        ],
        "b64_payloads": list(B64_MULTILINGUAL_PAYLOADS.keys()),
    })
