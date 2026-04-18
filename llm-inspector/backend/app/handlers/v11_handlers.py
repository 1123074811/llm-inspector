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

import json
import urllib.parse
from pathlib import Path

from app.core.circuit_breaker import circuit_breaker
from app.core.tracer import get_tracer, remove_tracer, get_all_tracer_progress, _get_trace_path
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


# ── Phase 4: Timeline SVG ─────────────────────────────────────────────────────

# Color palette per span type
_SPAN_COLORS = {
    "predetect":    "#4A90D9",
    "connectivity": "#27AE60",
    "phase1":       "#F39C12",
    "phase2":       "#E74C3C",
    "report":       "#8E44AD",
    "trace":        "#95A5A6",
}
_DEFAULT_COLOR = "#7F8C8D"

# Event marker colors
_EVENT_COLORS = {
    "run.started":   "#2ECC71",
    "run.completed": "#2980B9",
    "run.failed":    "#C0392B",
    "cb.open":       "#E74C3C",
    "case.result":   "#F39C12",
}
_EVENT_DEFAULT_COLOR = "#BDC3C7"


def _load_trace_events(run_id: str) -> list[dict]:
    """Load JSONL trace events for a run. Returns [] on any failure."""
    try:
        p = _get_trace_path(run_id)
        if not p.exists():
            return []
        events = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except Exception:
                        pass
        return events
    except Exception:
        return []


def _build_timeline_svg(run_id: str, events: list[dict]) -> str:
    """
    Build a pure-Python SVG timeline from trace events.

    Layout:
      - Title bar at top
      - One row per detected span
      - X-axis = elapsed time in seconds from earliest timestamp
      - Event markers as small vertical triangles
    """
    # ── Collect spans and stand-alone events ────────────────────────────────
    span_starts: dict[str, float] = {}
    span_ends: dict[str, float] = {}
    span_statuses: dict[str, str] = {}
    standalone_events: list[tuple[float, str]] = []  # (t, kind)

    t_min: float | None = None
    t_max: float | None = None

    for ev in events:
        ts = ev.get("timestamp")
        if ts is None:
            continue
        if t_min is None or ts < t_min:
            t_min = ts
        if t_max is None or ts > t_max:
            t_max = ts

        trace_event = ev.get("trace_event")
        kind = ev.get("type") or ev.get("kind", "")

        if trace_event == "span_start":
            span = ev.get("span", "")
            span_starts[span] = ts
        elif trace_event == "span_end":
            span = ev.get("span", "")
            span_ends[span] = ts
            span_statuses[span] = ev.get("status", "ok")
        elif kind and kind not in ("trace",):
            standalone_events.append((ts, kind))

    # Fallback: if no span data use a single timeline row
    all_span_names = list(dict.fromkeys(list(span_starts.keys()) + list(span_ends.keys())))

    if t_min is None:
        t_min = 0.0
    if t_max is None or t_max == t_min:
        t_max = t_min + 1.0

    duration_total = t_max - t_min
    case_count = sum(1 for _, k in standalone_events if "case" in k)

    # ── SVG dimensions ───────────────────────────────────────────────────────
    width = 900
    row_h = 40
    title_h = 60
    axis_h = 30
    left_margin = 160
    right_margin = 20
    chart_w = width - left_margin - right_margin
    n_rows = max(len(all_span_names), 1)
    height = title_h + n_rows * row_h + axis_h + 20

    def t_to_x(ts: float) -> float:
        if duration_total == 0:
            return left_margin
        return left_margin + (ts - t_min) / duration_total * chart_w

    lines: list[str] = []
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'font-family="monospace" font-size="12">'
    )

    # Background
    lines.append(f'<rect width="{width}" height="{height}" fill="#1E2130"/>')

    # Title bar
    run_id_short = run_id[:16] + ("..." if len(run_id) > 16 else "")
    dur_s = f"{duration_total:.1f}s"
    lines.append(f'<rect x="0" y="0" width="{width}" height="{title_h}" fill="#252A40"/>')
    lines.append(
        f'<text x="12" y="22" fill="#ECF0F1" font-size="14" font-weight="bold">'
        f'Pipeline Timeline — {run_id_short}</text>'
    )
    lines.append(
        f'<text x="12" y="42" fill="#95A5A6" font-size="11">'
        f'Total: {dur_s}  |  Cases: {case_count}  |  Spans: {len(all_span_names)}</text>'
    )

    # Grid lines + X-axis ticks
    n_ticks = 5
    for i in range(n_ticks + 1):
        tick_t = t_min + i / n_ticks * duration_total
        x = t_to_x(tick_t)
        lines.append(
            f'<line x1="{x:.1f}" y1="{title_h}" x2="{x:.1f}" '
            f'y2="{title_h + n_rows * row_h}" stroke="#2C3150" stroke-width="1"/>'
        )
        tick_label = f"{(tick_t - t_min):.1f}s"
        lines.append(
            f'<text x="{x:.1f}" y="{title_h + n_rows * row_h + 18}" '
            f'fill="#7F8C8D" text-anchor="middle" font-size="10">{tick_label}</text>'
        )

    # Span rows
    for row_idx, span_name in enumerate(all_span_names):
        y_top = title_h + row_idx * row_h
        y_center = y_top + row_h / 2

        color = _SPAN_COLORS.get(span_name, _DEFAULT_COLOR)

        # Row background (alternating)
        bg_color = "#222840" if row_idx % 2 == 0 else "#1E2130"
        lines.append(f'<rect x="0" y="{y_top}" width="{width}" height="{row_h}" fill="{bg_color}"/>')

        # Label
        lines.append(
            f'<text x="{left_margin - 8}" y="{y_center + 4:.1f}" fill="#BDC3C7" '
            f'text-anchor="end" font-size="11">{span_name}</text>'
        )

        # Span bar
        x_start = t_to_x(span_starts.get(span_name, t_min))
        x_end = t_to_x(span_ends.get(span_name, t_max))
        bar_w = max(x_end - x_start, 2)
        bar_h = row_h * 0.45
        bar_y = y_center - bar_h / 2
        status = span_statuses.get(span_name, "ok")
        bar_color = "#E74C3C" if status == "error" else color
        lines.append(
            f'<rect x="{x_start:.1f}" y="{bar_y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" '
            f'fill="{bar_color}" rx="3" opacity="0.85"/>'
        )

        # Duration label inside bar if wide enough
        if bar_w > 40:
            span_dur = span_ends.get(span_name, t_max) - span_starts.get(span_name, t_min)
            dur_label = f"{span_dur:.2f}s"
            mid_x = x_start + bar_w / 2
            lines.append(
                f'<text x="{mid_x:.1f}" y="{y_center + 4:.1f}" fill="white" '
                f'text-anchor="middle" font-size="10">{dur_label}</text>'
            )

    # Standalone event markers (small triangles on top of chart)
    for ts, kind in standalone_events:
        x = t_to_x(ts)
        color = _EVENT_COLORS.get(kind, _EVENT_DEFAULT_COLOR)
        # Draw a small downward-pointing triangle at y = title_h
        ty = title_h + 4
        pts = f"{x:.1f},{ty} {x - 4:.1f},{ty + 8} {x + 4:.1f},{ty + 8}"
        lines.append(f'<polygon points="{pts}" fill="{color}" opacity="0.7"/>')

    lines.append("</svg>")
    return "\n".join(lines)


def handle_run_timeline_svg(path: str, qs: dict, body: dict):
    """
    GET /api/v1/runs/{run_id}/timeline.svg

    Returns an SVG timeline showing trace spans/events for the run.
    Server-side generated, no external libraries needed.

    Data source:
    - If run is active: uses live tracer progress + any partial JSONL
    - If run completed: reads the persisted JSONL trace file
    - If no data: returns an empty timeline placeholder
    """
    parts = path.strip("/").split("/")
    # /api/v1/runs/{run_id}/timeline.svg
    if len(parts) < 4:
        return _error("Invalid timeline path", 400)
    run_id = parts[3]

    # Try JSONL first (works for both completed and active runs if flushed)
    events = _load_trace_events(run_id)

    # If run is still active, supplement with live progress events
    if not events:
        try:
            tracer = get_tracer(run_id)
            if tracer:
                progress = tracer.get_progress()
                import time as _time
                now = _time.time()
                for span_info in progress.get("completed_spans", []):
                    events.append({
                        "trace_event": "span_end",
                        "span": span_info["name"],
                        "timestamp": now,
                        "status": span_info.get("status", "ok"),
                    })
        except Exception:
            pass

    svg_content = _build_timeline_svg(run_id, events)
    return (200, svg_content.encode("utf-8"), "image/svg+xml")
