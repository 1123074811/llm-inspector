"""Report and export handlers."""
from __future__ import annotations

import io
import math
import urllib.parse
from app.handlers.helpers import _json, _error, _extract_id, _load_report_or_error
from app.repository import repo
from app.tasks.worker import submit_run
from app.core.logging import get_logger

logger = get_logger(__name__)


def handle_get_report(path, _qs, _body) -> tuple:
    run_id = _extract_id(path, r"/api/v1/runs/([^/]+)/report$")
    if not run_id:
        return _error("Invalid run ID", 400)
    report, err = _load_report_or_error(run_id)
    if err:
        return err
    return _json(report)


def _build_radar_svg_bytes(report: dict) -> bytes:
    score = report.get("scorecard", {})
    breakdown = score.get("breakdown", {})
    dims = [
        ("能力分", float(score.get("capability_score", 0.0))),
        ("真实性", float(score.get("authenticity_score", 0.0))),
        ("性能分", float(score.get("performance_score", 0.0))),
        ("推理", float(breakdown.get("reasoning", 0.0))),
        ("指令", float(breakdown.get("instruction", 0.0))),
        ("一致性", float(breakdown.get("consistency", 0.0))),
    ]

    w, h = 760, 560
    cx, cy = 400, 300
    max_r = 200
    n = len(dims)
    angles = [(-math.pi / 2) + i * (2 * math.pi / n) for i in range(n)]

    def p2xy(r: float, a: float):
        return cx + r * math.cos(a), cy + r * math.sin(a)

    rings = []
    for lv in [2000, 4000, 6000, 8000, 10000]:
        r = max_r * lv / 10000
        pts = ["{:.1f},{:.1f}".format(*p2xy(r, a)) for a in angles]
        rings.append(f'<polygon points="{" ".join(pts)}" fill="none" stroke="#ddd" stroke-width="1" />')

    axes = []
    labels = []
    poly_pts = []
    dots = []
    for (name, val), a in zip(dims, angles):
        x2, y2 = p2xy(max_r, a)
        axes.append(f'<line x1="{cx}" y1="{cy}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#bbb" stroke-width="1" />')
        lx, ly = p2xy(max_r + 24, a)
        labels.append(f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="14" fill="#333" text-anchor="middle">{name}</text>')
        r = max_r * max(0.0, min(10000.0, val)) / 10000
        x, y = p2xy(r, a)
        poly_pts.append(f"{x:.1f},{y:.1f}")
        dots.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#c8401a" />')

    poly = f'<polygon points="{" ".join(poly_pts)}" fill="rgba(200,64,26,0.2)" stroke="#c8401a" stroke-width="2" />'

    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">'
    svg += '<rect width="100%" height="100%" fill="white" />'
    svg += '\n'.join(rings)
    svg += '\n'.join(axes)
    svg += '\n'.join(labels)
    svg += poly
    svg += '\n'.join(dots)
    svg += '</svg>'
    return svg.encode("utf-8")


def handle_export_radar_svg(path, _qs, _body) -> tuple:
    run_id = _extract_id(path, r"/api/v1/runs/([^/]+)/radar\.svg$")
    if not run_id:
        return _error("Invalid run ID", 400)
    report, err = _load_report_or_error(run_id)
    if err:
        return err
    body = _build_radar_svg_bytes(report)
    return 200, body, "image/svg+xml; charset=utf-8"


def handle_export_runs_zip(_path, qs, _body) -> tuple:
    run_ids = qs.get("run_ids", [[]])[0]
    if not run_ids:
        return _error("run_ids required")
    run_ids = run_ids.split(",")

    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for rid in run_ids:
            report, err = _load_report_or_error(rid)
            if err:
                continue
            zf.writestr(f"{rid}/report.json", json.dumps(report, ensure_ascii=False, default=str))
    return 200, buf.getvalue(), "application/zip"


def handle_get_responses(path, _qs, _body) -> tuple:
    run_id = _extract_id(path, r"/api/v1/runs/([^/]+)/responses$")
    if not run_id:
        return _error("Invalid run ID", 400)
    responses = repo.get_responses(run_id)
    slim = []
    for r in responses:
        req = r.get("request_payload") or {}
        preview = {}
        if isinstance(req, dict):
            msgs = req.get("messages") or []
            user_msg = ""
            for m in msgs:
                if isinstance(m, dict) and m.get("role") == "user":
                    user_msg = str(m.get("content") or "")
            preview = {
                "temperature": req.get("temperature"),
                "max_tokens": req.get("max_tokens"),
                "user_prompt": user_msg[:160],
            }
        slim.append({
            "id": r["id"],
            "case_id": r["case_id"],
            "sample_index": r["sample_index"],
            "response_text": (r.get("response_text") or "")[:300],
            "status_code": r.get("status_code"),
            "latency_ms": r.get("latency_ms"),
            "judge_passed": r.get("judge_passed"),
            "error_type": r.get("error_type"),
            "request_preview": preview,
        })
    return _json(slim)


def handle_get_scorecard(path, _qs, _body) -> tuple:
    run_id = _extract_id(path, r"/api/v1/runs/([^/]+)/scorecard$")
    if not run_id:
        return _error("Invalid run ID", 400)
    report_row = repo.get_report(run_id)
    if not report_row:
        return _error("Report not found", 404)
    details = report_row.get("details") or {}
    scorecard = details.get("scorecard")
    verdict = details.get("verdict")
    if not scorecard:
        return _error("Scorecard not found", 404)
    return _json({
        "run_id": run_id,
        "scorecard": scorecard,
        "verdict": verdict,
        "breakdown": repo.get_score_breakdown(run_id),
    })


def handle_get_extraction_audit(path, _qs, _body) -> tuple:
    run_id = _extract_id(path, r"/api/v1/runs/([^/]+)/extraction-audit$")
    if not run_id:
        return _error("Invalid run ID", 400)
    report_row = repo.get_report(run_id)
    if not report_row:
        return _error("Report not found", 404)
    details = report_row.get("details") or {}
    extraction_audit = details.get("extraction_audit")
    proxy_analysis = details.get("proxy_latency_analysis")
    if not extraction_audit:
        return _error("Extraction audit not available (run in extraction mode)", 404)
    return _json({
        "run_id": run_id,
        "extraction_audit": extraction_audit,
        "proxy_latency_analysis": proxy_analysis,
    })


def handle_get_theta_report(path, _qs, _body) -> tuple:
    run_id = _extract_id(path, r"/api/v1/runs/([^/]+)/theta-report$")
    if not run_id:
        return _error("Invalid run ID", 400)
    row = repo.get_theta_by_run(run_id)
    if not row:
        return _error("Theta report not found", 404)
    return _json({
        "run_id": run_id,
        "theta_global": row.get("theta_global"),
        "theta_global_ci_low": row.get("theta_global_ci_low"),
        "theta_global_ci_high": row.get("theta_global_ci_high"),
    })


def handle_get_pairwise(path, _qs, _body) -> tuple:
    run_id = _extract_id(path, r"/api/v1/runs/([^/]+)/pairwise$")
    if not run_id:
        return _error("Invalid run ID", 400)
    row = repo.get_pairwise_by_run(run_id)
    if not row:
        return _error("Pairwise not found", 404)
    return _json({"run_id": run_id, **row})


import json
