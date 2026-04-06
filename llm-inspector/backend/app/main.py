"""
LLM Inspector API Server — stdlib http.server, no FastAPI required.
Implements all endpoints from the spec.
"""
from __future__ import annotations

import json
import re
import pathlib
import urllib.parse
import io
import csv
import zipfile
from http.server import HTTPServer, BaseHTTPRequestHandler
from app.core.config import settings
from app.core.db import init_db
from app.core.logging import setup_logging, get_logger
from app.core.security import validate_and_sanitize_url, get_key_manager
from app.core.schemas import ScoreCard
from app.analysis.pipeline import AnalysisPipeline
from app.tasks.seeder import seed_all
from app.tasks.calibration import recalibrate_and_snapshot, snapshot_calibration
from app.tasks.worker import submit_run, submit_compare, submit_calibration_replay, submit_continue, submit_skip_testing, active_count
from app.repository import repo

logger = get_logger(__name__)

# ── Response helpers ───────────────────────────────────────────────────────────

def _json(data, status: int = 200) -> tuple[int, bytes, str]:
    body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
    return status, body, "application/json"


def _error(msg: str, status: int = 400) -> tuple[int, bytes, str]:
    return _json({"error": msg}, status)


# ── Route handlers ─────────────────────────────────────────────────────────────

def handle_health(_path, _qs, _body) -> tuple:
    db_state = "ok"
    case_stats: dict[str, int] = {}
    try:
        conn = repo.get_conn()
        rows = conn.execute(
            "SELECT suite_version, COUNT(*) as n FROM test_cases GROUP BY suite_version"
        ).fetchall()
        case_stats = {row["suite_version"]: row["n"] for row in rows}
    except Exception:
        db_state = "degraded"

    return _json({
        "status": "ok",
        "db": db_state,
        "workers_active": active_count(),
        "version": "1.0.0",
        "test_cases": case_stats,
    })


def handle_create_run(_path, _qs, body: dict) -> tuple:
    # Validate required fields
    for field in ("base_url", "api_key", "model"):
        if not body.get(field):
            return _error(f"Missing required field: {field}")

    # SSRF-safe URL validation
    try:
        clean_url = validate_and_sanitize_url(body["base_url"])
    except ValueError as e:
        return _error(str(e))

    api_key: str = str(body["api_key"]).strip()
    if api_key.lower().startswith("bearer "):
        api_key = api_key[7:].strip()
    if len(api_key) < 4:
        return _error("api_key too short")

    # Encrypt API key
    km = get_key_manager()
    encrypted, key_hash = km.encrypt(api_key)

    test_mode = body.get("test_mode", "standard")
    if test_mode not in ("quick", "standard", "full", "extraction"):
        test_mode = "standard"

    evaluation_mode = str(body.get("evaluation_mode", "normal") or "normal").strip().lower()
    if evaluation_mode not in ("normal", "calibration"):
        evaluation_mode = "normal"

    scoring_profile_version = str(body.get("scoring_profile_version", settings.CALIBRATION_VERSION) or settings.CALIBRATION_VERSION).strip()
    if not scoring_profile_version:
        scoring_profile_version = settings.CALIBRATION_VERSION

    calibration_case_id = body.get("calibration_case_id")

    suite_version = body.get("suite_version", "v3")
    # Project now uses v3 suite by default for all runs
    suite_version = "v3"

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

    # Submit to background worker
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

    metadata = run.get("metadata") or {}

    # Count completed responses for progress
    responses = repo.get_responses(run_id)
    cases = repo.load_cases(run.get("suite_version", "v1"), run.get("test_mode", "standard"))
    completed = len(set(r["case_id"] for r in responses))
    total = len(cases)

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
        "evaluation_mode": metadata.get("evaluation_mode", "normal"),
        "calibration_case_id": metadata.get("calibration_case_id"),
        "scoring_profile_version": metadata.get("scoring_profile_version", settings.CALIBRATION_VERSION),
        "calibration_tag": metadata.get("calibration_tag"),
    })


def _load_report_or_error(run_id: str):
    run = repo.get_run(run_id)
    if not run:
        return None, _error("Run not found", 404)
    if run["status"] not in ("completed", "partial_failed"):
        return None, _error("Report not ready yet", 404)
    report_row = repo.get_report(run_id)
    if not report_row:
        return None, _error("Report not found", 404)
    return report_row["details"], None


def handle_get_report(path, _qs, _body) -> tuple:
    run_id = _extract_id(path, r"/api/v1/runs/([^/]+)/report$")
    if not run_id:
        return _error("Invalid run ID", 400)

    report, err = _load_report_or_error(run_id)
    if err:
        return err
    return _json(report)


def handle_export_report_csv(path, _qs, _body) -> tuple:
    run_id = _extract_id(path, r"/api/v1/runs/([^/]+)/report\.csv$")
    if not run_id:
        return _error("Invalid run ID", 400)

    report, err = _load_report_or_error(run_id)
    if err:
        return err

    scorecard = report.get("scorecard", {})
    verdict = report.get("verdict", {})
    risk = report.get("risk", {})
    target = report.get("target", {})

    rows = [
        ("run_id", report.get("run_id")),
        ("model", target.get("model")),
        ("base_url", target.get("base_url")),
        ("test_mode", target.get("test_mode")),
        ("total_score", scorecard.get("total_score")),
        ("capability_score", scorecard.get("capability_score")),
        ("authenticity_score", scorecard.get("authenticity_score")),
        ("performance_score", scorecard.get("performance_score")),
        ("reasoning_score", scorecard.get("reasoning_score")),
        ("instruction_score", scorecard.get("instruction_score")),
        ("coding_score", scorecard.get("coding_score")),
        ("safety_score", scorecard.get("safety_score")),
        ("protocol_score", scorecard.get("protocol_score")),
        ("consistency_score", scorecard.get("consistency_score")),
        ("speed_score", scorecard.get("speed_score")),
        ("stability_score", scorecard.get("stability_score")),
        ("cost_efficiency", scorecard.get("cost_efficiency")),
        ("verdict_level", verdict.get("level")),
        ("risk_level", risk.get("level")),
    ]

    body = _build_csv_bytes(report)
    return 200, body, "text/csv; charset=utf-8"


def _build_csv_bytes(report: dict) -> bytes:
    scorecard = report.get("scorecard", {})
    verdict = report.get("verdict", {})
    risk = report.get("risk", {})
    target = report.get("target", {})

    rows = [
        ("run_id", report.get("run_id")),
        ("model", target.get("model")),
        ("base_url", target.get("base_url")),
        ("test_mode", target.get("test_mode")),
        ("total_score", scorecard.get("total_score")),
        ("capability_score", scorecard.get("capability_score")),
        ("authenticity_score", scorecard.get("authenticity_score")),
        ("performance_score", scorecard.get("performance_score")),
        ("reasoning_score", scorecard.get("reasoning_score")),
        ("instruction_score", scorecard.get("instruction_score")),
        ("coding_score", scorecard.get("coding_score")),
        ("safety_score", scorecard.get("safety_score")),
        ("protocol_score", scorecard.get("protocol_score")),
        ("consistency_score", scorecard.get("consistency_score")),
        ("speed_score", scorecard.get("speed_score")),
        ("stability_score", scorecard.get("stability_score")),
        ("cost_efficiency", scorecard.get("cost_efficiency")),
        ("verdict_level", verdict.get("level")),
        ("risk_level", risk.get("level")),
    ]

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["metric", "value"])
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


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

    import math
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
        dots.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#2563eb" />')

    target = report.get("target") or {}
    title = f"综合能力雷达图 · {target.get('model', 'unknown')}"
    svg = f"""<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {w} {h}\" width=\"100%\" height=\"100%\">\n  <rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>\n  <text x=\"24\" y=\"36\" font-size=\"24\" font-weight=\"700\" fill=\"#111\">{title}</text>\n  <text x=\"24\" y=\"62\" font-size=\"14\" fill=\"#666\">run_id: {report.get('run_id')}</text>\n  {''.join(rings)}\n  {''.join(axes)}\n  <polygon points=\"{' '.join(poly_pts)}\" fill=\"rgba(37,99,235,0.20)\" stroke=\"#2563eb\" stroke-width=\"2\"/>\n  {''.join(dots)}\n  {''.join(labels)}\n</svg>\n"""
    return svg.encode("utf-8")


def handle_export_radar_svg(path, _qs, _body) -> tuple:
    run_id = _extract_id(path, r"/api/v1/runs/([^/]+)/radar\.svg$")
    if not run_id:
        return _error("Invalid run ID", 400)

    report, err = _load_report_or_error(run_id)
    if err:
        return err

    return 200, _build_radar_svg_bytes(report), "image/svg+xml"


def handle_export_runs_zip(_path, qs, _body) -> tuple:
    run_ids_raw = qs.get("run_ids", [""])[0]
    export_types_raw = qs.get("types", ["csv,svg"])[0]

    run_ids = [x.strip() for x in run_ids_raw.split(",") if x.strip()]
    if not run_ids:
        return _error("Missing run_ids query param", 400)

    requested_types = {x.strip().lower() for x in export_types_raw.split(",") if x.strip()}
    if not requested_types:
        requested_types = {"csv", "svg"}
    allowed_types = requested_types.intersection({"csv", "svg"})
    if not allowed_types:
        return _error("types must include csv and/or svg", 400)

    mem = io.BytesIO()
    zf = zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED)

    added = []
    errors = []
    for run_id in run_ids[:100]:
        report, err = _load_report_or_error(run_id)
        if err:
            errors.append(run_id)
            continue

        if "csv" in allowed_types:
            zf.writestr(f"{run_id}/report.csv", _build_csv_bytes(report))
            added.append(f"{run_id}/report.csv")
        if "svg" in allowed_types:
            zf.writestr(f"{run_id}/radar.svg", _build_radar_svg_bytes(report))
            added.append(f"{run_id}/radar.svg")

    manifest = {
        "requested_run_ids": run_ids,
        "types": sorted(list(allowed_types)),
        "added_files": added,
        "skipped_run_ids": errors,
    }
    zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    zf.close()

    return 200, mem.getvalue(), "application/zip"


def handle_get_responses(path, _qs, _body) -> tuple:
    run_id = _extract_id(path, r"/api/v1/runs/([^/]+)/responses$")
    if not run_id:
        return _error("Invalid run ID", 400)

    responses = repo.get_responses(run_id)
    # Strip large raw_response to keep payload small
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


def handle_delete_run(path, _qs, _body) -> tuple:
    run_id = _extract_id(path, r"/api/v1/runs/([^/]+)$")
    if not run_id:
        return _error("Invalid run ID", 400)

    conn = repo.get_conn()
    conn.execute("DELETE FROM test_runs WHERE id=?", (run_id,))
    conn.commit()
    return 204, b"", "application/json"


def handle_list_runs(_path, qs, _body) -> tuple:
    limit = int(qs.get("limit", ["20"])[0])
    runs = repo.list_runs(min(limit, 100))
    return _json([
        {
            "run_id": r["id"],
            "status": r["status"],
            "model": r["model_name"],
            "base_url": r["base_url"],
            "created_at": r["created_at"],
            "predetect_identified": bool(r.get("predetect_identified")),
        }
        for r in runs
    ])


def handle_benchmarks(_path, qs, _body) -> tuple:
    suite_version = qs.get("suite_version", ["v1"])[0]
    benchmarks = repo.get_benchmarks(suite_version)
    return _json([
        {
            "name": b["benchmark_name"],
            "suite_version": b["suite_version"],
            "generated_at": b["generated_at"],
            "sample_count": b.get("sample_count", 3),
        }
        for b in benchmarks
    ])


def handle_create_compare_run(_path, _qs, body: dict) -> tuple:
    golden_run_id = body.get("golden_run_id")
    candidate_run_id = body.get("candidate_run_id")
    if not golden_run_id or not candidate_run_id:
        return _error("Missing required field: golden_run_id/candidate_run_id")

    golden = repo.get_run(golden_run_id)
    candidate = repo.get_run(candidate_run_id)
    if not golden or not candidate:
        return _error("Run not found", 404)

    if golden.get("status") not in ("completed", "partial_failed"):
        return _error("golden_run is not completed", 400)
    if candidate.get("status") not in ("completed", "partial_failed"):
        return _error("candidate_run is not completed", 400)

    compare_id = repo.create_compare_run(golden_run_id, candidate_run_id)
    submit_compare(compare_id)
    return _json({"compare_id": compare_id, "status": "queued"}, 201)


def handle_get_compare_run(path, _qs, _body) -> tuple:
    compare_id = _extract_id(path, r"/api/v1/compare-runs/([^/]+)$")
    if not compare_id:
        return _error("Invalid compare ID", 400)

    row = repo.get_compare_run(compare_id)
    if not row:
        return _error("Compare run not found", 404)

    return _json({
        "compare_id": row["id"],
        "golden_run_id": row["golden_run_id"],
        "candidate_run_id": row["candidate_run_id"],
        "status": row["status"],
        "created_at": row.get("created_at"),
        "completed_at": row.get("completed_at"),
        "details": row.get("details"),
    })


def handle_list_compare_runs(_path, qs, _body) -> tuple:
    limit = int(qs.get("limit", ["20"])[0])
    rows = repo.list_compare_runs(min(limit, 100))
    return _json(rows)


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


def handle_model_trend(path, qs, _body) -> tuple:
    model_name = _extract_id(path, r"/api/v1/models/([^/]+)/trend$")
    if not model_name:
        return _error("Invalid model name", 400)
    model_name = urllib.parse.unquote(model_name)
    limit = int(qs.get("limit", ["20"])[0])
    return _json(repo.get_model_trend(model_name, min(limit, 200)))


def handle_leaderboard(_path, qs, _body) -> tuple:
    sort_by = qs.get("sort_by", ["total_score"])[0]
    limit = int(qs.get("limit", ["50"])[0])
    return _json(repo.get_leaderboard(sort_by=sort_by, limit=min(limit, 200)))


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
        "theta_dims": row.get("theta_dims_json") or {},
        "percentile_global": row.get("percentile_global"),
        "percentile_dims": row.get("percentile_dims_json") or {},
        "calibration_version": row.get("calibration_version"),
        "method": row.get("method"),
        "created_at": row.get("created_at"),
    })


def handle_model_theta_trend(path, qs, _body) -> tuple:
    model_name = _extract_id(path, r"/api/v1/models/([^/]+)/theta-trend$")
    if not model_name:
        return _error("Invalid model name", 400)
    model_name = urllib.parse.unquote(model_name)
    limit = int(qs.get("limit", ["50"])[0])
    return _json(repo.get_model_theta_trend(model_name, min(limit, 200)))


def handle_theta_leaderboard(_path, qs, _body) -> tuple:
    dimension = qs.get("dimension", ["global"])[0]
    limit = int(qs.get("limit", ["50"])[0])
    return _json(repo.get_theta_leaderboard(dimension=dimension, limit=min(limit, 200)))


def handle_get_pairwise(path, _qs, _body) -> tuple:
    run_id = _extract_id(path, r"/api/v1/runs/([^/]+)/pairwise$")
    if not run_id:
        return _error("Invalid run ID", 400)
    return _json(repo.get_pairwise_by_run(run_id))


def handle_calibration_rebuild(_path, qs, _body) -> tuple:
    version = qs.get("version", [None])[0]
    try:
        result = recalibrate_and_snapshot(version=version)
        return _json({"status": "ok", **result})
    except Exception as e:
        logger.error("Calibration rebuild failed", error=str(e))
        return _error(f"Calibration rebuild failed: {e}", 500)


def handle_calibration_snapshot_only(_path, qs, _body) -> tuple:
    version = qs.get("version", [None])[0]
    notes = qs.get("notes", [""])[0]
    try:
        result = snapshot_calibration(version=version, notes=notes)
        return _json({"status": "ok", "snapshot": result})
    except Exception as e:
        logger.error("Calibration snapshot failed", error=str(e))
        return _error(f"Calibration snapshot failed: {e}", 500)


def handle_create_calibration_replay(_path, _qs, body: dict) -> tuple:
    cases = body.get("cases")
    if not isinstance(cases, list) or not cases:
        return _error("Missing required field: cases (non-empty list)", 400)

    replay_id = repo.create_calibration_replay({"cases": cases})
    submit_calibration_replay(replay_id)
    return _json({"replay_id": replay_id, "status": "queued"}, 201)


def handle_get_calibration_replay(path, _qs, _body) -> tuple:
    replay_id = _extract_id(path, r"/api/v1/calibration/replay/([^/]+)$")
    if not replay_id:
        return _error("Invalid replay ID", 400)

    row = repo.get_calibration_replay(replay_id)
    if not row:
        return _error("Calibration replay not found", 404)

    return _json({
        "replay_id": row["id"],
        "status": row["status"],
        "created_at": row.get("created_at"),
        "started_at": row.get("started_at"),
        "completed_at": row.get("completed_at"),
        "error_message": row.get("error_message"),
        "cases": (row.get("cases_json") or {}).get("cases", []),
        "result": row.get("result_json"),
    })


def handle_list_calibration_replays(_path, qs, _body) -> tuple:
    limit = int(qs.get("limit", ["20"])[0])
    rows = repo.list_calibration_replays(min(limit, 100))
    return _json([
        {
            "replay_id": r["id"],
            "status": r["status"],
            "created_at": r.get("created_at"),
            "started_at": r.get("started_at"),
            "completed_at": r.get("completed_at"),
            "error_message": r.get("error_message"),
            "case_count": len((r.get("cases_json") or {}).get("cases", [])),
            "has_result": bool(r.get("result_json")),
        }
        for r in rows
    ])


def _build_isomorphic_cases() -> list[dict]:
    return [
        {
            "id": "reason_candy_iso_007",
            "category": "reasoning",
            "dimension": "reasoning",
            "tags": ["constraint_reasoning", "isomorphic"],
            "name": "candy_shape_flavor_iso_a",
            "system_prompt": "Focus on constraints and boundary proof. Keep answer concise.",
            "user_prompt": "袋中有三种口味糖果，每种有圆形和星形可触摸区分。请给出最少抓取总数，保证手里一定同时存在跨形状的苹果味与桃子味配对（圆苹果+星桃子，或圆桃子+星苹果）。请给出最不利边界证明。",
            "expected_type": "text",
            "judge_method": "constraint_reasoning",
            "max_tokens": 220,
            "n_samples": 2,
            "temperature": 0.0,
            "params": {
                "target_pattern": "\\b21\\b",
                "key_constraints": ["可触摸区分", "最不利", "边界", "形状"],
                "boundary_signals": ["最不利", "上界", "保证", "边界"],
                "anti_pattern_signals": ["盲抽", "随机抽取即可"],
                "min_constraint_hits": 1,
                "require_boundary": True,
                "allow_anti_pattern": False,
            },
            "weight": 2.4,
        },
        {
            "id": "reason_rope_iso_007",
            "category": "reasoning",
            "dimension": "reasoning",
            "tags": ["constraint_reasoning", "isomorphic"],
            "name": "rope_unsat_iso_a",
            "system_prompt": "Focus on constraints and boundary proof. Keep answer concise.",
            "user_prompt": "有一根燃烧不均匀的绳子，整根烧完60分钟，只有一个打火机。如何精确测15分钟？若无解请说明约束冲突与边界理由。",
            "expected_type": "text",
            "judge_method": "constraint_reasoning",
            "max_tokens": 220,
            "n_samples": 2,
            "temperature": 0.0,
            "params": {
                "target_pattern": "无解|无法",
                "key_constraints": ["不均匀", "一根", "一个打火机"],
                "boundary_signals": ["无解", "无法", "约束冲突"],
                "anti_pattern_signals": ["对折", "两头点燃后再"],
                "min_constraint_hits": 1,
                "require_boundary": True,
                "allow_anti_pattern": False,
            },
            "weight": 2.4,
        },
        {
            "id": "reason_mice_iso_007",
            "category": "reasoning",
            "dimension": "reasoning",
            "tags": ["constraint_reasoning", "isomorphic"],
            "name": "mice_ternary_iso_a",
            "system_prompt": "Focus on constraints and boundary proof. Keep answer concise.",
            "user_prompt": "240瓶酒有1瓶毒酒，毒发固定24小时。离宴会48小时，最少需要几只小白鼠确保定位毒酒？要求说明状态数建模。",
            "expected_type": "text",
            "judge_method": "constraint_reasoning",
            "max_tokens": 220,
            "n_samples": 2,
            "temperature": 0.0,
            "params": {
                "target_pattern": "\\b5\\b",
                "key_constraints": ["48小时", "24小时", "两轮", "三种状态"],
                "boundary_signals": ["3^5", "243", "编码"],
                "anti_pattern_signals": ["二进制", "8只"],
                "min_constraint_hits": 1,
                "require_boundary": True,
                "allow_anti_pattern": False,
            },
            "weight": 2.4,
        },
        {
            "id": "instr_token_iso_007",
            "category": "instruction",
            "dimension": "instruction",
            "tags": ["token_control", "isomorphic", "char_count", "forbidden_chars"],
            "name": "apple_50char_no_de_shi_iso",
            "system_prompt": "严格执行硬约束，不要解释。",
            "user_prompt": "写一段介绍苹果的中文短文，要求：恰好50个中文字符（不计标点），且不能出现“的”和“是”。",
            "expected_type": "text",
            "judge_method": "text_constraints",
            "max_tokens": 180,
            "n_samples": 3,
            "temperature": 0.0,
            "params": {"exact_chars": 50, "forbidden_chars": ["的", "是"]},
            "weight": 2.2,
        },
    ]


def handle_generate_isomorphic(_path, qs, _body) -> tuple:
    apply_flag = (qs.get("apply", ["false"])[0] or "false").lower() == "true"
    suite_path = pathlib.Path(__file__).parent / "fixtures" / "suite_v2.json"
    if not suite_path.exists():
        return _error("suite_v2.json not found", 404)

    data = json.loads(suite_path.read_text(encoding="utf-8"))
    cases = data.get("cases", [])
    existing_ids = {c.get("id") for c in cases}
    generated = _build_isomorphic_cases()
    to_add = [c for c in generated if c["id"] not in existing_ids]

    if apply_flag and to_add:
        data["cases"] = cases + to_add
        suite_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return _json({
        "apply": apply_flag,
        "added_count": len(to_add) if apply_flag else 0,
        "preview_count": len(to_add),
        "to_add_ids": [c["id"] for c in to_add],
    })


def handle_create_baseline(_path, _qs, body: dict) -> tuple:
    run_id = (body.get("run_id") or "").strip()
    model_name = (body.get("model_name") or "").strip().lower()
    display_name = (body.get("display_name") or "").strip()
    notes = (body.get("notes") or "").strip()

    if not run_id or not model_name or not display_name:
        return _error("Missing required field: run_id/model_name/display_name", 400)

    try:
        baseline = repo.create_baseline(
            run_id=run_id,
            model_name=model_name,
            display_name=display_name,
            notes=notes,
        )
    except ValueError as e:
        return _error(str(e), 400)

    return _json(
        {
            "baseline_id": baseline["id"],
            "model_name": baseline["model_name"],
            "display_name": baseline["display_name"],
            "total_score": baseline["total_score"],
            "created_at": baseline["created_at"],
        },
        201,
    )


def handle_list_baselines(_path, qs, _body) -> tuple:
    model_name = (qs.get("model_name", [""])[0] or "").strip().lower() or None
    active_only = (qs.get("active_only", ["true"])[0] or "true").lower() == "true"

    baselines = repo.list_baselines(model_name=model_name, active_only=active_only)
    return _json(
        {
            "baselines": [
                {
                    "id": b["id"],
                    "model_name": b["model_name"],
                    "display_name": b["display_name"],
                    "total_score": b["total_score"],
                    "capability_score": b["capability_score"],
                    "authenticity_score": b["authenticity_score"],
                    "performance_score": b["performance_score"],
                    "sample_count": b.get("sample_count", 1),
                    "notes": b.get("notes", ""),
                    "created_at": b.get("created_at"),
                }
                for b in baselines
            ]
        }
    )


def handle_compare_baseline(_path, _qs, body: dict) -> tuple:
    run_id = (body.get("run_id") or "").strip()
    baseline_id = (body.get("baseline_id") or "").strip() or None
    if not run_id:
        return _error("Missing required field: run_id", 400)

    run = repo.get_run(run_id)
    if not run:
        return _error("Run not found", 404)
    if run.get("status") not in ("completed", "partial_failed"):
        return _error("Run is not completed", 400)

    features = repo.get_features(run_id)
    report_row = repo.get_report(run_id)
    if not report_row:
        return _error("Report not found", 404)
    details = report_row.get("details") or {}
    scorecard = details.get("scorecard") or {}

    current_card = ScoreCard(
        total_score=float(scorecard.get("total_score", 0.0)) / 100.0,
        capability_score=float(scorecard.get("capability_score", 0.0)) / 100.0,
        authenticity_score=float(scorecard.get("authenticity_score", 0.0)) / 100.0,
        performance_score=float(scorecard.get("performance_score", 0.0)) / 100.0,
    )

    baseline = repo.get_baseline(baseline_id) if baseline_id else None
    if not baseline:
        baseline = repo.get_latest_active_baseline((run.get("model_name") or "").strip().lower())
    if not baseline:
        return _error("No active baseline found", 404)

    cmp = AnalysisPipeline.compare_with_baseline(
        current_features=features,
        baseline_feature_vector=baseline.get("feature_vector") or {},
        current_card=current_card,
        baseline_scores={
            "total_score": float(baseline.get("total_score", 0.0)),
            "capability_score": float(baseline.get("capability_score", 0.0)),
            "authenticity_score": float(baseline.get("authenticity_score", 0.0)),
            "performance_score": float(baseline.get("performance_score", 0.0)),
        },
    )

    reason = (
        f"余弦相似度 {cmp['cosine_similarity']:.3f}，总分偏差 {cmp['score_delta']['total']:+d}，"
        f"判定为 {cmp['verdict']}。"
    )

    comparison_id = repo.save_baseline_comparison(
        run_id=run_id,
        baseline_id=baseline["id"],
        cosine_similarity=float(cmp["cosine_similarity"]),
        score_delta_total=float(cmp["score_delta"]["total"]) / 100.0,
        score_delta_capability=float(cmp["score_delta"]["capability"]) / 100.0,
        score_delta_authenticity=float(cmp["score_delta"]["authenticity"]) / 100.0,
        score_delta_performance=float(cmp["score_delta"]["performance"]) / 100.0,
        verdict=cmp["verdict"],
        details={
            "feature_drift_top5": cmp["feature_drift_top5"],
            "score_delta": cmp["score_delta"],
            "baseline_id": baseline["id"],
            "baseline_model_name": baseline.get("model_name"),
            "verdict_reason": reason,
        },
    )

    return _json(
        {
            "comparison_id": comparison_id,
            "verdict": cmp["verdict"],
            "cosine_similarity": cmp["cosine_similarity"],
            "score_delta": cmp["score_delta"],
            "feature_drift_top5": cmp["feature_drift_top5"],
            "verdict_reason": reason,
        }
    )


def handle_delete_baseline(path, _qs, _body) -> tuple:
    baseline_id = _extract_id(path, r"/api/v1/baselines/([^/]+)$")
    if not baseline_id:
        return _error("Invalid baseline ID", 400)

    baseline = repo.get_baseline(baseline_id)
    if not baseline:
        return _error("Baseline not found", 404)

    repo.deactivate_baseline(baseline_id)
    return 204, b"", "application/json"


def handle_static(path) -> tuple:
    """Serve the frontend single-page app."""
    frontend_dir = pathlib.Path(__file__).parent.parent.parent / "frontend"
    if path == "/" or path == "":
        file_path = frontend_dir / "index.html"
    else:
        file_path = frontend_dir / path.lstrip("/")

    if not file_path.exists() or not file_path.is_file():
        file_path = frontend_dir / "index.html"  # SPA fallback

    if not file_path.exists():
        return 404, b"Not found", "text/plain"

    content_type = {
        ".html": "text/html; charset=utf-8",
        ".js": "application/javascript",
        ".css": "text/css",
        ".json": "application/json",
        ".png": "image/png",
        ".ico": "image/x-icon",
    }.get(file_path.suffix, "application/octet-stream")

    return 200, file_path.read_bytes(), content_type


# ── Router ─────────────────────────────────────────────────────────────────────

def handle_cancel_run(path, _qs, _body) -> tuple:
    run_id = _extract_id(path, r"/api/v1/runs/([^/]+)/cancel$")
    if not run_id:
        return _error("Invalid run ID", 400)
    run = repo.get_run(run_id)
    if not run:
        return _error("Run not found", 404)
    if run.get("status") not in ("queued", "pre_detecting", "running"):
        return _error("Run is not cancellable in current status", 400)
    repo.set_run_cancel_requested(run_id, True)
    return _json({"run_id": run_id, "status": "cancelling"})


def handle_retry_run(path, _qs, _body) -> tuple:
    run_id = _extract_id(path, r"/api/v1/runs/([^/]+)/retry$")
    if not run_id:
        return _error("Invalid run ID", 400)
    run = repo.get_run(run_id)
    if not run:
        return _error("Run not found", 404)
    if run.get("status") not in ("failed", "partial_failed"):
        return _error("Only failed/partial_failed runs can be retried", 400)
    repo.mark_run_retry(run_id)
    submit_run(run_id)
    return _json({"run_id": run_id, "status": "queued", "resume": True})


def handle_continue_run(path, _qs, _body) -> tuple:
    """User chose to continue full testing after pre_detected state."""
    run_id = _extract_id(path, r"/api/v1/runs/([^/]+)/continue$")
    if not run_id:
        return _error("Invalid run ID", 400)
    run = repo.get_run(run_id)
    if not run:
        return _error("Run not found", 404)
    if run.get("status") != "pre_detected":
        return _error("Run is not in pre_detected state", 400)
    submit_continue(run_id)
    return _json({"run_id": run_id, "status": "running"})


def handle_skip_testing(path, _qs, _body) -> tuple:
    """User chose to skip full testing and generate report from predetect only."""
    run_id = _extract_id(path, r"/api/v1/runs/([^/]+)/skip-testing$")
    if not run_id:
        return _error("Invalid run ID", 400)
    run = repo.get_run(run_id)
    if not run:
        return _error("Run not found", 404)
    if run.get("status") != "pre_detected":
        return _error("Run is not in pre_detected state", 400)
    submit_skip_testing(run_id)
    return _json({"run_id": run_id, "status": "completing"})


ROUTES: list[tuple[str, str, callable]] = [
    ("GET",    r"^/api/v1/health$",                  handle_health),
    ("GET",    r"^/api/v1/runs$",                    handle_list_runs),
    ("POST",   r"^/api/v1/runs$",                    handle_create_run),
    ("GET",    r"^/api/v1/runs/[^/]+$",              handle_get_run),
    ("DELETE", r"^/api/v1/runs/[^/]+$",              handle_delete_run),
    ("POST",   r"^/api/v1/runs/[^/]+/cancel$",       handle_cancel_run),
    ("POST",   r"^/api/v1/runs/[^/]+/retry$",        handle_retry_run),
    ("POST",   r"^/api/v1/runs/[^/]+/continue$",     handle_continue_run),
    ("POST",   r"^/api/v1/runs/[^/]+/skip-testing$",  handle_skip_testing),
    ("GET",    r"^/api/v1/runs/[^/]+/report$",       handle_get_report),
    ("GET",    r"^/api/v1/runs/[^/]+/report\.csv$",  handle_export_report_csv),
    ("GET",    r"^/api/v1/runs/[^/]+/radar\.svg$",   handle_export_radar_svg),
    ("GET",    r"^/api/v1/runs/[^/]+/responses$",    handle_get_responses),
    ("GET",    r"^/api/v1/runs/[^/]+/scorecard$",    handle_get_scorecard),
    ("GET",    r"^/api/v1/runs/[^/]+/extraction-audit$", handle_get_extraction_audit),
    ("GET",    r"^/api/v1/runs/[^/]+/theta-report$", handle_get_theta_report),
    ("GET",    r"^/api/v1/runs/[^/]+/pairwise$",     handle_get_pairwise),
    ("GET",    r"^/api/v1/benchmarks$",              handle_benchmarks),
    ("POST",   r"^/api/v1/baselines$",               handle_create_baseline),
    ("GET",    r"^/api/v1/baselines$",               handle_list_baselines),
    ("POST",   r"^/api/v1/baselines/compare$",       handle_compare_baseline),
    ("DELETE", r"^/api/v1/baselines/[^/]+$",         handle_delete_baseline),
    ("POST",   r"^/api/v1/compare-runs$",            handle_create_compare_run),
    ("GET",    r"^/api/v1/compare-runs$",            handle_list_compare_runs),
    ("GET",    r"^/api/v1/compare-runs/[^/]+$",      handle_get_compare_run),
    ("GET",    r"^/api/v1/models/[^/]+/trend$",      handle_model_trend),
    ("GET",    r"^/api/v1/models/[^/]+/theta-trend$",handle_model_theta_trend),
    ("GET",    r"^/api/v1/leaderboard$",             handle_leaderboard),
    ("GET",    r"^/api/v1/theta-leaderboard$",       handle_theta_leaderboard),
    ("GET",    r"^/api/v1/exports/runs\.zip$",       handle_export_runs_zip),
    ("POST",   r"^/api/v1/calibration/rebuild$",     handle_calibration_rebuild),
    ("POST",   r"^/api/v1/calibration/snapshot$",    handle_calibration_snapshot_only),
    ("POST",   r"^/api/v1/calibration/replay$",      handle_create_calibration_replay),
    ("GET",    r"^/api/v1/calibration/replay$",      handle_list_calibration_replays),
    ("GET",    r"^/api/v1/calibration/replay/[^/]+$", handle_get_calibration_replay),
    ("POST",   r"^/api/v1/tools/generate-isomorphic$", handle_generate_isomorphic),
]


def _extract_id(path: str, pattern: str) -> str | None:
    m = re.search(pattern, path)
    return m.group(1) if m else None


# ── Request handler ────────────────────────────────────────────────────────────

class InspectorHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        logger.info("HTTP " + fmt % args)

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # CORS
        origin = self.headers.get("Origin", "*")
        if origin in settings.CORS_ORIGINS or "*" in settings.CORS_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
        else:
            self.send_header("Access-Control-Allow-Origin",
                             settings.CORS_ORIGINS[0] if settings.CORS_ORIGINS else "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204, b"", "text/plain")

    def _dispatch(self, method: str) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        qs = urllib.parse.parse_qs(parsed.query)

        # Read body for POST
        body: dict = {}
        if method == "POST":
            length = int(self.headers.get("Content-Length", 0))
            if length:
                try:
                    body = json.loads(self.rfile.read(length).decode("utf-8"))
                except json.JSONDecodeError:
                    self._send(*_error("Invalid JSON body"))
                    return

        # API routes
        for route_method, pattern, handler in ROUTES:
            if route_method == method and re.match(pattern, path):
                try:
                    result = handler(path, qs, body)
                    self._send(*result)
                except Exception as e:
                    logger.error("Handler error", path=path, error=str(e))
                    self._send(*_error(f"Internal error: {e}", 500))
                return

        # Static files (frontend)
        if method == "GET":
            self._send(*handle_static(path))
            return

        self._send(*_error("Not found", 404))

    def do_GET(self):    self._dispatch("GET")
    def do_POST(self):   self._dispatch("POST")
    def do_DELETE(self): self._dispatch("DELETE")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    setup_logging()
    logger.info("Initialising database")
    init_db()
    logger.info("Seeding fixtures")
    seed_all()

    host = settings.HOST
    port = settings.PORT
    server = HTTPServer((host, port), InspectorHandler)
    logger.info("Server ready", host=host, port=port,
                url=f"http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down")
        server.server_close()


if __name__ == "__main__":
    main()
