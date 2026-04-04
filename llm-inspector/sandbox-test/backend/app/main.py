"""
LLM Inspector API Server — stdlib http.server, no FastAPI required.
Implements all endpoints from the spec.
"""
from __future__ import annotations

import json
import re
import sys
import pathlib
import threading
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from app.core.config import settings
from app.core.db import init_db
from app.core.logging import setup_logging, get_logger
from app.core.security import validate_and_sanitize_url, get_key_manager
from app.tasks.seeder import seed_all
from app.tasks.worker import submit_run, active_count
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
    return _json({
        "status": "ok",
        "db": "ok",
        "workers_active": active_count(),
        "version": "1.0.0",
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

    api_key: str = body["api_key"]
    if len(api_key) < 4:
        return _error("api_key too short")

    # Encrypt API key
    km = get_key_manager()
    encrypted, key_hash = km.encrypt(api_key)

    test_mode = body.get("test_mode", "standard")
    if test_mode not in ("quick", "standard", "full"):
        test_mode = "standard"

    run_id = repo.create_run(
        base_url=clean_url,
        api_key_encrypted=encrypted,
        api_key_hash=key_hash,
        model_name=body["model"],
        test_mode=test_mode,
        suite_version=body.get("suite_version", "v1"),
    )

    # Submit to background worker
    submit_run(run_id)
    logger.info("Run created", run_id=run_id, model=body["model"])

    return _json({"run_id": run_id, "status": "queued"}, 201)


def handle_get_run(path, _qs, _body) -> tuple:
    run_id = _extract_id(path, r"/api/v1/runs/([^/]+)$")
    if not run_id:
        return _error("Invalid run ID", 400)

    run = repo.get_run(run_id)
    if not run:
        return _error("Run not found", 404)

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
    })


def handle_get_report(path, _qs, _body) -> tuple:
    run_id = _extract_id(path, r"/api/v1/runs/([^/]+)/report$")
    if not run_id:
        return _error("Invalid run ID", 400)

    run = repo.get_run(run_id)
    if not run:
        return _error("Run not found", 404)

    if run["status"] not in ("completed", "partial_failed"):
        return _error("Report not ready yet", 404)

    report_row = repo.get_report(run_id)
    if not report_row:
        return _error("Report not found", 404)

    return _json(report_row["details"])


def handle_get_responses(path, _qs, _body) -> tuple:
    run_id = _extract_id(path, r"/api/v1/runs/([^/]+)/responses$")
    if not run_id:
        return _error("Invalid run ID", 400)

    responses = repo.get_responses(run_id)
    # Strip large raw_response to keep payload small
    slim = []
    for r in responses:
        slim.append({
            "id": r["id"],
            "case_id": r["case_id"],
            "sample_index": r["sample_index"],
            "response_text": (r.get("response_text") or "")[:300],
            "status_code": r.get("status_code"),
            "latency_ms": r.get("latency_ms"),
            "judge_passed": r.get("judge_passed"),
            "error_type": r.get("error_type"),
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


def handle_benchmarks(_path, _qs, _body) -> tuple:
    benchmarks = repo.get_benchmarks("v1")
    return _json([
        {
            "name": b["benchmark_name"],
            "suite_version": b["suite_version"],
            "generated_at": b["generated_at"],
            "sample_count": b.get("sample_count", 3),
        }
        for b in benchmarks
    ])


def handle_static(path) -> tuple:
    """Serve the frontend single-page app."""
    frontend_dir = pathlib.Path(__file__).parent.parent.parent.parent / "frontend"
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

ROUTES: list[tuple[str, str, callable]] = [
    ("GET",    r"^/api/v1/health$",                  handle_health),
    ("GET",    r"^/api/v1/runs$",                    handle_list_runs),
    ("POST",   r"^/api/v1/runs$",                    handle_create_run),
    ("GET",    r"^/api/v1/runs/[^/]+$",              handle_get_run),
    ("DELETE", r"^/api/v1/runs/[^/]+$",              handle_delete_run),
    ("GET",    r"^/api/v1/runs/[^/]+/report$",       handle_get_report),
    ("GET",    r"^/api/v1/runs/[^/]+/responses$",    handle_get_responses),
    ("GET",    r"^/api/v1/benchmarks$",              handle_benchmarks),
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
