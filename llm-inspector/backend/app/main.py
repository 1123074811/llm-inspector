"""
LLM Inspector API Server — stdlib http.server, no FastAPI required.
Implements all endpoints from the spec.
"""
from __future__ import annotations

import json
import re
import pathlib
import urllib.parse
import zipfile
from http.server import HTTPServer, BaseHTTPRequestHandler
from app.core.config import settings
from app.core.db import init_db
from app.core.db_migrations import migrate as run_migrations
from app.core.logging import setup_logging, get_logger
from app.handlers.runs import (
    handle_list_runs, handle_create_run, handle_get_run,
    handle_delete_run, handle_cancel_run, handle_retry_run,
    handle_continue_run, handle_skip_testing,
)
from app.handlers.reports import (
    handle_get_report, handle_export_radar_svg,
    handle_get_responses, handle_get_scorecard, handle_get_extraction_audit,
    handle_get_theta_report, handle_get_pairwise, handle_export_runs_zip,
)
from app.handlers.baselines import (
    handle_benchmarks, handle_create_baseline, handle_list_baselines,
    handle_compare_baseline, handle_delete_baseline, handle_get_baseline,
)
from app.handlers.compare import (
    handle_create_compare_run, handle_get_compare_run, handle_list_compare_runs,
)
from app.handlers.calibration import (
    handle_calibration_rebuild, handle_calibration_snapshot_only,
    handle_create_calibration_replay, handle_get_calibration_replay,
    handle_list_calibration_replays,
)
from app.handlers.models import (
    handle_model_trend, handle_leaderboard,
    handle_model_theta_trend, handle_theta_leaderboard,
)
from app.handlers.misc import handle_health, handle_generate_isomorphic, handle_static
from app.handlers.helpers import _json, _error, _extract_id, _load_report_or_error

logger = get_logger(__name__)


ROUTES: list[tuple[str, str, callable]] = [
    ("GET",    r"^/api/v1/health$",                  handle_health),
    ("GET",    r"^/api/v1/runs$",                    handle_list_runs),
    ("POST",   r"^/api/v1/runs$",                    handle_create_run),
    ("GET",    r"^/api/v1/runs/[^/]+$",             handle_get_run),
    ("DELETE", r"^/api/v1/runs/[^/]+$",             handle_delete_run),
    ("POST",   r"^/api/v1/runs/[^/]+/cancel$",      handle_cancel_run),
    ("POST",   r"^/api/v1/runs/[^/]+/retry$",       handle_retry_run),
    ("POST",   r"^/api/v1/runs/[^/]+/continue$",    handle_continue_run),
    ("POST",   r"^/api/v1/runs/[^/]+/skip-testing$",handle_skip_testing),
    ("GET",    r"^/api/v1/runs/[^/]+/report$",      handle_get_report),
    ("GET",    r"^/api/v1/runs/[^/]+/radar\.svg$",  handle_export_radar_svg),
    ("GET",    r"^/api/v1/runs/[^/]+/responses$",   handle_get_responses),
    ("GET",    r"^/api/v1/runs/[^/]+/scorecard$",   handle_get_scorecard),
    ("GET",    r"^/api/v1/runs/[^/]+/extraction-audit$", handle_get_extraction_audit),
    ("GET",    r"^/api/v1/runs/[^/]+/theta-report$", handle_get_theta_report),
    ("GET",    r"^/api/v1/runs/[^/]+/pairwise$",     handle_get_pairwise),
    ("GET",    r"^/api/v1/benchmarks$",              handle_benchmarks),
    ("POST",   r"^/api/v1/baselines$",               handle_create_baseline),
    ("GET",    r"^/api/v1/baselines$",               handle_list_baselines),
    ("POST",   r"^/api/v1/baselines/compare$",       handle_compare_baseline),
    ("DELETE", r"^/api/v1/baselines/[^/]+$",         handle_delete_baseline),
    ("GET",    r"^/api/v1/baselines/[^/]+$",         handle_get_baseline),
    ("POST",   r"^/api/v1/compare-runs$",            handle_create_compare_run),
    ("GET",    r"^/api/v1/compare-runs$",            handle_list_compare_runs),
    ("GET",    r"^/api/v1/compare-runs/[^/]+$",      handle_get_compare_run),
    ("GET",    r"^/api/v1/models/[^/]+/trend$",      handle_model_trend),
    ("GET",    r"^/api/v1/models/[^/]+/theta-trend$",handle_model_theta_trend),
    ("GET",    r"^/api/v1/leaderboard$",             handle_leaderboard),
    ("GET",    r"^/api/v1/theta-leaderboard$",       handle_theta_leaderboard),
    ("GET",    r"^/api/v1/exports/runs\.zip$",       handle_export_runs_zip),
    ("POST",   r"^/api/v1/calibration/rebuild$",      handle_calibration_rebuild),
    ("POST",   r"^/api/v1/calibration/snapshot$",     handle_calibration_snapshot_only),
    ("POST",   r"^/api/v1/calibration/replay$",      handle_create_calibration_replay),
    ("GET",    r"^/api/v1/calibration/replay$",       handle_list_calibration_replays),
    ("GET",    r"^/api/v1/calibration/replay/[^/]+$",handle_get_calibration_replay),
    ("POST",   r"^/api/v1/tools/generate-isomorphic$", handle_generate_isomorphic),
]


class InspectorHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        logger.info("HTTP " + fmt % args)

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
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

        body: dict = {}
        if method == "POST":
            length = int(self.headers.get("Content-Length", 0))
            if length:
                try:
                    body = json.loads(self.rfile.read(length).decode("utf-8"))
                except json.JSONDecodeError:
                    self._send(*_error("Invalid JSON body"))
                    return

        for route_method, pattern, handler in ROUTES:
            if route_method == method and re.match(pattern, path):
                try:
                    result = handler(path, qs, body)
                    self._send(*result)
                except Exception as e:
                    logger.error("Handler error", path=path, error=str(e))
                    self._send(*_error(f"Internal error: {e}", 500))
                return

        if method == "GET":
            status, body, content_type = handle_static(path)
            if status == 404:
                self._send(status, body, content_type)
                return
            self._send(status, body, content_type)
            return

        self._send(*_error("Not found", 404))

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_DELETE(self):
        self._dispatch("DELETE")


def main() -> None:
    setup_logging()
    logger.info("Initialising database")
    init_db()
    logger.info("Running database migrations")
    from app.core.db import get_conn
    run_migrations(get_conn())
    logger.info("Seeding fixtures")
    from app.tasks.seeder import seed_all
    seed_all()

    host = settings.HOST
    port = settings.PORT
    server = HTTPServer((host, port), InspectorHandler)
    logger.info("Server ready", host=host, port=port,
                url=f"http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
