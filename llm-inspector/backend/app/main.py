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
    handle_continue_run, handle_skip_testing, handle_batch_delete_runs,
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
    handle_elo_leaderboard,
)
from app.handlers.misc import handle_health, handle_generate_isomorphic, handle_static
from app.handlers.v8_handlers import (
    handle_v8_health,
    handle_v8_judgment_logs,
    handle_v8_case_provenance,
    handle_v8_data_lineage,
    handle_v8_plugin_stats,
    handle_v8_plugin_metadata,
    handle_v8_threshold_references,
    handle_v8_list_plugins,
)
from app.handlers.v11_handlers import (
    handle_circuit_breaker_status,
    handle_circuit_breaker_reset,
    handle_run_trace,
    handle_tracer_progress_all,
    handle_run_cdm,
    handle_run_attribution,
    handle_cdm_skills,
    # Phase 3
    handle_suite_prune,
    handle_suite_pruning_report,
    handle_prompt_optimizer_report,
    handle_gpqa_questions,
    handle_multilingual_attacks,
)

from app.handlers.helpers import _json, _error, _extract_id, _load_report_or_error

logger = get_logger(__name__)

# v10 SSE handler
from app.core.sse import publisher as sse_publisher
from app.core.logging import set_sse_publisher
import time

def handle_sse_logs(path: str, qs: dict, body: dict):
    """v10 SSE endpoint for real-time logs.
    Handled specially in _dispatch due to streaming nature.
    """
    pass

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
    ("GET",    r"^/api/v1/elo-leaderboard$",         handle_elo_leaderboard),
    ("POST",   r"^/api/v1/runs/batch-delete$",      handle_batch_delete_runs),
    ("GET",    r"^/api/v1/exports/runs\.zip$",       handle_export_runs_zip),
    ("POST",   r"^/api/v1/calibration/rebuild$",      handle_calibration_rebuild),
    ("POST",   r"^/api/v1/calibration/snapshot$",     handle_calibration_snapshot_only),
    ("POST",   r"^/api/v1/calibration/replay$",      handle_create_calibration_replay),
    ("GET",    r"^/api/v1/calibration/replay$",       handle_list_calibration_replays),
    ("GET",    r"^/api/v1/calibration/replay/[^/]+$",handle_get_calibration_replay),
    ("POST",   r"^/api/v1/tools/generate-isomorphic$", handle_generate_isomorphic),
    # v8.0 Phase 5 routes
    ("GET",    r"^/api/v8/health$",                    handle_v8_health),
    ("GET",    r"^/api/v8/plugins$",                    handle_v8_list_plugins),
    ("GET",    r"^/api/v8/plugins/[^/]+/metadata$",   handle_v8_plugin_metadata),
    ("GET",    r"^/api/v8/plugin-stats$",              handle_v8_plugin_stats),
    ("GET",    r"^/api/v8/runs/[^/]+/judgment-logs$", handle_v8_judgment_logs),
    ("GET",    r"^/api/v8/runs/[^/]+/case/[^/]+/provenance$", handle_v8_case_provenance),
    ("GET",    r"^/api/v8/runs/[^/]+/data-lineage$",  handle_v8_data_lineage),
    ("GET",    r"^/api/v8/references/thresholds$",     handle_v8_threshold_references),
    # v11 Phase 1 routes
    ("GET",    r"^/api/v1/circuit-breaker$",           handle_circuit_breaker_status),
    ("POST",   r"^/api/v1/circuit-breaker/reset$",     handle_circuit_breaker_reset),
    ("GET",    r"^/api/v1/runs/[^/]+/trace$",          handle_run_trace),
    ("GET",    r"^/api/v1/tracers/progress$",           handle_tracer_progress_all),
    # v11 Phase 2: CDM + Attribution
    ("GET",    r"^/api/v1/runs/[^/]+/cdm$",            handle_run_cdm),
    ("GET",    r"^/api/v1/runs/[^/]+/attribution$",    handle_run_attribution),
    ("GET",    r"^/api/v1/cdm/skills$",                 handle_cdm_skills),
    # v11 Phase 3: Suite Pruning + Prompt Optimizer + GPQA + Multilingual
    ("POST",   r"^/api/v1/suite/prune$",                handle_suite_prune),
    ("GET",    r"^/api/v1/suite/pruning-report$",       handle_suite_pruning_report),
    ("GET",    r"^/api/v1/prompt-optimizer/report$",    handle_prompt_optimizer_report),
    ("GET",    r"^/api/v1/gpqa/questions$",             handle_gpqa_questions),
    ("GET",    r"^/api/v1/attacks/multilingual$",       handle_multilingual_attacks),
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

        # v10: Handle SSE specially because it streams
        if method == "GET" and re.match(r"^/api/v10/runs/[^/]+/logs/stream$", path):
            run_id = path.split("/")[-3]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            import queue
            q = queue.Queue()
            def _listener(data: bytes):
                q.put(data)
            
            sse_publisher.subscribe(run_id, _listener)
            try:
                # Send initial ping
                self.wfile.write(b"event: ping\ndata: connected\n\n")
                self.wfile.flush()
                
                # Keep connection alive and stream logs
                while True:
                    try:
                        data = q.get(timeout=15)
                        self.wfile.write(data)
                        self.wfile.flush()
                    except queue.Empty:
                        self.wfile.write(b"event: ping\ndata: heartbeat\n\n")
                        self.wfile.flush()
            except Exception:
                pass
            finally:
                sse_publisher.unsubscribe(run_id, _listener)
            return

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


def run_server(host: str = None, port: int = None) -> None:
    """Run the HTTP server with optional host/port override."""
    # Wire up SSE to logger
    set_sse_publisher(sse_publisher)

    setup_logging()
    logger.info("Initialising database")
    init_db()
    logger.info("Running database migrations")
    from app.core.db import get_conn
    run_migrations(get_conn())
    logger.info("Seeding fixtures")
    from app.tasks.seeder import seed_all
    seed_all()

    host = host or settings.HOST
    port = port or settings.PORT
    server = HTTPServer((host, port), InspectorHandler)
    logger.info("Server ready", host=host, port=port,
                url=f"http://{host}:{port}")
    server.serve_forever()


def main() -> None:
    """Main entry point - uses settings from config."""
    run_server()


if __name__ == "__main__":
    main()
