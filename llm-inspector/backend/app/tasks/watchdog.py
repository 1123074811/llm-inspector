"""
tasks/watchdog.py — Run watchdog: marks stale 'running' tasks as partial_failed.

Called periodically (or once at worker startup) to find runs that have been
in a non-terminal state for longer than settings.RUN_MAX_DURATION_SEC (default: 1800s).

Usage:
    from app.tasks.watchdog import RunWatchdog
    RunWatchdog().run_once()  # called at startup and optionally in a background thread
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timezone

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Statuses that are considered non-terminal (i.e., the run is still "in progress")
_STALE_STATUSES = ("running", "pre_detecting")

# Statuses that are already terminal — watchdog must not touch these
_TERMINAL_STATUSES = frozenset(("completed", "failed", "partial_failed", "cancelled", "pre_detected", "suspended"))


def _parse_iso(ts: str | None) -> float | None:
    """Parse an ISO-8601 timestamp string to a Unix epoch float. Returns None on failure."""
    if not ts:
        return None
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            dt = datetime.strptime(ts, fmt)
            return dt.replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    return None


class RunWatchdog:
    """
    Scans for stale runs and transitions them to partial_failed.

    A run is considered stale if it has been in a non-terminal status for
    longer than settings.RUN_MAX_DURATION_SEC seconds.
    """

    def __init__(self, max_duration_sec: int | None = None):
        self._max_duration_sec = max_duration_sec if max_duration_sec is not None else settings.RUN_MAX_DURATION_SEC

    def run_once(self) -> int:
        """
        Scan for stale runs and mark them as partial_failed.

        Returns the number of runs that were transitioned.
        """
        from app.repository import repo

        marked = 0
        now = time.time()

        try:
            all_runs = repo.list_runs(limit=500)
        except Exception as exc:
            logger.error("Watchdog: failed to list runs", error=str(exc))
            return 0

        for run in all_runs:
            status = run.get("status", "")
            if status not in _STALE_STATUSES:
                continue

            run_id = run.get("id", "")

            # Prefer updated_at, fall back to created_at for age computation
            ref_ts = _parse_iso(run.get("updated_at")) or _parse_iso(run.get("created_at"))
            if ref_ts is None:
                continue

            age_sec = now - ref_ts
            if age_sec <= self._max_duration_sec:
                continue

            # Double-check current status in DB before mutating
            try:
                current = repo.get_run(run_id)
                if not current:
                    continue
                if current.get("status") in _TERMINAL_STATUSES:
                    continue
                repo.update_run_status(
                    run_id,
                    "partial_failed",
                    error_message="Watchdog: run exceeded max duration",
                    error_code="E_WATCHDOG_TIMEOUT",
                )
                logger.warning(
                    "Watchdog: marked stale run as partial_failed",
                    run_id=run_id,
                    age_sec=round(age_sec, 1),
                    status_was=status,
                )
                marked += 1
            except Exception as exc:
                logger.error("Watchdog: failed to update run", run_id=run_id, error=str(exc))

        if marked:
            logger.info("Watchdog: scan complete", stale_marked=marked)
        return marked


# ── Background watchdog thread ────────────────────────────────────────────────

_watchdog_thread: threading.Thread | None = None
_watchdog_lock = threading.Lock()


def start_background_watchdog(interval_sec: int = 300) -> None:
    """
    Launch a daemon thread that runs RunWatchdog().run_once() every interval_sec seconds.

    Safe to call multiple times — only one background watchdog thread is started.
    """
    global _watchdog_thread

    with _watchdog_lock:
        if _watchdog_thread is not None and _watchdog_thread.is_alive():
            logger.debug("Background watchdog already running")
            return

        def _loop():
            logger.info("Watchdog background thread started", interval_sec=interval_sec)
            # Run immediately at startup to catch any stale runs from a previous crash
            try:
                RunWatchdog().run_once()
            except Exception as exc:
                logger.error("Watchdog initial scan failed", error=str(exc))

            while True:
                time.sleep(interval_sec)
                try:
                    RunWatchdog().run_once()
                except Exception as exc:
                    logger.error("Watchdog periodic scan failed", error=str(exc))

        t = threading.Thread(target=_loop, name="run-watchdog", daemon=True)
        t.start()
        _watchdog_thread = t
        logger.info("Background watchdog thread launched", interval_sec=interval_sec)
