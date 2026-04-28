"""
tasks/maintenance_jobs.py — v17 Phase 12.x: background maintenance daemon.

Wraps the four "living suite + living model intelligence" jobs introduced
in Phases 6–10 (model_registry_sync / changelog_harvester / dataset_sync
/ pruner_job) into idempotent periodic tasks running in a single daemon
thread, similar in spirit to ``tasks.watchdog``.

Design
------
* One global daemon thread (`maintenance-jobs`) loops every 30s checking
  per-job ``next_run_at`` timestamps; eligible jobs are run sequentially
  inside the loop so that a slow OpenRouter pull cannot starve another
  job — the next job simply gets a small delay.
* Every job is wrapped in `try/except`: a failing job logs the error
  and is rescheduled normally — it never blocks other jobs or the loop.
* Total opt-in via ``MAINTENANCE_JOBS_ENABLED=1`` env var.  The default
  is **off** to preserve current behaviour for existing deployments;
  ``start.bat``/``start.sh`` set it to ``1`` so fresh installs benefit.

Public surface
--------------
  start_maintenance_jobs(jobs=None, *, check_interval_sec=30) -> bool
      Returns True when the daemon thread was started, False when it
      was skipped (already running / opt-out).

  default_jobs() -> list[MaintenanceJob]
      The four production jobs.

  MaintenanceJob (dataclass)
      Convenient unit-test handle: any callable + interval can be
      registered without touching the real Phase 6/7/9/10 modules.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

from app.core.logging import get_logger

logger = get_logger(__name__)


# ── Tunables ────────────────────────────────────────────────────────────────


_REGISTRY_SYNC_INTERVAL_SEC = 6 * 3600         # 6 hours
_CHANGELOG_HARVEST_INTERVAL_SEC = 24 * 3600    # 1 day
_DATASET_SYNC_INTERVAL_SEC = 7 * 24 * 3600     # 1 week
_PRUNER_INTERVAL_SEC = 3600                    # 1 hour
_DEFAULT_CHECK_INTERVAL_SEC = 30


# ── Job definition ─────────────────────────────────────────────────────────


@dataclass
class MaintenanceJob:
    """One periodic maintenance task.

    ``run_fn`` is invoked with no arguments; its return value is logged
    but otherwise ignored.  Exceptions are caught and counted.
    """
    name: str
    interval_sec: int
    run_fn: Callable[[], object]
    initial_delay_sec: int = 0
    next_run_at: float = field(default=0.0)
    successes: int = field(default=0)
    failures: int = field(default=0)
    last_error: str | None = field(default=None)

    def schedule_initial(self, now: float) -> None:
        self.next_run_at = now + max(0, self.initial_delay_sec)

    def due(self, now: float) -> bool:
        return now >= self.next_run_at

    def run_and_reschedule(self, now: float) -> None:
        try:
            t0 = time.monotonic()
            result = self.run_fn()
            dt = time.monotonic() - t0
            self.successes += 1
            logger.info(
                "maintenance job ok",
                job=self.name,
                duration_sec=round(dt, 2),
                summary=_summarise(result),
            )
        except Exception as e:
            self.failures += 1
            self.last_error = str(e)[:300]
            logger.warning("maintenance job failed", job=self.name, error=self.last_error)
        finally:
            self.next_run_at = now + self.interval_sec


def _summarise(result: object) -> object:
    """Collapse Phase 6/7/9/10 report objects into a small dict for logs."""
    if result is None:
        return None
    # Reports expose ``to_dict``; otherwise return the raw repr (truncated).
    if hasattr(result, "to_dict") and callable(getattr(result, "to_dict")):
        try:
            d = result.to_dict()
        except Exception:
            return None
        # keep only short numeric/string fields
        out: dict[str, object] = {}
        for k, v in d.items():
            if isinstance(v, (int, float, bool)) or v is None:
                out[k] = v
            elif isinstance(v, str) and len(v) <= 80:
                out[k] = v
        return out
    return None


# ── Default job factory ────────────────────────────────────────────────────


def _registry_sync_job() -> object:
    from app.runner.model_registry_sync import run_full_sync
    return run_full_sync(sweep_deprecated=True)


def _changelog_harvest_job() -> object:
    from app.runner.changelog_harvester import run_harvest
    return run_harvest()


def _dataset_sync_job() -> object:
    from app.runner.dataset_sync import run_dataset_sync
    return run_dataset_sync(max_rows_per_source=200)


def _pruner_job() -> object:
    from app.tasks.pruner_job import run_pruner_job
    return run_pruner_job()


def _harvester_enabled() -> bool:
    """``changelog_harvester`` calls ``JUDGE_API_URL`` (a paid LLM) once a
    day. Personal users can run for a week before noticing the bill, so
    the harvester is opt-in via its own env var even when the umbrella
    ``MAINTENANCE_JOBS_ENABLED`` is on.
    """
    raw = os.environ.get("MAINTENANCE_HARVESTER_ENABLED", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def default_jobs() -> list[MaintenanceJob]:
    """Production job set."""
    jobs = [
        MaintenanceJob(
            name="model_registry_sync",
            interval_sec=_REGISTRY_SYNC_INTERVAL_SEC,
            run_fn=_registry_sync_job,
            initial_delay_sec=60,            # let server warm up first
        ),
    ]
    if _harvester_enabled():
        jobs.append(MaintenanceJob(
            name="changelog_harvester",
            interval_sec=_CHANGELOG_HARVEST_INTERVAL_SEC,
            run_fn=_changelog_harvest_job,
            initial_delay_sec=300,           # 5min
        ))
    else:
        logger.info(
            "maintenance jobs: changelog_harvester skipped "
            "(set MAINTENANCE_HARVESTER_ENABLED=1 to enable; "
            "this job calls JUDGE_API_URL once per day, which incurs LLM cost)"
        )
    jobs.extend([
        MaintenanceJob(
            name="dataset_sync",
            interval_sec=_DATASET_SYNC_INTERVAL_SEC,
            run_fn=_dataset_sync_job,
            initial_delay_sec=600,           # 10min
        ),
        MaintenanceJob(
            name="pruner_job",
            interval_sec=_PRUNER_INTERVAL_SEC,
            run_fn=_pruner_job,
            initial_delay_sec=120,           # 2min
        ),
    ])
    return jobs


# ── Daemon thread ──────────────────────────────────────────────────────────


_thread: threading.Thread | None = None
_lock = threading.Lock()
_stop_event = threading.Event()
_active_jobs: list[MaintenanceJob] = []   # populated by start_maintenance_jobs


def get_status() -> dict:
    """Return a JSON-friendly snapshot of every registered maintenance job.

    Used by ``/api/v17/maintenance/status`` so operators can verify the
    daemon is running and inspect per-job success / failure counters.

    Always returns a stable schema, even when the daemon is disabled or
    no jobs are registered::

      {
        "enabled":        bool,           # MAINTENANCE_JOBS_ENABLED
        "running":        bool,           # daemon thread alive
        "jobs": [
            {
              "name":            str,
              "interval_sec":    int,
              "successes":       int,
              "failures":        int,
              "last_error":      str | None,
              "next_run_at":     float (epoch),
              "next_run_in_sec": int  (>=0; 0 = ready to run),
            },
            ...
        ],
      }
    """
    now = time.time()
    running = _thread is not None and _thread.is_alive()
    jobs_view: list[dict] = []
    for j in _active_jobs:
        jobs_view.append({
            "name": j.name,
            "interval_sec": j.interval_sec,
            "successes": j.successes,
            "failures": j.failures,
            "last_error": j.last_error,
            "next_run_at": round(j.next_run_at, 3),
            "next_run_in_sec": max(0, int(j.next_run_at - now)),
        })
    return {
        "enabled": is_enabled(),
        "running": running,
        "jobs": jobs_view,
    }


def is_enabled() -> bool:
    """Return True when MAINTENANCE_JOBS_ENABLED is set to a truthy value."""
    raw = os.environ.get("MAINTENANCE_JOBS_ENABLED", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def stop_maintenance_jobs() -> None:
    """Signal the daemon thread to exit at its next iteration."""
    _stop_event.set()


def start_maintenance_jobs(
    jobs: Iterable[MaintenanceJob] | None = None,
    *,
    check_interval_sec: int = _DEFAULT_CHECK_INTERVAL_SEC,
    force: bool = False,
) -> bool:
    """Start the maintenance daemon.

    Returns True when a thread is launched; False if skipped because
    ``MAINTENANCE_JOBS_ENABLED`` is not set (override with ``force=True``)
    or a daemon is already running.
    """
    global _thread

    if not force and not is_enabled():
        logger.info(
            "maintenance jobs: skipped (MAINTENANCE_JOBS_ENABLED is not set)"
        )
        return False

    with _lock:
        if _thread is not None and _thread.is_alive():
            logger.debug("maintenance jobs: already running")
            return False

        job_list = list(jobs) if jobs is not None else default_jobs()
        if not job_list:
            logger.info("maintenance jobs: no jobs registered, daemon not started")
            return False
        _stop_event.clear()
        now = time.time()
        for j in job_list:
            j.schedule_initial(now)
        # Expose the live job list so ``get_status()`` can introspect it.
        _active_jobs.clear()
        _active_jobs.extend(job_list)
        names = [j.name for j in job_list]

        def _loop():
            logger.info(
                "maintenance jobs: daemon started",
                jobs=names,
                check_interval_sec=check_interval_sec,
            )
            while not _stop_event.is_set():
                now = time.time()
                for job in job_list:
                    if _stop_event.is_set():
                        break
                    if job.due(now):
                        job.run_and_reschedule(now)
                # Sleep in small chunks so stop_event responds quickly
                slept = 0.0
                while slept < check_interval_sec and not _stop_event.is_set():
                    step = min(1.0, check_interval_sec - slept)
                    time.sleep(step)
                    slept += step
            logger.info("maintenance jobs: daemon stopped")

        t = threading.Thread(target=_loop, name="maintenance-jobs", daemon=True)
        t.start()
        _thread = t
        return True


__all__ = [
    "MaintenanceJob",
    "default_jobs",
    "get_status",
    "is_enabled",
    "start_maintenance_jobs",
    "stop_maintenance_jobs",
]
