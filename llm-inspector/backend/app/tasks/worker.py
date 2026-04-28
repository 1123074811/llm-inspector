"""
Task worker — runs pipelines in a background thread pool.

v17 Phase 0: removed async pipeline branch (ASYNC_PIPELINE_ENABLED was always
false) and Celery distributed queue path (no celery dependency in pyproject).
Local ThreadPoolExecutor via app.tasks.queue is the only execution backend.
"""
from __future__ import annotations

import threading

from app.core.logging import get_logger
from app.tasks.queue import get_queue, submit_task
from app.tasks.watchdog import start_background_watchdog

logger = get_logger(__name__)

# Fallback lock for local task tracking when not using the unified queue abstraction
_local_lock = threading.Lock()
_local_running: dict[str, bool] = {}

# v13 Phase 4: Start watchdog at module load time so stale runs are cleaned up
# even after a server restart.
start_background_watchdog(interval_sec=300)

# v17 Phase 12.x: opt-in periodic maintenance daemon.
# Controlled by env var MAINTENANCE_JOBS_ENABLED — start.bat / start.sh set
# it to 1 by default, but existing deployments stay opt-in.
try:
    from app.tasks.maintenance_jobs import (
        start_maintenance_jobs,
        stop_maintenance_jobs,
    )
    start_maintenance_jobs()

    # Without an explicit signal handler, the daemon's 30s polling loop
    # delays SIGTERM/SIGINT response — Ctrl+C can hang the process for
    # up to half a minute. Register handlers so shutdown is prompt.
    # signal.signal must be called from the main thread; guard for
    # non-main-thread imports (e.g. test harnesses).
    import signal as _signal
    if threading.current_thread() is threading.main_thread():
        try:
            _prev_term = _signal.getsignal(_signal.SIGTERM)
            _prev_int = _signal.getsignal(_signal.SIGINT)

            def _shutdown_handler(signum, frame):  # type: ignore[no-untyped-def]
                try:
                    stop_maintenance_jobs()
                finally:
                    prev = _prev_term if signum == _signal.SIGTERM else _prev_int
                    if callable(prev) and prev not in (_signal.SIG_DFL, _signal.SIG_IGN):
                        prev(signum, frame)
                    else:
                        # Default behaviour: re-raise the signal so the
                        # interpreter exits as the user expects.
                        _signal.signal(signum, _signal.SIG_DFL)
                        import os as _os
                        _os.kill(_os.getpid(), signum)

            _signal.signal(_signal.SIGTERM, _shutdown_handler)
            _signal.signal(_signal.SIGINT, _shutdown_handler)
        except (ValueError, OSError) as _sig_exc:
            # ValueError: not in main thread; OSError: signal unsupported on platform
            logger.debug("maintenance shutdown signal handlers skipped", error=str(_sig_exc))
except Exception as _maint_exc:
    logger.warning("maintenance jobs failed to start", error=str(_maint_exc))


def submit_run(run_id: str) -> None:
    """Submit a run pipeline to the background task queue."""

    def _task():
        with _local_lock:
            _local_running[run_id] = True
        try:
            from app.runner.orchestrator import run_pipeline
            run_pipeline(run_id)
            logger.info("Run finished", run_id=run_id)
        except Exception as e:
            logger.error("Pipeline exception", run_id=run_id, error=str(e))
            from app.repository import repo
            repo.update_run_status(run_id, "failed", error_message=str(e)[:500])
        finally:
            with _local_lock:
                _local_running.pop(run_id, None)

    submit_task(run_id, _task)
    logger.info("Run submitted to task queue", run_id=run_id)



def submit_compare(compare_id: str) -> None:
    """Submit a compare pipeline task."""
    from app.runner.orchestrator import run_compare_pipeline

    task_key = f"compare:{compare_id}"

    def _task():
        with _local_lock:
            _local_running[task_key] = True
        try:
            run_compare_pipeline(compare_id)
        except Exception as e:
            logger.error("Compare pipeline exception", compare_id=compare_id, error=str(e))
            from app.repository import repo
            repo.update_compare_run(compare_id, status="failed", details={"error": str(e)[:500]})
        finally:
            with _local_lock:
                _local_running.pop(task_key, None)

    submit_task(task_key, _task)
    logger.info("Compare run submitted to task queue", compare_id=compare_id)


def is_running(run_id: str) -> bool:
    """v10: Use unified queue to check running status."""
    return get_queue().is_running(run_id) or _local_running.get(run_id, False)


def submit_calibration_replay(replay_id: str) -> None:
    """Submit a calibration replay task."""
    from app.tasks.calibration_replay import run_calibration_replay

    task_key = f"calibration:{replay_id}"

    def _task():
        with _local_lock:
            _local_running[task_key] = True
        try:
            run_calibration_replay(replay_id)
        except Exception as e:
            logger.error("Calibration replay exception", replay_id=replay_id, error=str(e))
            from app.repository import repo
            repo.update_calibration_replay(replay_id, status="failed", error_message=str(e)[:500])
        finally:
            with _local_lock:
                _local_running.pop(task_key, None)

    submit_task(task_key, _task)
    logger.info("Calibration replay submitted to task queue", replay_id=replay_id)


def submit_continue(run_id: str) -> None:
    """Submit continue pipeline (from pre_detected state)."""
    from app.runner.orchestrator import continue_pipeline

    def _task():
        with _local_lock:
            _local_running[run_id] = True
        try:
            continue_pipeline(run_id)
        except Exception as e:
            logger.error("Continue pipeline exception", run_id=run_id, error=str(e))
            from app.repository import repo
            repo.update_run_status(run_id, "failed", error_message=str(e)[:500])
        finally:
            with _local_lock:
                _local_running.pop(run_id, None)

    submit_task(f"continue:{run_id}", _task)
    logger.info("Continue run submitted to task queue", run_id=run_id)


def submit_skip_testing(run_id: str) -> None:
    """Submit skip-testing pipeline (generate report from predetect only)."""
    from app.runner.orchestrator import skip_testing_pipeline

    def _task():
        with _local_lock:
            _local_running[run_id] = True
        try:
            skip_testing_pipeline(run_id)
        except Exception as e:
            logger.error("Skip testing pipeline exception", run_id=run_id, error=str(e))
            from app.repository import repo
            repo.update_run_status(run_id, "failed", error_message=str(e)[:500])
        finally:
            with _local_lock:
                _local_running.pop(run_id, None)

    submit_task(f"skip:{run_id}", _task)
    logger.info("Skip testing submitted to task queue", run_id=run_id)


def active_count() -> int:
    with _local_lock:
        return len(_local_running)


