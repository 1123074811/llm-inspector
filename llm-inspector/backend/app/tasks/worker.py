"""
Task worker — runs pipelines in a background thread pool.
No Celery required. Swap to Celery when Redis is available.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from app.core.logging import get_logger

logger = get_logger(__name__)

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="inspector-worker")
_running: dict[str, bool] = {}
_lock = threading.Lock()


def submit_run(run_id: str) -> None:
    """Submit a run pipeline to the background thread pool."""
    from app.runner.orchestrator import run_pipeline

    def _task():
        with _lock:
            _running[run_id] = True
        try:
            run_pipeline(run_id)
        except Exception as e:
            logger.error("Pipeline exception", run_id=run_id, error=str(e))
            from app.repository import repo
            repo.update_run_status(run_id, "failed", error_message=str(e)[:500])
        finally:
            with _lock:
                _running.pop(run_id, None)

    _executor.submit(_task)
    logger.info("Run submitted to worker pool", run_id=run_id)


def submit_compare(compare_id: str) -> None:
    """Submit a compare pipeline task."""
    from app.runner.orchestrator import run_compare_pipeline

    task_key = f"compare:{compare_id}"

    def _task():
        with _lock:
            _running[task_key] = True
        try:
            run_compare_pipeline(compare_id)
        except Exception as e:
            logger.error("Compare pipeline exception", compare_id=compare_id, error=str(e))
            from app.repository import repo
            repo.update_compare_run(compare_id, status="failed", details={"error": str(e)[:500]})
        finally:
            with _lock:
                _running.pop(task_key, None)

    _executor.submit(_task)
    logger.info("Compare run submitted to worker pool", compare_id=compare_id)


def is_running(run_id: str) -> bool:
    with _lock:
        return _running.get(run_id, False)


def submit_calibration_replay(replay_id: str) -> None:
    """Submit a calibration replay task."""
    from app.tasks.calibration_replay import run_calibration_replay

    task_key = f"calibration:{replay_id}"

    def _task():
        with _lock:
            _running[task_key] = True
        try:
            run_calibration_replay(replay_id)
        except Exception as e:
            logger.error("Calibration replay exception", replay_id=replay_id, error=str(e))
            from app.repository import repo
            repo.update_calibration_replay(replay_id, status="failed", error_message=str(e)[:500])
        finally:
            with _lock:
                _running.pop(task_key, None)

    _executor.submit(_task)
    logger.info("Calibration replay submitted to worker pool", replay_id=replay_id)


def submit_continue(run_id: str) -> None:
    """Submit continue pipeline (from pre_detected state)."""
    from app.runner.orchestrator import continue_pipeline

    def _task():
        with _lock:
            _running[run_id] = True
        try:
            continue_pipeline(run_id)
        except Exception as e:
            logger.error("Continue pipeline exception", run_id=run_id, error=str(e))
            from app.repository import repo
            repo.update_run_status(run_id, "failed", error_message=str(e)[:500])
        finally:
            with _lock:
                _running.pop(run_id, None)

    _executor.submit(_task)
    logger.info("Continue run submitted to worker pool", run_id=run_id)


def submit_skip_testing(run_id: str) -> None:
    """Submit skip-testing pipeline (generate report from predetect only)."""
    from app.runner.orchestrator import skip_testing_pipeline

    def _task():
        with _lock:
            _running[run_id] = True
        try:
            skip_testing_pipeline(run_id)
        except Exception as e:
            logger.error("Skip testing pipeline exception", run_id=run_id, error=str(e))
            from app.repository import repo
            repo.update_run_status(run_id, "failed", error_message=str(e)[:500])
        finally:
            with _lock:
                _running.pop(run_id, None)

    _executor.submit(_task)
    logger.info("Skip testing submitted to worker pool", run_id=run_id)


def active_count() -> int:
    with _lock:
        return len(_running)
