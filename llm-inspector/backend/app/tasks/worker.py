"""
Task worker — runs pipelines in a background thread pool.
Supports local (ThreadPoolExecutor) and distributed (Redis/Celery) backends.

To enable distributed mode, set:
    CELERY_BROKER_URL=redis://localhost:6379/0
    CELERY_RESULT_BACKEND=redis://localhost:6379/0
Then implement CeleryTaskQueue in tasks/queue.py and call init_queue(CeleryTaskQueue()).
"""
from __future__ import annotations

import asyncio
import threading
import os

from app.core.logging import get_logger
from app.core.config import settings
from app.tasks.queue import get_queue, submit_task
from app.tasks.watchdog import start_background_watchdog

logger = get_logger(__name__)

# Fallback lock for local task tracking when not using the unified queue abstraction
_local_lock = threading.Lock()
_local_running: dict[str, bool] = {}

# v13 Phase 4: Start watchdog at module load time so stale runs are cleaned up
# even after a server restart.
start_background_watchdog(interval_sec=300)


def submit_run(run_id: str) -> None:
    """Submit a run pipeline to the background task queue.

    v10: Replaced raw ThreadPoolExecutor with unified TaskQueue abstraction
    that natively supports Celery.
    """
    use_async = settings.ASYNC_PIPELINE_ENABLED

    def _task():
        with _local_lock:
            _local_running[run_id] = True
        try:
            if use_async:
                from app.runner.orchestrator import run_pipeline_async
                try:
                    asyncio.run(run_pipeline_async(run_id))
                    logger.info("Run finished with async pipeline", run_id=run_id)
                    return
                except Exception as async_err:
                    logger.error("Async pipeline exception", run_id=run_id, error=str(async_err))
                    if not settings.ASYNC_PIPELINE_FALLBACK_SYNC:
                        from app.repository import repo
                        repo.update_run_status(run_id, "failed", error_message=str(async_err)[:500])
                        return
                    logger.warning("Falling back to sync pipeline", run_id=run_id)

            from app.runner.orchestrator import run_pipeline
            run_pipeline(run_id)
            logger.info("Run finished with sync pipeline", run_id=run_id)
        except Exception as e:
            logger.error("Pipeline exception", run_id=run_id, error=str(e))
            from app.repository import repo
            repo.update_run_status(run_id, "failed", error_message=str(e)[:500])
        finally:
            with _local_lock:
                _local_running.pop(run_id, None)

    submit_task(run_id, _task)
    logger.info(
        "Run submitted to task queue",
        run_id=run_id,
        async_enabled=use_async,
        async_fallback_sync=settings.ASYNC_PIPELINE_FALLBACK_SYNC,
    )



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


def init_distributed_queue() -> bool:
    """
    Initialize distributed queue if CELERY_BROKER_URL is configured.
    Returns True if distributed mode is enabled.
    """
    broker_url = os.environ.get("CELERY_BROKER_URL")
    if not broker_url:
        logger.info("Using local task queue (set CELERY_BROKER_URL for distributed mode)")
        return False

    try:
        from app.tasks.celery_queue import CeleryTaskQueue
        from app.tasks.queue import init_queue
        init_queue(CeleryTaskQueue(broker_url))
        logger.info("Distributed task queue initialized", broker_url=broker_url)
        return True
    except ImportError as e:
        logger.warning("Celery not available, falling back to local queue", error=str(e))
        return False
    logger.info("Task queue initialized")
