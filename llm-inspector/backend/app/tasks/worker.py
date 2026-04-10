"""
Task worker — runs pipelines in a background thread pool.
Supports local (ThreadPoolExecutor) and distributed (Redis/Celery) backends.

To enable distributed mode, set:
    CELERY_BROKER_URL=redis://localhost:6379/0
    CELERY_RESULT_BACKEND=redis://localhost:6379/0
Then implement CeleryTaskQueue in tasks/queue.py and call init_queue(CeleryTaskQueue()).
"""
from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor

from app.core.logging import get_logger
from app.tasks.queue import LocalTaskQueue, get_queue

logger = get_logger(__name__)

_local_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="inspector-worker")
_local_running: dict[str, bool] = {}
_local_lock = threading.Lock()


def submit_run(run_id: str) -> None:
    """Submit a run pipeline to the background thread pool.

    When the environment variable ASYNCIO_MODE=1 is set, the async pipeline
    (run_pipeline_async) is used instead of the sync pipeline, giving higher
    concurrency and lower per-task overhead.
    """
    use_async = os.environ.get("ASYNCIO_MODE", "0") == "1"

    if use_async:
        from app.runner.orchestrator import run_pipeline_async
        import asyncio

        def _task():
            with _local_lock:
                _local_running[run_id] = True
            try:
                asyncio.run(run_pipeline_async(run_id))
            except Exception as e:
                logger.error("Async pipeline exception", run_id=run_id, error=str(e))
                from app.repository import repo
                repo.update_run_status(run_id, "failed", error_message=str(e)[:500])
            finally:
                with _local_lock:
                    _local_running.pop(run_id, None)

        _local_executor.submit(_task)
        logger.info("Run submitted to async worker pool", run_id=run_id)
        return

    from app.runner.orchestrator import run_pipeline

    def _task():
        with _local_lock:
            _local_running[run_id] = True
        try:
            run_pipeline(run_id)
        except Exception as e:
            logger.error("Pipeline exception", run_id=run_id, error=str(e))
            from app.repository import repo
            repo.update_run_status(run_id, "failed", error_message=str(e)[:500])
        finally:
            with _local_lock:
                _local_running.pop(run_id, None)

    _local_executor.submit(_task)
    logger.info("Run submitted to worker pool", run_id=run_id)



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

    _local_executor.submit(_task)
    logger.info("Compare run submitted to worker pool", compare_id=compare_id)


def is_running(run_id: str) -> bool:
    with _local_lock:
        return _local_running.get(run_id, False)


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

    _local_executor.submit(_task)
    logger.info("Calibration replay submitted to worker pool", replay_id=replay_id)


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

    _local_executor.submit(_task)
    logger.info("Continue run submitted to worker pool", run_id=run_id)


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

    _local_executor.submit(_task)
    logger.info("Skip testing submitted to worker pool", run_id=run_id)


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
        init_queue(CeleryTaskQueue(broker_url))
        logger.info("Distributed task queue initialized", broker_url=broker_url)
        return True
    except ImportError as e:
        logger.warning("Celery not available, falling back to local queue", error=str(e))
        return False


def init_queue(queue) -> None:
    """Initialize the global task queue."""
    global _local_executor, _local_running
    if _local_executor:
        _local_executor.shutdown(wait=False)
        _local_running.clear()
    from app.tasks import queue
    queue.init_queue(queue)
    logger.info("Task queue initialized")
