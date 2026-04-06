"""
Celery task queue implementation (distributed mode).
Enable by setting CELERY_BROKER_URL and calling init_queue().

Example:
    from app.tasks.worker import init_distributed_queue
    init_distributed_queue()  # Reads CELERY_BROKER_URL from env
"""
from __future__ import annotations

import os
import threading
from typing import Callable

from app.core.logging import get_logger
from app.tasks.queue import TaskQueue

logger = get_logger(__name__)


class CeleryTaskQueue(TaskQueue):
    """
    Distributed task queue using Celery with Redis broker.
    Requires: celery[redis]>=5.0
    """

    PENDING_STATES = {"PENDING", "STARTED", "RECEIVED"}
    FINISHED_STATES = {"SUCCESS", "FAILURE", "REVOKED"}

    def __init__(self, broker_url: str | None = None, result_backend: str | None = None):
        self.broker_url = broker_url or os.environ.get(
            "CELERY_BROKER_URL", "redis://localhost:6379/0"
        )
        self.result_backend = result_backend or os.environ.get(
            "CELERY_RESULT_BACKEND", "redis://localhost:6379/0"
        )
        self._app = None
        self._results: dict[str, object] = {}
        self._lock = threading.Lock()

    def _get_app(self):
        if self._app is None:
            from celery import Celery
            self._app = Celery(
                "llm_inspector",
                broker=self.broker_url,
                backend=self.result_backend,
            )
            self._app.conf.update(
                task_serializer="json",
                accept_content=["json"],
                result_serializer="json",
                timezone="UTC",
                enable_utc=True,
            )
        return self._app

    def submit(self, task_id: str, fn: Callable, *args, **kwargs) -> None:
        from celery.result import AsyncResult

        app = self._get_app()

        @app.task(name=f"llm_inspector.{task_id}", bind=True, ignore_result=False)
        def _celery_task(*args, **kwargs):
            return fn(*args, **kwargs)

        result = _celery_task.apply_async(args=args, kwargs=kwargs)

        with self._lock:
            self._results[task_id] = result

        logger.info("Task submitted to Celery", task_id=task_id, celery_task_id=result.id)

    def is_running(self, task_id: str) -> bool:
        with self._lock:
            result = self._results.get(task_id)

        if result is None:
            return False

        state = result.state
        return state in self.PENDING_STATES

    def shutdown(self, wait: bool = True) -> None:
        with self._lock:
            self._results.clear()

        if self._app:
            self._app.close()
            self._app = None
