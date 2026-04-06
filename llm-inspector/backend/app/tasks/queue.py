"""
Task queue abstraction layer.
Supports local (ThreadPoolExecutor) and distributed (Redis/Celery) backends.
"""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from app.core.logging import get_logger

logger = get_logger(__name__)


class TaskQueue(ABC):
    """Abstract task queue interface."""

    @abstractmethod
    def submit(self, task_id: str, fn: Callable, *args, **kwargs) -> None:
        """Submit a task for execution."""
        raise NotImplementedError

    @abstractmethod
    def is_running(self, task_id: str) -> bool:
        """Check if a task is currently running."""
        raise NotImplementedError

    @abstractmethod
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the queue."""
        raise NotImplementedError


class LocalTaskQueue(TaskQueue):
    """Local task queue using ThreadPoolExecutor."""

    def __init__(self, max_workers: int = 4):
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="inspector-worker"
        )
        self._running: dict[str, bool] = {}
        self._lock = threading.Lock()

    def submit(self, task_id: str, fn: Callable, *args, **kwargs) -> None:
        def _task():
            with self._lock:
                self._running[task_id] = True
            try:
                fn(*args, **kwargs)
            except Exception as e:
                logger.error("Task exception", task_id=task_id, error=str(e))
                raise
            finally:
                with self._lock:
                    self._running.pop(task_id, None)

        self._executor.submit(_task)
        logger.info("Task submitted to local queue", task_id=task_id)

    def is_running(self, task_id: str) -> bool:
        with self._lock:
            return self._running.get(task_id, False)

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)


_queue: TaskQueue | None = None
_queue_lock = threading.Lock()


def get_queue() -> TaskQueue:
    """Get the global task queue instance."""
    global _queue
    with _queue_lock:
        if _queue is None:
            _queue = LocalTaskQueue()
        return _queue


def init_queue(queue: TaskQueue | None = None) -> None:
    """Initialize the global task queue with a custom implementation."""
    global _queue
    with _queue_lock:
        if _queue is not None:
            _queue.shutdown(wait=False)
        _queue = queue or LocalTaskQueue()


def submit_task(task_id: str, fn: Callable, *args, **kwargs) -> None:
    """Submit a task to the global queue."""
    get_queue().submit(task_id, fn, *args, **kwargs)
