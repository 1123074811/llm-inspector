"""
core/logging.py — Structured JSON logging (v15 shim layer).

All runtime logging (get_logger / setup_logging / set_sse_publisher) is
implemented here.  The v8 StructuredLogger singleton lives in
core/structured_logger.py and is re-exported for backward compatibility.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from app.core.config import settings

# Global SSE publisher (initialized in routes/main)
_sse_publisher = None


def set_sse_publisher(publisher) -> None:
    """Set the global SSE publisher for real-time logs."""
    global _sse_publisher
    _sse_publisher = publisher


class StructuredLogger:
    """Wraps stdlib Logger to accept keyword args like structlog."""

    def __init__(self, name: str):
        self._log = logging.getLogger(name)

    def _fmt(self, msg: str, kwargs: dict) -> str:
        if kwargs:
            extras = " ".join(f"{k}={v!r}" for k, v in kwargs.items())
            return f"{msg} | {extras}"
        return msg

    def _publish_sse(self, level: str, msg: str, kwargs: dict) -> None:
        if _sse_publisher and "run_id" in kwargs:
            try:
                _sse_publisher.publish(kwargs["run_id"], {
                    "level": level,
                    "msg": msg,
                    "kwargs": kwargs,
                    "ts": time.time(),
                })
            except Exception:
                pass

    def info(self, msg: str, **kwargs) -> None:
        self._log.info(self._fmt(msg, kwargs))
        self._publish_sse("info", msg, kwargs)

    def warning(self, msg: str, **kwargs) -> None:
        self._log.warning(self._fmt(msg, kwargs))
        self._publish_sse("warning", msg, kwargs)

    def error(self, msg: str, **kwargs) -> None:
        self._log.error(self._fmt(msg, kwargs))
        self._publish_sse("error", msg, kwargs)

    def debug(self, msg: str, **kwargs) -> None:
        self._log.debug(self._fmt(msg, kwargs))
        self._publish_sse("debug", msg, kwargs)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            log["exception"] = self.formatException(record.exc_info)
        return json.dumps(log, ensure_ascii=False)


def setup_logging() -> None:
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> StructuredLogger:
    return StructuredLogger(name)


# -- Re-export v8 audit logger (renamed from StructuredLogger in v15) ----------
try:
    from app.core.structured_logger import (  # noqa: F401
        AuditLogger,
        get_structured_logger,
        LogLevel as AuditLogLevel,
        LogEventType as AuditLogEventType,
        StructuredLogEntry,
    )
except Exception:
    pass


__all__ = [
    "StructuredLogger",
    "get_logger",
    "set_sse_publisher",
    "setup_logging",
    "JsonFormatter",
]
