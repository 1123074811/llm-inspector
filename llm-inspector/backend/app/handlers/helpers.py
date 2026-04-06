"""Shared helpers for handlers."""
from __future__ import annotations

import json
import re
from typing import Callable

from app.core.logging import get_logger
from app.core.schemas import ScoreCard
from app.repository import repo

logger = get_logger(__name__)


def _json(data, status: int = 200) -> tuple[int, bytes, str]:
    body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
    return status, body, "application/json"


def _error(msg: str, status: int = 400) -> tuple[int, bytes, str]:
    return _json({"error": msg}, status)


def _extract_id(path: str, pattern: str) -> str | None:
    m = re.search(pattern, path)
    return m.group(1) if m else None


def _load_report_or_error(run_id: str):
    run = repo.get_run(run_id)
    if not run:
        return None, _error("Run not found", 404)
    if run["status"] not in ("completed", "partial_failed"):
        return None, _error("Report not ready yet", 404)
    report_row = repo.get_report(run_id)
    if not report_row:
        return None, _error("Report not found", 404)
    return report_row["details"], None
