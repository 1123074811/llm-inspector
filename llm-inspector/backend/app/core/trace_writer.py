"""
core/trace_writer.py — v16 Phase 8: Standardized JSONL Trace Writer

Writes structured trace files to data/traces/{run_id}/:
  - preflight.jsonl
  - predetect.jsonl
  - judge_chain.jsonl
  - errors.jsonl
  - token_audit.jsonl (Phase 7)
"""
from __future__ import annotations

import json
import pathlib as _pl
from datetime import datetime, timezone
from threading import Lock

from app.core.logging import get_logger

logger = get_logger(__name__)

_DATA_DIR = _pl.Path(__file__).resolve().parent.parent.parent / "data" / "traces"


class TraceWriter:
    """Thread-safe JSONL trace writer for a single run."""

    SCHEMAS = {
        "preflight": {"required": ["step", "name", "passed", "duration_ms"]},
        "predetect": {"required": ["layer", "name", "started_at", "duration_ms"]},
        "judge_chain": {"required": ["case_id", "judge_chain_step", "method", "verdict"]},
        "errors": {"required": ["attempt", "error_type", "action"]},
        "token_audit": {"required": ["case_id", "prompt_tokens", "completion_tokens"]},
    }

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._run_dir = _DATA_DIR / run_id
        self._locks: dict[str, Lock] = {}
        self._counts: dict[str, int] = {}

    def _get_lock(self, trace_type: str) -> Lock:
        if trace_type not in self._locks:
            self._locks[trace_type] = Lock()
        return self._locks[trace_type]

    def write(self, trace_type: str, record: dict) -> int:
        """
        Write a single JSONL record to the specified trace file.

        Args:
            trace_type: One of "preflight", "predetect", "judge_chain", "errors", "token_audit".
            record: Dict to serialize as JSONL.

        Returns:
            Number of lines written (1 on success, 0 on failure).
        """
        if trace_type not in self.SCHEMAS:
            logger.warning("Unknown trace type", trace_type=trace_type)
            return 0

        # Add timestamp if not present
        if "timestamp" not in record:
            record["timestamp"] = datetime.now(timezone.utc).isoformat()

        # Validate required fields (soft — warn but don't block)
        schema = self.SCHEMAS[trace_type]
        for field_name in schema.get("required", []):
            if field_name not in record:
                logger.warning(
                    "Trace record missing field",
                    trace_type=trace_type,
                    field=field_name,
                )

        lock = self._get_lock(trace_type)
        with lock:
            self._run_dir.mkdir(parents=True, exist_ok=True)
            path = self._run_dir / f"{trace_type}.jsonl"
            line = json.dumps(record, ensure_ascii=False) + "\n"
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
            self._counts[trace_type] = self._counts.get(trace_type, 0) + 1

        return 1

    def write_batch(self, trace_type: str, records: list[dict]) -> int:
        """Write multiple records to a trace file."""
        count = 0
        for record in records:
            count += self.write(trace_type, record)
        return count

    def count(self, trace_type: str) -> int:
        """Return number of records written for a trace type."""
        return self._counts.get(trace_type, 0)

    def close(self) -> dict[str, int]:
        """Return final counts and clean up."""
        return dict(self._counts)


# ── Global registry ──────────────────────────────────────────────────────────

_active_writers: dict[str, TraceWriter] = {}
_writers_lock = Lock()


def get_writer(run_id: str) -> TraceWriter:
    """Get or create a TraceWriter for a run."""
    with _writers_lock:
        if run_id not in _active_writers:
            _active_writers[run_id] = TraceWriter(run_id)
        return _active_writers[run_id]


def close_writer(run_id: str) -> dict[str, int] | None:
    """Close and remove a TraceWriter, returning final counts."""
    with _writers_lock:
        writer = _active_writers.pop(run_id, None)
    if writer:
        return writer.close()
    return None
