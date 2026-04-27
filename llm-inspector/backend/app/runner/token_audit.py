"""
runner/token_audit.py — v16 Phase 7: Token Audit Tracker

Tracks per-case token consumption, cache hits, and finish reasons.
Writes to data/traces/{run_id}/token_audit.jsonl.
"""
from __future__ import annotations

import json
import pathlib as _pl
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from threading import Lock

from app.core.logging import get_logger

logger = get_logger(__name__)

_DATA_DIR = _pl.Path(__file__).resolve().parent.parent.parent / "data" / "traces"


@dataclass
class TokenAuditEntry:
    """Single token audit record for one case execution."""
    case_id: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cache_hit: bool = False
    finish_reason: str = ""
    layer: str = ""           # e.g. "predetect_L5", "capability", "rescue"
    timestamp: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TokenAuditSummary:
    """Summary of token audit for a run."""
    run_id: str
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate: float = 0.0
    entry_count: int = 0
    by_layer: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class TokenAuditTracker:
    """Thread-safe token audit tracker for a single run."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._entries: list[TokenAuditEntry] = []
        self._lock = Lock()

    def record(
        self,
        case_id: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        cache_hit: bool = False,
        finish_reason: str = "",
        layer: str = "",
    ) -> None:
        """Record a token audit entry."""
        entry = TokenAuditEntry(
            case_id=case_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cache_hit=cache_hit,
            finish_reason=finish_reason,
            layer=layer,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        with self._lock:
            self._entries.append(entry)

    def summary(self) -> TokenAuditSummary:
        """Compute audit summary."""
        with self._lock:
            entries = list(self._entries)

        total_prompt = sum(e.prompt_tokens for e in entries)
        total_completion = sum(e.completion_tokens for e in entries)
        total = sum(e.total_tokens for e in entries)
        hits = sum(1 for e in entries if e.cache_hit)
        misses = len(entries) - hits
        hit_rate = hits / len(entries) if entries else 0.0

        by_layer: dict[str, dict] = {}
        for e in entries:
            layer = e.layer or "unknown"
            if layer not in by_layer:
                by_layer[layer] = {"prompt": 0, "completion": 0, "total": 0, "count": 0}
            by_layer[layer]["prompt"] += e.prompt_tokens
            by_layer[layer]["completion"] += e.completion_tokens
            by_layer[layer]["total"] += e.total_tokens
            by_layer[layer]["count"] += 1

        return TokenAuditSummary(
            run_id=self.run_id,
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
            total_tokens=total,
            cache_hits=hits,
            cache_misses=misses,
            cache_hit_rate=round(hit_rate, 4),
            entry_count=len(entries),
            by_layer=by_layer,
        )

    def flush_to_jsonl(self) -> _pl.Path | None:
        """Write all entries to token_audit.jsonl."""
        if not self._entries:
            return None

        run_dir = _DATA_DIR / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "token_audit.jsonl"

        with self._lock:
            entries = list(self._entries)

        with open(path, "a", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

        logger.info("Token audit flushed", path=str(path), entries=len(entries))
        return path


# ── Global registry ──────────────────────────────────────────────────────────

_active_trackers: dict[str, TokenAuditTracker] = {}
_trackers_lock = Lock()


def get_tracker(run_id: str) -> TokenAuditTracker:
    """Get or create a TokenAuditTracker for a run."""
    with _trackers_lock:
        if run_id not in _active_trackers:
            _active_trackers[run_id] = TokenAuditTracker(run_id)
        return _active_trackers[run_id]


def remove_tracker(run_id: str) -> None:
    """Remove and flush a tracker after run completes."""
    with _trackers_lock:
        tracker = _active_trackers.pop(run_id, None)
    if tracker:
        tracker.flush_to_jsonl()
