"""
Pipeline Tracer — v11 lightweight observability layer.

Implements a structured tracing system inspired by OpenTelemetry but
without the heavy dependency. Provides:

1. Span-based timing: track duration of each pipeline stage
   (predetect, connectivity, phase1, phase2, judge, analysis, report)
2. Token accounting: per-span and cumulative token usage
3. Event streaming: push trace events to frontend via SSE in real-time
4. Trace persistence: save complete trace to DB for post-hoc analysis

The tracer is per-run: each run_pipeline() call creates a new trace context.

Usage:
    from app.core.tracer import get_tracer

    tracer = get_tracer(run_id)

    with tracer.span("predetect") as span:
        result = PreDetectionPipeline().run(...)
        span.set_attribute("tokens_used", result.total_tokens_used)
        span.set_attribute("identified_as", result.identified_as)

    # Trace events are automatically pushed to SSE subscribers.
"""
from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from app.core.logging import get_logger
from app.core.sse import publisher as sse_publisher

logger = get_logger(__name__)


# ── JSONL trace path helper ───────────────────────────────────────────────────

def _get_trace_path(run_id: str) -> Path:
    """Return the JSONL trace file path for a given run_id."""
    from app.core.config import settings
    d = Path(settings.DATA_DIR) / "traces"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{run_id}.jsonl"


@dataclass
class SpanEvent:
    """A single event within a span (e.g., "case_completed", "judge_finished")."""
    name: str
    timestamp: float
    attributes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "timestamp": round(self.timestamp, 3),
            "attributes": self.attributes,
        }


@dataclass
class Span:
    """
    A timed operation within the pipeline trace.

    Like OpenTelemetry Span but lightweight — no parent/child linking
    (all spans are top-level within a trace).
    """
    name: str
    start_time: float = 0.0
    end_time: float = 0.0
    status: str = "ok"  # "ok" | "error" | "cancelled"
    attributes: dict = field(default_factory=dict)
    events: list[SpanEvent] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        if self.end_time and self.start_time:
            return round((self.end_time - self.start_time) * 1000, 1)
        return 0.0

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, **attrs) -> None:
        self.events.append(SpanEvent(
            name=name,
            timestamp=time.monotonic(),
            attributes=attrs,
        ))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "attributes": self.attributes,
            "events": [e.to_dict() for e in self.events],
        }


@dataclass
class PipelineTrace:
    """
    Complete trace of a single pipeline run.

    Contains all spans and a summary with total duration and token usage.
    """
    run_id: str
    spans: list[Span] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    total_tokens: int = 0

    @property
    def duration_ms(self) -> float:
        if self.end_time and self.start_time:
            return round((self.end_time - self.start_time) * 1000, 1)
        return 0.0

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "duration_ms": self.duration_ms,
            "total_tokens": self.total_tokens,
            "spans": [s.to_dict() for s in self.spans],
            "span_summary": {
                s.name: {
                    "duration_ms": s.duration_ms,
                    "status": s.status,
                    "tokens": s.attributes.get("tokens_used", 0),
                }
                for s in self.spans
            },
        }


class PipelineTracer:
    """
    Per-run pipeline tracer.

    Thread-safe: spans can be created from any thread (e.g., concurrent
    case executors). Events are pushed to SSE subscribers in real-time.
    """

    def __init__(self, run_id: str):
        self._run_id = run_id
        self._trace = PipelineTrace(run_id=run_id)
        self._lock = threading.Lock()
        self._active_spans: dict[str, Span] = {}

        # JSONL persistence — open in append mode so that partial traces survive
        self._jsonl_path: Path | None = None
        self._jsonl_file = None
        self._jsonl_lock = threading.Lock()
        try:
            self._jsonl_path = _get_trace_path(run_id)
            self._jsonl_file = open(self._jsonl_path, "a", encoding="utf-8")
        except Exception as exc:
            logger.warning("Could not open JSONL trace file", run_id=run_id, error=str(exc))

    @property
    def run_id(self) -> str:
        return self._run_id

    @contextmanager
    def span(self, name: str, **initial_attrs):
        """
        Context manager for a timed span.

        Usage:
            with tracer.span("predetect") as span:
                span.set_attribute("confidence", 0.85)
        """
        s = Span(name=name, start_time=time.monotonic())
        s.attributes.update(initial_attrs)

        with self._lock:
            self._active_spans[name] = s

        # Push start event to SSE
        self._push_sse("span_start", {
            "span": name,
            "attributes": initial_attrs,
        })

        try:
            yield s
            s.status = "ok"
        except Exception as e:
            s.status = "error"
            s.set_attribute("error", str(e)[:500])
            raise
        finally:
            s.end_time = time.monotonic()
            with self._lock:
                self._trace.spans.append(s)
                if name in self._active_spans:
                    del self._active_spans[name]

            # Push end event to SSE with timing
            self._push_sse("span_end", {
                "span": name,
                "duration_ms": s.duration_ms,
                "status": s.status,
                "tokens_used": s.attributes.get("tokens_used", 0),
            })

    def record_tokens(self, span_name: str, tokens: int) -> None:
        """Add token usage to the current span and cumulative total."""
        with self._lock:
            self._trace.total_tokens += tokens
            if span_name in self._active_spans:
                current = self._active_spans[span_name].attributes.get("tokens_used", 0)
                self._active_spans[span_name].set_attribute("tokens_used", current + tokens)

        # Push token event
        self._push_sse("tokens_consumed", {
            "span": span_name,
            "delta_tokens": tokens,
            "cumulative_tokens": self._trace.total_tokens,
        })

    def add_event(self, name: str, **attrs) -> None:
        """Add a standalone trace event (not tied to a specific span)."""
        self._push_sse(name, attrs)

    def start(self) -> None:
        """Mark the trace as started."""
        self._trace.start_time = time.monotonic()
        self._push_sse("trace_start", {"run_id": self._run_id})

    def flush_to_jsonl(self, event_dict: dict) -> None:
        """Append one event dict as a JSONL line to the trace file. Thread-safe."""
        if self._jsonl_file is None:
            return
        with self._jsonl_lock:
            try:
                self._jsonl_file.write(json.dumps(event_dict) + "\n")
                self._jsonl_file.flush()
            except Exception:
                pass  # JSONL writes are best-effort

    def finish(self) -> PipelineTrace:
        """Mark the trace as finished and return the complete trace."""
        self._trace.end_time = time.monotonic()
        summary = {
            "run_id": self._run_id,
            "duration_ms": self._trace.duration_ms,
            "total_tokens": self._trace.total_tokens,
            "span_summary": {
                s.name: {"duration_ms": s.duration_ms, "status": s.status}
                for s in self._trace.spans
            },
        }
        self._push_sse("trace_end", summary)

        # Close JSONL file handle
        with self._jsonl_lock:
            if self._jsonl_file is not None:
                try:
                    self._jsonl_file.flush()
                    self._jsonl_file.close()
                except Exception:
                    pass
                finally:
                    self._jsonl_file = None
        if self._jsonl_path:
            self._trace.spans  # no-op; path available in _jsonl_path

        return self._trace

    def get_progress(self) -> dict:
        """
        Get current progress info for frontend rendering.

        Returns the latest span status and timing info.
        """
        with self._lock:
            active = list(self._active_spans.keys())
            completed = [
                {"name": s.name, "duration_ms": s.duration_ms, "status": s.status}
                for s in self._trace.spans
            ]
        return {
            "run_id": self._run_id,
            "active_spans": active,
            "completed_spans": completed,
            "total_tokens": self._trace.total_tokens,
            "elapsed_ms": self._trace.duration_ms,
        }

    def _push_sse(self, event_type: str, data: dict) -> None:
        """Push a trace event to SSE subscribers and persist to JSONL."""
        payload = {
            "type": "trace",
            "trace_event": event_type,
            "run_id": self._run_id,
            "timestamp": time.time(),
            **data,
        }
        try:
            sse_publisher.publish(self._run_id, payload)
        except Exception:
            pass  # SSE is best-effort; don't fail the pipeline
        # Persist to JSONL trace file
        self.flush_to_jsonl(payload)


# ── Tracer registry ─────────────────────────────────────────────────────────

_tracers: dict[str, PipelineTracer] = {}
_tracers_lock = threading.Lock()


def get_tracer(run_id: str) -> PipelineTracer:
    """Get or create a tracer for a given run_id."""
    with _tracers_lock:
        if run_id not in _tracers:
            _tracers[run_id] = PipelineTracer(run_id)
        return _tracers[run_id]


def remove_tracer(run_id: str) -> Optional[PipelineTrace]:
    """Remove and return the tracer's trace (for cleanup after run completes)."""
    with _tracers_lock:
        tracer = _tracers.pop(run_id, None)
        if tracer:
            return tracer._trace
    return None


def get_all_tracer_progress() -> dict[str, dict]:
    """Get progress info for all active tracers (for admin monitoring)."""
    with _tracers_lock:
        return {rid: t.get_progress() for rid, t in _tracers.items()}
