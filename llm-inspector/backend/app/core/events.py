"""
core/events.py — Structured event bus for the v13 pipeline.

Defines typed event kinds that are emitted throughout the pipeline and
consumed by the SSE publisher and the structured logger.

Usage:
    from app.core.events import emit, EventKind
    emit(run_id, EventKind.CASE_RESULT, case_id=..., passed=True, tokens=50)

All events are automatically:
  1. Forwarded to the SSE publisher (so the frontend can filter by type)
  2. Written to the structlog logger at DEBUG level
  3. (Optionally) appended to the in-memory PipelineTracer for this run
"""
from __future__ import annotations

import enum
import json
import time
from dataclasses import dataclass, field

from app.core.logging import get_logger
from app.core.sse import publisher as sse_publisher

logger = get_logger(__name__)


# ── EventKind enum ────────────────────────────────────────────────────────────

class EventKind(str, enum.Enum):
    # Pre-detection probe lifecycle
    PROBE_REQUEST    = "probe.request"
    PROBE_RESPONSE   = "probe.response"
    PROBE_TIMEOUT    = "probe.timeout"

    # Test case lifecycle
    CASE_START       = "case.start"
    CASE_SAMPLE      = "case.sample"
    CASE_RESULT      = "case.result"
    CASE_SKIP        = "case.skip"

    # Judge lifecycle
    JUDGE_RULE       = "judge.rule"
    JUDGE_SEMANTIC   = "judge.semantic"
    JUDGE_CONSENSUS  = "judge.consensus"
    JUDGE_KAPPA      = "judge.kappa"

    # Circuit breaker
    CB_OPEN          = "cb.open"
    CB_HALF_OPEN     = "cb.half_open"
    CB_CLOSE         = "cb.close"

    # Retry/backoff
    RETRY_SCHEDULED  = "retry.scheduled"
    RETRY_EXECUTED   = "retry.executed"

    # Phase transitions
    PHASE_TRANSITION = "phase.transition"
    RUN_STARTED      = "run.started"
    RUN_COMPLETED    = "run.completed"
    RUN_FAILED       = "run.failed"

    # PreDetect layer tracing (v14 Phase 5)
    PREDETECT_LAYER_TRACE = "predetect_layer_trace"

    # Preflight (v15)
    PREFLIGHT_STARTED    = "preflight.started"
    PREFLIGHT_STEP       = "preflight.step"
    PREFLIGHT_PASSED     = "preflight.passed"
    PREFLIGHT_FAILED     = "preflight.failed"
    # Measurement / evidence (v15)
    MEASUREMENT_SKIPPED  = "measurement.skipped"
    EVIDENCE_ADDED       = "authenticity.evidence_added"
    JUDGE_FALLBACK       = "judge.fallback"


# ── Event dataclass ───────────────────────────────────────────────────────────

@dataclass
class Event:
    """A single structured event emitted during a pipeline run."""

    run_id: str
    kind: EventKind
    timestamp: float          # time.time()
    payload: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "kind": self.kind.value,
            "timestamp": self.timestamp,
            "payload": self.payload,
        }

    def to_jsonl(self) -> str:
        """Serialize to a single JSONL line (newline-terminated)."""
        return json.dumps(self.to_dict()) + "\n"


# ── emit() ────────────────────────────────────────────────────────────────────

def emit(run_id: str, kind: EventKind, **payload) -> Event:
    """
    Emit a structured event for a run.

    - Forwards to SSE publisher under event type = kind.value
    - Logs at DEBUG via structlog
    - Appends to PipelineTracer if one exists for this run
    """
    event = Event(
        run_id=run_id,
        kind=kind,
        timestamp=time.time(),
        payload=payload,
    )

    # 1. Forward to SSE publisher
    try:
        sse_data = {
            "type": kind.value,
            "run_id": run_id,
            "timestamp": event.timestamp,
            **payload,
        }
        sse_publisher.publish(run_id, sse_data)
    except Exception:
        pass  # SSE is best-effort

    # 2. Log at DEBUG level
    try:
        logger.debug(
            "event",
            run_id=run_id,
            kind=kind.value,
            **{k: v for k, v in payload.items() if not isinstance(v, (dict, list))},
        )
    except Exception:
        pass

    # 3. Append to PipelineTracer if active for this run
    try:
        from app.core.tracer import get_tracer  # local import to avoid circular
        tracer = get_tracer(run_id)
        if tracer is not None:
            tracer.add_event(kind.value, **payload)
            # Also flush to JSONL if the tracer supports it
            if hasattr(tracer, "flush_to_jsonl"):
                tracer.flush_to_jsonl(event.to_dict())
    except Exception:
        pass

    return event
