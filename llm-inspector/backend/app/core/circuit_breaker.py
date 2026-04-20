"""
CircuitBreaker — v11 resilient API call wrapper.

Replaces the non-existent pyfailsafe dependency with a lightweight,
self-contained circuit breaker that:

1. Tracks consecutive failures per API endpoint
2. Opens the circuit after N consecutive failures (configurable threshold)
3. In half-open state, allows a single probe request through
4. On probe success, closes the circuit; on failure, re-opens
5. Integrates with the orchestrator to suspend/resume failed runs
   instead of dropping them

Thread-safe: all state mutations are protected by a lock.

Usage:
    from app.core.circuit_breaker import circuit_breaker

    # Before making an API call
    if circuit_breaker.is_open(base_url):
        # Circuit is open — skip this request, schedule retry later
        handle_suspended_run(run_id, reason="API circuit open")
    else:
        try:
            resp = adapter.chat(req)
            circuit_breaker.record_success(base_url)
        except Exception as e:
            circuit_breaker.record_failure(base_url, str(e))
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from app.core.logging import get_logger

logger = get_logger(__name__)

# -- Event history ring buffer ------------------------------------------------

_CB_HISTORY_MAX = 100  # keep last 100 trip/recovery events in memory

# Module-level ring buffer, protected by the circuit_breaker's lock below.
_cb_event_history: deque[dict] = deque(maxlen=_CB_HISTORY_MAX)


def _record_cb_event(event_type: str, base_url: str, state: str, **extra) -> None:
    """Append a state-change event to the in-memory ring buffer."""
    _cb_event_history.append({
        "event_type": event_type,  # "opened" | "closed" | "half_open" | "reset"
        "base_url": base_url,
        "state": state,
        "timestamp": time.time(),
        **extra,
    })


def get_cb_event_history(limit: int = 50, state_filter: str | None = None) -> list[dict]:
    """Return recent circuit breaker state-change events, newest first."""
    events = list(_cb_event_history)
    events.reverse()
    if state_filter:
        events = [e for e in events if e.get("state") == state_filter]
    return events[:limit]


class CircuitState(Enum):
    CLOSED = "closed"       # Normal operation — requests flow through
    OPEN = "open"           # Circuit tripped — requests are rejected
    HALF_OPEN = "half_open" # Probing — one request allowed to test recovery


@dataclass
class CircuitMetrics:
    """Per-endpoint circuit breaker state."""
    base_url: str
    state: CircuitState = CircuitState.CLOSED
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    total_failures: int = 0
    total_successes: int = 0
    last_failure_time: float = 0.0
    last_failure_reason: str = ""
    opened_at: float = 0.0
    half_open_probe_sent: bool = False

    @property
    def failure_rate(self) -> float:
        total = self.total_failures + self.total_successes
        if total == 0:
            return 0.0
        return self.total_failures / total

    def to_dict(self) -> dict:
        return {
            "base_url": self.base_url,
            "state": self.state.value,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "failure_rate": round(self.failure_rate, 3),
            "last_failure_reason": self.last_failure_reason,
            "opened_at": self.opened_at,
        }


class CircuitBreaker:
    """
    Thread-safe circuit breaker for LLM API endpoints.

    Configuration:
        - failure_threshold: consecutive failures before opening (default: 5)
        - recovery_timeout_sec: seconds before half-open probe (default: 30)
        - success_threshold: consecutive successes to close from half-open (default: 3)
        - per_endpoint: track state per base_url independently
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_sec: float = 30.0,
        success_threshold: int = 3,
    ):
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout_sec
        self._success_threshold = success_threshold
        self._circuits: dict[str, CircuitMetrics] = {}
        self._lock = threading.Lock()

    def _get_circuit(self, base_url: str) -> CircuitMetrics:
        if base_url not in self._circuits:
            self._circuits[base_url] = CircuitMetrics(base_url=base_url)
        return self._circuits[base_url]

    def is_open(self, base_url: str) -> bool:
        """
        Check if the circuit is open (requests should be rejected/suspended).

        Returns True if circuit is OPEN, False if CLOSED or HALF_OPEN.
        In HALF_OPEN state, allows one probe request through.
        """
        with self._lock:
            circuit = self._get_circuit(base_url)

            if circuit.state == CircuitState.CLOSED:
                return False

            if circuit.state == CircuitState.OPEN:
                # Check if recovery timeout has elapsed
                elapsed = time.monotonic() - circuit.opened_at
                if elapsed >= self._recovery_timeout:
                    # Transition to half-open
                    circuit.state = CircuitState.HALF_OPEN
                    circuit.half_open_probe_sent = False
                    _record_cb_event("half_open", base_url, "half_open", elapsed_since_open=round(elapsed, 1))
                    logger.info(
                        "Circuit transitioning to HALF_OPEN",
                        base_url=base_url,
                        elapsed_since_open=round(elapsed, 1),
                    )
                    return False  # Allow probe
                return True  # Still open

            if circuit.state == CircuitState.HALF_OPEN:
                if not circuit.half_open_probe_sent:
                    # Allow one probe request
                    circuit.half_open_probe_sent = True
                    return False
                # Probe already sent, still waiting — block
                return True

            return False

    def record_success(self, base_url: str) -> None:
        """Record a successful API call."""
        with self._lock:
            circuit = self._get_circuit(base_url)
            circuit.consecutive_failures = 0
            circuit.consecutive_successes += 1
            circuit.total_successes += 1

            if circuit.state == CircuitState.HALF_OPEN:
                if circuit.consecutive_successes >= self._success_threshold:
                    # Enough successes in half-open — close the circuit
                    circuit.state = CircuitState.CLOSED
                    circuit.half_open_probe_sent = False
                    _record_cb_event("closed", base_url, "closed", reason="recovered_from_half_open")
                    logger.info(
                        "Circuit CLOSED (recovered)",
                        base_url=base_url,
                        consecutive_successes=circuit.consecutive_successes,
                    )

            elif circuit.state == CircuitState.OPEN:
                # Shouldn't happen — success while open means something
                # bypassed the check. Close immediately.
                circuit.state = CircuitState.CLOSED
                _record_cb_event("closed", base_url, "closed", reason="success_while_open_unexpected")
                logger.warning(
                    "Circuit CLOSED (success while open — unexpected)",
                    base_url=base_url,
                )

    def record_failure(self, base_url: str, reason: str = "") -> None:
        """Record a failed API call."""
        with self._lock:
            circuit = self._get_circuit(base_url)
            circuit.consecutive_failures += 1
            circuit.consecutive_successes = 0
            circuit.total_failures += 1
            circuit.last_failure_time = time.monotonic()
            circuit.last_failure_reason = reason[:200]

            if circuit.state == CircuitState.HALF_OPEN:
                # Probe failed — re-open
                circuit.state = CircuitState.OPEN
                circuit.opened_at = time.monotonic()
                circuit.half_open_probe_sent = False
                _record_cb_event("opened", base_url, "open", reason="half_open_probe_failed", failure_reason=reason[:100])
                logger.warning(
                    "Circuit re-OPENED (half-open probe failed)",
                    base_url=base_url,
                    reason=reason,
                )
                return

            if circuit.state == CircuitState.CLOSED:
                if circuit.consecutive_failures >= self._failure_threshold:
                    circuit.state = CircuitState.OPEN
                    circuit.opened_at = time.monotonic()
                    _record_cb_event("opened", base_url, "open", reason="failure_threshold_reached",
                                     consecutive_failures=circuit.consecutive_failures,
                                     threshold=self._failure_threshold)
                    logger.warning(
                        "Circuit OPENED (failure threshold reached)",
                        base_url=base_url,
                        consecutive_failures=circuit.consecutive_failures,
                        threshold=self._failure_threshold,
                        reason=reason,
                    )

    def get_metrics(self, base_url: str | None = None) -> dict | list[dict]:
        """Get circuit breaker metrics for monitoring."""
        with self._lock:
            if base_url:
                circuit = self._circuits.get(base_url)
                return circuit.to_dict() if circuit else {}
            return [c.to_dict() for c in self._circuits.values()]

    def reset(self, base_url: str | None = None) -> None:
        """Reset circuit state (for admin override or testing)."""
        with self._lock:
            if base_url:
                if base_url in self._circuits:
                    del self._circuits[base_url]
                _record_cb_event("reset", base_url, "closed", reason="admin_reset")
            else:
                self._circuits.clear()
                _record_cb_event("reset", "*", "closed", reason="admin_reset_all")

    @property
    def stats(self) -> dict:
        """Global stats summary."""
        with self._lock:
            total = len(self._circuits)
            open_count = sum(
                1 for c in self._circuits.values()
                if c.state == CircuitState.OPEN
            )
            half_open_count = sum(
                1 for c in self._circuits.values()
                if c.state == CircuitState.HALF_OPEN
            )
            return {
                "total_endpoints": total,
                "open": open_count,
                "half_open": half_open_count,
                "closed": total - open_count - half_open_count,
            }


# ── Global singleton ────────────────────────────────────────────────────────

circuit_breaker = CircuitBreaker()
