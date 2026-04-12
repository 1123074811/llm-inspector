"""
SSE (Server-Sent Events) Publisher for Real-time Logs.
"""
import json
import threading
from typing import Dict, List, Callable

class SSEPublisher:
    """Manages SSE connections and message broadcasting per run_id."""
    def __init__(self):
        self._listeners: Dict[str, List[Callable]] = {}
        self._lock = threading.Lock()

    def subscribe(self, run_id: str, listener: Callable):
        with self._lock:
            if run_id not in self._listeners:
                self._listeners[run_id] = []
            self._listeners[run_id].append(listener)

    def unsubscribe(self, run_id: str, listener: Callable):
        with self._lock:
            if run_id in self._listeners:
                if listener in self._listeners[run_id]:
                    self._listeners[run_id].remove(listener)
                if not self._listeners[run_id]:
                    del self._listeners[run_id]

    def publish(self, run_id: str, event_data: dict):
        """Publish event to all listeners of run_id."""
        with self._lock:
            listeners = self._listeners.get(run_id, []).copy()
        
        if not listeners:
            return

        event_str = f"data: {json.dumps(event_data)}\n\n"
        event_bytes = event_str.encode('utf-8')
        
        for listener in listeners:
            try:
                listener(event_bytes)
            except Exception:
                # Fire and forget; if a listener fails, we ignore
                pass

publisher = SSEPublisher()
