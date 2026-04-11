"""
Structured Logger — v8.0 Transparent Logging System

Provides structured, queryable logs for:
- Judgment process transparency
- Debugging and auditing
- Performance monitoring

Reference: V8_UPGRADE_PLAN.md Section 1.4
"""
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
import json
import threading


class LogLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class LogEventType(Enum):
    """Types of log events for filtering and analysis."""
    # Judgment events
    JUDGE_START = "judge_start"
    JUDGE_STEP = "judge_step"
    JUDGE_COMPLETE = "judge_complete"
    JUDGE_ERROR = "judge_error"
    
    # Pipeline events
    TEST_START = "test_start"
    TEST_COMPLETE = "test_complete"
    CASE_EXECUTE = "case_execute"
    
    # Pre-detection events
    PREDETECT_START = "predetect_start"
    PREDETECT_LAYER = "predetect_layer"
    PREDETECT_COMPLETE = "predetect_complete"
    
    # System events
    PLUGIN_REGISTER = "plugin_register"
    PLUGIN_ERROR = "plugin_error"
    CONFIG_LOAD = "config_load"
    
    # Data events
    PROVENANCE_RECORD = "provenance_record"
    THRESHOLD_APPLY = "threshold_apply"


@dataclass
class StructuredLogEntry:
    """A single structured log entry."""
    timestamp: str
    event_type: str
    level: str
    component: str  # e.g., "judge", "pipeline", "predetect"
    message: str
    data: Dict[str, Any]
    trace_id: Optional[str] = None  # For distributed tracing
    parent_event: Optional[str] = None  # For event hierarchy
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "level": self.level,
            "component": self.component,
            "message": self.message,
            "data": self.data,
            "trace_id": self.trace_id,
            "parent_event": self.parent_event,
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class StructuredLogger:
    """
    v8.0 structured logging system.
    
    Features:
    - Event type classification
    - Hierarchical event tracing
    - Queryable log storage
    - Export to JSON
    """
    
    _instance: Optional['StructuredLogger'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._log: List[StructuredLogEntry] = []
        self._max_size = 10000
        self._initialized = True
        self._handlers: List[Callable[[StructuredLogEntry], None]] = []
        self._level = LogLevel.INFO
    
    def add_handler(self, handler: Callable[[StructuredLogEntry], None]):
        """Add a custom log handler."""
        self._handlers.append(handler)
    
    def set_level(self, level: LogLevel):
        """Set minimum log level."""
        self._level = level
    
    def _should_log(self, level: LogLevel) -> bool:
        """Check if level should be logged."""
        level_order = [LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARNING, LogLevel.ERROR]
        return level_order.index(level) >= level_order.index(self._level)
    
    def log(self,
            event_type: LogEventType,
            level: LogLevel,
            component: str,
            message: str,
            data: Dict[str, Any] = None,
            trace_id: Optional[str] = None,
            parent_event: Optional[str] = None):
        """
        Log a structured event.
        
        Args:
            event_type: Type of event
            level: Log level
            component: Component name (judge, pipeline, etc.)
            message: Human-readable message
            data: Structured data
            trace_id: Optional trace identifier
            parent_event: Optional parent event ID
        """
        if not self._should_log(level):
            return
        
        entry = StructuredLogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type.value,
            level=level.value,
            component=component,
            message=message,
            data=data or {},
            trace_id=trace_id,
            parent_event=parent_event
        )
        
        self._log.append(entry)
        
        # Trim log if too large
        if len(self._log) > self._max_size:
            self._log = self._log[-self._max_size:]
        
        # Notify handlers
        for handler in self._handlers:
            try:
                handler(entry)
            except Exception:
                pass
    
    # Convenience methods
    def debug(self, event_type: LogEventType, component: str, message: str, **kwargs):
        self.log(event_type, LogLevel.DEBUG, component, message, kwargs)
    
    def info(self, event_type: LogEventType, component: str, message: str, **kwargs):
        self.log(event_type, LogLevel.INFO, component, message, kwargs)
    
    def warning(self, event_type: LogEventType, component: str, message: str, **kwargs):
        self.log(event_type, LogLevel.WARNING, component, message, kwargs)
    
    def error(self, event_type: LogEventType, component: str, message: str, **kwargs):
        self.log(event_type, LogLevel.ERROR, component, message, kwargs)
    
    # Specialized logging methods
    def log_judge_start(self, case_id: str, method: str, trace_id: Optional[str] = None):
        """Log start of judgment."""
        self.info(
            LogEventType.JUDGE_START,
            "judge",
            f"Starting judgment for case {case_id}",
            case_id=case_id,
            method=method,
            trace_id=trace_id
        )
    
    def log_judge_step(self,
                       case_id: str,
                       step: str,
                       input_data: Dict,
                       output_data: Dict,
                       threshold: Optional[float] = None,
                       threshold_source: Optional[str] = None,
                       parent_event: Optional[str] = None):
        """Log a judgment step with full transparency."""
        data = {
            "case_id": case_id,
            "step": step,
            "input": input_data,
            "output": output_data,
        }
        if threshold is not None:
            data["threshold"] = threshold
        if threshold_source:
            data["threshold_source"] = threshold_source
        
        self.info(
            LogEventType.JUDGE_STEP,
            "judge",
            f"Judgment step {step} for case {case_id}",
            **data,
            parent_event=parent_event
        )
    
    def log_judge_complete(self,
                          case_id: str,
                          method: str,
                          passed: bool,
                          confidence: float,
                          detail: Dict,
                          tokens_used: int = 0,
                          latency_ms: int = 0):
        """Log completion of judgment."""
        self.info(
            LogEventType.JUDGE_COMPLETE,
            "judge",
            f"Judgment complete for case {case_id}: {'passed' if passed else 'failed'}",
            case_id=case_id,
            method=method,
            passed=passed,
            confidence=confidence,
            detail=detail,
            tokens_used=tokens_used,
            latency_ms=latency_ms
        )
    
    def log_threshold_apply(self,
                           component: str,
                           threshold_name: str,
                           value: float,
                           source: str,
                           context: Dict[str, Any]):
        """Log threshold application with provenance."""
        self.info(
            LogEventType.THRESHOLD_APPLY,
            component,
            f"Applying threshold {threshold_name}={value} (source: {source})",
            threshold_name=threshold_name,
            value=value,
            source=source,
            context=context
        )
    
    def query(self,
              event_type: Optional[LogEventType] = None,
              component: Optional[str] = None,
              level: Optional[LogLevel] = None,
              trace_id: Optional[str] = None,
              start_time: Optional[str] = None,
              end_time: Optional[str] = None) -> List[StructuredLogEntry]:
        """
        Query logs with filters.
        
        Returns matching log entries.
        """
        results = self._log
        
        if event_type:
            results = [e for e in results if e.event_type == event_type.value]
        
        if component:
            results = [e for e in results if e.component == component]
        
        if level:
            results = [e for e in results if e.level == level.value]
        
        if trace_id:
            results = [e for e in results if e.trace_id == trace_id]
        
        if start_time:
            results = [e for e in results if e.timestamp >= start_time]
        
        if end_time:
            results = [e for e in results if e.timestamp <= end_time]
        
        return results
    
    def export_json(self, filepath: str, 
                   event_type: Optional[LogEventType] = None,
                   start_time: Optional[str] = None,
                   end_time: Optional[str] = None):
        """Export logs to JSON file."""
        entries = self.query(event_type=event_type, 
                           start_time=start_time, 
                           end_time=end_time)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump([e.to_dict() for e in entries], f, 
                     ensure_ascii=False, indent=2)
    
    def get_recent(self, n: int = 100) -> List[StructuredLogEntry]:
        """Get n most recent entries."""
        return self._log[-n:]
    
    def clear(self):
        """Clear all logs."""
        self._log = []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get log statistics."""
        return {
            "total_entries": len(self._log),
            "by_level": {
                level.value: len([e for e in self._log if e.level == level.value])
                for level in LogLevel
            },
            "by_component": self._count_by("component"),
            "by_event_type": self._count_by("event_type"),
        }
    
    def _count_by(self, field: str) -> Dict[str, int]:
        """Count entries by field value."""
        counts = {}
        for entry in self._log:
            value = getattr(entry, field)
            counts[value] = counts.get(value, 0) + 1
        return counts


# Global accessor
def get_structured_logger() -> StructuredLogger:
    """Get the global structured logger instance."""
    return StructuredLogger()
