"""Core package - data structures, configuration, and utilities.

v8.0 新增组件：
- DataProvenance: 数据血缘追踪
- ReferenceDatabase: 参考文献数据库
- StructuredLogger: 结构化日志系统
"""
from app.core.provenance import (
    DataProvenance,
    ProvenanceTracker,
    get_provenance_tracker,
    reset_provenance_tracker,
)
from app.core.references import (
    Reference,
    ReferenceType,
    ReferenceDatabase,
    get_reference_database,
    validate_formula_source,
)
from app.core.structured_logger import (
    AuditLogger,
    StructuredLogEntry,
    LogLevel,
    LogEventType,
    get_structured_logger,
)
# v15: AuditLogger replaces StructuredLogger; keep alias for backward compat
StructuredLogger = AuditLogger

__all__ = [
    # 数据血缘
    "DataProvenance",
    "ProvenanceTracker",
    "get_provenance_tracker",
    "reset_provenance_tracker",
    # 参考文献
    "Reference",
    "ReferenceType",
    "ReferenceDatabase",
    "get_reference_database",
    "validate_formula_source",
    # 结构化日志 (Phase 4)
    "StructuredLogger",
    "StructuredLogEntry",
    "LogLevel",
    "LogEventType",
    "get_structured_logger",
]
