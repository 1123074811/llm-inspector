"""Runner package - detection execution.

v5.0 新增组件：
- AsyncDetectionPipeline: 异步检测流水线（背压+优先级）
"""
# v5.0 新增组件
from app.runner.async_pipeline import (
    AsyncDetectionPipeline,
    PipelineMetrics,
    PipelineTask,
    get_async_pipeline,
    run_detection_async,
)

__all__ = [
    # v5.0 异步流水线组件
    "AsyncDetectionPipeline",
    "PipelineMetrics",
    "PipelineTask",
    "get_async_pipeline",
    "run_detection_async",
]
