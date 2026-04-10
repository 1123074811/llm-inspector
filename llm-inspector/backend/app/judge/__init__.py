"""Judge package - response evaluation methods.

v5.0 新增组件：
- SemanticJudgeV2: 新一代语义判题引擎
- HallucinationDetectorV2: 增强版幻觉检测
"""
from app.judge.methods import judge
from app.judge.semantic import llm_judge, local_semantic_judge, llm_judge_available

# v5.0 新增组件
from app.judge.semantic_v2 import (
    SemanticJudgeV2,
    SemanticJudgeResult,
    get_semantic_judge_v2,
    semantic_judge_v2,
)
from app.judge.hallucination_v2 import (
    HallucinationDetectorV2,
    UncertaintyMetrics,
    get_hallucination_detector_v2,
    hallucination_detect_v2,
)

__all__ = [
    # 原有组件
    "judge",
    "llm_judge",
    "local_semantic_judge",
    "llm_judge_available",
    # v5.0 新增
    "SemanticJudgeV2",
    "SemanticJudgeResult",
    "get_semantic_judge_v2",
    "semantic_judge_v2",
    "HallucinationDetectorV2",
    "UncertaintyMetrics",
    "get_hallucination_detector_v2",
    "hallucination_detect_v2",
]
