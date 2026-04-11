"""Judge package - response evaluation methods.

v5.0 新增组件：
- SemanticJudgeV2: 新一代语义判题引擎
- HallucinationDetectorV2: 增强版幻觉检测

v8.0 新增组件（Phase 4 - 架构优化）：
- JudgePlugin: 判题方法插件接口
- PluginManager: 插件管理器
- TransparentJudgeWrapper: 透明判题包装器
- StructuredLogger: 结构化日志系统
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

# v8.0 Phase 4: 插件化架构
from app.judge.plugin_interface import (
    JudgePlugin,
    JudgeResult,
    JudgeMetadata,
    JudgeTier,
    TieredJudgePlugin,
)
from app.judge.plugin_manager import (
    PluginManager,
    get_plugin_manager,
    RegisteredPlugin,
    PluginStats,
)
from app.judge.builtin_plugins import (
    ExactMatchPlugin,
    RegexMatchPlugin,
    JSONSchemaPlugin,
    ConstraintReasoningPlugin,
    RefusalDetectPlugin,
    LineCountPlugin,
    BUILTIN_PLUGINS,
    register_builtin_plugins,
)
from app.judge.transparent_judge import (
    TransparentJudgeWrapper,
    JudgmentLogger,
    create_transparent_judge,
    judge_with_transparency,
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
    # v8.0 Phase 4: 插件化架构
    "JudgePlugin",
    "JudgeResult",
    "JudgeMetadata",
    "JudgeTier",
    "TieredJudgePlugin",
    "PluginManager",
    "get_plugin_manager",
    "RegisteredPlugin",
    "PluginStats",
    "ExactMatchPlugin",
    "RegexMatchPlugin",
    "JSONSchemaPlugin",
    "ConstraintReasoningPlugin",
    "RefusalDetectPlugin",
    "LineCountPlugin",
    "BUILTIN_PLUGINS",
    "register_builtin_plugins",
    "TransparentJudgeWrapper",
    "JudgmentLogger",
    "create_transparent_judge",
    "judge_with_transparency",
]
