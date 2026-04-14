"""Analysis package - feature extraction, scoring, similarity, and reporting.

v5.0 新增组件：
- AdaptiveScoreCalibrator: 自适应评分校准
- IRTEngine: IRT 2PL 模型引擎
- ScoreAttributionAnalyzer: 分数归因分析
- NeuralSimilarityEngine: 神经网络相似度
"""
from app.analysis.pipeline import (
    AnalysisPipeline,
    FeatureExtractor,
    ScoreCalculator,
    SimilarityEngine,
    VerdictEngine,
    RiskEngine,
    ThetaEstimator,
    UncertaintyEstimator,
    PercentileMapper,
    PairwiseEngine,
    NarrativeBuilder,
    ReportBuilder,
    ProxyLatencyAnalyzer,
    ExtractionAuditBuilder,
)
from app.analysis.pipeline import ScoreCardCalculator  # authoritative definition lives in pipeline.py

# v5.0 新增组件
from app.analysis.adaptive_scoring import (
    AdaptiveScoreCalibrator,
    ScoreConfidenceEstimator,
    get_calibrator,
    get_confidence_estimator,
    calculate_score_with_confidence,
)
from app.analysis.irt_engine import (
    IRTEngine,
    IRTItemStats,
    AbilityEstimate,
    get_irt_engine,
)
from app.analysis.attribution import (
    ScoreAttributionAnalyzer,
    AttributionReport,
    get_attribution_analyzer,
    analyze_score_attribution,
)
from app.analysis.neural_similarity import (
    BehavioralEmbeddingExtractor,
    MultiModalSimilarityFusion,
    get_embedding_extractor,
    get_similarity_fusion,
    compute_neural_similarity,
)
from app.analysis.irt_params import (
    IRTParameters,
    IRTParameterDB,
    ThetaScoreConverter,
    get_irt_db,
)
# v11 Phase 3: Suite pruning + GPQA
from app.analysis.suite_pruner import (
    SuitePruner,
    CaseQualityMetrics,
    PruningReport,
    GPQAAdapter,
    GPQAQuestion,
    suite_pruner,
    gpqa_adapter,
    get_pruner,
    get_gpqa_adapter,
)

__all__ = [
    # 原有组件
    "AnalysisPipeline",
    "FeatureExtractor",
    "ScoreCalculator",
    "ScoreCardCalculator",
    "SimilarityEngine",
    "VerdictEngine",
    "RiskEngine",
    "ThetaEstimator",
    "UncertaintyEstimator",
    "PercentileMapper",
    "PairwiseEngine",
    "NarrativeBuilder",
    "ReportBuilder",
    "ProxyLatencyAnalyzer",
    "ExtractionAuditBuilder",
    # v5.0 新增
    "AdaptiveScoreCalibrator",
    "ScoreConfidenceEstimator",
    "IRTEngine",
    "IRTItemStats",
    "AbilityEstimate",
    "ScoreAttributionAnalyzer",
    "AttributionReport",
    "BehavioralEmbeddingExtractor",
    "MultiModalSimilarityFusion",
    # v8.0 新增 - IRT参数数据库
    "IRTParameters",
    "IRTParameterDB",
    "ThetaScoreConverter",
    "get_irt_db",
    # v11 Phase 3 新增 - Suite pruning + GPQA
    "SuitePruner",
    "CaseQualityMetrics",
    "PruningReport",
    "GPQAAdapter",
    "GPQAQuestion",
    "suite_pruner",
    "gpqa_adapter",
    "get_pruner",
    "get_gpqa_adapter",
    # 便捷函数
    "get_calibrator",
    "get_confidence_estimator",
    "calculate_score_with_confidence",
    "get_irt_engine",
    "get_attribution_analyzer",
    "analyze_score_attribution",
    "get_embedding_extractor",
    "get_similarity_fusion",
    "compute_neural_similarity",
]
