"""Repository package - data persistence and retrieval.

v5.0 新增组件：
- FeatureStatisticsRepository: 特征值数据库（替代硬编码统计值）
"""
from app.repository.feature_stats import (
    FeatureStatisticsRepository,
    FeatureStat,
    ScoreDriftAlert,
    get_feature_repository,
    get_feature_mean,
)

__all__ = [
    "FeatureStatisticsRepository",
    "FeatureStat",
    "ScoreDriftAlert",
    "get_feature_repository",
    "get_feature_mean",
]
