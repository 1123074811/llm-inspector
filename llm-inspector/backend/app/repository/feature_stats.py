"""
Feature Statistics Repository — 特征值数据库（从硬编码到数据驱动）

替代原有的硬编码 GLOBAL_FEATURE_MEANS 和 KNOWN_TTFT_BASELINES

v5.0 升级组件 - P6 数据基础设施重建
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureStat:
    """特征统计值"""
    feature_name: str
    model_family: str | None
    model_name: str | None
    mean: float
    median: float
    std_dev: float
    p5: float
    p95: float
    sample_count: int
    calculated_at: str
    expires_at: str | None
    data_quality_tag: str  # "golden" | "verified" | "estimated"


@dataclass
class ScoreDriftAlert:
    """分数漂移告警"""
    dimension: str
    baseline: float
    current: float
    drift_pct: float
    severity: str  # "high" | "medium"


class FeatureStatisticsRepository:
    """
    特征统计值的数据库存储与自动更新
    替代原有的硬编码 GLOBAL_FEATURE_MEANS
    """
    
    def __init__(self, db_path: str = "llm_inspector.db"):
        self.db_path = db_path
        self._ensure_table_exists()
    
    def _ensure_table_exists(self):
        """确保数据库表结构存在"""
        schema = """
        CREATE TABLE IF NOT EXISTS feature_statistics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            feature_name TEXT NOT NULL,
            model_family TEXT,
            model_name TEXT,
            
            -- 统计值
            mean REAL,
            median REAL,
            std_dev REAL,
            p5 REAL,
            p95 REAL,
            
            -- 样本信息
            sample_count INTEGER,
            sample_sources TEXT,
            
            -- 时间戳
            calculated_at TEXT,
            expires_at TEXT,
            
            -- 元数据
            data_quality_tag TEXT,
            notes TEXT,
            
            UNIQUE(feature_name, model_family, model_name)
        );
        
        CREATE INDEX IF NOT EXISTS idx_feature_name ON feature_statistics(feature_name);
        CREATE INDEX IF NOT EXISTS idx_model_family ON feature_statistics(model_family);
        CREATE INDEX IF NOT EXISTS idx_expires ON feature_statistics(expires_at);
        """
        
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(schema)
            conn.commit()
            logger.info("Feature statistics table initialized")
    
    def get_feature_mean(
        self,
        feature_name: str,
        model_family: str | None = None,
        fallback_to_global: bool = True,
    ) -> dict | None:
        """
        获取特征均值（替代原有的字典查找）
        
        Args:
            feature_name: 特征名称
            model_family: 模型家族（如 "gpt-4", "claude"），None表示全局统计
            fallback_to_global: 如果特定模型家族无数据，是否回退到全局统计
        
        Returns:
            dict: 包含mean, median, std_dev, sample_count, quality的字典，或None
        """
        query = """
        SELECT mean, median, std_dev, p5, p95, sample_count, data_quality_tag
        FROM feature_statistics
        WHERE feature_name = ?
        AND (model_family = ? OR (? IS NULL AND model_family IS NULL))
        AND (expires_at IS NULL OR expires_at > datetime('now'))
        ORDER BY 
            CASE data_quality_tag
                WHEN 'golden' THEN 1
                WHEN 'verified' THEN 2
                WHEN 'estimated' THEN 3
                ELSE 4
            END,
            calculated_at DESC
        LIMIT 1
        """
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            result = conn.execute(query, (feature_name, model_family, model_family)).fetchone()
        
        if result:
            return {
                "mean": result["mean"],
                "median": result["median"],
                "std_dev": result["std_dev"],
                "p5": result["p5"],
                "p95": result["p95"],
                "sample_count": result["sample_count"],
                "quality": result["data_quality_tag"],
            }
        
        # 尝试回退到全局统计
        if fallback_to_global and model_family is not None:
            return self.get_feature_mean(feature_name, None, False)
        
        return None
    
    def update_statistics(
        self,
        feature_name: str,
        new_values: list[float],
        model_family: str | None = None,
        model_name: str | None = None,
        source_run_ids: list[str] | None = None,
        notes: str = "",
    ) -> FeatureStat:
        """
        基于新检测数据更新统计值
        
        Args:
            feature_name: 特征名称
            new_values: 新观测值列表
            model_family: 模型家族
            model_name: 具体模型名称
            source_run_ids: 数据来源的run_id列表
            notes: 备注
        
        Returns:
            FeatureStat: 更新后的统计值
        """
        if len(new_values) < 3:
            logger.warning(f"Insufficient samples for {feature_name}: {len(new_values)}")
            # 使用现有数据或创建估计值
            existing = self.get_feature_mean(feature_name, model_family, False)
            if existing:
                return self._dict_to_feature_stat(feature_name, model_family, model_name, existing)
            # 创建最小估计值
            new_values = list(new_values) + [np.mean(new_values)] * (3 - len(new_values))
        
        arr = np.array(new_values)
        
        stats = {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std_dev": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "p5": float(np.percentile(arr, 5)),
            "p95": float(np.percentile(arr, 95)),
            "sample_count": len(new_values),
            "calculated_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(days=30)).isoformat(),
        }
        
        # 确定数据质量标签
        if len(new_values) >= 100 and source_run_ids and len(source_run_ids) >= 10:
            quality = "golden"
        elif len(new_values) >= 20:
            quality = "verified"
        else:
            quality = "estimated"
        
        # 插入或更新
        upsert_sql = """
        INSERT INTO feature_statistics (
            feature_name, model_family, model_name,
            mean, median, std_dev, p5, p95,
            sample_count, sample_sources, calculated_at, expires_at,
            data_quality_tag, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(feature_name, model_family, model_name) DO UPDATE SET
            mean = excluded.mean,
            median = excluded.median,
            std_dev = excluded.std_dev,
            p5 = excluded.p5,
            p95 = excluded.p95,
            sample_count = excluded.sample_count,
            sample_sources = excluded.sample_sources,
            calculated_at = excluded.calculated_at,
            expires_at = excluded.expires_at,
            data_quality_tag = excluded.data_quality_tag,
            notes = excluded.notes
        """
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(upsert_sql, (
                feature_name, model_family, model_name,
                stats["mean"], stats["median"], stats["std_dev"],
                stats["p5"], stats["p95"],
                stats["sample_count"],
                json.dumps(source_run_ids or []),
                stats["calculated_at"], stats["expires_at"],
                quality, notes,
            ))
            conn.commit()
        
        logger.info(
            f"Updated statistics for {feature_name}",
            model_family=model_family,
            mean=stats["mean"],
            n=len(new_values),
            quality=quality,
        )
        
        return FeatureStat(
            feature_name=feature_name,
            model_family=model_family,
            model_name=model_name,
            mean=stats["mean"],
            median=stats["median"],
            std_dev=stats["std_dev"],
            p5=stats["p5"],
            p95=stats["p95"],
            sample_count=stats["sample_count"],
            calculated_at=stats["calculated_at"],
            expires_at=stats["expires_at"],
            data_quality_tag=quality,
        )
    
    def batch_update_from_detection_run(
        self,
        run_id: str,
        features: dict[str, float],
        model_family: str,
        model_name: str | None = None,
    ):
        """
        从单次检测运行批量更新特征统计
        
        Args:
            run_id: 检测运行ID
            features: 特征值字典
            model_family: 模型家族
            model_name: 具体模型名称
        """
        for feature_name, value in features.items():
            # 先获取现有统计值
            existing = self.get_feature_mean(feature_name, model_family, False)
            
            if existing:
                # 合并新旧数据
                # 简化处理：使用加权平均，保留历史数据
                old_mean = existing["mean"]
                old_n = existing["sample_count"]
                new_values = [old_mean] * min(old_n, 50) + [value]  # 限制历史数据权重
            else:
                new_values = [value]
            
            self.update_statistics(
                feature_name=feature_name,
                new_values=new_values,
                model_family=model_family,
                model_name=model_name,
                source_run_ids=[run_id],
                notes=f"Auto-updated from run {run_id}",
            )
    
    def detect_score_drift(
        self,
        current_scores: dict[str, float],
        model_family: str,
        threshold: float = 0.15,
    ) -> list[ScoreDriftAlert]:
        """
        检测分数相对于基线的漂移
        
        Args:
            current_scores: 当前分数字典 {dimension: score}
            model_family: 模型家族
            threshold: 漂移阈值（默认15%）
        
        Returns:
            list[ScoreDriftAlert]: 漂移告警列表
        """
        alerts = []
        
        for dimension, current in current_scores.items():
            baseline_data = self.get_feature_mean(dimension, model_family, True)
            
            if not baseline_data:
                continue
            
            baseline = baseline_data["mean"]
            
            # 计算相对漂移
            if baseline != 0:
                drift = abs(current - baseline) / abs(baseline)
            else:
                drift = abs(current) / 100.0  # 避免除以零
            
            if drift > threshold:
                alerts.append(ScoreDriftAlert(
                    dimension=dimension,
                    baseline=baseline,
                    current=current,
                    drift_pct=drift * 100,
                    severity="high" if drift > 0.3 else "medium",
                ))
                
                logger.warning(
                    f"Score drift detected: {dimension}",
                    baseline=baseline,
                    current=current,
                    drift=f"{drift*100:.1f}%",
                )
        
        return alerts
    
    def get_all_statistics(
        self,
        model_family: str | None = None,
        min_quality: str = "estimated",
    ) -> list[FeatureStat]:
        """
        获取所有统计值
        
        Args:
            model_family: 模型家族筛选（None表示全部）
            min_quality: 最低数据质量要求
        
        Returns:
            list[FeatureStat]: 统计值列表
        """
        quality_order = {"golden": 1, "verified": 2, "estimated": 3}
        min_quality_rank = quality_order.get(min_quality, 3)
        
        query = """
        SELECT * FROM feature_statistics
        WHERE (model_family = ? OR (? IS NULL))
        AND (expires_at IS NULL OR expires_at > datetime('now'))
        ORDER BY feature_name
        """
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, (model_family, model_family)).fetchall()
        
        results = []
        for row in rows:
            quality_rank = quality_order.get(row["data_quality_tag"], 4)
            if quality_rank > min_quality_rank:
                continue
            
            results.append(FeatureStat(
                feature_name=row["feature_name"],
                model_family=row["model_family"],
                model_name=row["model_name"],
                mean=row["mean"],
                median=row["median"],
                std_dev=row["std_dev"],
                p5=row["p5"],
                p95=row["p95"],
                sample_count=row["sample_count"],
                calculated_at=row["calculated_at"],
                expires_at=row["expires_at"],
                data_quality_tag=row["data_quality_tag"],
            ))
        
        return results
    
    def _dict_to_feature_stat(
        self,
        feature_name: str,
        model_family: str | None,
        model_name: str | None,
        data: dict,
    ) -> FeatureStat:
        """将字典转换为FeatureStat"""
        return FeatureStat(
            feature_name=feature_name,
            model_family=model_family,
            model_name=model_name,
            mean=data.get("mean", 0.0),
            median=data.get("median", 0.0),
            std_dev=data.get("std_dev", 0.0),
            p5=data.get("p5", 0.0),
            p95=data.get("p95", 0.0),
            sample_count=data.get("sample_count", 0),
            calculated_at=datetime.now().isoformat(),
            expires_at=None,
            data_quality_tag=data.get("quality", "estimated"),
        )
    
    def cleanup_expired(self, days_to_keep: int = 90):
        """清理过期数据"""
        cutoff = (datetime.now() - timedelta(days=days_to_keep)).isoformat()
        
        query = """
        DELETE FROM feature_statistics
        WHERE expires_at < ?
        """
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, (cutoff,))
            conn.commit()
            deleted = cursor.rowcount
        
        logger.info(f"Cleaned up {deleted} expired feature statistics")
        return deleted


# 全局仓库实例
_repository: Optional[FeatureStatisticsRepository] = None


def get_feature_repository(db_path: str = "llm_inspector.db") -> FeatureStatisticsRepository:
    """获取全局特征统计仓库实例"""
    global _repository
    if _repository is None:
        _repository = FeatureStatisticsRepository(db_path)
    return _repository


# 兼容性函数：替代原有的 GLOBAL_FEATURE_MEANS 字典访问
def get_feature_mean(
    feature_name: str,
    model_family: str | None = None,
    db_path: str = "llm_inspector.db",
) -> float | None:
    """
    获取特征均值的便捷函数（向后兼容）
    
    如果数据库中无数据，返回None
    """
    repo = get_feature_repository(db_path)
    data = repo.get_feature_mean(feature_name, model_family, True)
    
    if data:
        return data["mean"]
    return None
