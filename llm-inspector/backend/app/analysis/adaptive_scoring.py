"""
Adaptive Score Calibrator — 评分体系自适应校准

基于历史检测数据的贝叶斯权重校准

v5.0 升级组件 - P1 评分体系自适应校准
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import minimize

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class WeightCalibration:
    """权重校准结果"""
    model_family: str
    weights: dict[str, float]
    auc: float  # 校准后的AUC
    calibrated_at: str
    sample_count: int


@dataclass
class ScoreWithConfidence:
    """带置信区间的分数"""
    point_estimate: float
    ci_95_low: float
    ci_95_high: float
    std_error: float
    sample_size: int
    reliability: str  # "high" | "medium" | "low" | "insufficient"
    
    def to_dict(self) -> dict:
        return {
            "score": round(self.point_estimate, 1),
            "ci_95": [round(self.ci_95_low, 1), round(self.ci_95_high, 1)],
            "std_error": round(self.std_error, 2),
            "n_samples": self.sample_size,
            "reliability": self.reliability,
        }


class AdaptiveScoreCalibrator:
    """
    基于历史检测数据的贝叶斯权重校准
    """
    
    # 默认能力维度权重
    DEFAULT_CAPABILITY_WEIGHTS = {
        "reasoning": 0.25,
        "coding": 0.20,
        "instruction": 0.20,
        "safety": 0.15,
        "knowledge": 0.20,
    }
    
    def __init__(self, db_path: str = "llm_inspector.db"):
        self.db_path = db_path
        self._cached_weights: dict[str, dict[str, float]] = {}
    
    def calibrate_capability_weights(
        self,
        claimed_model_family: str,
        min_samples: int = 50,
    ) -> dict[str, float]:
        """
        基于历史数据校准能力维度权重
        
        优化目标：最大化真实模型的区分度（AUC）
        
        Args:
            claimed_model_family: 声称的模型家族（如 "gpt-4", "claude"）
            min_samples: 最少样本数（不足时使用默认权重）
        
        Returns:
            dict[str, float]: 优化后的权重字典
        """
        # 加载历史数据
        historical_data = self._load_historical_data(claimed_model_family)
        
        if len(historical_data) < min_samples:
            logger.info(
                f"Insufficient samples for {claimed_model_family}: {len(historical_data)} < {min_samples}",
                using="default_weights",
            )
            return self.DEFAULT_CAPABILITY_WEIGHTS.copy()
        
        # 使用梯度下降优化权重
        # 目标函数：最大化真实模型的AUC，最小化误判率
        
        def objective(weights_array):
            """负AUC作为损失函数"""
            weights = dict(zip(self.DEFAULT_CAPABILITY_WEIGHTS.keys(), weights_array))
            
            weighted_scores = []
            true_labels = []
            
            for record in historical_data:
                dim_scores = record.get("dimension_scores", {})
                
                # 计算加权总分
                weighted = sum(
                    weights.get(dim, 0) * dim_scores.get(dim, 50)
                    for dim in self.DEFAULT_CAPABILITY_WEIGHTS.keys()
                )
                weighted_scores.append(weighted)
                
                # 真实标签：根据verdict推断
                is_real = record.get("final_verdict") == "trusted"
                true_labels.append(1 if is_real else 0)
            
            # 计算AUC
            try:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(true_labels, weighted_scores)
            except (ValueError, ImportError):
                auc = 0.5  # 无法计算时返回随机水平
            
            return -auc  # 最小化负AUC = 最大化AUC
        
        # 约束：权重和为1，每个权重>=0.05
        def constraint_sum(weights):
            return sum(weights) - 1.0
        
        constraints = [
            {"type": "eq", "fun": constraint_sum},
        ]
        
        bounds = [(0.05, 0.5) for _ in range(len(self.DEFAULT_CAPABILITY_WEIGHTS))]
        
        # 优化
        result = minimize(
            objective,
            x0=list(self.DEFAULT_CAPABILITY_WEIGHTS.values()),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 100, "ftol": 1e-6},
        )
        
        # 返回优化后的权重
        optimized = dict(zip(self.DEFAULT_CAPABILITY_WEIGHTS.keys(), result.x))
        
        # 记录校准历史
        final_auc = -result.fun
        self._log_calibration(claimed_model_family, optimized, final_auc, len(historical_data))
        
        # 缓存结果
        self._cached_weights[claimed_model_family] = optimized
        
        logger.info(
            f"Calibrated weights for {claimed_model_family}",
            auc=f"{final_auc:.3f}",
            n_samples=len(historical_data),
        )
        
        return optimized
    
    def _load_historical_data(self, model_family: str) -> list[dict]:
        """加载历史检测数据用于校准"""
        query = """
        SELECT 
            run_id,
            claimed_model,
            final_verdict,
            dimension_scores,
            total_score
        FROM detection_runs
        WHERE claimed_model LIKE ?
        AND created_at >= date('now', '-90 days')
        AND status = 'completed'
        """
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, (f"{model_family}%",)).fetchall()
        
        results = []
        for row in rows:
            dim_scores = {}
            try:
                dim_scores = json.loads(row["dimension_scores"] or "{}")
            except json.JSONDecodeError:
                pass
            
            results.append({
                "run_id": row["run_id"],
                "claimed_model": row["claimed_model"],
                "final_verdict": row["final_verdict"],
                "dimension_scores": dim_scores,
                "total_score": row["total_score"],
            })
        
        return results
    
    def _log_calibration(
        self,
        model_family: str,
        weights: dict[str, float],
        auc: float,
        sample_count: int,
    ):
        """记录校准历史"""
        # 确保校准记录表存在
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS weight_calibration_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_family TEXT NOT NULL,
            weights TEXT,
            auc REAL,
            sample_count INTEGER,
            calibrated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        insert_sql = """
        INSERT INTO weight_calibration_history (model_family, weights, auc, sample_count)
        VALUES (?, ?, ?, ?)
        """
        
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(create_table_sql)
            conn.execute(insert_sql, (
                model_family,
                json.dumps(weights),
                auc,
                sample_count,
            ))
            conn.commit()
    
    def get_cached_weights(self, model_family: str) -> dict[str, float] | None:
        """获取缓存的权重"""
        return self._cached_weights.get(model_family)
    
    def get_latest_calibration(self, model_family: str) -> WeightCalibration | None:
        """获取最新的校准记录"""
        query = """
        SELECT model_family, weights, auc, sample_count, calibrated_at
        FROM weight_calibration_history
        WHERE model_family = ?
        ORDER BY calibrated_at DESC
        LIMIT 1
        """
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(query, (model_family,)).fetchone()
        
        if row:
            return WeightCalibration(
                model_family=row["model_family"],
                weights=json.loads(row["weights"]),
                auc=row["auc"],
                calibrated_at=row["calibrated_at"],
                sample_count=row["sample_count"],
            )
        return None


class ScoreConfidenceEstimator:
    """
    基于Bootstrap和统计模型的评分置信度估计
    """
    
    def estimate_confidence(
        self,
        case_results: list,
        dimension: str,
        n_bootstrap: int = 1000,
    ) -> ScoreWithConfidence:
        """
        估计维度分数的置信区间
        
        Args:
            case_results: 用例结果列表
            dimension: 维度名称
            n_bootstrap: Bootstrap重采样次数
        
        Returns:
            ScoreWithConfidence: 带置信区间的分数
        """
        # 筛选相关用例
        relevant_cases = [
            r for r in case_results
            if getattr(r.case, "dimension", None) == dimension or 
               getattr(r.case, "category", None) == dimension
        ]
        
        if not relevant_cases:
            return ScoreWithConfidence(
                point_estimate=50.0,
                ci_95_low=40.0,
                ci_95_high=60.0,
                std_error=5.0,
                sample_size=0,
                reliability="insufficient",
            )
        
        # Bootstrap重采样
        bootstrap_scores = []
        n_cases = len(relevant_cases)
        
        rng = np.random.RandomState(42)
        for _ in range(n_bootstrap):
            # 有放回采样
            sampled_indices = rng.choice(n_cases, size=n_cases, replace=True)
            sampled_cases = [relevant_cases[i] for i in sampled_indices]
            
            # 计算该次采样的分数
            score = self._calculate_weighted_pass_rate(sampled_cases)
            bootstrap_scores.append(score)
        
        # 计算统计量
        point_estimate = float(np.mean(bootstrap_scores))
        std_error = float(np.std(bootstrap_scores, ddof=1))
        ci_95_low, ci_95_high = np.percentile(bootstrap_scores, [2.5, 97.5])
        
        # 可靠性等级
        reliability = self._assess_reliability(n_cases, std_error, bootstrap_scores)
        
        return ScoreWithConfidence(
            point_estimate=point_estimate,
            ci_95_low=float(ci_95_low),
            ci_95_high=float(ci_95_high),
            std_error=std_error,
            sample_size=n_cases,
            reliability=reliability,
        )
    
    def _calculate_weighted_pass_rate(self, case_results: list) -> float:
        """计算加权的通过率分数"""
        if not case_results:
            return 50.0
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for case_result in case_results:
            # 计算该用例的通过率
            samples = getattr(case_result, "samples", [])
            if not samples:
                continue
            
            passed_count = sum(1 for s in samples if getattr(s, "judge_passed", False))
            pass_rate = passed_count / len(samples) if samples else 0.0
            
            # 获取用例权重
            case = case_result.case
            weight = getattr(case, "weight", 1.0)
            difficulty = getattr(case, "difficulty", 0.5)
            
            # 困难用例的权重调整
            adjusted_weight = weight * (1 + difficulty)
            
            weighted_sum += pass_rate * 100 * adjusted_weight
            total_weight += adjusted_weight
        
        if total_weight == 0:
            return 50.0
        
        return weighted_sum / total_weight
    
    def _assess_reliability(
        self,
        n_cases: int,
        std_error: float,
        bootstrap_scores: list,
    ) -> str:
        """评估分数可靠性等级"""
        
        # 检查Bootstrap分布的稳定性
        score_variance = np.var(bootstrap_scores)
        
        if n_cases < 5 or std_error > 15:
            return "insufficient"
        elif n_cases >= 10 and std_error < 5 and score_variance < 100:
            return "high"
        elif n_cases >= 8 and std_error < 10:
            return "medium"
        else:
            return "low"


# 全局实例
_calibrator: Optional[AdaptiveScoreCalibrator] = None
_estimator: Optional[ScoreConfidenceEstimator] = None


def get_calibrator(db_path: str = "llm_inspector.db") -> AdaptiveScoreCalibrator:
    """获取全局校准器实例"""
    global _calibrator
    if _calibrator is None:
        _calibrator = AdaptiveScoreCalibrator(db_path)
    return _calibrator


def get_confidence_estimator() -> ScoreConfidenceEstimator:
    """获取全局置信度估计器实例"""
    global _estimator
    if _estimator is None:
        _estimator = ScoreConfidenceEstimator()
    return _estimator


def calculate_score_with_confidence(
    case_results: list,
    dimension: str,
) -> dict:
    """
    便捷函数：计算带置信区间的分数
    
    Args:
        case_results: 用例结果列表
        dimension: 维度名称
    
    Returns:
        dict: 包含分数和置信区间的字典
    """
    estimator = get_confidence_estimator()
    result = estimator.estimate_confidence(case_results, dimension)
    return result.to_dict()
