"""
IRT (Item Response Theory) Engine — IRT 2PL模型实现

用于：
1. 校准测试用例参数（a, b）
2. 估计被测模型能力(theta)
3. 自适应选题(Adaptive Testing)

v5.0 升级组件 - P2 测试用例IRT校准与动态生成
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class IRTItemStats:
    """IRT 2PL模型参数"""
    item_id: str
    a: float  # 区分度 (discrimination)
    b: float  # 难度 (difficulty)
    c: float = 0.0  # 猜测参数（可选3PL）
    
    information: float = 0.0  # 项目信息函数值
    sample_size: int = 0
    last_calibrated: str = ""
    
    def __post_init__(self):
        if not self.last_calibrated:
            self.last_calibrated = datetime.now().isoformat()


@dataclass
class AbilityEstimate:
    """能力参数估计结果"""
    theta: float  # 能力估计值
    se: float     # 标准误
    ci_95_low: float
    ci_95_high: float
    n_items: int  # 使用的题目数
    reliability: str  # "high" | "medium" | "low"


class IRTEngine:
    """
    IRT (Item Response Theory) 引擎
    
    实现2PL模型（区分度+难度）
    """
    
    # IRT参数约束
    A_MIN, A_MAX = 0.1, 3.0  # 区分度范围
    B_MIN, B_MAX = -3.0, 3.0  # 难度范围（标准正态分布）
    
    def __init__(self):
        self._item_params: dict[str, IRTItemStats] = {}
    
    def calibrate_items(
        self,
        historical_results: list[dict],
    ) -> dict[str, IRTItemStats]:
        """
        基于历史检测数据校准测试用例参数
        
        使用EM算法估计a(区分度)和b(难度)
        
        Args:
            historical_results: 历史检测结果列表
                每项格式: {
                    "run_id": str,
                    "model_id": str,
                    "responses": [{"item_id": str, "passed": bool}],
                    "theta_estimate": float (optional)
                }
        
        Returns:
            dict[str, IRTItemStats]: 校准后的项目参数
        """
        # 构建响应矩阵
        # 行：模型实例，列：测试用例，值：通过(1)/失败(0)
        response_matrix = self._build_response_matrix(historical_results)
        
        if response_matrix.empty:
            logger.warning("No valid response data for IRT calibration")
            return {}
        
        # 估计或获取能力参数
        abilities = self._estimate_abilities_for_calibration(historical_results, response_matrix)
        
        # 对每个项目估计参数
        item_params = {}
        
        for item_id in response_matrix.columns:
            if item_id == "theta_estimate":
                continue
                
            # 提取该题目的响应数据
            responses = response_matrix[item_id].dropna().values
            
            if len(responses) < 5:  # 样本不足，跳过
                logger.warning(f"Insufficient samples for item {item_id}: {len(responses)}")
                continue
            
            # 获取对应的能力估计
            valid_abilities = []
            valid_responses = []
            for idx, passed in enumerate(response_matrix[item_id].values):
                if not np.isnan(passed) and idx < len(abilities):
                    valid_abilities.append(abilities[idx])
                    valid_responses.append(int(passed))
            
            if len(valid_responses) < 5:
                continue
            
            # 最大似然估计 a 和 b
            a_opt, b_opt = self._estimate_2pl_parameters(
                np.array(valid_abilities),
                np.array(valid_responses),
            )
            
            # 计算项目信息函数（在theta=0处的信息）
            info = self._calculate_information(a_opt, b_opt, 0.0)
            
            item_params[item_id] = IRTItemStats(
                item_id=item_id,
                a=round(a_opt, 3),
                b=round(b_opt, 3),
                information=round(info, 4),
                sample_size=len(valid_responses),
                last_calibrated=datetime.now().isoformat(),
            )
            
            logger.info(
                f"Calibrated item {item_id}",
                a=a_opt,
                b=b_opt,
                n=len(valid_responses),
            )
        
        self._item_params.update(item_params)
        return item_params
    
    def _build_response_matrix(
        self,
        historical_results: list[dict],
    ) -> "pd.DataFrame":
        """构建响应矩阵"""
        import pandas as pd
        
        # 收集所有项目和模型
        all_items = set()
        responses = []
        
        for result in historical_results:
            run_id = result.get("run_id", "unknown")
            model_id = result.get("model_id", "unknown")
            
            response_dict = {"run_id": run_id, "model_id": model_id}
            
            for resp in result.get("responses", []):
                item_id = resp.get("item_id")
                passed = resp.get("passed")
                if item_id and passed is not None:
                    all_items.add(item_id)
                    response_dict[item_id] = 1.0 if passed else 0.0
            
            # 如果提供，添加能力估计
            if "theta_estimate" in result:
                response_dict["theta_estimate"] = result["theta_estimate"]
            
            responses.append(response_dict)
        
        # 创建DataFrame
        df = pd.DataFrame(responses)
        
        # 确保所有项目列存在
        for item_id in all_items:
            if item_id not in df.columns:
                df[item_id] = np.nan
        
        return df
    
    def _estimate_abilities_for_calibration(
        self,
        historical_results: list[dict],
        response_matrix: "pd.DataFrame",
    ) -> np.ndarray:
        """为校准估计能力参数"""
        # 如果数据中已有theta_estimate，使用它
        if "theta_estimate" in response_matrix.columns:
            return response_matrix["theta_estimate"].fillna(0.0).values
        
        # 否则使用简单比例转换为z分数
        abilities = []
        for idx, row in response_matrix.iterrows():
            # 计算通过率
            item_cols = [c for c in response_matrix.columns if c not in 
                        ["run_id", "model_id", "theta_estimate"]]
            scores = [row[c] for c in item_cols if not np.isnan(row[c])]
            
            if scores:
                p = np.mean(scores)
                # 转换为z分数（避免极端值）
                p = max(0.001, min(0.999, p))
                theta = math.log(p / (1 - p))
            else:
                theta = 0.0
            
            abilities.append(theta)
        
        return np.array(abilities)
    
    def _estimate_2pl_parameters(
        self,
        abilities: np.ndarray,
        responses: np.ndarray,
    ) -> tuple[float, float]:
        """
        估计2PL模型参数 a 和 b
        
        使用最大似然估计
        """
        def neg_log_likelihood(params):
            a, b = params
            # 2PL模型概率
            predictions = 1 / (1 + np.exp(-a * (abilities - b)))
            # 添加正则化防止过拟合
            reg = 0.01 * (a ** 2 + b ** 2)
            
            # 对数似然（避免log(0)）
            eps = 1e-10
            ll = np.sum(
                responses * np.log(predictions + eps) +
                (1 - responses) * np.log(1 - predictions + eps)
            )
            
            return -ll + reg
        
        # 优化
        result = minimize(
            neg_log_likelihood,
            x0=[1.0, 0.0],  # 初始值：中等区分度，平均难度
            bounds=[(self.A_MIN, self.A_MAX), (self.B_MIN, self.B_MAX)],
            method="L-BFGS-B",
        )
        
        if result.success:
            return result.x[0], result.x[1]
        else:
            logger.warning(f"IRT parameter estimation failed: {result.message}")
            return 1.0, 0.0  # 返回默认值
    
    def _calculate_information(self, a: float, b: float, theta: float) -> float:
        """
        计算项目信息函数
        
        I(theta) = a^2 * P(theta) * (1 - P(theta))
        其中 P(theta) = 1 / (1 + exp(-a*(theta-b)))
        """
        p = 1 / (1 + math.exp(-a * (theta - b)))
        return a ** 2 * p * (1 - p)
    
    def select_next_item(
        self,
        current_theta: float,
        available_items: list[IRTItemStats],
        administered_items: list[str],
    ) -> IRTItemStats | None:
        """
        自适应选题：选择能提供最大信息量的未使用题目
        
        信息量函数：I(theta) = a^2 * P(theta) * (1 - P(theta))
        """
        # 过滤已使用题目
        available = [i for i in available_items if i.item_id not in administered_items]
        
        if not available:
            return None
        
        # 计算每个可用题目在当前theta处的信息量
        best_item = None
        max_info = -1.0
        
        for item in available:
            info = self._calculate_information(item.a, item.b, current_theta)
            
            if info > max_info:
                max_info = info
                best_item = item
        
        logger.debug(
            f"Selected item {best_item.item_id if best_item else None}",
            info=max_info,
            theta=current_theta,
        )
        
        return best_item
    
    def estimate_ability(
        self,
        responses: list[tuple[str, bool]],  # (item_id, passed)
        item_params: dict[str, IRTItemStats] | None = None,
    ) -> AbilityEstimate:
        """
        最大似然估计被测模型的能力参数theta
        
        Args:
            responses: 已回答的题目及结果列表
            item_params: 项目参数（如果不提供，使用已缓存的参数）
        
        Returns:
            AbilityEstimate: 能力估计结果
        """
        params = item_params or self._item_params
        
        if not params:
            logger.warning("No item parameters available for ability estimation")
            return AbilityEstimate(
                theta=0.0,
                se=1.0,
                ci_95_low=-2.0,
                ci_95_high=2.0,
                n_items=0,
                reliability="low",
            )
        
        def neg_log_likelihood(theta):
            ll = 0.0
            for item_id, passed in responses:
                item = params.get(item_id)
                if not item:
                    continue
                
                # 2PL模型概率
                p = 1 / (1 + math.exp(-item.a * (theta - item.b)))
                
                # 对数似然
                eps = 1e-10
                if passed:
                    ll += math.log(p + eps)
                else:
                    ll += math.log(1 - p + eps)
            
            return -ll
        
        # 优化求解最大似然
        result = minimize_scalar(
            neg_log_likelihood,
            bounds=(-4.0, 4.0),
            method="bounded",
        )
        
        theta = result.x
        
        # 计算标准误（使用Fisher信息）
        fisher_info = 0.0
        for item_id, passed in responses:
            item = params.get(item_id)
            if item:
                fisher_info += self._calculate_information(item.a, item.b, theta)
        
        if fisher_info > 0:
            se = 1.0 / math.sqrt(fisher_info)
        else:
            se = 1.0
        
        # 计算95%置信区间
        ci_95_low = theta - 1.96 * se
        ci_95_high = theta + 1.96 * se
        
        # 评估可靠性
        reliability = self._assess_reliability(len(responses), se)
        
        return AbilityEstimate(
            theta=round(theta, 4),
            se=round(se, 4),
            ci_95_low=round(ci_95_low, 4),
            ci_95_high=round(ci_95_high, 4),
            n_items=len(responses),
            reliability=reliability,
        )
    
    def _assess_reliability(self, n_items: int, se: float) -> str:
        """评估能力估计的可靠性"""
        if n_items >= 10 and se < 0.5:
            return "high"
        elif n_items >= 5 and se < 1.0:
            return "medium"
        else:
            return "low"
    
    def get_item_parameters(self) -> dict[str, IRTItemStats]:
        """获取当前缓存的项目参数"""
        return self._item_params.copy()
    
    def update_item_parameters(self, params: dict[str, IRTItemStats]):
        """更新项目参数"""
        self._item_params.update(params)


# 全局引擎实例
_irt_engine: Optional[IRTEngine] = None


def get_irt_engine() -> IRTEngine:
    """获取全局IRT引擎实例"""
    global _irt_engine
    if _irt_engine is None:
        _irt_engine = IRTEngine()
    return _irt_engine
