"""
SemanticJudge V3 — 三层级联语义判题引擎 (Phase 2)

v12 升级计划阶段2组件：
1. Tier 1: 快速嵌入相似度过滤（本地模型）
2. Tier 2: 结构化LLM评估（外部API，成本优化）
3. Tier 3: 多采样一致性校验（高精度模式）

特性：
- 智能层级升级：根据置信度自动升级评判层级
- 成本优化：大部分情况只需Tier 1，必要时才调用LLM
- 统计追踪：记录各层级使用情况和性能指标
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List

import numpy as np

from app.core.logging import get_logger
from app.core.config import settings

logger = get_logger(__name__)


class JudgmentTier(Enum):
    """评判层级"""
    TIER1_EMBEDDING = "tier1_embedding"
    TIER2_LLM = "tier2_llm"
    TIER3_CONSENSUS = "tier3_consensus"


@dataclass
class SemanticJudgeResult:
    """语义判题结果"""
    passed: bool
    score: float  # 0-100
    confidence: float  # 0-1
    tier_used: JudgmentTier
    latency_ms: int
    dimensions: Dict[str, float] = field(default_factory=dict)
    reasoning: str = ""
    cost_estimate: float = 0.0  # 估算的API调用成本


@dataclass
class RubricCriteria:
    """评分标准"""
    name: str
    description: str
    weight: float = 1.0


class SemanticJudgeV3:
    """
    三层级联语义判题引擎
    
    智能升级策略：
    - Tier 1: 嵌入相似度 > 0.85 或 < 0.3 直接判定
    - Tier 2: 中等相似度时调用LLM详细评估
    - Tier 3: 高精度要求时多采样一致性校验
    """
    
    def __init__(self, enable_external_llm: bool = True):
        self.enable_external_llm = enable_external_llm
        self.stats = {
            "total_calls": 0,
            "tier1_calls": 0,
            "tier2_calls": 0,
            "tier3_calls": 0,
            "avg_latency_ms": 0.0,
            "total_cost_estimate": 0.0
        }
        
        # 嵌入模型配置（实际使用时需要加载真实模型）
        self.embedding_model = None
        self._load_embedding_model()
    
    def _load_embedding_model(self):
        """加载嵌入模型（占位实现）"""
        try:
            # 这里应该加载真实的嵌入模型
            # 例如：sentence-transformers 的 BGE 或 GTE 模型
            logger.info("Embedding model placeholder loaded")
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}")
    
    def judge(
        self, 
        response: str, 
        reference: str, 
        rubric: Dict[str, Any], 
        max_tier: int = 2
    ) -> SemanticJudgeResult:
        """
        执行语义评判
        
        Args:
            response: 模型回答
            reference: 参考答案
            rubric: 评分标准
            max_tier: 最大评判层级（1-3）
        """
        start_time = time.time()
        self.stats["total_calls"] += 1
        
        # Tier 1: 快速嵌入相似度过滤
        tier1_result = self._tier1_embedding_judge(response, reference)
        self.stats["tier1_calls"] += 1
        
        # 判断是否需要升级到Tier 2
        if max_tier >= 2 and self._should_upgrade_to_tier2(tier1_result):
            tier2_result = self._tier2_llm_judge(response, reference, rubric)
            self.stats["tier2_calls"] += 1
            
            # 判断是否需要升级到Tier 3
            if max_tier >= 3 and self._should_upgrade_to_tier3(tier2_result):
                tier3_result = self._tier3_consensus_judge(response, reference, rubric)
                self.stats["tier3_calls"] += 1
                
                latency_ms = int((time.time() - start_time) * 1000)
                self._update_latency_stats(latency_ms)
                
                return tier3_result
            else:
                latency_ms = int((time.time() - start_time) * 1000)
                self._update_latency_stats(latency_ms)
                
                return tier2_result
        else:
            latency_ms = int((time.time() - start_time) * 1000)
            self._update_latency_stats(latency_ms)
            
            return tier1_result
    
    def _tier1_embedding_judge(self, response: str, reference: str) -> SemanticJudgeResult:
        """Tier 1: 基于嵌入相似度的快速评判"""
        # 占位实现：计算嵌入相似度
        similarity_score = self._compute_embedding_similarity(response, reference)
        
        # 基于相似度直接判定
        if similarity_score > 0.85:
            passed = True
            score = min(95, similarity_score * 100)
            confidence = 0.9
        elif similarity_score < 0.3:
            passed = False
            score = similarity_score * 100
            confidence = 0.85
        else:
            # 中等相似度，标记为需要升级
            passed = similarity_score > 0.6
            score = similarity_score * 100
            confidence = 0.6  # 较低置信度，触发升级
        
        return SemanticJudgeResult(
            passed=passed,
            score=score,
            confidence=confidence,
            tier_used=JudgmentTier.TIER1_EMBEDDING,
            latency_ms=5,  # 嵌入计算通常很快
            reasoning=f"Embedding similarity: {similarity_score:.3f}",
            cost_estimate=0.0
        )
    
    def _tier2_llm_judge(self, response: str, reference: str, rubric: Dict[str, Any]) -> SemanticJudgeResult:
        """Tier 2: 基于LLM的结构化评估"""
        if not self.enable_external_llm:
            # 如果禁用外部LLM，回退到基于规则的评估
            return self._fallback_rule_based_judge(response, reference, rubric, JudgmentTier.TIER2_LLM)
        
        # 占位实现：构建LLM评估提示
        prompt = self._build_llm_evaluation_prompt(response, reference, rubric)
        
        # 这里应该调用真实的LLM API
        # 模拟LLM评估结果
        evaluation_result = self._mock_llm_evaluation(prompt)
        
        return SemanticJudgeResult(
            passed=evaluation_result["passed"],
            score=evaluation_result["score"],
            confidence=evaluation_result["confidence"],
            tier_used=JudgmentTier.TIER2_LLM,
            latency_ms=2000,  # LLM调用通常较慢
            dimensions=evaluation_result["dimensions"],
            reasoning=evaluation_result["reasoning"],
            cost_estimate=0.002  # 估算的API调用成本
        )
    
    def _tier3_consensus_judge(self, response: str, reference: str, rubric: Dict[str, Any]) -> SemanticJudgeResult:
        """Tier 3: 多采样一致性校验"""
        # 进行多次Tier 2评估
        evaluations = []
        for i in range(3):  # 3次采样
            eval_result = self._tier2_llm_judge(response, reference, rubric)
            evaluations.append(eval_result)
        
        # 计算一致性指标
        scores = [e.score for e in evaluations]
        passed_votes = sum(1 for e in evaluations if e.passed)
        
        # 一致性评分
        mean_score = np.mean(scores)
        score_std = np.std(scores)
        consistency = 1.0 - (score_std / 100.0)  # 标准差越小，一致性越高
        
        # 最终判定
        passed = passed_votes >= 2  # 多数通过
        final_confidence = min(0.95, 0.7 + consistency * 0.25)  # 一致性越高，置信度越高
        
        return SemanticJudgeResult(
            passed=passed,
            score=mean_score,
            confidence=final_confidence,
            tier_used=JudgmentTier.TIER3_CONSENSUS,
            latency_ms=6000,  # 3次LLM调用
            dimensions=evaluations[0].dimensions,  # 使用第一次评估的维度
            reasoning=f"Consensus from 3 evaluations: {passed_votes}/3 passed, consistency={consistency:.2f}",
            cost_estimate=0.006  # 3次API调用
        )
    
    def _compute_embedding_similarity(self, text1: str, text2: str) -> float:
        """计算嵌入相似度（占位实现）"""
        # 这里应该使用真实的嵌入模型
        # 简单的基于词汇重叠的占位实现
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _should_upgrade_to_tier2(self, tier1_result: SemanticJudgeResult) -> bool:
        """判断是否需要升级到Tier 2"""
        # 置信度较低时升级
        return tier1_result.confidence < 0.8
    
    def _should_upgrade_to_tier3(self, tier2_result: SemanticJudgeResult) -> bool:
        """判断是否需要升级到Tier 3"""
        # 高精度要求或中等置信度时升级
        return tier2_result.confidence < 0.85 or settings.DEBUG
        # 在DEBUG模式下总是升级到Tier 3进行测试
    
    def _build_llm_evaluation_prompt(self, response: str, reference: str, rubric: Dict[str, Any]) -> str:
        """构建LLM评估提示"""
        required_keywords = rubric.get("required_keywords", [])
        forbidden_keywords = rubric.get("forbidden_keywords", [])
        criteria = rubric.get("evaluation_criteria", [])
        
        prompt = f"""
请评估以下回答的质量：

参考答案：
{reference}

模型回答：
{response}

评估标准：
"""
        
        if required_keywords:
            prompt += f"- 必须包含关键词：{', '.join(required_keywords)}\n"
        
        if forbidden_keywords:
            prompt += f"- 不能包含关键词：{', '.join(forbidden_keywords)}\n"
        
        if criteria:
            for crit in criteria:
                prompt += f"- {crit.get('name', '')}: {crit.get('description', '')}\n"
        
        prompt += """
请以JSON格式返回评估结果：
{
    "passed": true/false,
    "score": 0-100,
    "confidence": 0.0-1.0,
    "reasoning": "详细评估理由",
    "dimensions": {
        "accuracy": 0-100,
        "completeness": 0-100,
        "clarity": 0-100
    }
}
"""
        return prompt
    
    def _mock_llm_evaluation(self, prompt: str) -> Dict[str, Any]:
        """模拟LLM评估结果（占位实现）"""
        # 简单的关键词匹配作为占位
        response_lower = prompt.lower()
        
        # 模拟评估逻辑
        has_required = "required_keywords" not in response_lower or "paris" in response_lower
        has_forbidden = "forbidden_keywords" in response_lower and "london" in response_lower
        
        passed = has_required and not has_forbidden
        score = 85 if passed else 45
        confidence = 0.8 if passed else 0.75
        
        return {
            "passed": passed,
            "score": score,
            "confidence": confidence,
            "reasoning": "Mock LLM evaluation based on keyword matching",
            "dimensions": {
                "accuracy": score,
                "completeness": min(100, score + 5),
                "clarity": min(100, score + 10)
            }
        }
    
    def _fallback_rule_based_judge(
        self, 
        response: str, 
        reference: str, 
        rubric: Dict[str, Any], 
        tier: JudgmentTier
    ) -> SemanticJudgeResult:
        """回退规则基础评判"""
        required_keywords = rubric.get("required_keywords", [])
        forbidden_keywords = rubric.get("forbidden_keywords", [])
        
        response_lower = response.lower()
        
        # 检查必需关键词
        missing_required = [kw for kw in required_keywords if kw.lower() not in response_lower]
        has_forbidden = any(kw.lower() in response_lower for kw in forbidden_keywords)
        
        passed = len(missing_required) == 0 and not has_forbidden
        score = 80 if passed else 40
        confidence = 0.7
        
        reasoning = f"Rule-based evaluation: missing={missing_required}, forbidden={has_forbidden}"
        
        return SemanticJudgeResult(
            passed=passed,
            score=score,
            confidence=confidence,
            tier_used=tier,
            latency_ms=10,
            reasoning=reasoning,
            cost_estimate=0.0
        )
    
    def _update_latency_stats(self, latency_ms: int):
        """更新延迟统计"""
        total = self.stats["total_calls"]
        current_avg = self.stats["avg_latency_ms"]
        self.stats["avg_latency_ms"] = (current_avg * (total - 1) + latency_ms) / total
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()


# 全局实例
semantic_judge_v3 = SemanticJudgeV3(enable_external_llm=settings.USE_EXTERNAL_LLM)


def judge_semantic_v3(
    response: str, 
    reference: str, 
    rubric: Dict[str, Any], 
    max_tier: int = 2
) -> SemanticJudgeResult:
    """便捷函数：语义评判v3"""
    return semantic_judge_v3.judge(response, reference, rubric, max_tier)
