"""
Score Attribution Analyzer — 自动归因分析

自动分析分数构成，归因到具体用例

v5.0 升级组件 - P7 可解释性与审计系统
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CaseContribution:
    """单个用例的贡献分析"""
    case_id: str
    case_name: str
    pass_rate: float
    contribution: float
    impact: str  # "high" | "medium" | "low"
    weight: float = 1.0
    difficulty: float = 0.5


@dataclass
class DimensionAttribution:
    """维度归因分析"""
    dimension: str
    dimension_score: float
    top_positive_contributors: list[CaseContribution]
    top_negative_contributors: list[CaseContribution]
    total_contribution: float = 0.0


@dataclass
class AttributionReport:
    """分数归因报告"""
    total_score: float
    dimension_attributions: list[DimensionAttribution]
    summary: str
    key_findings: list[str] = field(default_factory=list)


class ScoreAttributionAnalyzer:
    """
    自动分析分数构成，归因到具体用例
    """
    
    # 默认维度列表
    DIMENSIONS = ["reasoning", "coding", "instruction", "safety", "knowledge"]
    
    def attribute_score(
        self,
        scorecard,
        case_results: list,
    ) -> AttributionReport:
        """
        分析总分是如何由各个用例贡献的
        
        Args:
            scorecard: 评分卡对象（有 dimension_score 属性）
            case_results: 用例结果列表
        
        Returns:
            AttributionReport: 归因分析报告
        """
        attributions = []
        
        for dimension in self.DIMENSIONS:
            # 获取维度分数
            dim_score = self._get_dimension_score(scorecard, dimension)
            
            # 获取该维度的用例
            dim_cases = [
                r for r in case_results
                if self._get_case_dimension(r) == dimension
            ]
            
            if not dim_cases:
                continue
            
            # 计算每个用例的贡献
            case_contributions = []
            for case_result in dim_cases:
                contribution = self._calculate_case_contribution(case_result, dim_score, len(dim_cases))
                case_contributions.append(contribution)
            
            # 按贡献排序
            positive = [c for c in case_contributions if c.contribution > 0]
            negative = [c for c in case_contributions if c.contribution < 0]
            
            positive.sort(key=lambda x: abs(x.contribution), reverse=True)
            negative.sort(key=lambda x: abs(x.contribution), reverse=True)
            
            attributions.append(DimensionAttribution(
                dimension=dimension,
                dimension_score=dim_score,
                top_positive_contributors=positive[:3],
                top_negative_contributors=negative[:3],
                total_contribution=sum(c.contribution for c in case_contributions),
            ))
        
        # 生成摘要
        summary = self._generate_attribution_summary(attributions)
        
        # 生成关键发现
        key_findings = self._generate_key_findings(attributions)
        
        return AttributionReport(
            total_score=self._get_total_score(scorecard),
            dimension_attributions=attributions,
            summary=summary,
            key_findings=key_findings,
        )
    
    def _get_dimension_score(self, scorecard, dimension: str) -> float:
        """从评分卡获取维度分数"""
        # 尝试各种可能的属性名
        attr_names = [
            f"{dimension}_score",
            dimension,
            f"{dimension.capitalize()}Score",
        ]
        
        for attr in attr_names:
            if hasattr(scorecard, attr):
                val = getattr(scorecard, attr)
                if isinstance(val, (int, float)):
                    return float(val)
        
        # 默认返回50
        return 50.0
    
    def _get_total_score(self, scorecard) -> float:
        """获取总分"""
        if hasattr(scorecard, "total_score"):
            return float(scorecard.total_score)
        
        # 计算所有维度平均
        total = 0.0
        count = 0
        for dim in self.DIMENSIONS:
            score = self._get_dimension_score(scorecard, dim)
            if score > 0:
                total += score
                count += 1
        
        return total / max(count, 1)
    
    def _get_case_dimension(self, case_result) -> str | None:
        """获取用例所属的维度"""
        case = getattr(case_result, "case", None)
        if not case:
            return None
        
        # 尝试 dimension 属性
        dim = getattr(case, "dimension", None)
        if dim:
            return dim
        
        # 尝试 category 属性
        cat = getattr(case, "category", None)
        if cat and cat in self.DIMENSIONS:
            return cat
        
        # 根据 category 映射
        category_map = {
            "reasoning": "reasoning",
            "coding": "coding",
            "code": "coding",
            "instruction": "instruction",
            "instruction_following": "instruction",
            "safety": "safety",
            "jailbreak": "safety",
            "knowledge": "knowledge",
            "knowledge_qa": "knowledge",
        }
        
        return category_map.get(cat)
    
    def _calculate_case_contribution(
        self,
        case_result,
        dimension_score: float,
        n_cases: int,
    ) -> CaseContribution:
        """
        计算单个用例对维度分数的贡献
        
        简化模型：假设维度分数是各用例通过率的加权平均
        """
        case = case_result.case
        
        # 获取用例信息
        case_id = getattr(case, "id", "unknown")
        case_name = getattr(case, "name", case_id)
        weight = getattr(case, "weight", 1.0)
        difficulty = getattr(case, "difficulty", 0.5)
        
        # 计算通过率
        samples = getattr(case_result, "samples", [])
        if not samples:
            pass_rate = 0.0
        else:
            passed = sum(1 for s in samples if getattr(s, "judge_passed", False))
            pass_rate = passed / len(samples)
        
        # 基础贡献：偏离50分基准
        base_contribution = ((pass_rate or 0) * 100 - 50) * (weight / max(n_cases, 1))
        
        # 考虑用例难度（困难用例的贡献权重更高）
        difficulty_multiplier = 1 + difficulty * 0.5
        
        contribution = base_contribution * difficulty_multiplier
        
        # 确定影响等级
        abs_contrib = abs(contribution)
        if abs_contrib > 5:
            impact = "high"
        elif abs_contrib > 2:
            impact = "medium"
        else:
            impact = "low"
        
        return CaseContribution(
            case_id=case_id,
            case_name=case_name,
            pass_rate=pass_rate,
            contribution=round(contribution, 2),
            impact=impact,
            weight=weight,
            difficulty=difficulty,
        )
    
    def _generate_attribution_summary(self, attributions: list[DimensionAttribution]) -> str:
        """生成归因摘要"""
        if not attributions:
            return "无可用的归因分析数据。"
        
        parts = []
        
        # 找出最强和最弱维度
        sorted_dims = sorted(attributions, key=lambda x: x.dimension_score, reverse=True)
        strongest = sorted_dims[0]
        weakest = sorted_dims[-1]
        
        parts.append(f"最强维度：{strongest.dimension}（{strongest.dimension_score:.1f}分）")
        parts.append(f"最弱维度：{weakest.dimension}（{weakest.dimension_score:.1f}分）")
        
        # 分析维度平衡性
        scores = [a.dimension_score for a in attributions]
        score_range = max(scores) - min(scores)
        
        if score_range > 20:
            parts.append(f"维度发展不平衡，极差达{score_range:.1f}分。")
        elif score_range < 10:
            parts.append("各维度发展较为均衡。")
        
        return "；".join(parts)
    
    def _generate_key_findings(self, attributions: list[DimensionAttribution]) -> list[str]:
        """生成关键发现列表"""
        findings = []
        
        for attr in attributions:
            # 低分维度分析
            if attr.dimension_score < 40:
                # 查找负贡献最大的用例
                if attr.top_negative_contributors:
                    top_neg = attr.top_negative_contributors[0]
                    findings.append(
                        f"{attr.dimension}维度得分较低（{attr.dimension_score:.1f}分），"
                        f"主要受用例'{top_neg.case_name}'影响（贡献{top_neg.contribution:.1f}分）"
                    )
            
            # 高分维度分析
            if attr.dimension_score > 80:
                if attr.top_positive_contributors:
                    top_pos = attr.top_positive_contributors[0]
                    findings.append(
                        f"{attr.dimension}维度表现优秀（{attr.dimension_score:.1f}分），"
                        f"用例'{top_pos.case_name}'贡献突出（+{top_pos.contribution:.1f}分）"
                    )
            
            # 困难用例表现
            hard_cases = [
                c for c in (attr.top_positive_contributors + attr.top_negative_contributors)
                if c.difficulty > 0.7 and c.contribution > 0
            ]
            if hard_cases:
                findings.append(
                    f"{attr.dimension}维度成功通过高难度用例'{hard_cases[0].case_name}'，"
                    f"体现较强的{attr.dimension}能力"
                )
        
        return findings[:5]  # 限制发现数量
    
    def to_dict(self, report: AttributionReport) -> dict:
        """将报告转换为字典格式"""
        return {
            "total_score": round(report.total_score, 1),
            "summary": report.summary,
            "key_findings": report.key_findings,
            "dimensions": [
                {
                    "dimension": a.dimension,
                    "score": round(a.dimension_score, 1),
                    "positive_contributors": [
                        {
                            "case_id": c.case_id,
                            "case_name": c.case_name,
                            "pass_rate": round(c.pass_rate, 2),
                            "contribution": c.contribution,
                            "impact": c.impact,
                        }
                        for c in a.top_positive_contributors
                    ],
                    "negative_contributors": [
                        {
                            "case_id": c.case_id,
                            "case_name": c.case_name,
                            "pass_rate": round(c.pass_rate, 2),
                            "contribution": c.contribution,
                            "impact": c.impact,
                        }
                        for c in a.top_negative_contributors
                    ],
                }
                for a in report.dimension_attributions
            ],
        }


# 全局实例
_analyzer: Optional[ScoreAttributionAnalyzer] = None


def get_attribution_analyzer() -> ScoreAttributionAnalyzer:
    """获取全局归因分析器实例"""
    global _analyzer
    if _analyzer is None:
        _analyzer = ScoreAttributionAnalyzer()
    return _analyzer


def analyze_score_attribution(scorecard, case_results: list) -> dict:
    """
    便捷函数：分析分数归因
    
    Args:
        scorecard: 评分卡对象
        case_results: 用例结果列表
    
    Returns:
        dict: 归因分析报告字典
    """
    analyzer = get_attribution_analyzer()
    report = analyzer.attribute_score(scorecard, case_results)
    return analyzer.to_dict(report)
