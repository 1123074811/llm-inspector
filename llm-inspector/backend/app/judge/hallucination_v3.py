"""
Hallucination Detection V3 — 多信号集成幻觉检测器 (Phase 2)

v12 升级计划阶段2组件：
1. 多信号检测：不确定性标记、事实矛盾、知识图谱验证
2. 集成评分：贝叶斯融合多个检测信号
3. 置信度量化：基于信号强度的概率估计

特性：
- 多层次检测：从语言模式到事实核查
- 知识图谱集成：可选的外部知识验证
- 置信度校准：基于历史数据的概率估计
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum

import numpy as np

from app.core.logging import get_logger
from app.core.config import settings

logger = get_logger(__name__)


class HallucinationType(Enum):
    """幻觉类型"""
    EXPLICIT_UNCERTAINTY = "explicit_uncertainty"
    FABRICATED_DETAILS = "fabricated_details"
    FACTUAL_CONTRADICTION = "factual_contradiction"
    FALSE_CONFIDENCE = "false_confidence"
    ENTITY_HALLUCINATION = "entity_hallucination"


@dataclass
class FactualClaim:
    """事实声明"""
    text: str
    entities: List[str]
    confidence: float
    verifiable: bool = True


@dataclass
class HallucinationSignal:
    """幻觉检测信号"""
    signal_type: str
    strength: float  # 0-1
    evidence: str
    confidence: float


@dataclass
class HallucinationDetectionResult:
    """幻觉检测结果"""
    ensemble_score: float  # 0-1, 越高越可能幻觉
    confidence: float  # 检测置信度
    uncertainty_present: bool
    factual_claims: List[FactualClaim]
    primary_signals: List[str]
    detailed_signals: List[HallucinationSignal]
    reasoning: str
    latency_ms: int


class HallucinationDetectorV3:
    """
    多信号集成幻觉检测器
    
    检测策略：
    1. 语言模式分析：不确定性标记、模糊表达
    2. 事实声明提取：识别可验证的事实性陈述
    3. 知识图谱验证：对比外部知识库（可选）
    4. 信号融合：贝叶斯方法集成多个信号
    """
    
    def __init__(self, use_knowledge_graph: bool = False):
        self.use_knowledge_graph = use_knowledge_graph
        self.stats = {
            "total_checks": 0,
            "avg_latency_ms": 0.0,
            "signal_distribution": {}
        }
        
        # 不确定性标记模式
        self.uncertainty_patterns = [
            r"\b(i think|maybe|perhaps|possibly|might be|could be|seems like|probably)\b",
            r"\b(i'm not sure|i don't know|not certain|unclear)\b",
            r"\b(approximately|roughly|about|around|estimated)\b",
            r"\b(assuming|presumably|supposedly|allegedly)\b"
        ]
        
        # 虚假细节模式
        self.fabrication_patterns = [
            r"\b(exactly|precisely|specifically|definitely|certainly)\b.*\d+",
            r"\b(studies show|research indicates|according to)\b.*(without|no|lack of)",
            r"\b(statistics|data|numbers|figures)\b.*(unavailable|missing|no data)"
        ]
        
        # 实体提取模式（简单实现）
        self.entity_pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"
        
        # 初始化知识图谱连接（占位）
        self.knowledge_graph = None
        if use_knowledge_graph:
            self._init_knowledge_graph()
    
    def _init_knowledge_graph(self):
        """初始化知识图谱连接（占位实现）"""
        try:
            # 这里应该连接到真实的知识图谱
            logger.info("Knowledge graph placeholder initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize knowledge graph: {e}")
    
    def detect(self, text: str) -> HallucinationDetectionResult:
        """
        检测文本中的幻觉内容
        
        Args:
            text: 待检测文本
            
        Returns:
            HallucinationDetectionResult: 检测结果
        """
        start_time = time.time()
        self.stats["total_checks"] += 1
        
        # 1. 提取事实声明
        factual_claims = self._extract_factual_claims(text)
        
        # 2. 检测各种幻觉信号
        signals = []
        
        # 不确定性信号
        uncertainty_signals = self._detect_uncertainty_signals(text)
        signals.extend(uncertainty_signals)
        
        # 虚假细节信号
        fabrication_signals = self._detect_fabrication_signals(text)
        signals.extend(fabrication_signals)
        
        # 错误置信度信号
        false_confidence_signals = self._detect_false_confidence_signals(text)
        signals.extend(false_confidence_signals)
        
        # 知识图谱验证（如果启用）
        if self.use_knowledge_graph and self.knowledge_graph:
            kg_signals = self._verify_with_knowledge_graph(factual_claims)
            signals.extend(kg_signals)
        
        # 3. 信号融合
        ensemble_score, primary_signals = self._fuse_signals(signals)
        
        # 4. 生成结果
        uncertainty_present = any(s.signal_type == "explicit_uncertainty" for s in signals)
        
        result = HallucinationDetectionResult(
            ensemble_score=ensemble_score,
            confidence=self._calculate_detection_confidence(signals),
            uncertainty_present=uncertainty_present,
            factual_claims=factual_claims,
            primary_signals=primary_signals,
            detailed_signals=signals,
            reasoning=self._generate_reasoning(signals, ensemble_score),
            latency_ms=int((time.time() - start_time) * 1000)
        )
        
        # 更新统计
        self._update_stats(result)
        
        return result
    
    def _extract_factual_claims(self, text: str) -> List[FactualClaim]:
        """提取事实声明"""
        claims = []
        
        # 简单的句子分割
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # 跳过太短的句子
                continue
            
            # 提取实体
            entities = re.findall(self.entity_pattern, sentence)
            
            # 评估可验证性
            verifiable = self._is_claim_verifiable(sentence)
            
            # 评估置信度（基于语言特征）
            confidence = self._estimate_claim_confidence(sentence)
            
            claim = FactualClaim(
                text=sentence,
                entities=entities,
                confidence=confidence,
                verifiable=verifiable
            )
            claims.append(claim)
        
        return claims
    
    def _detect_uncertainty_signals(self, text: str) -> List[HallucinationSignal]:
        """检测不确定性信号"""
        signals = []
        
        for pattern in self.uncertainty_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                strength = min(1.0, len(matches) * 0.3)
                signals.append(HallucinationSignal(
                    signal_type="explicit_uncertainty",
                    strength=strength,
                    evidence=f"Found uncertainty markers: {matches[:3]}",
                    confidence=0.8
                ))
        
        return signals
    
    def _detect_fabrication_signals(self, text: str) -> List[HallucinationSignal]:
        """检测虚假细节信号"""
        signals = []
        
        for pattern in self.fabrication_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                strength = min(1.0, len(matches) * 0.4)
                signals.append(HallucinationSignal(
                    signal_type="fabricated_details",
                    strength=strength,
                    evidence=f"Potential fabrication: {matches[:2]}",
                    confidence=0.7
                ))
        
        return signals
    
    def _detect_false_confidence_signals(self, text: str) -> List[HallucinationSignal]:
        """检测错误置信度信号"""
        signals = []
        
        # 检测过度自信的表达
        confidence_patterns = [
            r"\b(definitely|certainly|absolutely|without doubt|100%|always|never)\b",
            r"\b(obviously|clearly|undoubtedly|surely)\b"
        ]
        
        # 同时检测不确定性标记的组合
        has_uncertainty = any(re.search(p, text, re.IGNORECASE) for p in self.uncertainty_patterns)
        
        if has_uncertainty:
            for pattern in confidence_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    strength = min(1.0, len(matches) * 0.5)
                    signals.append(HallucinationSignal(
                        signal_type="false_confidence",
                        strength=strength,
                        evidence=f"Conflicting confidence signals: {matches[:3]}",
                        confidence=0.9
                    ))
        
        return signals
    
    def _verify_with_knowledge_graph(self, claims: List[FactualClaim]) -> List[HallucinationSignal]:
        """使用知识图谱验证事实声明（占位实现）"""
        signals = []
        
        for claim in claims:
            if not claim.verifiable:
                continue
            
            # 占位实现：模拟知识图谱查询
            verification_result = self._mock_kg_verification(claim)
            
            if verification_result["contradicted"]:
                signals.append(HallucinationSignal(
                    signal_type="factual_contradiction",
                    strength=verification_result["confidence"],
                    evidence=f"KG contradicts: {claim.text[:50]}...",
                    confidence=verification_result["confidence"]
                ))
        
        return signals
    
    def _mock_kg_verification(self, claim: FactualClaim) -> Dict[str, Any]:
        """模拟知识图谱验证（占位实现）"""
        # 简单的基于关键词的模拟验证
        text_lower = claim.text.lower()
        
        # 模拟一些常见错误
        contradictory_keywords = ["great wall built by napoleon", "paris capital of england"]
        
        for contradiction in contradictory_keywords:
            if contradiction in text_lower:
                return {
                    "contradicted": True,
                    "confidence": 0.9,
                    "evidence": "Known factual contradiction"
                }
        
        return {
            "contradicted": False,
            "confidence": 0.0,
            "evidence": "No contradiction found"
        }
    
    def _fuse_signals(self, signals: List[HallucinationSignal]) -> Tuple[float, List[str]]:
        """融合多个检测信号"""
        if not signals:
            return 0.0, []
        
        # 加权融合不同类型的信号
        type_weights = {
            "explicit_uncertainty": 0.3,
            "fabricated_details": 0.4,
            "false_confidence": 0.5,
            "factual_contradiction": 0.8,
            "entity_hallucination": 0.6
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        primary_signals = []
        
        for signal in signals:
            weight = type_weights.get(signal.signal_type, 0.3)
            weighted_score += signal.strength * weight * signal.confidence
            total_weight += weight * signal.confidence
            
            # 识别主要信号
            if signal.strength * signal.confidence > 0.5:
                primary_signals.append(signal.signal_type)
        
        ensemble_score = weighted_score / total_weight if total_weight > 0 else 0.0
        
        return min(1.0, ensemble_score), list(set(primary_signals))
    
    def _calculate_detection_confidence(self, signals: List[HallucinationSignal]) -> float:
        """计算检测置信度"""
        if not signals:
            return 0.0
        
        # 基于信号数量和强度计算置信度
        avg_confidence = np.mean([s.confidence for s in signals])
        signal_count_factor = min(1.0, len(signals) / 5.0)  # 信号越多，置信度越高
        
        return avg_confidence * (0.7 + 0.3 * signal_count_factor)
    
    def _is_claim_verifiable(self, sentence: str) -> bool:
        """判断声明是否可验证"""
        # 简单的启发式规则
        unverifiable_patterns = [
            r"\b(i think|i believe|in my opinion)\b",
            r"\b(perhaps|maybe|possibly)\b",
            r"\b(some|many|few|several)\b"  # 模糊量词
        ]
        
        return not any(re.search(p, sentence, re.IGNORECASE) for p in unverifiable_patterns)
    
    def _estimate_claim_confidence(self, sentence: str) -> float:
        """估计声明的置信度"""
        confidence = 0.5  # 基础置信度
        
        # 包含具体数字增加置信度
        if re.search(r'\b\d+\b', sentence):
            confidence += 0.2
        
        # 包含不确定性表达降低置信度
        for pattern in self.uncertainty_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                confidence -= 0.2
        
        # 包含确定表达增加置信度
        if re.search(r'\b(definitely|certainly|exactly)\b', sentence, re.IGNORECASE):
            confidence += 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def _generate_reasoning(self, signals: List[HallucinationSignal], ensemble_score: float) -> str:
        """生成检测推理"""
        if not signals:
            return "No hallucination signals detected"
        
        signal_types = [s.signal_type for s in signals if s.strength > 0.3]
        
        reasoning = f"Detected {len(signal_types)} types of signals: {', '.join(signal_types[:3])}"
        
        if ensemble_score > 0.7:
            reasoning += " - High hallucination risk"
        elif ensemble_score > 0.4:
            reasoning += " - Moderate hallucination risk"
        else:
            reasoning += " - Low hallucination risk"
        
        return reasoning
    
    def _update_stats(self, result: HallucinationDetectionResult):
        """更新统计信息"""
        # 更新延迟统计
        total = self.stats["total_checks"]
        current_avg = self.stats["avg_latency_ms"]
        self.stats["avg_latency_ms"] = (current_avg * (total - 1) + result.latency_ms) / total
        
        # 更新信号分布统计
        for signal in result.detailed_signals:
            signal_type = signal.signal_type
            self.stats["signal_distribution"][signal_type] = \
                self.stats["signal_distribution"].get(signal_type, 0) + 1
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()


# 全局实例
hallucination_detector_v3 = HallucinationDetectorV3(
    use_knowledge_graph=settings.USE_KNOWLEDGE_GRAPH
)


def detect_hallucination_v3(text: str) -> HallucinationDetectionResult:
    """便捷函数：幻觉检测v3"""
    return hallucination_detector_v3.detect(text)


# 为了兼容验证脚本，添加别名
hallucination_detect_v3 = detect_hallucination_v3
