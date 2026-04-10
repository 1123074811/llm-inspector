"""
Neural Similarity Engine — 神经网络相似度基础版

基于对比学习的模型行为嵌入提取

v5.0 升级组件 - P3 相似度引擎深度学习增强
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class NeuralSimilarityResult:
    """神经网络相似度计算结果"""
    similarity: float  # 0-1
    component_scores: dict[str, float]
    confidence: float
    method: str


class BehavioralEmbeddingExtractor:
    """
    基于对比学习的模型行为嵌入提取器（基础版）
    
    使用传统特征工程作为神经网络的简化替代
    """
    
    # 特征顺序（与FEATURE_ORDER对齐）
    FEATURE_NAMES = [
        "protocol_success_rate",
        "instruction_pass_rate",
        "system_obedience_rate",
        "param_compliance_rate",
        "refusal_rate",
        "identity_consistency_pass_rate",
        "avg_response_length",
        "avg_markdown_score",
        "ttft_mean",
        "ttft_std",
    ]
    
    def __init__(self):
        self._embedding_dim = 32  # 简化版嵌入维度
    
    def extract_embedding(
        self,
        features: dict[str, float],
        case_results: list,
    ) -> np.ndarray:
        """
        提取模型行为嵌入
        
        Args:
            features: 特征字典
            case_results: 用例结果列表
        
        Returns:
            np.ndarray: 行为嵌入向量
        """
        # 1. 构建基础特征向量
        base_features = self._build_feature_vector(features)
        
        # 2. 提取时序模式特征
        temporal_features = self._extract_temporal_patterns(case_results)
        
        # 3. 提取风格指纹
        style_fingerprint = self._extract_style_fingerprint(case_results)
        
        # 4. 合并并降维（使用PCA简化）
        combined = np.concatenate([
            base_features,
            temporal_features,
            style_fingerprint,
        ])
        
        # 5. 投影到低维空间（简化版：取前32个重要维度）
        embedding = self._project_embedding(combined)
        
        # L2归一化
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _build_feature_vector(self, features: dict[str, float]) -> np.ndarray:
        """从特征字典构建特征向量"""
        vec = []
        for name in self.FEATURE_NAMES:
            val = features.get(name, 0.0)
            # 归一化到0-1
            if "rate" in name or "score" in name:
                vec.append(max(0.0, min(1.0, val)))
            elif "length" in name:
                vec.append(min(1.0, val / 5000))  # 假设最大5000字符
            elif "ttft" in name:
                vec.append(min(1.0, val / 5000))  # 假设最大5000ms
            else:
                vec.append(max(0.0, min(1.0, val)))
        
        # 填充到固定长度
        while len(vec) < len(self.FEATURE_NAMES):
            vec.append(0.0)
        
        return np.array(vec[:len(self.FEATURE_NAMES)], dtype=np.float32)
    
    def _extract_temporal_patterns(self, case_results: list) -> np.ndarray:
        """
        提取时序行为模式特征
        
        包括：
        - TTFT分布的统计特征
        - 延迟-长度关系曲线
        - token生成速率模式
        """
        patterns = []
        
        # TTFT分布特征
        ttfts = []
        latencies = []
        lengths = []
        
        for r in case_results:
            for s in getattr(r, "samples", []):
                resp = getattr(s, "response", None)
                if resp:
                    ft = getattr(resp, "first_token_ms", None)
                    if ft:
                        ttfts.append(ft)
                    
                    lat = getattr(resp, "latency_ms", None)
                    if lat:
                        latencies.append(lat)
                    
                    content = getattr(resp, "content", "") or ""
                    if content:
                        lengths.append(len(content))
        
        # TTFT统计特征
        if ttfts:
            patterns.extend([
                np.mean(ttfts) / 1000,
                np.std(ttfts) / 1000 if len(ttfts) > 1 else 0,
                np.percentile(ttfts, 25) / 1000,
                np.percentile(ttfts, 75) / 1000,
            ])
        else:
            patterns.extend([0, 0, 0, 0])
        
        # 延迟-长度关系
        if len(latencies) >= 5 and len(lengths) >= 5:
            # 计算相关性
            min_len = min(len(latencies), len(lengths))
            corr = np.corrcoef(latencies[:min_len], lengths[:min_len])[0, 1]
            patterns.extend([
                corr if not np.isnan(corr) else 0,
                np.mean(latencies) / max(np.mean(lengths), 1),
            ])
        else:
            patterns.extend([0, 0])
        
        # Token生成速率估算
        if latencies and lengths:
            # 假设1字符≈0.5token（中文）或0.25token（英文）
            est_tokens = np.mean(lengths) * 0.4
            est_time = np.mean(latencies) / 1000  # 秒
            if est_time > 0:
                patterns.append(est_tokens / est_time / 100)  # 归一化
            else:
                patterns.append(0)
        else:
            patterns.append(0)
        
        return np.array(patterns, dtype=np.float32)
    
    def _extract_style_fingerprint(self, case_results: list) -> np.ndarray:
        """
        提取响应风格指纹
        
        包括：
        - 标点和格式偏好
        - 词汇多样性
        - 句子长度分布
        """
        fingerprint = []
        
        all_texts = []
        for r in case_results:
            for s in getattr(r, "samples", []):
                resp = getattr(s, "response", None)
                if resp:
                    content = getattr(resp, "content", "") or ""
                    if content:
                        all_texts.append(content)
        
        if not all_texts:
            return np.zeros(6, dtype=np.float32)
        
        # 合并所有文本进行分析
        combined_text = " ".join(all_texts)
        
        # 1. 标点使用模式
        punct_counts = {
            "。": combined_text.count("。"),
            "，": combined_text.count("，"),
            ",": combined_text.count(","),
            ".": combined_text.count("."),
            "!": combined_text.count("!"),
            "?": combined_text.count("?"),
        }
        total_chars = len(combined_text)
        if total_chars > 0:
            fingerprint.extend([
                punct_counts["。"] / total_chars * 100,
                punct_counts["，"] / total_chars * 100,
                (punct_counts[","] + punct_counts["."]) / total_chars * 100,
            ])
        else:
            fingerprint.extend([0, 0, 0])
        
        # 2. 词汇多样性（简化版）
        words = combined_text.split()
        unique_words = set(words)
        if words:
            fingerprint.append(len(unique_words) / len(words))
        else:
            fingerprint.append(0)
        
        # 3. 平均句子长度
        sentences = combined_text.replace("。", ".").replace("！", "!").replace("？", "?").split(".")
        sentence_lengths = [len(s.strip()) for s in sentences if s.strip()]
        if sentence_lengths:
            fingerprint.append(np.mean(sentence_lengths) / 100)
        else:
            fingerprint.append(0)
        
        # 4. Markdown使用频率
        md_patterns = ["```", "**", "#", "- ", "|"]
        md_count = sum(combined_text.count(p) for p in md_patterns)
        fingerprint.append(min(1.0, md_count / max(len(all_texts), 1)))
        
        return np.array(fingerprint, dtype=np.float32)
    
    def _project_embedding(self, combined: np.ndarray) -> np.ndarray:
        """将组合特征投影到低维嵌入空间"""
        # 简化版：使用随机投影矩阵（实际可使用预训练的PCA或神经网络）
        target_dim = self._embedding_dim
        
        if len(combined) <= target_dim:
            # 如果原始维度已经够低，直接返回填充后的结果
            result = np.zeros(target_dim, dtype=np.float32)
            result[:len(combined)] = combined
            return result
        
        # 使用确定性随机投影（保证可重复性）
        rng = np.random.RandomState(42)
        projection_matrix = rng.randn(len(combined), target_dim) / math.sqrt(len(combined))
        
        embedded = combined @ projection_matrix
        
        return embedded.astype(np.float32)
    
    def compute_similarity(
        self,
        embedding_a: np.ndarray,
        embedding_b: np.ndarray,
    ) -> float:
        """计算余弦相似度"""
        dot_product = np.dot(embedding_a, embedding_b)
        norm_a = np.linalg.norm(embedding_a)
        norm_b = np.linalg.norm(embedding_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        similarity = dot_product / (norm_a * norm_b)
        return float(max(0.0, min(1.0, similarity)))


class MultiModalSimilarityFusion:
    """
    融合多维度相似度信号
    """
    
    def __init__(self):
        self.extractor = BehavioralEmbeddingExtractor()
    
    def compute_fused_similarity(
        self,
        target_features: dict[str, float],
        target_results: list,
        benchmark_features: dict[str, float],
        benchmark_results: list,
    ) -> NeuralSimilarityResult:
        """
        计算融合相似度
        
        Args:
            target_features: 目标模型特征
            target_results: 目标模型用例结果
            benchmark_features: 基准模型特征
            benchmark_results: 基准模型用例结果
        
        Returns:
            NeuralSimilarityResult: 融合相似度结果
        """
        # 1. 行为向量相似度（传统）
        behavioral_sim = self._behavioral_cosine_similarity(
            target_features, benchmark_features
        )
        
        # 2. 神经网络嵌入相似度
        emb_a = self.extractor.extract_embedding(target_features, target_results)
        emb_b = self.extractor.extract_embedding(benchmark_features, benchmark_results)
        neural_sim = self.extractor.compute_similarity(emb_a, emb_b)
        
        # 3. 响应风格相似度
        style_sim = self._response_style_similarity(target_results, benchmark_results)
        
        # 4. 时间指纹相似度
        temporal_sim = self._temporal_pattern_similarity(target_results, benchmark_results)
        
        # 5. 融合（使用固定权重）
        similarities = {
            "behavioral": behavioral_sim,
            "neural": neural_sim,
            "style": style_sim,
            "temporal": temporal_sim,
        }
        
        # 固定权重配置
        weights = {
            "behavioral": 0.30,
            "neural": 0.35,
            "style": 0.20,
            "temporal": 0.15,
        }
        
        fused = sum(
            weights.get(modality, 0.25) * score
            for modality, score in similarities.items()
        )
        
        # 计算融合置信度
        confidence = self._compute_fusion_confidence(similarities, weights)
        
        return NeuralSimilarityResult(
            similarity=round(fused, 4),
            component_scores={k: round(v, 4) for k, v in similarities.items()},
            confidence=round(confidence, 4),
            method="multi_modal_fusion_v1",
        )
    
    def _behavioral_cosine_similarity(
        self,
        features_a: dict[str, float],
        features_b: dict[str, float],
    ) -> float:
        """计算传统行为向量余弦相似度"""
        # 提取共同特征
        common_keys = set(features_a.keys()) & set(features_b.keys())
        
        if not common_keys:
            return 0.0
        
        vec_a = np.array([features_a.get(k, 0) for k in common_keys])
        vec_b = np.array([features_b.get(k, 0) for k in common_keys])
        
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    
    def _response_style_similarity(
        self,
        results_a: list,
        results_b: list,
    ) -> float:
        """计算响应风格相似度"""
        # 提取文本特征
        texts_a = self._extract_texts_from_results(results_a)
        texts_b = self._extract_texts_from_results(results_b)
        
        if not texts_a or not texts_b:
            return 0.5  # 中性值
        
        # 计算简单特征相似度
        feat_a = self._compute_text_features(texts_a)
        feat_b = self._compute_text_features(texts_b)
        
        # 余弦相似度
        norm_a = np.linalg.norm(feat_a)
        norm_b = np.linalg.norm(feat_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(np.dot(feat_a, feat_b) / (norm_a * norm_b))
    
    def _temporal_pattern_similarity(
        self,
        results_a: list,
        results_b: list,
    ) -> float:
        """计算时间模式相似度"""
        # 提取延迟分布
        ttfts_a = self._extract_ttfts(results_a)
        ttfts_b = self._extract_ttfts(results_b)
        
        if len(ttfts_a) < 3 or len(ttfts_b) < 3:
            return 0.5  # 数据不足
        
        # 比较统计特征
        stats_a = [np.mean(ttfts_a), np.std(ttfts_a), np.median(ttfts_a)]
        stats_b = [np.mean(ttfts_b), np.std(ttfts_b), np.median(ttfts_b)]
        
        # 标准化
        all_vals = ttfts_a + ttfts_b
        if all_vals:
            scale = np.std(all_vals)
            if scale > 0:
                stats_a = [s / scale for s in stats_a]
                stats_b = [s / scale for s in stats_b]
        
        # 欧氏距离转换相似度
        dist = np.linalg.norm(np.array(stats_a) - np.array(stats_b))
        similarity = max(0, 1 - dist / 3)  # 归一化
        
        return float(similarity)
    
    def _extract_texts_from_results(self, results: list) -> list[str]:
        """从结果中提取文本"""
        texts = []
        for r in results:
            for s in getattr(r, "samples", []):
                resp = getattr(s, "response", None)
                if resp:
                    content = getattr(resp, "content", "")
                    if content:
                        texts.append(content)
        return texts
    
    def _compute_text_features(self, texts: list[str]) -> np.ndarray:
        """计算文本特征向量"""
        if not texts:
            return np.zeros(4)
        
        combined = " ".join(texts)
        
        # 简单特征
        features = [
            len(combined) / 1000,  # 总长度
            len(set(combined.split())) / max(len(combined.split()), 1),  # 词汇多样性
            combined.count(".") / max(len(texts), 1),  # 句子密度
            sum(c.isupper() for c in combined) / max(len(combined), 1),  # 大写比例
        ]
        
        return np.array(features)
    
    def _extract_ttfts(self, results: list) -> list[float]:
        """提取TTFT值"""
        ttfts = []
        for r in results:
            for s in getattr(r, "samples", []):
                resp = getattr(s, "response", None)
                if resp:
                    ft = getattr(resp, "first_token_ms", None)
                    if ft is not None:
                        ttfts.append(float(ft))
        return ttfts
    
    def _compute_fusion_confidence(
        self,
        similarities: dict[str, float],
        weights: dict[str, float],
    ) -> float:
        """计算融合置信度"""
        # 基于各信号方差的简单估计
        values = list(similarities.values())
        variance = np.var(values)
        
        # 方差小 = 一致性好 = 高置信度
        confidence = max(0.3, 1 - variance)
        
        return float(confidence)


# 全局实例
_extractor: Optional[BehavioralEmbeddingExtractor] = None
_fusion: Optional[MultiModalSimilarityFusion] = None


def get_embedding_extractor() -> BehavioralEmbeddingExtractor:
    """获取全局嵌入提取器实例"""
    global _extractor
    if _extractor is None:
        _extractor = BehavioralEmbeddingExtractor()
    return _extractor


def get_similarity_fusion() -> MultiModalSimilarityFusion:
    """获取全局相似度融合器实例"""
    global _fusion
    if _fusion is None:
        _fusion = MultiModalSimilarityFusion()
    return _fusion


def compute_neural_similarity(
    features_a: dict[str, float],
    results_a: list,
    features_b: dict[str, float],
    results_b: list,
) -> dict:
    """
    便捷函数：计算神经网络增强的相似度
    
    Args:
        features_a: 模型A的特征
        results_a: 模型A的用例结果
        features_b: 模型B的特征
        results_b: 模型B的用例结果
    
    Returns:
        dict: 包含相似度和组件分数的字典
    """
    fusion = get_similarity_fusion()
    result = fusion.compute_fused_similarity(
        features_a, results_a,
        features_b, results_b,
    )
    
    return {
        "similarity": result.similarity,
        "confidence": result.confidence,
        "components": result.component_scores,
        "method": result.method,
    }
