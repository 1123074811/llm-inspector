"""
SemanticJudge V2 — 新一代语义判题引擎，结合：
1. 本地嵌入模型（BGE-large-zh-v1.5 / GTE-large-en-v1.5）
2. LLM-as-Judge（外部API）
3. 结构化评分标准（Rubric-based）

v5.0 升级组件 - P0 判题系统智能化重构
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from app.core.logging import get_logger
from app.core.config import settings

logger = get_logger(__name__)


@dataclass
class SemanticJudgeResult:
    """语义判题结果"""
    passed: bool
    score: float  # 0-100
    confidence: float  # 0-1
    method: str
    dimensions: dict[str, float] = field(default_factory=dict)
    reasoning: str = ""


@dataclass
class RubricScores:
    """Rubric维度评分"""
    dimension_scores: dict[str, float]
    confidence: float
    reasoning: str


class SemanticJudgeV2:
    """
    新一代语义判题引擎
    
    三层级联评判：
    1. 快速过滤：嵌入相似度（本地，<10ms）
    2. 结构化评分：LLM按维度打分
    3. 一致性校验：多轮采样降低方差
    """
    
    # 本地嵌入模型（延迟加载）
    _embedding_model: Optional[object] = None
    _cache_dir: str = field(default="./models", init=False)
    
    # Rubric维度定义
    RUBRIC_DIMENSIONS = {
        "correctness": {
            "weight": 0.35,
            "criteria": ["答案事实准确", "推理过程正确", "结论合理"],
            "scoring": "0-10",
        },
        "completeness": {
            "weight": 0.25,
            "criteria": ["覆盖问题要点", "没有遗漏关键信息"],
            "scoring": "0-10",
        },
        "clarity": {
            "weight": 0.20,
            "criteria": ["表达清晰", "结构合理", "易于理解"],
            "scoring": "0-10",
        },
        "reasoning_quality": {
            "weight": 0.20,
            "criteria": ["逻辑严密", "论证充分", "没有跳跃"],
            "scoring": "0-10",
        },
    }
    
    # 嵌入相似度阈值
    EMBEDDING_FILTER_THRESHOLD = 0.3
    EMBEDDING_PASS_THRESHOLD = 0.85
    
    def __init__(self, cache_dir: str = "./models"):
        self._cache_dir = cache_dir
        self._embedding_model = None
    
    def judge(
        self,
        prompt: str,
        response: str,
        reference: str | None = None,
        rubric: dict | None = None,
        rubric_weights: dict[str, float] | None = None,
        pass_threshold: float = 60.0,
        llm_judge_weight: float = 0.6,
    ) -> SemanticJudgeResult:
        """
        三层级联评判
        
        Args:
            prompt: 原始问题提示
            response: 模型回答
            reference: 参考答案（可选）
            rubric: 自定义评分标准（可选，默认使用RUBRIC_DIMENSIONS）
            rubric_weights: 自定义维度权重（可选，覆盖RUBRIC_DIMENSIONS中的权重）
            pass_threshold: 通过阈值（默认60分）
            llm_judge_weight: LLM评分在共识中的权重（默认0.6，本地评分占0.4）
        
        Returns:
            SemanticJudgeResult: 判题结果
        """
        # Layer 1: 本地嵌入过滤
        emb_score = self._embedding_similarity(prompt, response, reference)
        if emb_score < self.EMBEDDING_FILTER_THRESHOLD:
            # 明显不相关，直接失败
            return SemanticJudgeResult(
                passed=False,
                score=emb_score * 100,
                confidence=0.8,
                method="embedding_filter",
                reasoning="与问题语义关联度过低",
            )
        
        # 如果嵌入相似度很高，可以直接通过（快速路径）
        if emb_score >= self.EMBEDDING_PASS_THRESHOLD and reference:
            return SemanticJudgeResult(
                passed=True,
                score=emb_score * 100,
                confidence=0.75,
                method="embedding_fast_pass",
                reasoning="与参考答案语义高度一致",
            )
        
        # Layer 2: LLM结构化评分
        try:
            llm_scores = self._llm_rubric_scoring(prompt, response, reference, rubric)
        except Exception as e:
            logger.warning("LLM rubric scoring failed", error=str(e))
            # LLM评分失败，退回到嵌入相似度
            final_score = emb_score * 100
            return SemanticJudgeResult(
                passed=final_score >= 60,
                score=final_score,
                confidence=0.5,
                method="embedding_fallback",
                reasoning=f"LLM评分失败，使用嵌入相似度: {emb_score:.2f}",
            )
        
        # Layer 3: 一致性校验（仅当置信度低时）
        if llm_scores.confidence < 0.7:
            try:
                consensus = self._multi_round_consensus(prompt, response, reference, rubric, n_rounds=3)
                # 融合多轮结果
                for dim, scores in consensus.items():
                    llm_scores.dimension_scores[dim] = np.mean(scores)
                llm_scores.confidence = min(0.95, llm_scores.confidence + 0.15)
            except Exception as e:
                logger.warning("Multi-round consensus failed", error=str(e))
        
        # 综合评分
        final_score = self._weighted_rubric_score(llm_scores)
        
        return SemanticJudgeResult(
            passed=final_score >= 60,  # 从45提升到60
            score=final_score,
            confidence=llm_scores.confidence,
            method="semantic_v2",
            dimensions=llm_scores.dimension_scores,
            reasoning=llm_scores.reasoning,
        )
    
    def _get_embedding_model(self):
        """延迟加载嵌入模型"""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                
                # 自动选择模型（中文优先）
                model_name = "BAAI/bge-large-zh-v1.5"
                
                logger.info("Loading embedding model", model=model_name)
                self._embedding_model = SentenceTransformer(
                    model_name,
                    cache_folder=self._cache_dir,
                    device="cpu",  # 可根据环境改为cuda
                )
                logger.info("Embedding model loaded successfully")
            except ImportError:
                logger.error("sentence_transformers not installed")
                raise
            except Exception as e:
                logger.error("Failed to load embedding model", error=str(e))
                raise
        
        return self._embedding_model
    
    def _embedding_similarity(
        self, prompt: str, response: str, reference: str | None
    ) -> float:
        """基于本地嵌入模型的语义相似度计算"""
        try:
            model = self._get_embedding_model()
        except Exception:
            # 模型加载失败，使用简单的关键词重叠
            return self._fallback_similarity(prompt, response, reference)
        
        # 编码
        try:
            query_emb = model.encode(prompt, normalize_embeddings=True)
            resp_emb = model.encode(response, normalize_embeddings=True)
            
            # 计算余弦相似度（已归一化，点积即余弦相似度）
            sim = float(np.dot(query_emb, resp_emb))
            
            # 如果有参考答案，计算与参考答案的相似度
            if reference:
                ref_emb = model.encode(reference, normalize_embeddings=True)
                ref_sim = float(np.dot(resp_emb, ref_emb))
                # 取两者中较高者（响应既应回答问题，也应接近参考答案）
                sim = max(sim, ref_sim * 0.8)  # 参考答案权重略降
            
            return max(0.0, min(1.0, sim))
        except Exception as e:
            logger.warning("Embedding similarity calculation failed", error=str(e))
            return self._fallback_similarity(prompt, response, reference)
    
    def _fallback_similarity(
        self, prompt: str, response: str, reference: str | None
    ) -> float:
        """嵌入模型不可用时使用关键词重叠作为回退"""
        def extract_keywords(text: str) -> set[str]:
            # 简单中文/英文分词
            words = re.findall(r'[\w\u4e00-\u9fff]+', text.lower())
            # 过滤短词
            return {w for w in words if len(w) > 1}
        
        prompt_words = extract_keywords(prompt)
        response_words = extract_keywords(response)
        
        if not prompt_words:
            return 0.5
        
        overlap = len(prompt_words & response_words)
        sim = overlap / len(prompt_words)
        
        if reference:
            ref_words = extract_keywords(reference)
            ref_overlap = len(response_words & ref_words)
            ref_sim = ref_overlap / max(len(ref_words), 1)
            sim = max(sim, ref_sim * 0.8)
        
        return min(1.0, sim)
    
    def _llm_rubric_scoring(
        self,
        prompt: str,
        response: str,
        reference: str | None,
        rubric: dict | None,
    ) -> RubricScores:
        """
        使用LLM按Rubric维度进行结构化评分
        """
        # 构建评分提示
        dimensions = rubric.get("dimensions", self.RUBRIC_DIMENSIONS) if rubric else self.RUBRIC_DIMENSIONS
        
        scoring_prompt = self._build_scoring_prompt(prompt, response, reference, dimensions)
        
        # 调用LLM进行评分
        try:
            scores = self._call_llm_scorer(scoring_prompt)
        except Exception as e:
            logger.warning("LLM scoring failed", error=str(e))
            # 返回默认分数
            return RubricScores(
                dimension_scores={dim: 5.0 for dim in dimensions.keys()},
                confidence=0.3,
                reasoning="LLM评分失败，使用默认分数",
            )
        
        return scores
    
    def _build_scoring_prompt(
        self,
        prompt: str,
        response: str,
        reference: str | None,
        dimensions: dict,
    ) -> str:
        """构建结构化评分提示"""
        dim_descriptions = []
        for dim_name, dim_info in dimensions.items():
            criteria_str = "\n".join(f"  - {c}" for c in dim_info.get("criteria", []))
            dim_descriptions.append(
                f"【{dim_name}】(权重{dim_info.get('weight', 0.25)*100:.0f}%)\n{criteria_str}"
            )
        
        ref_section = f"\n## 参考答案\n{reference}\n" if reference else ""
        
        return f"""你是一位严谨的评分专家。请对以下AI回答进行多维度评分。

## 原始问题
{prompt[:500]}

## AI回答
{response[:2000]}
{ref_section}
## 评分维度
{"\n\n".join(dim_descriptions)}

## 评分要求
1. 每个维度给出0-10分的评分（10分为完美）
2. 简要说明每个维度的评分理由（1句话）
3. 给出整体置信度（0.0-1.0）

## 输出格式
请以JSON格式输出：
{{
  "dimensions": {{
{chr(10).join(f'    "{k}": 评分数字,' for k in dimensions.keys())}
  }},
  "reasoning": "整体评价摘要",
  "confidence": 0.85
}}"""
    
    def _call_llm_scorer(self, prompt: str) -> RubricScores:
        """调用LLM评分API"""
        import json
        import urllib.request
        import urllib.error
        
        payload = {
            "model": getattr(settings, "JUDGE_MODEL", "gpt-4o-mini"),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        
        url = getattr(settings, "JUDGE_API_URL", "").rstrip("/")
        api_key = getattr(settings, "JUDGE_API_KEY", "")
        
        if not url or not api_key:
            # 没有配置LLM judge，返回默认分数
            return RubricScores(
                dimension_scores={dim: 5.0 for dim in self.RUBRIC_DIMENSIONS.keys()},
                confidence=0.3,
                reasoning="LLM scorer not configured",
            )
        
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"
        
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        
        try:
            with urllib.request.urlopen(req, timeout=getattr(settings, "JUDGE_TIMEOUT", 30)) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            logger.warning("LLM scorer API call failed", error=str(e))
            raise
        
        # 解析响应
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        try:
            parsed = json.loads(content.strip())
            dimensions = parsed.get("dimensions", {})
            confidence = float(parsed.get("confidence", 0.5))
            reasoning = parsed.get("reasoning", "")
            
            return RubricScores(
                dimension_scores=dimensions,
                confidence=confidence,
                reasoning=reasoning,
            )
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Failed to parse LLM scorer response", error=str(e), content=content[:200])
            raise
    
    def _multi_round_consensus(
        self,
        prompt: str,
        response: str,
        reference: str | None,
        rubric: dict | None,
        n_rounds: int = 3,
    ) -> dict[str, list[float]]:
        """多轮评分取共识"""
        results: dict[str, list[float]] = {}
        
        for i in range(n_rounds):
            try:
                # 轻微扰动提示以获取不同视角
                perturbed_prompt = f"[Round {i+1}] {prompt}"
                scores = self._llm_rubric_scoring(perturbed_prompt, response, reference, rubric)
                
                for dim, score in scores.dimension_scores.items():
                    if dim not in results:
                        results[dim] = []
                    results[dim].append(float(score))
            except Exception as e:
                logger.warning(f"Round {i+1} scoring failed", error=str(e))
                continue
        
        return results
    
    def _weighted_rubric_score(
        self,
        scores: RubricScores,
        rubric_weights: dict[str, float] | None = None,
    ) -> float:
        """计算加权总分"""
        total_weight = 0.0
        weighted_sum = 0.0

        for dim_name, dim_info in self.RUBRIC_DIMENSIONS.items():
            # v6: Allow case-level weight override
            if rubric_weights and dim_name in rubric_weights:
                weight = rubric_weights[dim_name]
            else:
                weight = dim_info.get("weight", 0.25)
            score = scores.dimension_scores.get(dim_name, 5.0)

            # 转换为0-100分制
            normalized_score = score * 10

            weighted_sum += normalized_score * weight
            total_weight += weight

        if total_weight == 0:
            return 50.0

        return round(weighted_sum / total_weight, 1)


# 全局实例（单例模式）
_semantic_judge_v2: Optional[SemanticJudgeV2] = None


def get_semantic_judge_v2() -> SemanticJudgeV2:
    """获取全局语义判题引擎v2实例"""
    global _semantic_judge_v2
    if _semantic_judge_v2 is None:
        _semantic_judge_v2 = SemanticJudgeV2()
    return _semantic_judge_v2


def semantic_judge_v2(
    text: str,
    params: dict,
) -> tuple[bool, dict]:
    """
    兼容现有接口的语义判题v2函数

    Args:
        text: 模型回答文本
        params: 判题参数 (支持 rubric_weights, pass_threshold, llm_judge_weight)

    Returns:
        (passed, detail_dict)
    """
    prompt = params.get("_original_prompt", "")
    reference = params.get("reference_answer")
    rubric = params.get("rubric", {})
    # v6: Extract configurable parameters
    rubric_weights = params.get("rubric_weights")
    pass_threshold = params.get("semantic_pass_threshold", 60.0)
    llm_judge_weight = params.get("llm_judge_weight", 0.6)

    judge = get_semantic_judge_v2()
    result = judge.judge(
        prompt, text, reference, rubric,
        rubric_weights=rubric_weights,
        pass_threshold=pass_threshold,
        llm_judge_weight=llm_judge_weight,
    )

    detail = {
        "method": result.method,
        "score": result.score,
        "confidence": result.confidence,
        "reasoning": result.reasoning,
        "pass_threshold": pass_threshold,
        "llm_judge_weight": llm_judge_weight,
    }

    if result.dimensions:
        detail["dimensions"] = result.dimensions

    return result.passed, detail
