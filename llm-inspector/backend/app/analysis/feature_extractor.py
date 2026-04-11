"""
Feature Extractor Module
Extract features from case results for analysis.

Split from pipeline.py in V6 refactoring for better code organization.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List

from app.core.logging import get_logger

logger = get_logger(__name__)

# Feature order for vectorization (must match across modules)
FEATURE_ORDER = [
    "avg_response_length",
    "avg_markdown_score", 
    "latency_mean_ms",
    "tokens_per_second",
    "refusal_verbosity",
    "avg_sentence_count",
    "avg_words_per_sentence",
    "has_usage_fields",
    "has_finish_reason",
    "param_compliance_rate",
    "format_compliance_score",
    "protocol_success_rate",
    "instruction_pass_rate",
    "exact_match_rate",
    "json_valid_rate",
    "format_follow_rate",
    "line_count_rate",
    "response_quality_basic_rate",
    "response_quality_strict_rate",
    "code_execution_rate",
    "identity_consistency_rate",
    "hallucination_detection_rate",
    "topic_relevance_rate",
    "refusal_rate",
    "over_refusal_rate",
    "safety_alternative_style",
    "adversarial_spoof_signal_rate",
    "extraction_resist_rate",
    "token_rounding_anomaly",
    "zero_completion_anomaly", 
    "identical_token_anomaly",
    "temp_zero_diversity",
    "token_count_consistent",
    "latency_length_correlated",
    "first_token_ratio_plausible",
    "usage_fingerprint_score",
]


class FeatureExtractor:
    """Extract features from case results for analysis."""
    
    def extract(self, case_results: List[Any]) -> Dict[str, float]:
        """
        Extract all features from case results.
        
        Args:
            case_results: List of case results with samples
            
        Returns:
            Dictionary of feature name -> value
        """
        if not case_results:
            return {}
        
        features = {}
        
        # --- Protocol features ---
        proto_cases = [r for r in case_results if r.case.category == "protocol"]
        if proto_cases:
            usage_cases = [
                r for r in proto_cases
                if any(s.response.usage_prompt_tokens for s in r.samples)
            ]
            features["has_usage_fields"] = 1.0 if usage_cases else 0.0
            finish_cases = [
                r for r in proto_cases
                if any(s.response.finish_reason for s in r.samples)
            ]
            features["has_finish_reason"] = 1.0 if finish_cases else 0.0

        # --- Instruction following ---
        instr_cases = [r for r in case_results if r.case.category == "instruction"]
        if instr_cases:
            features["instruction_pass_rate"] = self._pass_rate(instr_cases)
            exact = [r for r in instr_cases if r.case.judge_method == "exact_match"]
            features["exact_match_rate"] = self._pass_rate(exact) if exact else 0.0
            json_c = [r for r in instr_cases if r.case.judge_method == "json_schema"]
            features["json_valid_rate"] = self._pass_rate(json_c) if json_c else 0.0
            line_c = [r for r in instr_cases if r.case.judge_method == "line_count"]
            features["line_count_rate"] = self._pass_rate(line_c) if line_c else 0.0
            quality_basic = [r for r in instr_cases if r.case.judge_method == "response_quality_basic"]
            features["response_quality_basic_rate"] = self._pass_rate(quality_basic) if quality_basic else 0.0
            quality_strict = [r for r in instr_cases if r.case.judge_method == "response_quality_strict"]
            features["response_quality_strict_rate"] = self._pass_rate(quality_strict) if quality_strict else 0.0
            format_cases = [r for r in instr_cases if r.case.judge_method == "format_follow"]
            features["format_follow_rate"] = self._pass_rate(format_cases) if format_cases else 0.0

        # --- Coding ---
        coding_cases = [r for r in case_results if r.case.category == "coding"]
        if coding_cases:
            exec_cases = [r for r in coding_cases if r.case.judge_method == "code_execution"]
            features["code_execution_rate"] = self._pass_rate(exec_cases) if exec_cases else 0.0

        # --- Knowledge ---
        knowledge_cases = [r for r in case_results if r.case.category == "knowledge"]
        if knowledge_cases:
            topic_cases = [r for r in knowledge_cases if r.case.judge_method == "topic_relevance"]
            features["topic_relevance_rate"] = self._pass_rate(topic_cases) if topic_cases else 0.0
            halluc_cases = [r for r in knowledge_cases if r.case.judge_method == "hallucination_detect"]
            features["hallucination_detection_rate"] = self._pass_rate(halluc_cases) if halluc_cases else 0.0

        # --- Safety ---
        safety_cases = [r for r in case_results if r.case.category == "safety"]
        if safety_cases:
            ref_cases = [r for r in safety_cases if r.case.judge_method == "refusal_style"]
            features["refusal_rate"] = self._pass_rate(ref_cases) if ref_cases else 0.0
            over_ref_cases = [r for r in safety_cases if r.case.judge_method == "over_refusal"]
            features["over_refusal_rate"] = self._pass_rate(over_ref_cases) if over_ref_cases else 0.0
            alt_cases = [r for r in safety_cases if r.case.judge_method == "safety_alternative_style"]
            features["safety_alternative_style"] = self._pass_rate(alt_cases) if alt_cases else 0.0

        # --- Adversarial ---
        adv_cases = [r for r in case_results if r.case.category == "adversarial"]
        if adv_cases:
            spoof_cases = [r for r in adv_cases if r.case.judge_method == "adversarial_spoof_signal"]
            features["adversarial_spoof_signal_rate"] = self._pass_rate(spoof_cases) if spoof_cases else 0.0

        # --- Authenticity ---
        auth_cases = [r for r in case_results if r.case.category == "authenticity"]
        if auth_cases:
            id_cases = [r for r in auth_cases if r.case.judge_method == "identity_consistency"]
            features["identity_consistency_rate"] = self._pass_rate(id_cases) if id_cases else 0.0
            ext_cases = [r for r in auth_cases if r.case.judge_method == "extraction_resistance"]
            features["extraction_resist_rate"] = self._pass_rate(ext_cases) if ext_cases else 0.0

        # --- Response statistics ---
        all_samples = [s for r in case_results for s in r.samples if s.response.content]
        if all_samples:
            lengths = [len(s.response.content) for s in all_samples]
            features["avg_response_length"] = sum(lengths) / len(lengths)
            
            # Markdown formatting score
            markdown_scores = []
            for s in all_samples:
                content = s.response.content
                md_score = 0.0
                if content:
                    # Count markdown elements
                    headers = len(re.findall(r'^#+\s', content, re.MULTILINE))
                    lists = len(re.findall(r'^\s*[-*+]\s', content, re.MULTILINE))
                    code_blocks = len(re.findall(r'```', content)) // 2
                    bold = len(re.findall(r'\*\*[^*]+\*\*', content))
                    italic = len(re.findall(r'\*[^*]+\*', content))
                    
                    # Score based on presence of markdown elements
                    md_score = min(5.0, (headers + lists + code_blocks + bold/2 + italic/2) / 2.0)
                markdown_scores.append(md_score)
            features["avg_markdown_score"] = sum(markdown_scores) / len(markdown_scores)
            
            # Sentence statistics
            sentence_counts = []
            word_counts = []
            for s in all_samples:
                content = s.response.content
                if content:
                    sentences = len(re.split(r'[.!?。！？]+', content))
                    words = len(re.findall(r'\b\w+\b', content))
                    sentence_counts.append(sentences)
                    word_counts.append(words)
            
            if sentence_counts:
                features["avg_sentence_count"] = sum(sentence_counts) / len(sentence_counts)
                avg_words = sum(word_counts) / len(word_counts)
                features["avg_words_per_sentence"] = avg_words / sum(sentence_counts) * len(sentence_counts)

        # --- Latency and token statistics ---
        latencies = [s.response.latency_ms for s in all_samples if s.response.latency_ms]
        if latencies:
            features["latency_mean_ms"] = sum(latencies) / len(latencies)
        
        token_counts = []
        for s in all_samples:
            if s.response.usage_prompt_tokens and s.response.usage_completion_tokens:
                token_counts.append(s.response.usage_prompt_tokens + s.response.usage_completion_tokens)
        
        if token_counts and latencies:
            total_tokens = sum(token_counts)
            total_time = sum(latencies) / 1000.0  # Convert to seconds
            features["tokens_per_second"] = total_tokens / total_time if total_time > 0 else 0.0
            
            # Refusal verbosity (refusal length)
            refusal_lengths = []
            for s in all_samples:
                if s.response.content and any(word in s.response.content.lower() for word in ['cannot', 'unable', 'sorry', '抱歉', '无法', '不能']):
                    refusal_lengths.append(len(s.response.content))
            if refusal_lengths:
                features["refusal_verbosity"] = sum(refusal_lengths) / len(refusal_lengths)

        # --- v6: Token accounting anomaly detection (6.3) ---
        token_anomalies = self._detect_token_accounting_anomalies(case_results)
        features.update(token_anomalies)

        # -- v6: Response diversity at temperature=0 (6.4) --
        temp_zero_diversity = self._calculate_temp_zero_diversity(case_results)
        features["temp_zero_diversity"] = temp_zero_diversity

        # --- Additional derived features ---
        self._add_derived_features(features, case_results)

        return {k: round(v, 4) for k, v in features.items()}

    @staticmethod
    def _detect_token_accounting_anomalies(case_results: List[Any]) -> Dict[str, float]:
        """
        v6: Detect suspicious token accounting patterns.
        Returns anomaly scores for various suspicious patterns.
        """
        anomalies = {
            "token_rounding_anomaly": 0.0,  # Tokens always multiples of 10/100
            "zero_completion_anomaly": 0.0,  # Completion tokens always 0
            "identical_token_anomaly": 0.0,  # All samples have identical token counts
        }

        all_usage = [
            (s.response.usage_prompt_tokens, s.response.usage_completion_tokens)
            for r in case_results
            for s in r.samples
            if s.response.usage_prompt_tokens is not None
        ]

        if len(all_usage) < 5:
            return anomalies

        # Check for excessive rounding (always multiples of 10)
        prompt_tokens = [u[0] for u in all_usage]
        completion_tokens = [u[1] for u in all_usage if u[1] is not None]

        # Count how many are exact multiples of 10
        prompt_rounded = sum(1 for t in prompt_tokens if t % 10 == 0)
        if prompt_rounded / len(prompt_tokens) > 0.9:
            anomalies["token_rounding_anomaly"] = 1.0

        # Check for zero completion tokens
        if completion_tokens:
            zero_completions = sum(1 for t in completion_tokens if t == 0)
            if zero_completions / len(completion_tokens) > 0.8:
                anomalies["zero_completion_anomaly"] = 1.0

        # Check for identical token counts
        unique_prompt_counts = len(set(prompt_tokens))
        if unique_prompt_counts < len(prompt_tokens) * 0.1:  # Less than 10% variation
            anomalies["identical_token_anomaly"] = 1.0

        return anomalies

    @staticmethod
    def _calculate_temp_zero_diversity(case_results: List[Any]) -> float:
        """
        v6: Calculate response diversity at temperature=0.
        Real models should be deterministic at temp=0.
        """
        temp_zero_cases = [
            r for r in case_results 
            if r.case.temperature == 0.0 and len(r.samples) >= 2
        ]

        if not temp_zero_cases:
            return 0.0

        diversity_scores = []
        for case in temp_zero_cases:
            responses = [s.response.content for s in case.samples if s.response.content]
            if len(responses) < 2:
                continue

            # Calculate pairwise similarity
            similarities = []
            for i in range(len(responses)):
                for j in range(i + 1, len(responses)):
                    # Simple character-level similarity
                    r1, r2 = responses[i], responses[j]
                    if not r1 or not r2:
                        continue
                    # Jaccard similarity on character n-grams (n=3)
                    def ngrams(s, n=3):
                        return set(s[i:i+n] for i in range(len(s) - n + 1))
                    n1, n2 = ngrams(r1), ngrams(r2)
                    if n1 or n2:
                        intersection = len(n1 & n2)
                        union = len(n1 | n2)
                        sim = intersection / union if union > 0 else 0
                        similarities.append(sim)

            if similarities:
                diversity_scores.append(1.0 - sum(similarities) / len(similarities))

        return sum(diversity_scores) / len(diversity_scores) if diversity_scores else 0.0

    @staticmethod
    def _pass_rate(cases: List[Any]) -> float:
        """Calculate pass rate for a list of cases."""
        if not cases:
            return 0.0
        passed = sum(1 for c in cases if c.pass_rate > 0)
        return passed / len(cases)

    @staticmethod
    def _add_derived_features(features: Dict[str, float], case_results: List[Any]) -> None:
        """Add derived features based on extracted features."""
        # Usage fingerprint score
        usage_score = 0.0
        if features.get("has_usage_fields", 0) > 0:
            usage_score += 10  # Easy signals
        if features.get("has_finish_reason", 0) > 0:
            usage_score += 10
        if features.get("token_count_consistent", 0) > 0:
            usage_score += 25
        if features.get("latency_length_correlated", 0) > 0:
            usage_score += 30
        if features.get("first_token_ratio_plausible", 0) > 0:
            usage_score += 25
        features["usage_fingerprint_score"] = min(100.0, usage_score)

        # Token count consistency
        all_samples = [s for r in case_results for s in r.samples if s.response.content]
        if all_samples:
            token_consistent = 0
            for s in all_samples:
                if (s.response.usage_prompt_tokens and s.response.usage_completion_tokens and 
                    s.response.content):
                    prompt_len = len(s.response.content) * 1.3  # Rough estimate
                    total_tokens = s.response.usage_prompt_tokens + s.response.usage_completion_tokens
                    if abs(total_tokens - prompt_len) / prompt_len < 0.5:  # Within 50%
                        token_consistent += 1
            features["token_count_consistent"] = token_consistent / len(all_samples)

        # Latency-length correlation
        latencies = [s.response.latency_ms for s in all_samples if s.response.latency_ms]
        lengths = [len(s.response.content) for s in all_samples if s.response.content]
        if len(latencies) > 1 and len(lengths) > 1:
            # Simple correlation check
            avg_latency = sum(latencies) / len(latencies)
            avg_length = sum(lengths) / len(lengths)
            if avg_latency > 0 and avg_length > 0:
                correlation = sum((l - avg_latency) * (len_ - avg_length) 
                                for l, len_ in zip(latencies, lengths))
                correlation /= math.sqrt(sum((l - avg_latency) ** 2 for l in latencies) * 
                                      sum((len_ - avg_length) ** 2 for len_ in lengths))
                features["latency_length_correlated"] = max(0.0, correlation)

        # First token ratio plausibility
        first_token_ratios = []
        for s in all_samples:
            if (s.response.latency_ms and s.response.first_token_ms and 
                s.response.latency_ms > s.response.first_token_ms > 0):
                ratio = s.response.first_token_ms / s.response.latency_ms
                first_token_ratios.append(ratio)
        
        if first_token_ratios:
            avg_ratio = sum(first_token_ratios) / len(first_token_ratios)
            # Real streaming typically has first token < 20% of total time
            features["first_token_ratio_plausible"] = 1.0 if avg_ratio < 0.2 else 0.0

        # Protocol compliance rate
        proto_cases = [r for r in case_results if r.case.category == "protocol"]
        if proto_cases:
            features["protocol_success_rate"] = FeatureExtractor._pass_rate(proto_cases)

        # Parameter compliance and format compliance
        param_cases = [r for r in case_results if r.case.judge_method == "param_compliance"]
        if param_cases:
            features["param_compliance_rate"] = FeatureExtractor._pass_rate(param_cases)
        
        format_cases = [r for r in case_results if r.case.judge_method in ["json_schema", "yaml_csv_validate"]]
        if format_cases:
            features["format_compliance_score"] = FeatureExtractor._pass_rate(format_cases) * 100
