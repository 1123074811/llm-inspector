"""
analysis/feature_engine.py — FeatureExtractor

Extracts a flat dict of named numeric features from all CaseResult objects.
Extracted from pipeline.py to keep individual files under ~750 lines.
"""
from __future__ import annotations

import math
import re

import numpy as np

from app.core.schemas import CaseResult
from app.core.logging import get_logger

logger = get_logger(__name__)


# ── Feature Extractor ─────────────────────────────────────────────────────────

class FeatureExtractor:
    """
    Extracts a flat dict of named numeric features from all case results.
    """

    def extract(self, case_results: list[CaseResult]) -> dict[str, float]:
        features: dict[str, float] = {}
        dim_stats = self._dimension_stats(case_results)
        tag_stats = self._tag_stats(case_results)
        failure_stats = self._failure_attribution(case_results)

        # --- Protocol features (from protocol category) ---
        proto_cases = [r for r in case_results if r.case.category == "protocol"]
        if proto_cases:
            features["protocol_success_rate"] = self._pass_rate(proto_cases)
            usage_cases = [
                r for r in proto_cases
                if any(s.response.usage_total_tokens for s in r.samples)
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
            features["line_count_follow_rate"] = self._pass_rate(line_c) if line_c else 0.0
            regex_c = [r for r in instr_cases if r.case.judge_method == "regex_match"]
            features["format_follow_rate"] = self._pass_rate(regex_c) if regex_c else 0.0

        # --- System prompt obedience ---
        sys_cases = [r for r in case_results if r.case.category == "system"]
        if sys_cases:
            features["system_obedience_rate"] = self._pass_rate(sys_cases)

        # --- Parameter compliance ---
        param_cases = [r for r in case_results if r.case.category == "param"]
        if param_cases:
            features["param_compliance_rate"] = self._pass_rate(
                [r for r in param_cases if r.samples and r.samples[-1].judge_passed is not None]
            )
            temp_cases = [
                r for r in param_cases
                if r.case.judge_method == "heuristic_style"
                and isinstance(r.case.params, dict)
                and "also_run_at" in r.case.params
            ]
            if temp_cases:
                for r in temp_cases:
                    for s in r.samples:
                        if "temperature_param_effective" in s.judge_detail:
                            features["temperature_param_effective"] = (
                                1.0 if s.judge_detail["temperature_param_effective"] else 0.0
                            )

        # --- Style features ---
        style_cases = [r for r in case_results if r.case.category == "style"]
        if style_cases:
            markdown_scores = []
            lengths = []
            has_disclaimer_count = 0
            for r in style_cases:
                for s in r.samples:
                    d = s.judge_detail
                    if "markdown_score" in d:
                        markdown_scores.append(d["markdown_score"])
                    if "length" in d:
                        lengths.append(d["length"])
                    if d.get("has_disclaimer"):
                        has_disclaimer_count += 1
            if markdown_scores:
                features["avg_markdown_score"] = sum(markdown_scores) / len(markdown_scores)
            if lengths:
                features["avg_response_length"] = sum(lengths) / len(lengths)
            total_style_samples = sum(len(r.samples) for r in style_cases)
            if total_style_samples > 0:
                features["disclaimer_rate"] = has_disclaimer_count / total_style_samples

        # --- Refusal features ---
        refusal_cases = [r for r in case_results if r.case.category == "refusal"]
        if refusal_cases:
            refusal_count = 0
            alt_count = 0
            total = 0
            for r in refusal_cases:
                for s in r.samples:
                    total += 1
                    if s.judge_detail.get("refusal_detected"):
                        refusal_count += 1
                    if s.judge_detail.get("offers_alternative"):
                        alt_count += 1
            if total > 0:
                features["refusal_rate"] = refusal_count / total
                features["alt_suggestion_rate"] = alt_count / total

        # --- Antispoof identity features ---
        identity_cases = [r for r in case_results if r.case.judge_method == "identity_consistency"]
        if identity_cases:
            judged = [
                s for r in identity_cases for s in r.samples
                if s.judge_passed is not None
            ]
            if judged:
                features["identity_consistency_pass_rate"] = (
                    sum(1 for s in judged if s.judge_passed) / len(judged)
                )

        antispoof_cases = [r for r in case_results if r.case.category == "antispoof"]
        if antispoof_cases:
            extract_samples = []
            contradiction_samples = []
            for r in antispoof_cases:
                for s in r.samples:
                    detail = s.judge_detail or {}
                    detected = detail.get("detected_identities") or []
                    if detected:
                        extract_samples.append(1)
                    elif "detected_identities" in detail:
                        extract_samples.append(0)

                    if "leaked_identity" in detail:
                        leaked = detail.get("leaked_identity") or []
                        contradiction_samples.append(1 if leaked else 0)

            if extract_samples:
                features["antispoof_identity_detect_rate"] = sum(extract_samples) / len(extract_samples)
            if contradiction_samples:
                features["antispoof_override_leak_rate"] = (
                    sum(contradiction_samples) / len(contradiction_samples)
                )

        # --- Latency features ---
        all_latencies = [
            s.response.latency_ms
            for r in case_results
            for s in r.samples
            if s.response.latency_ms is not None
        ]
        if all_latencies:
            features["latency_mean_ms"] = sum(all_latencies) / len(all_latencies)
            sorted_lat = sorted(all_latencies)
            idx_p95 = max(0, int(len(sorted_lat) * 0.95) - 1)
            features["latency_p95_ms"] = sorted_lat[idx_p95]

        features.update(dim_stats)
        features.update(tag_stats)
        features.update(failure_stats)

        # --- Adversarial spoof signal ---
        # v11 fix: Use category=="adversarial" instead of dimension=="adversarial_reasoning".
        # Suite cases have category=adversarial but dimension=safety/reasoning, so the old
        # dimension filter matched zero cases and adversarial_spoof_signal_rate was always 0.
        adv_cases = [
            r for r in case_results
            if r.case.category == "adversarial"
        ]
        if adv_cases:
            features["adversarial_spoof_signal_rate"] = self._adversarial_spoof_rate(
                adv_cases, case_results,
            )

        # --- Usage fingerprint signals (add at end of extract()) ---

        # 1. token_count_consistent: prompt+completion ≈ total (within 5%)
        all_usage = [
            (s.response.usage_prompt_tokens,
             s.response.usage_completion_tokens,
             s.response.usage_total_tokens)
            for r in case_results
            for s in r.samples
            if s.response.usage_prompt_tokens and s.response.usage_total_tokens
        ]
        if len(all_usage) >= 3:
            consistent_count = sum(
                1 for p, c, t in all_usage
                if c is not None and abs((p + c) - t) / max(t, 1) < 0.05
            )
            features["token_count_consistent"] = consistent_count / len(all_usage)

        # 2. latency_length_correlated: check if longer responses take more time
        #    Use short-answer cases (max_tokens<=5) vs long cases as two groups.
        short_latencies = [
            s.response.latency_ms for r in case_results for s in r.samples
            if r.case.max_tokens <= 5 and s.response.latency_ms
        ]
        long_latencies = [
            s.response.latency_ms for r in case_results for s in r.samples
            if r.case.max_tokens > 50 and s.response.latency_ms
        ]
        if short_latencies and long_latencies:
            avg_short = sum(short_latencies) / len(short_latencies)
            avg_long = sum(long_latencies) / len(long_latencies)
            # Real model: long should be at least 20% slower than short
            features["latency_length_correlated"] = 1.0 if avg_long > avg_short * 1.2 else 0.0

        # 3. first_token_ratio_plausible: first_token_ms < 60% of total latency
        ftms_samples = [
            (s.response.first_token_ms, s.response.latency_ms)
            for r in case_results for s in r.samples
            if s.response.first_token_ms and s.response.latency_ms
               and s.response.latency_ms > 200  # ignore trivially short responses
        ]
        if len(ftms_samples) >= 3:
            plausible = sum(
                1 for ft, total in ftms_samples if ft < total * 0.6
            )
            features["first_token_ratio_plausible"] = plausible / len(ftms_samples)

        # --- Refined refusal features ---
        # Split refusal/safety cases into harmful vs legitimate expected groups
        # v7 fix: Include category=="safety" cases (v3/v10 suites) and handle
        # expect_refusal, require_refusal, and forbid_compliance param names
        # v11 fix: Also include category=="adversarial" (jailbreak/override cases
        # with require_refusal=True were invisible to safety scoring)
        harmful_refusal_cases = [
            r for r in case_results
            if r.case.category in ("refusal", "safety", "adversarial")
            and (r.case.params.get("expect_refusal", False)
                 or r.case.params.get("require_refusal", False)
                 or r.case.params.get("forbid_compliance", False))
        ]
        legit_compliance_cases = [
            r for r in case_results
            if r.case.category in ("refusal", "safety")
            and r.case.params.get("expect_compliance", False)
        ]
        if harmful_refusal_cases:
            features["harmful_refusal_rate"] = self._pass_rate(harmful_refusal_cases)
        if legit_compliance_cases:
            features["legit_compliance_rate"] = self._pass_rate(legit_compliance_cases)
            features["over_refusal_rate"] = 1.0 - features["legit_compliance_rate"]

        ttfts = [
            s.response.first_token_ms
            for r in case_results
            for s in r.samples
            if s.response.first_token_ms is not None and s.response.first_token_ms > 0
        ]

        if len(ttfts) >= 8:
            arr = np.array(sorted(ttfts))
            mid = len(arr) // 2
            low_half, high_half = arr[:mid], arr[mid:]

            full_std = float(np.std(arr))
            half_std = float((np.std(low_half) + np.std(high_half)) / 2)

            if full_std > 0:
                bimodal_ratio = 1.0 - (half_std / full_std)
                features["ttft_bimodal_ratio"] = round(max(0.0, bimodal_ratio), 4)

                gap_ms = float(np.median(high_half) - np.median(low_half))
                features["ttft_cluster_gap_ms"] = round(gap_ms, 1)

                features["ttft_proxy_signal"] = (
                    1.0 if (bimodal_ratio > 0.4 and gap_ms > 300) else 0.0
                )
            else:
                features["ttft_bimodal_ratio"] = 0.0
                features["ttft_cluster_gap_ms"] = 0.0
                features["ttft_proxy_signal"] = 0.0

        features["ttft_sample_count"] = len(ttfts)
        features["ttft_mean_ms"] = round(sum(ttfts) / len(ttfts), 1) if ttfts else 0.0
        if ttfts:
            sorted_ttfts = sorted(ttfts)
            features["ttft_p95_ms"] = round(sorted_ttfts[int(len(sorted_ttfts) * 0.95)], 1)
        else:
            features["ttft_p95_ms"] = 0.0

        all_latencies = [
            s.response.latency_ms
            for r in case_results
            for s in r.samples
            if s.response.latency_ms is not None
        ]
        if len(all_latencies) >= 2:
            mean_lat = sum(all_latencies) / len(all_latencies)
            variance = sum((l - mean_lat) ** 2 for l in all_latencies) / len(all_latencies)
            std_lat = math.sqrt(variance)
            features["latency_cv"] = round(std_lat / mean_lat, 4) if mean_lat > 0 else 0.0
        else:
            features["latency_cv"] = 0.0

        total_tokens_gen = [
            s.response.usage_completion_tokens
            for r in case_results
            for s in r.samples
            if s.response.usage_completion_tokens is not None and s.response.latency_ms
        ]
        if total_tokens_gen and all_latencies:
            total_tokens = sum(total_tokens_gen)
            total_time_sec = sum(all_latencies) / 1000.0
            features["tokens_per_second"] = round(total_tokens / total_time_sec, 2) if total_time_sec > 0 else 0.0
        else:
            features["tokens_per_second"] = 0.0

        ftms_list = [
            s.response.first_token_ms
            for r in case_results
            for s in r.samples
            if s.response.first_token_ms is not None
        ]
        features["first_token_mean_ms"] = round(sum(ftms_list) / len(ftms_list), 1) if ftms_list else 0.0

        reasoning_cases = [r for r in case_results if r.case.category == "reasoning"]
        features["reasoning_pass_rate"] = self._pass_rate(reasoning_cases) if reasoning_cases else 0.0

        coding_cases = [r for r in case_results if r.case.category == "coding"]
        features["coding_pass_rate"] = self._pass_rate(coding_cases) if coding_cases else 0.0

        adv_cases = [r for r in case_results if (r.case.dimension or "").lower() == "adversarial_reasoning"]
        features["adversarial_pass_rate"] = self._pass_rate(adv_cases) if adv_cases else 0.0

        # v7 fix: Include safety-category cases for alternative style extraction
        refusal_samples = [
            (s.response.content or "", s.judge_detail)
            for r in case_results
            for s in r.samples
            if r.case.category in ("refusal", "safety") and s.response.content
        ]
        if refusal_samples:
            refusal_verbosities = [len(content) for content, _ in refusal_samples if content]
            features["refusal_verbosity"] = round(sum(refusal_verbosities) / len(refusal_verbosities), 1) if refusal_verbosities else 0.0
            alt_count = sum(1 for _, d in refusal_samples if d and d.get("offers_alternative"))
            features["safety_alternative_style"] = round(alt_count / len(refusal_samples), 4)
        else:
            features["refusal_verbosity"] = 0.0
            features["safety_alternative_style"] = 0.0

        sentence_counts = []
        words_per_sentence_list = []
        bullet_count = 0
        numbered_count = 0
        code_block_count = 0
        heading_count = 0
        total_style_samples = 0

        for r in case_results:
            if r.case.category in ("style", "reasoning", "coding", "instruction"):
                for s in r.samples:
                    if s.response.content:
                        total_style_samples += 1
                        content = s.response.content
                        sentences = re.split(r'[.!?。！？]+', content)
                        sentences = [ss.strip() for ss in sentences if ss.strip()]
                        sentence_counts.append(len(sentences))
                        words = content.split()
                        if sentences:
                            words_per_sentence_list.append(len(words) / len(sentences))
                        if re.search(r'^\s*[-*•]\s', content, re.MULTILINE):
                            bullet_count += 1
                        if re.search(r'^\s*\d+\.\s', content, re.MULTILINE):
                            numbered_count += 1
                        if '```' in content:
                            code_block_count += 1
                        if re.search(r'^#{1,4}\s', content, re.MULTILINE):
                            heading_count += 1

        if total_style_samples > 0:
            features["avg_sentence_count"] = round(sum(sentence_counts) / len(sentence_counts), 2) if sentence_counts else 0.0
            features["avg_words_per_sentence"] = round(sum(words_per_sentence_list) / len(words_per_sentence_list), 2) if words_per_sentence_list else 0.0
            features["bullet_list_rate"] = round(bullet_count / total_style_samples, 4)
            features["numbered_list_rate"] = round(numbered_count / total_style_samples, 4)
            features["code_block_rate"] = round(code_block_count / total_style_samples, 4)
            features["heading_rate"] = round(heading_count / total_style_samples, 4)
        else:
            features["avg_sentence_count"] = 0.0
            features["avg_words_per_sentence"] = 0.0
            features["bullet_list_rate"] = 0.0
            features["numbered_list_rate"] = 0.0
            features["code_block_rate"] = 0.0
            features["heading_rate"] = 0.0

        difficulty_results = []
        for r in case_results:
            diff = getattr(r.case, 'difficulty', None)
            if diff is not None:
                for s in r.samples:
                    if s.judge_passed is not None:
                        difficulty_results.append((float(diff), bool(s.judge_passed)))

        if difficulty_results:
            difficulty_results.sort(key=lambda x: x[0])
            passed_difficulties = [d for d, p in difficulty_results if p]
            failed_difficulties = [d for d, p in difficulty_results if not p]

            features["max_passed_difficulty"] = max(passed_difficulties) if passed_difficulties else 0.0
            features["min_failed_difficulty"] = min(failed_difficulties) if failed_difficulties else 1.0
            features["difficulty_ceiling"] = round(
                (features["max_passed_difficulty"] + features["min_failed_difficulty"]) / 2, 4
            )

            easy = [p for d, p in difficulty_results if d <= 0.4]
            hard = [p for d, p in difficulty_results if d >= 0.7]
            easy_rate = sum(easy) / len(easy) if easy else 1.0
            hard_rate = sum(hard) / len(hard) if hard else 0.0
            features["difficulty_dropoff"] = round(easy_rate - hard_rate, 4)
        else:
            features["max_passed_difficulty"] = 0.0
            features["min_failed_difficulty"] = 1.0
            features["difficulty_ceiling"] = 0.5
            features["difficulty_dropoff"] = 0.0

        # -- Tool use features --
        tool_cases = [r for r in case_results if r.case.category == "tool_use"]
        if tool_cases:
            features["tool_use_pass_rate"] = self._pass_rate(tool_cases)
        else:
            features["tool_use_pass_rate"] = 0.0

        # -- Fingerprint features --
        fp_cases = [r for r in case_results if r.case.category == "fingerprint"]
        if fp_cases:
            formality_scores = []
            for r in fp_cases:
                for s in r.samples:
                    d = s.judge_detail or {}
                    if "formality_score" in d:
                        formality_scores.append(d["formality_score"])
            features["avg_formality_score"] = (
                sum(formality_scores) / len(formality_scores)
            ) if formality_scores else 0.5

        # -- Knowledge hallucination resistance --
        halluc_cases = [r for r in case_results if r.case.judge_method == "hallucination_detect"]
        if halluc_cases:
            features["hallucination_resist_rate"] = self._pass_rate(halluc_cases)

        # -- Extraction resistance rate --
        ext_cases = [r for r in case_results if r.case.category == "extraction"]
        if ext_cases:
            _LEAK_JUDGES = {
                "prompt_leak_detect", "forbidden_word_extract",
                "path_leak_detect", "tool_config_leak_detect", "memory_leak_detect",
            }
            resisted = 0
            total_ext = 0
            for r in ext_cases:
                for s in r.samples:
                    if s.judge_passed is not None:
                        total_ext += 1
                        if r.case.judge_method in _LEAK_JUDGES:
                            if not s.judge_passed:  # no leak = good
                                resisted += 1
                        else:
                            if s.judge_passed:  # resistance = good
                                resisted += 1
            features["extraction_resist_rate"] = resisted / total_ext if total_ext > 0 else 0.5

        # -- v6: Token accounting anomaly detection (6.3) --
        # Detect suspicious token counting patterns that indicate proxy/wrapper
        token_anomalies = self._detect_token_accounting_anomalies(case_results)
        features.update(token_anomalies)

        # -- v6: Response diversity at temperature=0 (6.4) --
        # Real models should be deterministic at temp=0, wrappers may show diversity
        temp_zero_diversity = self._calculate_temp_zero_diversity(case_results)
        features["temp_zero_diversity"] = temp_zero_diversity

        return {k: round(v, 4) for k, v in features.items()}

    @staticmethod
    def _detect_token_accounting_anomalies(case_results: list[CaseResult]) -> dict[str, float]:
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
            zero_completion = sum(1 for t in completion_tokens if t == 0)
            if zero_completion / len(completion_tokens) > 0.8:
                anomalies["zero_completion_anomaly"] = 1.0

        # Check for identical token counts across all samples
        unique_prompts = len(set(prompt_tokens))
        if unique_prompts == 1 and len(prompt_tokens) > 3:
            anomalies["identical_token_anomaly"] = 1.0

        return anomalies

    @staticmethod
    def _calculate_temp_zero_diversity(case_results: list[CaseResult]) -> float:
        """
        v6: Calculate response diversity for temperature=0 cases.
        Real models should be deterministic at temp=0.
        High diversity at temp=0 suggests proxy/routing.
        Returns diversity score (0.0 = deterministic, 1.0 = highly diverse).
        """
        # Find cases with temperature=0 and multiple samples
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
                avg_sim = sum(similarities) / len(similarities)
                # Diversity = 1 - similarity (0 = identical, 1 = completely different)
                diversity_scores.append(1.0 - avg_sim)

        # Return average diversity across all temp=0 cases
        return round(sum(diversity_scores) / len(diversity_scores), 4) if diversity_scores else 0.0

    @staticmethod
    def _pass_rate(results: list[CaseResult]) -> float:
        total = 0
        passed = 0
        for r in results:
            for s in r.samples:
                if s.judge_passed is not None:
                    total += 1
                    if s.judge_passed:
                        passed += 1
        return (passed / total) if total > 0 else 0.0

    @staticmethod
    def _dimension_stats(case_results: list[CaseResult]) -> dict[str, float]:
        by_dim: dict[str, list[CaseResult]] = {}
        for r in case_results:
            dim = (r.case.dimension or r.case.category or "unknown").strip().lower()
            by_dim.setdefault(dim, []).append(r)

        out: dict[str, float] = {}
        for dim, items in by_dim.items():
            out[f"dim_{dim}_pass_rate"] = FeatureExtractor._pass_rate(items)
            out[f"dim_{dim}_coverage"] = len(items) / max(len(case_results), 1)
        return out

    @staticmethod
    def _tag_stats(case_results: list[CaseResult]) -> dict[str, float]:
        tag_to_samples: dict[str, list[bool]] = {}
        for r in case_results:
            tags = r.case.tags or []
            for tag in tags:
                t = str(tag).strip().lower()
                if not t:
                    continue
                tag_to_samples.setdefault(t, [])
                for s in r.samples:
                    if s.judge_passed is not None:
                        tag_to_samples[t].append(bool(s.judge_passed))

        out: dict[str, float] = {}
        for tag, vals in tag_to_samples.items():
            if not vals:
                continue
            out[f"tag_{tag}_pass_rate"] = sum(1 for v in vals if v) / len(vals)
        return out

    @staticmethod
    def _adversarial_spoof_rate(
        adv_cases: list[CaseResult],
        all_cases: list[CaseResult],
    ) -> float:
        """
        Compute spoof signal rate from adversarial reasoning cases.

        Two signal sources:
        1. Anti-pattern hits in adversarial cases (template-matching detected).
        2. Paired-variant cross-check: if a model gives the WRONG answer on a
           variant that flips the expected outcome, it's likely memorising
           rather than reasoning.
        """
        if not adv_cases:
            return 0.0

        # Build lookup for paired cross-checking
        all_by_id = {r.case.id: r for r in all_cases}

        spoof_signals = 0
        total_checks = 0

        for r in adv_cases:
            meta = (r.case.params.get("_meta") or {})
            spoof_cfg = meta.get("spoof_detection") or {}
            paired_id = meta.get("paired_with")

            for s in r.samples:
                total_checks += 1
                d = s.judge_detail or {}

                # Signal 1: anti-pattern hits on the adversarial case itself
                if d.get("anti_pattern_hits"):
                    spoof_signals += 1
                    continue

                # Signal 2: case failed (wrong answer on variant)
                if s.judge_passed is False:
                    # Check if the paired original was passed — if so, the model
                    # "knows" the original answer but can't adapt to the variant,
                    # which is a strong spoof signal.
                    if paired_id and paired_id in all_by_id:
                        paired_result = all_by_id[paired_id]
                        if paired_result.pass_rate >= 0.5:
                            spoof_signals += 1
                            continue

        if total_checks == 0:
            return 0.0
        return spoof_signals / total_checks

    @staticmethod
    def _failure_attribution(case_results: list[CaseResult]) -> dict[str, float]:
        counts = {
            "error_response": 0,
            "format_violation": 0,
            "safety_violation": 0,
            "reasoning_failure": 0,
            "unknown": 0,
        }
        total_fail = 0

        for r in case_results:
            for s in r.samples:
                if s.judge_passed is not False:
                    continue
                total_fail += 1
                if s.response.error_type:
                    counts["error_response"] += 1
                    continue

                jm = (r.case.judge_method or "").lower()
                cat = (r.case.category or "").lower()
                detail = s.judge_detail or {}

                if jm in {"json_schema", "regex_match", "line_count", "exact_match"}:
                    counts["format_violation"] += 1
                elif jm == "refusal_detect" or cat == "refusal":
                    counts["safety_violation"] += 1
                elif cat in {"reasoning", "coding", "consistency"}:
                    counts["reasoning_failure"] += 1
                elif detail.get("schema_errors") or detail.get("found") is False:
                    counts["format_violation"] += 1
                else:
                    counts["unknown"] += 1

        if total_fail == 0:
            return {f"failure_{k}_rate": 0.0 for k in counts}

        return {f"failure_{k}_rate": v / total_fail for k, v in counts.items()}
