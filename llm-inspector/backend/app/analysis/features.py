"""Feature extraction from case results."""
from __future__ import annotations

import math
from app.core.schemas import CaseResult


class FeatureExtractor:
    """Extracts a flat dict of named numeric features from all case results."""

    def extract(self, case_results: list[CaseResult]) -> dict[str, float]:
        features: dict[str, float] = {}
        dim_stats = self._dimension_stats(case_results)
        tag_stats = self._tag_stats(case_results)
        failure_stats = self._failure_attribution(case_results)

        proto_cases = [r for r in case_results if r.case.category == "protocol"]
        if proto_cases:
            features["protocol_success_rate"] = self._pass_rate(proto_cases)
            usage_cases = [r for r in proto_cases if any(s.response.usage_total_tokens for s in r.samples)]
            features["has_usage_fields"] = 1.0 if usage_cases else 0.0
            finish_cases = [r for r in proto_cases if any(s.response.finish_reason for s in r.samples)]
            features["has_finish_reason"] = 1.0 if finish_cases else 0.0
        else:
            features["protocol_success_rate"] = 0.0
            features["has_usage_fields"] = 0.0
            features["has_finish_reason"] = 0.0

        reason_cases = [r for r in case_results if r.case.category == "reasoning"]
        features["reasoning_pass_rate"] = self._pass_rate(reason_cases)
        features["reasoning_fail_rate"] = 1.0 - features["reasoning_pass_rate"]
        features["reasoning_avg_latency"] = self._avg_latency(reason_cases)
        features["reasoning_consistency"] = self._consistency_score(reason_cases)
        if dim_stats.get("reasoning"):
            features["reasoning_avg_confidence"] = dim_stats["reasoning"]["avg_confidence"]
            features["reasoning_avg_latency"] = dim_stats["reasoning"]["avg_latency"]
        else:
            features["reasoning_avg_confidence"] = 0.0
            features["reasoning_avg_latency"] = 0.0

        instr_cases = [r for r in case_results if r.case.category == "instruction"]
        features["instruction_pass_rate"] = self._pass_rate(instr_cases)
        features["instruction_fail_rate"] = 1.0 - features["instruction_pass_rate"]
        features["instruction_avg_latency"] = self._avg_latency(instr_cases)
        features["instruction_consistency"] = self._consistency_score(instr_cases)
        if dim_stats.get("instruction"):
            features["instruction_avg_confidence"] = dim_stats["instruction"]["avg_confidence"]
        else:
            features["instruction_avg_confidence"] = 0.0

        safety_cases = [r for r in case_results if r.case.category == "safety"]
        features["safety_pass_rate"] = self._pass_rate(safety_cases)
        features["safety_fail_rate"] = 1.0 - features["safety_pass_rate"]
        features["safety_avg_latency"] = self._avg_latency(safety_cases)
        if dim_stats.get("safety"):
            features["safety_avg_confidence"] = dim_stats["safety"]["avg_confidence"]
        else:
            features["safety_avg_confidence"] = 0.0

        code_cases = [r for r in case_results if r.case.category == "coding"]
        features["coding_pass_rate"] = self._pass_rate(code_cases)
        features["coding_fail_rate"] = 1.0 - features["coding_pass_rate"]
        features["coding_avg_latency"] = self._avg_latency(code_cases)
        features["coding_consistency"] = self._consistency_score(code_cases)
        if dim_stats.get("coding"):
            features["coding_avg_confidence"] = dim_stats["coding"]["avg_confidence"]
        else:
            features["coding_avg_confidence"] = 0.0

        extraction_cases = [r for r in case_results if r.case.category == "extraction"]
        features["extraction_pass_rate"] = self._pass_rate(extraction_cases)
        if dim_stats.get("extraction"):
            features["extraction_avg_confidence"] = dim_stats["extraction"]["avg_confidence"]
        else:
            features["extraction_avg_confidence"] = 0.0

        features["total_cases"] = float(len(case_results))
        features["total_passed"] = float(sum(1 for r in case_results if all(s.judge_passed for s in r.samples)))
        features["total_failed"] = float(sum(1 for r in case_results if any(not s.judge_passed for s in r.samples)))
        features["overall_pass_rate"] = features["total_passed"] / max(features["total_cases"], 1.0)

        tag_pass = tag_stats.get("pass", {})
        tag_fail = tag_stats.get("fail", {})
        all_tags = set(tag_pass) | set(tag_fail)
        for tag in all_tags:
            n_pass = tag_pass.get(tag, 0)
            n_fail = tag_fail.get(tag, 0)
            n_total = n_pass + n_fail
            features[f"tag_{tag}_pass_rate"] = n_pass / max(n_total, 1)
            features[f"tag_{tag}_count"] = float(n_total)

        for dim in ("reasoning", "instruction", "safety", "coding", "extraction"):
            if dim_stats.get(dim):
                features[f"{dim}_case_count"] = float(dim_stats[dim]["count"])
            else:
                features[f"{dim}_case_count"] = 0.0

        all_latencies = [s.response.latency_ms for r in case_results for s in r.samples if s.response.latency_ms]
        if all_latencies:
            sorted_lat = sorted(all_latencies)
            features["p50_latency"] = sorted_lat[len(sorted_lat) // 2]
            features["p95_latency"] = sorted_lat[int(len(sorted_lat) * 0.95)]
            features["p99_latency"] = sorted_lat[int(len(sorted_lat) * 0.99)]
            features["avg_latency"] = sum(all_latencies) / len(all_latencies)
        else:
            features["p50_latency"] = 0.0
            features["p95_latency"] = 0.0
            features["p99_latency"] = 0.0
            features["avg_latency"] = 0.0

        features["total_token_usage"] = sum(
            s.response.usage_total_tokens or 0
            for r in case_results for s in r.samples
        )
        features["avg_token_per_response"] = features["total_token_usage"] / max(features["total_cases"], 1.0)

        all_conf = [s.judge_confidence for r in case_results for s in r.samples if s.judge_confidence is not None]
        features["avg_judge_confidence"] = sum(all_conf) / max(len(all_conf), 1) if all_conf else 0.0

        for threshold in (0.5, 0.7, 0.85, 0.95):
            features[f"cases_above_{threshold}_confidence"] = float(sum(1 for c in all_conf if c >= threshold))

        error_types: dict[str, int] = {}
        for r in case_results:
            for s in r.samples:
                if not s.judge_passed and s.error_type:
                    error_types[s.error_type] = error_types.get(s.error_type, 0) + 1
        for et, count in error_types.items():
            features[f"error_{et}"] = float(count)
        features["total_errors"] = float(sum(error_types.values()))

        for dim, stats in dim_stats.items():
            features[f"{dim}_error_rate"] = 1.0 - stats["pass_rate"]
        features["overall_error_rate"] = 1.0 - features["overall_pass_rate"]

        high_entropy_cases = [r for r in case_results if self._response_entropy(r) > 1.5]
        features["high_entropy_count"] = float(len(high_entropy_cases))
        features["high_entropy_ratio"] = features["high_entropy_count"] / max(features["total_cases"], 1.0)

        if failure_stats:
            features["failure_api_errors"] = float(failure_stats.get("api_error", 0))
            features["failure_timeout"] = float(failure_stats.get("timeout", 0))
            features["failure_parse"] = float(failure_stats.get("parse_error", 0))
            features["failure_refuse"] = float(failure_stats.get("content_filter", 0))
        else:
            features["failure_api_errors"] = 0.0
            features["failure_timeout"] = 0.0
            features["failure_parse"] = 0.0
            features["failure_refuse"] = 0.0

        return features

    def _dimension_stats(self, case_results: list[CaseResult]) -> dict:
        stats: dict = {}
        for r in case_results:
            dim = r.case.dimension
            if dim not in stats:
                stats[dim] = {"count": 0, "passed": 0, "total_confidence": 0.0, "total_latency": 0.0}
            stats[dim]["count"] += 1
            if all(s.judge_passed for s in r.samples):
                stats[dim]["passed"] += 1
            confs = [s.judge_confidence for s in r.samples if s.judge_confidence is not None]
            if confs:
                stats[dim]["total_confidence"] += sum(confs) / len(confs)
            latencies = [s.response.latency_ms for s in r.samples if s.response.latency_ms]
            if latencies:
                stats[dim]["total_latency"] += sum(latencies) / len(latencies)

        for dim, s in stats.items():
            s["pass_rate"] = s["passed"] / max(s["count"], 1)
            s["avg_confidence"] = s["total_confidence"] / max(s["count"], 1)
            s["avg_latency"] = s["total_latency"] / max(s["count"], 1)
        return stats

    def _tag_stats(self, case_results: list[CaseResult]) -> dict:
        tag_stats: dict = {"pass": {}, "fail": {}}
        for r in case_results:
            passed = all(s.judge_passed for s in r.samples)
            for tag in r.case.tags:
                bucket = "pass" if passed else "fail"
                tag_stats[bucket][tag] = tag_stats[bucket].get(tag, 0) + 1
        return tag_stats

    def _failure_attribution(self, case_results: list[CaseResult]) -> dict:
        attrs: dict = {"api_error": 0, "timeout": 0, "parse_error": 0, "content_filter": 0}
        for r in case_results:
            if all(s.judge_passed for s in r.samples):
                continue
            for s in r.samples:
                if not s.judge_passed:
                    et = s.error_type or "unknown"
                    if "timeout" in et.lower():
                        attrs["timeout"] += 1
                    elif "过滤" in et or "filter" in et.lower() or "refuse" in et.lower():
                        attrs["content_filter"] += 1
                    elif "json" in et.lower() or "parse" in et.lower():
                        attrs["parse_error"] += 1
                    else:
                        attrs["api_error"] += 1
        return attrs

    def _pass_rate(self, cases: list[CaseResult]) -> float:
        if not cases:
            return 0.0
        return sum(1 for r in cases if all(s.judge_passed for s in r.samples)) / len(cases)

    def _avg_latency(self, cases: list[CaseResult]) -> float:
        latencies = [s.response.latency_ms for r in cases for s in r.samples if s.response.latency_ms]
        return sum(latencies) / len(latencies) if latencies else 0.0

    def _consistency_score(self, cases: list[CaseResult]) -> float:
        if not cases:
            return 0.0
        scores = []
        for r in cases:
            judgments = [s.judge_passed for s in r.samples]
            if len(judgments) > 1:
                scores.append(float(sum(judgments) / len(judgments)))
            else:
                scores.append(1.0)
        return sum(scores) / len(scores)

    def _response_entropy(self, case_result: CaseResult) -> float:
        judgments = [s.judge_passed for s in case_result.samples]
        if len(judgments) <= 1:
            return 0.0
        p = sum(judgments) / len(judgments)
        p = max(min(p, 0.999), 0.001)
        import math
        return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
