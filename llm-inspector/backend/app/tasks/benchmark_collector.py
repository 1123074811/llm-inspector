"""
Benchmark collector — automatically collect real model profiles
by running the full test suite against official APIs.

Usage:
    collector = BenchmarkCollector()
    collector.collect([
        {"name": "gpt-4o", "base_url": "https://api.openai.com/v1", "api_key": "sk-..."},
        {"name": "claude-3-5-sonnet", "base_url": "https://api.anthropic.com/v1", "api_key": "sk-ant-..."},
    ], n_runs=3)
"""
from __future__ import annotations

import json
import pathlib
import time
from datetime import datetime, timezone

from app.core.logging import get_logger
from app.core.schemas import TestCase, CaseResult
from app.adapters.openai_compat import OpenAICompatibleAdapter
from app.runner.case_executor import execute_case
from app.analysis.pipeline import FeatureExtractor
from app.tasks.seeder import load_suite

logger = get_logger(__name__)

_BENCHMARKS_PATH = (
    pathlib.Path(__file__).parent.parent / "fixtures" / "benchmarks" / "default_profiles.json"
)


class BenchmarkCollector:
    """
    Collects real benchmark profiles by running the test suite
    against official model APIs.
    """

    def __init__(self, suite_version: str = "v3"):
        self.suite_version = suite_version
        self.extractor = FeatureExtractor()

    def collect(
        self,
        model_configs: list[dict],
        n_runs: int = 3,
    ) -> list[dict]:
        """
        Run suite against each model config, extract features, save profiles.

        model_configs: [{"name": "gpt-4o", "base_url": "...", "api_key": "..."}]
        n_runs: number of runs to average over for stability.

        Returns list of generated profiles.
        """
        cases = self._load_cases()
        if not cases:
            logger.error("No test cases found", suite_version=self.suite_version)
            return []

        new_profiles = []

        for config in model_configs:
            model_name = config["name"]
            base_url = config["base_url"]
            api_key = config["api_key"]

            logger.info(
                "Collecting benchmark",
                model=model_name, base_url=base_url, n_runs=n_runs,
            )

            try:
                adapter = OpenAICompatibleAdapter(base_url, api_key)
                all_features: list[dict[str, float]] = []

                for run_i in range(n_runs):
                    logger.info("Run start", model=model_name, run=run_i + 1, total=n_runs)
                    case_results: list[CaseResult] = []

                    for case in cases:
                        try:
                            result = execute_case(adapter, model_name, case)
                            case_results.append(result)
                        except Exception as e:
                            logger.warning(
                                "Case execution failed",
                                model=model_name, case_id=case.id, error=str(e),
                            )
                        time.sleep(0.5)  # rate limit courtesy

                    features = self.extractor.extract(case_results)
                    all_features.append(features)
                    logger.info(
                        "Run complete",
                        model=model_name, run=run_i + 1,
                        feature_count=len(features),
                    )

                # Average features across runs
                avg_features = self._average_features(all_features)

                profile = {
                    "name": model_name,
                    "suite_version": self.suite_version,
                    "sample_count": n_runs,
                    "n_runs": n_runs,
                    "data_source": "measured",
                    "collected_at": datetime.now(timezone.utc).isoformat(),
                    "feature_vector": avg_features,
                    "provider": self._infer_provider(model_name),
                    "description": f"{model_name} 真实测量基准 (suite {self.suite_version}, {n_runs} 次运行均值)",
                }
                new_profiles.append(profile)
                logger.info("Profile generated", model=model_name, features=len(avg_features))

            except Exception as e:
                logger.error(
                    "Benchmark collection failed",
                    model=model_name, error=str(e),
                )

        if new_profiles:
            self._merge_and_save(new_profiles)

        return new_profiles

    def _load_cases(self) -> list[TestCase]:
        """Load test cases from suite JSON."""
        raw = load_suite(self.suite_version)
        if not raw:
            return []
        cases = []
        for c in raw:
            cases.append(TestCase(
                id=c["id"],
                category=c.get("category", ""),
                name=c.get("name", ""),
                system_prompt=c.get("system_prompt"),
                user_prompt=c.get("user_prompt", ""),
                expected_type=c.get("expected_type", "any"),
                judge_method=c.get("judge_method", "any_text"),
                max_tokens=c.get("max_tokens", 100),
                n_samples=c.get("n_samples", 1),
                temperature=c.get("temperature", 0.0),
                params=c.get("params", {}),
                weight=c.get("weight", 1.0),
                dimension=c.get("dimension"),
                tags=c.get("tags"),
                difficulty=c.get("difficulty"),
            ))
        return cases

    @staticmethod
    def _average_features(
        feature_list: list[dict[str, float]],
    ) -> dict[str, float]:
        """Average feature vectors across multiple runs."""
        if not feature_list:
            return {}
        if len(feature_list) == 1:
            return feature_list[0]

        all_keys = set()
        for f in feature_list:
            all_keys.update(f.keys())

        avg: dict[str, float] = {}
        for key in all_keys:
            values = [f[key] for f in feature_list if key in f]
            if values:
                avg[key] = round(sum(values) / len(values), 4)
        return avg

    @staticmethod
    def _infer_provider(model_name: str) -> str:
        """Infer provider from model name."""
        name = model_name.lower()
        if "gpt" in name or "o1" in name or "o3" in name:
            return "OpenAI"
        if "claude" in name:
            return "Anthropic"
        if "gemini" in name:
            return "Google"
        if "deepseek" in name:
            return "DeepSeek"
        if "qwen" in name:
            return "Alibaba"
        if "glm" in name:
            return "Zhipu"
        if "llama" in name:
            return "Meta"
        if "mistral" in name or "mixtral" in name:
            return "Mistral AI"
        return "Unknown"

    def _merge_and_save(self, new_profiles: list[dict]) -> None:
        """Merge new measured profiles into existing benchmark file."""
        existing = {"benchmarks": []}
        if _BENCHMARKS_PATH.exists():
            try:
                existing = json.loads(_BENCHMARKS_PATH.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass

        # Replace existing profiles with same name, append new ones
        existing_by_name = {p["name"]: p for p in existing.get("benchmarks", [])}
        for profile in new_profiles:
            existing_by_name[profile["name"]] = profile

        existing["benchmarks"] = list(existing_by_name.values())

        _BENCHMARKS_PATH.write_text(
            json.dumps(existing, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "Benchmark profiles saved",
            path=str(_BENCHMARKS_PATH),
            total=len(existing["benchmarks"]),
        )
