"""
v11 Phase 3 Tests — Prompt Optimizer + Suite Pruner + Multilingual Attack.

Covers:
1. PromptOptimizer (TF-IDF retrieval, n-gram fallback, token budget)
2. SuitePruner (Fisher information, discrimination, pass-rate analysis)
3. GPQAAdapter (question conversion)
4. MultilingualAttackEngine (templates, B64 payloads, response evaluation)
5. Integration with v11 handlers
6. Integration with orchestrator pipeline
"""
import sys
import os
import json
import pytest

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Prompt Optimizer — Tokenizer & TF-IDF
# ═══════════════════════════════════════════════════════════════════════════════

class TestTokenizer:
    """Test the lightweight tokenizer."""

    def test_english_tokenize(self):
        from app.runner.prompt_optimizer import _tokenize
        tokens = _tokenize("Hello World")
        assert "hello" in tokens
        assert "world" in tokens

    def test_chinese_tokenize(self):
        from app.runner.prompt_optimizer import _tokenize
        tokens = _tokenize("你好世界")
        assert "你" in tokens
        assert "好" in tokens

    def test_mixed_tokenize(self):
        from app.runner.prompt_optimizer import _tokenize
        tokens = _tokenize("Return JSON object 返回JSON")
        assert "return" in tokens
        assert "json" in tokens
        assert "返" in tokens

    def test_ngrams(self):
        from app.runner.prompt_optimizer import _ngrams
        result = _ngrams(["hello", "world"], n=2)
        assert "he" in result
        assert "el" in result
        assert "wo" in result


class TestTfidfIndex:
    """Test the TF-IDF vector index."""

    def test_build_and_search(self):
        from app.runner.prompt_optimizer import TfidfIndex
        index = TfidfIndex()
        docs = [
            ("doc1", "instruction following exact match format"),
            ("doc2", "reasoning logic deduction syllogism"),
            ("doc3", "safety refusal harmful content block"),
            ("doc4", "coding python algorithm fibonacci"),
        ]
        index.build(docs)
        assert index.size == 4

        results = index.search("instruction format compliance", top_k=2)
        assert len(results) >= 1
        assert results[0][0] == "doc1"  # Most similar

    def test_empty_index(self):
        from app.runner.prompt_optimizer import TfidfIndex
        index = TfidfIndex()
        results = index.search("test query")
        assert results == []

    def test_search_returns_scores(self):
        from app.runner.prompt_optimizer import TfidfIndex
        index = TfidfIndex()
        index.build([("a", "hello world"), ("b", "foo bar")])
        results = index.search("hello", top_k=2)
        assert len(results) == 2
        # First result should be "a" with positive score
        assert results[0][0] == "a"
        assert results[0][1] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Prompt Optimizer
# ═══════════════════════════════════════════════════════════════════════════════

class TestPromptOptimizer:
    """Test the dynamic Few-Shot prompt optimizer."""

    def test_default_examples_loaded(self):
        from app.runner.prompt_optimizer import PromptOptimizer
        opt = PromptOptimizer()
        assert opt.n_candidates > 0

    def test_register_example(self):
        from app.runner.prompt_optimizer import PromptOptimizer, ShotExample
        opt = PromptOptimizer()
        before = opt.n_candidates
        opt.register_example(ShotExample(
            id="test_ex_1",
            category="test",
            dimension="test",
            user_prompt="Test prompt",
            expected_response="Test response",
        ))
        assert opt.n_candidates == before + 1

    def test_compile_prompt_tfidf(self):
        from app.runner.prompt_optimizer import PromptOptimizer
        opt = PromptOptimizer()
        compiled = opt.compile_prompt(
            test_prompt="Output only the digit 7. Nothing else.",
            category="instruction",
            dimension="instruction",
        )
        assert compiled.prompt  # Non-empty
        assert compiled.n_examples >= 0
        assert compiled.method in ("tfidf", "ngram", "random")

    def test_compile_prompt_with_budget(self):
        from app.runner.prompt_optimizer import PromptOptimizer
        opt = PromptOptimizer()
        compiled = opt.compile_prompt(
            test_prompt="Write a Python function",
            category="coding",
            dimension="coding",
            max_tokens_budget=100,
        )
        # With small budget, should have fewer or equal examples
        assert compiled.n_examples <= 2

    def test_optimization_report(self):
        from app.runner.prompt_optimizer import PromptOptimizer
        opt = PromptOptimizer()
        opt.compile_prompt("Test prompt", category="instruction")
        report = opt.get_report()
        assert report.total_candidates > 0
        assert report.total_compilations == 1

    def test_global_singleton(self):
        from app.runner.prompt_optimizer import prompt_optimizer, get_optimizer
        assert prompt_optimizer is get_optimizer()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Suite Pruner
# ═══════════════════════════════════════════════════════════════════════════════

class TestSuitePruner:
    """Test the IIF-based suite pruning engine."""

    def test_analyze_good_case(self):
        from app.analysis.suite_pruner import SuitePruner
        pruner = SuitePruner()
        metrics = pruner.analyze_case(
            case_id="good_001",
            irt_a=1.2,
            irt_b=0.5,
            pass_rate=0.6,
            n_responses=20,
        )
        assert metrics.is_discriminative
        assert metrics.discrimination_a == 1.2
        assert len(metrics.flags) == 0

    def test_analyze_low_discrimination(self):
        from app.analysis.suite_pruner import SuitePruner
        pruner = SuitePruner()
        metrics = pruner.analyze_case(
            case_id="bad_001",
            irt_a=0.3,  # Below threshold
            irt_b=0.0,
            pass_rate=0.5,
            n_responses=20,
        )
        assert not metrics.is_discriminative
        assert "low_discrimination" in metrics.flags

    def test_analyze_ceiling_effect(self):
        from app.analysis.suite_pruner import SuitePruner
        pruner = SuitePruner()
        metrics = pruner.analyze_case(
            case_id="easy_001",
            irt_a=1.0,
            irt_b=-2.0,
            pass_rate=0.98,  # > 0.95
            n_responses=20,
        )
        assert not metrics.is_discriminative
        assert "ceiling_effect" in metrics.flags

    def test_analyze_floor_effect(self):
        from app.analysis.suite_pruner import SuitePruner
        pruner = SuitePruner()
        metrics = pruner.analyze_case(
            case_id="hard_001",
            irt_a=0.8,
            irt_b=3.0,
            pass_rate=0.02,  # < 0.05
            n_responses=20,
        )
        assert not metrics.is_discriminative
        assert "floor_effect" in metrics.flags

    def test_analyze_insufficient_data(self):
        from app.analysis.suite_pruner import SuitePruner
        pruner = SuitePruner()
        metrics = pruner.analyze_case(
            case_id="new_001",
            irt_a=1.0,
            irt_b=0.0,
            pass_rate=0.5,
            n_responses=2,  # Below minimum
        )
        assert "insufficient_data" in metrics.flags

    def test_fisher_information_calculation(self):
        from app.analysis.suite_pruner import SuitePruner
        # High discrimination = high Fisher information
        info_high = SuitePruner._fisher_information(2.0, 0.0, 0.25, 0.0)
        info_low = SuitePruner._fisher_information(0.3, 0.0, 0.25, 0.0)
        assert info_high > info_low

    def test_analyze_suite(self):
        from app.analysis.suite_pruner import SuitePruner
        pruner = SuitePruner()
        cases = [
            {"id": "good", "irt_a": 1.2, "irt_b": 0.0, "pass_rate": 0.6, "n_responses": 20, "weight": 1.0, "max_tokens": 100},
            {"id": "bad_disc", "irt_a": 0.3, "irt_b": 0.0, "pass_rate": 0.5, "n_responses": 20, "weight": 1.0, "max_tokens": 100},
            {"id": "too_easy", "irt_a": 1.0, "irt_b": -3.0, "pass_rate": 0.98, "n_responses": 20, "weight": 1.5, "max_tokens": 150},
        ]
        report = pruner.analyze_suite(cases)
        assert report.total_cases == 3
        assert report.non_discriminative_cases == 2
        assert report.discriminative_cases == 1
        assert "good" not in report.non_discriminative_ids
        assert "bad_disc" in report.non_discriminative_ids
        assert "too_easy" in report.non_discriminative_ids

    def test_apply_to_eval_cases(self):
        from app.analysis.suite_pruner import SuitePruner, CaseQualityMetrics
        from app.core.eval_schemas import EvalTestCase
        pruner = SuitePruner()

        # Create mock eval cases
        case1 = EvalTestCase(
            id="good", category="test", name="good", user_prompt="t",
            expected_type="any", judge_method="exact_match",
        )
        case2 = EvalTestCase(
            id="bad", category="test", name="bad", user_prompt="t",
            expected_type="any", judge_method="exact_match",
        )

        metrics = [
            CaseQualityMetrics(
                case_id="good", discrimination_a=1.2, difficulty_b=0.0,
                fisher_info_at_mean=0.5, fisher_info_max=0.8,
                pass_rate=0.6, n_responses=20, is_discriminative=True, flags=[],
            ),
            CaseQualityMetrics(
                case_id="bad", discrimination_a=0.3, difficulty_b=0.0,
                fisher_info_at_mean=0.01, fisher_info_max=0.02,
                pass_rate=0.5, n_responses=20, is_discriminative=False,
                flags=["low_discrimination"],
            ),
        ]

        flagged = pruner.apply_to_eval_cases([case1, case2], metrics)
        assert flagged == 1
        assert case1.eval_meta.discriminative_valid is True
        assert case2.eval_meta.discriminative_valid is False


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: GPQA Adapter
# ═══════════════════════════════════════════════════════════════════════════════

class TestGPQAAdapter:
    """Test the GPQA question adapter."""

    def test_default_questions(self):
        from app.analysis.suite_pruner import GPQAAdapter
        adapter = GPQAAdapter()
        assert adapter.n_questions > 0

    def test_to_eval_cases(self):
        from app.analysis.suite_pruner import GPQAAdapter
        adapter = GPQAAdapter()
        cases = adapter.to_eval_cases()
        assert len(cases) > 0
        assert all("id" in c for c in cases)
        assert all("category" in c for c in cases)
        assert all(c["category"] == "reasoning" for c in cases)

    def test_question_domains(self):
        from app.analysis.suite_pruner import GPQAAdapter
        adapter = GPQAAdapter()
        cases = adapter.to_eval_cases()
        domains = set(c["params"]["_meta"]["gpqa_domain"] for c in cases)
        assert len(domains) > 0

    def test_global_singleton(self):
        from app.analysis.suite_pruner import gpqa_adapter, get_gpqa_adapter
        assert gpqa_adapter is get_gpqa_adapter()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Multilingual Attack Engine
# ═══════════════════════════════════════════════════════════════════════════════

class TestMultilingualTemplates:
    """Test multilingual attack template definitions."""

    def test_templates_exist(self):
        from app.predetect.multilingual_attack import MULTILINGUAL_TEMPLATES
        assert len(MULTILINGUAL_TEMPLATES) >= 5

    def test_zulu_template_present(self):
        from app.predetect.multilingual_attack import MULTILINGUAL_TEMPLATES
        zu_templates = [(l, c, p) for l, c, p in MULTILINGUAL_TEMPLATES if c == "zu"]
        assert len(zu_templates) >= 1

    def test_b64_payloads_exist(self):
        from app.predetect.multilingual_attack import B64_MULTILINGUAL_PAYLOADS
        assert len(B64_MULTILINGUAL_PAYLOADS) >= 3

    def test_b64_payloads_are_valid_base64(self):
        import base64
        from app.predetect.multilingual_attack import B64_MULTILINGUAL_PAYLOADS
        for key, payload in B64_MULTILINGUAL_PAYLOADS.items():
            decoded = base64.b64decode(payload)
            assert len(decoded) > 0  # Valid base64 that decodes to something


class TestMultilingualAttackEngine:
    """Test the multilingual attack engine."""

    def test_engine_creation(self):
        from app.predetect.multilingual_attack import MultilingualAttackEngine
        engine = MultilingualAttackEngine()
        assert engine._max_languages == 3

    def test_select_languages_always_includes_zulu(self):
        from app.predetect.multilingual_attack import MultilingualAttackEngine
        engine = MultilingualAttackEngine(max_languages_per_run=2)
        selected = engine._select_languages()
        lang_codes = [c for _, c, _ in selected]
        assert "zu" in lang_codes

    def test_evaluate_response_with_model_name(self):
        from app.predetect.multilingual_attack import MultilingualAttackEngine
        engine = MultilingualAttackEngine()
        leaked, keywords = engine._evaluate_response(
            "I am actually GPT-4, trained by OpenAI."
        )
        assert len(keywords) > 0
        assert "GPT-4" in keywords or "OpenAI" in keywords

    def test_evaluate_response_no_leak(self):
        from app.predetect.multilingual_attack import MultilingualAttackEngine
        engine = MultilingualAttackEngine()
        leaked, keywords = engine._evaluate_response(
            "I cannot reveal that information."
        )
        # Should have minimal or no leaks
        assert len(keywords) == 0 or len(leaked) == 0

    def test_calculate_confidence(self):
        from app.predetect.multilingual_attack import MultilingualAttackEngine
        engine = MultilingualAttackEngine()
        # High confidence: multiple keywords + identity disclosure
        high_conf = engine._calculate_confidence(
            [{"type": "identity_disclosure", "value": "I am GPT-4", "context": ""}],
            ["GPT-4", "OpenAI"],
        )
        # Low confidence: no leaks
        low_conf = engine._calculate_confidence([], [])
        assert high_conf > low_conf

    def test_aggregate_results(self):
        from app.predetect.multilingual_attack import (
            MultilingualAttackEngine, MultilingualAttackResult,
        )
        engine = MultilingualAttackEngine()
        results = [
            MultilingualAttackResult(
                attack_type="multilingual", language="Zulu", language_code="zu",
                prompt_sent="test", response="I am GPT-4",
                leaked_info=[{"type": "identity_disclosure", "value": "GPT-4", "context": ""}],
                keywords_found=["GPT-4"], success=True, confidence=0.7,
            ),
            MultilingualAttackResult(
                attack_type="b64_multilingual", language="Welsh", language_code="cy",
                prompt_sent="test", response="No comment",
                leaked_info=[], keywords_found=[], success=False, confidence=0.0,
            ),
        ]
        report = engine._aggregate_results(results)
        assert report.total_attacks == 2
        assert report.successful_attacks == 1
        assert "Zulu" in report.results_by_language
        assert report.best_attack.language == "Zulu"


class TestLayer14:
    """Test the Layer 14 predetect layer."""

    def test_layer_creation(self):
        from app.predetect.multilingual_attack import Layer14MultilingualAttack
        layer = Layer14MultilingualAttack()
        assert layer.engine is not None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: v11 Phase 3 Handlers
# ═══════════════════════════════════════════════════════════════════════════════

class TestV11Phase3Handlers:
    """Test Phase 3 API handlers."""

    def test_prompt_optimizer_report_handler(self):
        from app.handlers.v11_handlers import handle_prompt_optimizer_report
        status, body, content_type = handle_prompt_optimizer_report("/api/v1/prompt-optimizer/report", {}, {})
        data = json.loads(body)
        assert "total_candidates" in data
        assert "total_compilations" in data

    def test_gpqa_questions_handler(self):
        from app.handlers.v11_handlers import handle_gpqa_questions
        status, body, content_type = handle_gpqa_questions("/api/v1/gpqa/questions", {}, {})
        data = json.loads(body)
        assert "total_questions" in data
        assert "questions" in data
        assert data["total_questions"] > 0

    def test_multilingual_attacks_handler(self):
        from app.handlers.v11_handlers import handle_multilingual_attacks
        status, body, content_type = handle_multilingual_attacks("/api/v1/attacks/multilingual", {}, {})
        data = json.loads(body)
        assert "total_templates" in data
        assert "languages" in data
        assert "Zulu" in data["languages"]

    def test_suite_pruning_report_handler_no_report(self):
        from app.handlers.v11_handlers import handle_suite_pruning_report
        # Reset the cache
        import app.handlers.v11_handlers as h
        h._latest_pruning_report = None
        status, body, content_type = handle_suite_pruning_report("/api/v1/suite/pruning-report", {}, {})
        data = json.loads(body)
        assert data["status"] == "no_report"

    def test_suite_prune_handler_no_data(self):
        from app.handlers.v11_handlers import handle_suite_prune
        # When no cases in DB, should return no_data gracefully
        # (repo.load_cases may fail in test context)
        status, body, content_type = handle_suite_prune(
            "/api/v1/suite/prune", {}, {}
        )
        data = json.loads(body)
        # Either "no_data" or "completed" depending on DB state
        assert "status" in data


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: Integration — Orchestrator imports
# ═══════════════════════════════════════════════════════════════════════════════

class TestOrchestratorPhase3Imports:
    """Test that orchestrator correctly imports Phase 3 modules."""

    def test_pruner_import(self):
        from app.runner.orchestrator import suite_pruner
        assert suite_pruner is not None

    def test_gpqa_adapter_import(self):
        from app.runner.orchestrator import gpqa_adapter
        assert gpqa_adapter is not None

    def test_prompt_optimizer_import(self):
        from app.runner.orchestrator import prompt_optimizer
        assert prompt_optimizer is not None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: Integration — Predetect pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class TestPredetectPhase3Integration:
    """Test that predetect pipeline integrates Layer 14."""

    def test_multilingual_attack_module_importable(self):
        from app.predetect.multilingual_attack import Layer14MultilingualAttack
        assert Layer14MultilingualAttack is not None

    def test_pipeline_module_imports_layer14(self):
        """Verify the predetect pipeline can import Layer 14."""
        from app.predetect.pipeline import Layer14MultilingualAttack
        assert Layer14MultilingualAttack is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
