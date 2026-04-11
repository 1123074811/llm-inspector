"""
Test Suite for v8.0 Phase 4 — Architecture Optimization

Tests:
- Plugin interface compliance
- Plugin manager functionality
- Structured logging
- Transparent judgment

Reference: V8_IMPLEMENTATION_GUIDE.md Section 6
"""
import pytest
from typing import Dict, Any

from app.judge.plugin_interface import (
    JudgePlugin, JudgeResult, JudgeMetadata, JudgeTier, TieredJudgePlugin
)
from app.judge.plugin_manager import PluginManager, get_plugin_manager, PluginStats
from app.judge.builtin_plugins import (
    ExactMatchPlugin, RegexMatchPlugin, JSONSchemaPlugin,
    ConstraintReasoningPlugin, RefusalDetectPlugin, LineCountPlugin,
    BUILTIN_PLUGINS, register_builtin_plugins
)
from app.judge.transparent_judge import (
    TransparentJudgeWrapper, JudgmentLogger, create_transparent_judge,
    judge_with_transparency
)
from app.core.structured_logger import (
    StructuredLogger, LogLevel, LogEventType, get_structured_logger
)


class TestJudgePluginInterface:
    """Test the plugin interface contracts."""
    
    def test_judge_result_structure(self):
        """JudgeResult must have required fields."""
        result = JudgeResult(
            passed=True,
            detail={"test": "data"},
            confidence=0.95,
            tokens_used=100,
            latency_ms=50,
            method="test"
        )
        
        assert result.passed is True
        assert result.detail["test"] == "data"
        assert result.confidence == 0.95
        assert result.tokens_used == 100
        assert result.latency_ms == 50
        assert result.method == "test"
    
    def test_judge_result_to_dict(self):
        """JudgeResult must serialize correctly."""
        result = JudgeResult(
            passed=True,
            detail={"key": "value"},
            confidence=0.9,
            method="test_method",
            version="1.0",
            threshold_source="irt_calibration",
            threshold_value=0.5
        )
        
        d = result.to_dict()
        assert d["passed"] is True
        assert d["confidence"] == 0.9
        assert d["threshold_source"] == "irt_calibration"
        assert d["threshold_value"] == 0.5
    
    def test_judge_metadata_structure(self):
        """JudgeMetadata must capture plugin info."""
        meta = JudgeMetadata(
            name="test_judge",
            version="1.0.0",
            tier=JudgeTier.LOCAL,
            supported_languages=["en", "zh"],
            description="Test judge method",
            deterministic=True,
            required_params=["param1"],
            optional_params=["param2"]
        )
        
        assert meta.name == "test_judge"
        assert meta.deterministic is True
        assert "param1" in meta.required_params
    
    def test_exact_match_plugin_compliance(self):
        """ExactMatchPlugin must implement interface correctly."""
        plugin = ExactMatchPlugin()
        
        # Check metadata
        meta = plugin.metadata
        assert meta.name == "exact_match"
        assert meta.tier == JudgeTier.LOCAL
        assert meta.deterministic is True
        
        # Test judgment
        result = plugin.judge("hello", {"target": "hello"})
        assert result.passed is True
        
        result = plugin.judge("world", {"target": "hello"})
        assert result.passed is False
    
    def test_regex_match_plugin(self):
        """RegexMatchPlugin must match patterns."""
        plugin = RegexMatchPlugin()
        
        result = plugin.judge("test123", {"pattern": r"\d+"})
        assert result.passed is True
        assert result.detail["found"] is True
        
        result = plugin.judge("test", {"pattern": r"\d+"})
        assert result.passed is False
    
    def test_json_schema_plugin(self):
        """JSONSchemaPlugin must validate JSON structure."""
        plugin = JSONSchemaPlugin()
        
        # Valid JSON
        result = plugin.judge('{"name": "test"}', {"schema": {"type": "object"}})
        assert result.passed is True
        
        # Invalid JSON
        result = plugin.judge('not json', {})
        assert result.passed is False
        assert "error" in result.detail
    
    def test_constraint_reasoning_threshold_tracking(self):
        """ConstraintReasoningPlugin must track threshold source."""
        plugin = ConstraintReasoningPlugin()
        
        response = "The answer is 42"
        params = {
            "target_pattern": r"42",
            "key_constraints": ["answer"],
            "coverage_threshold": 0.5,
            "threshold_source": "irt_calibration_v2026q1",
            "case_id": "test_case_001"
        }
        
        result = plugin.judge(response, params)
        
        # Check threshold tracking
        assert result.threshold_value == 0.5
        assert result.threshold_source == "irt_calibration_v2026q1"
        assert "keyword_coverage" in result.detail


class TestPluginManager:
    """Test the plugin manager functionality."""
    
    def test_singleton_pattern(self):
        """PluginManager must be a singleton."""
        manager1 = PluginManager()
        manager2 = PluginManager()
        assert manager1 is manager2
    
    def test_register_builtin_plugins(self):
        """All built-in plugins must register successfully."""
        manager = PluginManager()
        
        # Clear any existing state
        for name in list(manager.list_plugins()):
            manager.disable_plugin(name)
        
        # Register builtins
        register_builtin_plugins(manager)
        
        # Check registration
        plugins = manager.list_plugins()
        assert "exact_match" in plugins
        assert "regex_match" in plugins
        assert "json_schema" in plugins
        assert "constraint_reasoning" in plugins
        assert "refusal_detect" in plugins
        assert "line_count" in plugins
    
    def test_judge_execution(self):
        """Plugin manager must execute judgments."""
        manager = PluginManager()
        register_builtin_plugins(manager)
        
        result = manager.judge("exact_match", "hello", {"target": "hello"})
        assert result.passed is True
        
        result = manager.judge("exact_match", "world", {"target": "hello"})
        assert result.passed is False
    
    def test_unknown_method_handling(self):
        """Unknown methods must return error result."""
        manager = get_plugin_manager()
        
        result = manager.judge("nonexistent", "test", {})
        assert result.passed is None
        assert "error" in result.detail
    
    def test_plugin_metadata_access(self):
        """Manager must provide plugin metadata."""
        manager = get_plugin_manager()
        register_builtin_plugins(manager)
        
        meta = manager.get_metadata("exact_match")
        assert meta is not None
        assert meta.name == "exact_match"
        assert meta.deterministic is True
    
    def test_plugin_statistics(self):
        """Manager must track plugin statistics."""
        manager = PluginManager()
        register_builtin_plugins(manager)
        
        # Execute some judgments
        for _ in range(3):
            manager.judge("exact_match", "hello", {"target": "hello"})
        
        stats = manager.get_stats("exact_match")
        assert stats is not None
        assert stats.call_count >= 3


class TestStructuredLogger:
    """Test structured logging functionality."""
    
    def test_singleton_pattern(self):
        """StructuredLogger must be a singleton."""
        logger1 = StructuredLogger()
        logger2 = StructuredLogger()
        assert logger1 is logger2
    
    def test_log_entry_structure(self):
        """Log entries must have required fields."""
        logger = StructuredLogger()
        logger.clear()
        
        logger.info(
            LogEventType.JUDGE_START,
            "test",
            "Test message",
            test_data="value"
        )
        
        entries = logger.get_recent(1)
        assert len(entries) == 1
        
        entry = entries[0]
        assert entry.event_type == LogEventType.JUDGE_START.value
        assert entry.component == "test"
        assert entry.level == LogLevel.INFO.value
        assert "timestamp" in entry.to_dict()
    
    def test_log_query_by_event_type(self):
        """Must be able to query logs by event type."""
        logger = StructuredLogger()
        logger.clear()
        
        logger.info(LogEventType.JUDGE_START, "judge", "Start")
        logger.info(LogEventType.JUDGE_COMPLETE, "judge", "Complete")
        logger.info(LogEventType.TEST_START, "test", "Test start")
        
        judge_entries = logger.query(event_type=LogEventType.JUDGE_START)
        assert len(judge_entries) == 1
        
        complete_entries = logger.query(event_type=LogEventType.JUDGE_COMPLETE)
        assert len(complete_entries) == 1
    
    def test_log_judge_step(self):
        """Must log detailed judgment steps."""
        logger = StructuredLogger()
        logger.clear()
        
        logger.log_judge_step(
            case_id="case_001",
            step="keyword_coverage",
            input_data={"constraints": ["a", "b", "c"]},
            output_data={"coverage": 0.67, "matched": ["a", "b"]},
            threshold=0.50,
            threshold_source="irt_calibration_v2026q1"
        )
        
        entries = logger.query(event_type=LogEventType.JUDGE_STEP)
        assert len(entries) == 1
        
        entry = entries[0]
        assert entry.data["threshold"] == 0.50
        assert entry.data["threshold_source"] == "irt_calibration_v2026q1"
    
    def test_threshold_application_logging(self):
        """Must log threshold applications with provenance."""
        logger = StructuredLogger()
        logger.clear()
        
        logger.log_threshold_apply(
            "judge",
            "coverage_threshold",
            0.50,
            "irt_calibration_v2026q1",
            {"case_id": "case_001", "mode": "standard"}
        )
        
        entries = logger.query(event_type=LogEventType.THRESHOLD_APPLY)
        assert len(entries) == 1
        
        entry = entries[0]
        assert entry.data["value"] == 0.50
        assert entry.data["source"] == "irt_calibration_v2026q1"


class TestTransparentJudge:
    """Test transparent judgment wrapper."""
    
    def test_wrapper_delegates_to_plugin(self):
        """Wrapper must delegate to underlying plugin."""
        plugin = ExactMatchPlugin()
        wrapper = TransparentJudgeWrapper(plugin)
        
        result = wrapper.judge("hello", {"target": "hello", "case_id": "test"})
        assert result.passed is True
    
    def test_wrapper_logs_execution(self):
        """Wrapper must log judgment execution."""
        logger = get_structured_logger()
        logger.clear()
        
        plugin = ExactMatchPlugin()
        wrapper = TransparentJudgeWrapper(plugin)
        
        result = wrapper.judge("hello", {"target": "hello", "case_id": "test"})
        
        # Check logs were created
        recent = logger.get_recent(5)
        assert len(recent) > 0
    
    def test_judgment_logger_steps(self):
        """JudgmentLogger must track steps correctly."""
        jlogger = JudgmentLogger("case_001", "constraint_reasoning")
        
        jlogger.log_step(
            "validation",
            {"input": "test"},
            {"output": "valid"}
        )
        
        jlogger.log_coverage(
            constraints=["a", "b", "c"],
            response_keywords=["a", "b"],
            coverage=0.67,
            threshold=0.5,
            threshold_source="irt"
        )
        
        record = jlogger.finalize(passed=True, final_detail={"grade": "A"})
        
        assert record["case_id"] == "case_001"
        assert record["total_steps"] == 2
        assert record["passed"] is True
    
    def test_judge_with_transparency(self):
        """High-level function must return result and log."""
        manager = get_plugin_manager()
        register_builtin_plugins(manager)
        
        # Clear logs
        get_structured_logger().clear()
        
        result, transparency_log = judge_with_transparency(
            "exact_match",
            "hello",
            {"target": "hello"},
            case_id="test_case"
        )
        
        assert result.passed is True
        assert "method" in transparency_log
        assert "logs" in transparency_log


class TestCodeSimplification:
    """Test code cleanup and simplification."""
    
    def test_no_duplicate_plugins_registered(self):
        """Same plugin should not be registered twice."""
        manager = PluginManager()
        
        # Register twice
        register_builtin_plugins(manager)
        register_builtin_plugins(manager)
        
        # Should still work correctly
        plugins = manager.list_plugins()
        # exact_match should only appear once
        assert plugins.count("exact_match") == 1 if "exact_match" in plugins else True


class TestIntegration:
    """Integration tests for Phase 4 components."""
    
    def test_full_judgment_pipeline(self):
        """Complete judgment pipeline with all v8 features."""
        # Setup
        logger = get_structured_logger()
        logger.clear()
        
        manager = get_plugin_manager()
        register_builtin_plugins(manager)
        
        # Execute constraint reasoning with transparency
        result, transparency = judge_with_transparency(
            "constraint_reasoning",
            "The answer is 42, calculated using the boundary method",
            {
                "target_pattern": r"42",
                "key_constraints": ["boundary", "answer"],
                "boundary_signals": ["boundary"],
                "coverage_threshold": 0.5,
                "threshold_source": "irt_calibration_v2026q1",
                "case_id": "integration_test"
            },
            case_id="integration_test"
        )
        
        # Verify
        assert result.passed is True
        assert "keyword_coverage" in result.detail
        assert result.threshold_source == "irt_calibration_v2026q1"
        
        # Verify logging
        logs = logger.get_recent(10)
        assert len(logs) > 0
    
    def test_plugin_statistics_accumulation(self):
        """Plugin statistics must accumulate across calls."""
        manager = PluginManager()
        register_builtin_plugins(manager)
        
        # Execute multiple judgments
        for i in range(5):
            manager.judge("exact_match", f"test{i}", {"target": "test0", "case_id": f"case_{i}"})
        
        # Check stats
        stats = manager.get_stats("exact_match")
        assert stats is not None
        assert stats.call_count >= 5
        assert stats.avg_latency_ms >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
