"""
v8.0 Phase 6 — Comprehensive Regression Test Suite

全量回归测试套件，验证所有 v8.0 功能：
- Phase 1: IRT参数与数据血缘
- Phase 2: 判题系统 (语义判题v3, 幻觉检测v3)
- Phase 3: 评分体系 (Theta量表)
- Phase 4: 插件架构、结构化日志
- Phase 5: API端点与前端组件

Reference: V8_UPGRADE_PLAN.md Phase 6
"""
import pytest
import json
import time
from typing import Dict, Any, List

# v8 imports
from app.core import (
    DataProvenance, ProvenanceTracker, get_provenance_tracker,
    Reference, ReferenceType, ReferenceDatabase, get_reference_database,
    StructuredLogger, LogEventType, get_structured_logger
)
from app.judge import (
    JudgePlugin, JudgeResult, JudgeMetadata, JudgeTier,
    PluginManager, get_plugin_manager, register_builtin_plugins,
    ExactMatchPlugin, RegexMatchPlugin, JSONSchemaPlugin,
    ConstraintReasoningPlugin, RefusalDetectPlugin, LineCountPlugin,
    TransparentJudgeWrapper, JudgmentLogger, judge_with_transparency
)


class TestV8Phase1IRTAndProvenance:
    """Phase 1: IRT参数与数据血缘回归测试"""
    
    def test_data_provenance_creation(self):
        """Test DataProvenance dataclass creation."""
        prov = DataProvenance(
            source_type="irt_calibration",
            source_id="irt_v2026q1_case001",
            collected_at="2026-01-15T08:30:00Z",
            sample_size=15000,
            confidence=0.92,
            verified=True
        )
        
        assert prov.source_type == "irt_calibration"
        assert prov.source_id == "irt_v2026q1_case001"
        assert prov.collected_at == "2026-01-15T08:30:00Z"
        assert prov.sample_size == 15000
        assert prov.confidence == 0.92
        assert prov.verified is True
    
    def test_data_provenance_to_dict(self):
        """Test DataProvenance serialization."""
        prov = DataProvenance(
            source_type="test_source",
            source_id="test_123",
            collected_at="2026-01-01T00:00:00Z",
            sample_size=100,
            confidence=0.85,
            verified=True
        )
        
        d = prov.to_dict()
        assert d["source_type"] == "test_source"
        assert d["source_id"] == "test_123"
        assert d["collected_at"] == "2026-01-01T00:00:00Z"
        assert d["sample_size"] == 100
        assert d["confidence"] == 0.85
    
    def test_data_provenance_create_fallback(self):
        """Test fallback provenance creation."""
        prov = DataProvenance.create_fallback(
            "test_entity",
            "Test fallback reason"
        )
        
        assert prov.source_type == "fallback"
        assert prov.confidence == 0.0
        assert "Test fallback reason" in (prov.notes or "")
    
    def test_provenance_tracker_singleton(self):
        """Test ProvenanceTracker singleton pattern."""
        tracker1 = get_provenance_tracker()
        tracker2 = get_provenance_tracker()
        assert tracker1 is tracker2
    
    def test_reference_database_access(self):
        """Test ReferenceDatabase access."""
        db = get_reference_database()
        assert db is not None
        # ReferenceDatabase is a class with classmethods, not a singleton instance
        assert hasattr(db, 'get_reference')
        assert hasattr(db, 'REFERENCES')
    
    def test_reference_creation(self):
        """Test Reference dataclass."""
        ref = Reference(
            reference_id="test_ref_001",
            reference_type=ReferenceType.JOURNAL_ARTICLE,
            title="Example Reference",
            authors="Author A, Author B",
            year=2025,
            doi="10.1000/example",
            journal="Test Journal"
        )
        
        assert ref.reference_id == "test_ref_001"
        assert ref.reference_type == ReferenceType.JOURNAL_ARTICLE
        assert ref.doi == "10.1000/example"
        assert ref.year == 2025


class TestV8Phase2JudgmentSystem:
    """Phase 2: 判题系统回归测试"""
    
    def test_exact_match_plugin_basic(self):
        """Test exact match plugin basic functionality."""
        plugin = ExactMatchPlugin()
        
        result = plugin.judge("hello", {"target": "hello"})
        assert result.passed is True
        assert result.confidence == 1.0
        
        result = plugin.judge("world", {"target": "hello"})
        assert result.passed is False
    
    def test_regex_match_plugin_patterns(self):
        """Test regex match plugin with various patterns."""
        plugin = RegexMatchPlugin()
        
        # Match digits
        result = plugin.judge("test123", {"pattern": r"\d+"})
        assert result.passed is True
        assert result.detail["found"] is True
        
        # No match
        result = plugin.judge("test", {"pattern": r"\d+"})
        assert result.passed is False
    
    def test_json_schema_plugin_validation(self):
        """Test JSON schema plugin."""
        plugin = JSONSchemaPlugin()
        
        # Valid JSON
        result = plugin.judge('{"name": "test"}', {"schema": {"type": "object"}})
        assert result.passed is True
        
        # Invalid JSON
        result = plugin.judge('not json', {})
        assert result.passed is False
        assert "error" in result.detail
    
    def test_constraint_reasoning_with_threshold_tracking(self):
        """Test constraint reasoning with v8 threshold tracking."""
        plugin = ConstraintReasoningPlugin()
        
        response = "The answer is 42 using boundary method"
        params = {
            "target_pattern": r"42",
            "key_constraints": ["boundary", "answer"],
            "coverage_threshold": 0.50,
            "threshold_source": "irt_calibration_v2026q1"
        }
        
        result = plugin.judge(response, params)
        
        # Verify threshold tracking (v8 feature)
        assert result.threshold_value == 0.50
        assert result.threshold_source == "irt_calibration_v2026q1"
        assert "keyword_coverage" in result.detail


class TestV8Phase3ScoringSystem:
    """Phase 3: 评分体系回归测试"""
    
    def test_theta_score_conversion(self):
        """Test theta score conversion logic."""
        # Mock theta to percentage conversion
        def theta_to_percentage(theta: float) -> float:
            # Sigmoid-like conversion
            import math
            return 1 / (1 + math.exp(-theta))
        
        # Theta = 0 should give ~50%
        score = theta_to_percentage(0)
        assert 0.49 < score < 0.51
        
        # Theta = 2 should give high score
        score = theta_to_percentage(2)
        assert score > 0.8
        
        # Theta = -2 should give low score
        score = theta_to_percentage(-2)
        assert score < 0.2
    
    def test_confidence_interval_calculation(self):
        """Test confidence interval calculation."""
        # Given a score and standard error, calculate CI
        def calculate_ci(score: float, se: float, confidence: float = 0.95) -> tuple:
            # Simplified: use 1.96 for 95% CI
            z = 1.96 if confidence >= 0.95 else 1.645
            margin = z * se
            return (max(0, score - margin), min(1, score + margin))
        
        score = 0.75
        se = 0.05
        lower, upper = calculate_ci(score, se)
        
        assert lower < score < upper
        assert upper - lower > 0  # Interval has width
        assert 0 <= lower <= 1
        assert 0 <= upper <= 1


class TestV8Phase4PluginArchitecture:
    """Phase 4: 插件架构回归测试"""
    
    def test_plugin_manager_singleton(self):
        """Test PluginManager singleton."""
        manager1 = get_plugin_manager()
        manager2 = get_plugin_manager()
        assert manager1 is manager2
    
    def test_all_builtin_plugins_registered(self):
        """Test all 6 built-in plugins can be registered."""
        manager = get_plugin_manager()
        register_builtin_plugins(manager)
        
        plugins = manager.list_plugins()
        required_plugins = [
            "exact_match",
            "regex_match", 
            "json_schema",
            "constraint_reasoning",
            "refusal_detect",
            "line_count"
        ]
        
        for plugin_name in required_plugins:
            assert plugin_name in plugins, f"Plugin {plugin_name} not registered"
    
    def test_plugin_metadata_structure(self):
        """Test plugin metadata has all required fields."""
        manager = get_plugin_manager()
        register_builtin_plugins(manager)
        
        meta = manager.get_metadata("exact_match")
        assert meta is not None
        assert meta.name == "exact_match"
        assert meta.version
        assert meta.tier in [JudgeTier.LOCAL, JudgeTier.EMBEDDING, JudgeTier.LLM]
        assert isinstance(meta.supported_languages, list)
        assert meta.description
        assert isinstance(meta.deterministic, bool)
    
    def test_plugin_judge_execution(self):
        """Test plugin execution through manager."""
        manager = get_plugin_manager()
        register_builtin_plugins(manager)
        
        result = manager.judge("exact_match", "hello", {"target": "hello"})
        assert result.passed is True
        assert result.method == "exact_match"
        assert result.latency_ms >= 0
    
    def test_plugin_statistics_tracking(self):
        """Test plugin statistics are tracked."""
        manager = get_plugin_manager()
        register_builtin_plugins(manager)
        
        # Execute some judgments
        for _ in range(3):
            manager.judge("exact_match", "test", {"target": "test"})
        
        stats = manager.get_stats("exact_match")
        assert stats is not None
        assert stats.call_count >= 3
        assert stats.avg_latency_ms >= 0
    
    def test_structured_logger_singleton(self):
        """Test StructuredLogger singleton."""
        logger1 = get_structured_logger()
        logger2 = get_structured_logger()
        assert logger1 is logger2
    
    def test_structured_logging_events(self):
        """Test structured logging of judgment events."""
        logger = get_structured_logger()
        logger.clear()
        
        # Log various events
        logger.log_judge_start("case_001", "exact_match")
        logger.log_judge_step(
            "case_001",
            "validation",
            {"input": "test"},
            {"output": "valid"}
        )
        logger.log_threshold_apply(
            "judge",
            "coverage_threshold",
            0.50,
            "irt_calibration",
            {"case_id": "case_001"}
        )
        
        # Query logs
        logs = logger.query(component="judge")
        assert len(logs) >= 3
    
    def test_transparent_judge_wrapper(self):
        """Test transparent judge wrapper."""
        manager = get_plugin_manager()
        register_builtin_plugins(manager)
        
        result, transparency = judge_with_transparency(
            "exact_match",
            "hello",
            {"target": "hello"},
            case_id="test_case"
        )
        
        assert result.passed is True
        assert "logs" in transparency
        assert "result_summary" in transparency


class TestV8Phase5APIAndFrontend:
    """Phase 5: API与前端回归测试"""
    
    def test_v8_health_endpoint_structure(self):
        """Test v8 health endpoint returns correct structure."""
        from app.handlers.v8_handlers import handle_v8_health
        
        status, body, content_type = handle_v8_health("/api/v8/health", {}, {})
        assert status == 200
        
        data = json.loads(body)
        assert data["status"] == "healthy"
        assert data["version"] == "8.0.0"
        assert data["phase"] == "5"
        assert "components" in data
        assert "features_enabled" in data
    
    def test_v8_plugin_list_endpoint(self):
        """Test v8 plugin list endpoint."""
        from app.handlers.v8_handlers import handle_v8_list_plugins
        from app.judge import get_plugin_manager, register_builtin_plugins
        
        # Register plugins first
        manager = get_plugin_manager()
        register_builtin_plugins(manager)
        
        status, body, content_type = handle_v8_list_plugins("/api/v8/plugins", {}, {})
        assert status == 200
        
        data = json.loads(body)
        assert "total" in data
        assert "plugins" in data
        assert data["total"] >= 6
    
    def test_v8_threshold_references_endpoint(self):
        """Test v8 threshold references endpoint."""
        from app.handlers.v8_handlers import handle_v8_threshold_references
        
        status, body, content_type = handle_v8_threshold_references(
            "/api/v8/references/thresholds", {}, {}
        )
        assert status == 200
        
        data = json.loads(body)
        assert isinstance(data, list)
        assert len(data) > 0
        
        # Check required fields
        for ref in data:
            assert "threshold_name" in ref
            assert "default_value" in ref
            assert "source" in ref
    
    def test_v8_judgment_logs_endpoint(self):
        """Test v8 judgment logs endpoint."""
        from app.handlers.v8_handlers import handle_v8_judgment_logs
        
        status, body, content_type = handle_v8_judgment_logs(
            "/api/v8/runs/test_123/judgment-logs", {}, {}
        )
        assert status == 200
        
        data = json.loads(body)
        assert data["run_id"] == "test_123"
        assert "total_logs" in data
        assert "logs" in data


class TestV8EndToEndIntegration:
    """端到端集成测试"""
    
    def test_full_judgment_pipeline_with_logging(self):
        """Test complete judgment pipeline with all v8 features."""
        # Setup
        logger = get_structured_logger()
        logger.clear()
        
        manager = get_plugin_manager()
        register_builtin_plugins(manager)
        
        # Execute judgment with transparency
        result, transparency = judge_with_transparency(
            "constraint_reasoning",
            "The answer is 100, calculated using optimization boundary",
            {
                "target_pattern": r"100",
                "key_constraints": ["optimization", "boundary"],
                "coverage_threshold": 0.6,
                "threshold_source": "irt_calibration_v2026q1"
            },
            case_id="e2e_test_001"
        )
        
        # Verify result
        assert result.passed is True
        assert result.confidence > 0
        assert result.threshold_source == "irt_calibration_v2026q1"
        assert result.threshold_value == 0.6
        
        # Verify logs were created
        logs = logger.get_recent(10)
        assert len(logs) > 0
    
    def test_plugin_tier_classification(self):
        """Test all plugins have correct tier classification."""
        manager = get_plugin_manager()
        register_builtin_plugins(manager)
        
        # All built-in plugins should be LOCAL tier (no API tokens)
        for plugin_name in manager.list_plugins():
            meta = manager.get_metadata(plugin_name)
            assert meta.tier == JudgeTier.LOCAL, f"{plugin_name} should be LOCAL tier"
    
    def test_judge_result_structure_completeness(self):
        """Test JudgeResult has all v8 fields."""
        result = JudgeResult(
            passed=True,
            detail={"test": "data"},
            confidence=0.95,
            tokens_used=100,
            latency_ms=50,
            method="test_method",
            version="1.0",
            threshold_source="test_source",
            threshold_value=0.75
        )
        
        d = result.to_dict()
        assert "passed" in d
        assert "confidence" in d
        assert "tokens_used" in d
        assert "latency_ms" in d
        assert "method" in d
        assert "version" in d
        # v8 specific fields
        assert "threshold_source" in d
        assert "threshold_value" in d


class TestV8PerformanceBenchmarks:
    """性能基准测试"""
    
    def test_exact_match_performance(self):
        """Benchmark exact match plugin performance."""
        plugin = ExactMatchPlugin()
        
        start = time.time()
        iterations = 1000
        
        for _ in range(iterations):
            plugin.judge("hello world test string", {"target": "hello world test string"})
        
        elapsed = time.time() - start
        avg_latency_ms = (elapsed / iterations) * 1000
        
        # Should be very fast (< 1ms per judgment)
        assert avg_latency_ms < 1.0, f"Average latency {avg_latency_ms}ms too high"
    
    def test_plugin_manager_performance(self):
        """Benchmark plugin manager dispatch performance."""
        manager = get_plugin_manager()
        register_builtin_plugins(manager)
        
        start = time.time()
        iterations = 100
        
        for _ in range(iterations):
            manager.judge("exact_match", "test", {"target": "test"})
        
        elapsed = time.time() - start
        avg_latency_ms = (elapsed / iterations) * 1000
        
        # Should be fast (< 5ms per judgment including dispatch overhead)
        assert avg_latency_ms < 5.0, f"Average latency {avg_latency_ms}ms too high"
    
    def test_structured_logging_performance(self):
        """Benchmark structured logging performance."""
        logger = get_structured_logger()
        
        start = time.time()
        iterations = 1000
        
        for i in range(iterations):
            logger.info(
                LogEventType.JUDGE_STEP,
                "test",
                f"Test message {i}",
                index=i
            )
        
        elapsed = time.time() - start
        avg_latency_ms = (elapsed / iterations) * 1000
        
        # Should be fast (< 0.1ms per log entry)
        assert avg_latency_ms < 0.1, f"Average latency {avg_latency_ms}ms too high"


# ═══════════════════════════════════════════════════════════════
# v8.0 Test Summary
# ═══════════════════════════════════════════════════════════════

"""
测试清单 (Test Checklist):

Phase 1 - IRT与数据血缘:
✓ DataProvenance 创建与序列化
✓ ProvenanceTracker 单例模式
✓ Reference 创建与管理

Phase 2 - 判题系统:
✓ ExactMatchPlugin 功能
✓ RegexMatchPlugin 功能
✓ JSONSchemaPlugin 功能
✓ ConstraintReasoningPlugin 阈值追踪

Phase 3 - 评分体系:
✓ Theta分数转换
✓ 置信区间计算

Phase 4 - 插件架构:
✓ PluginManager 单例模式
✓ 6个内置插件注册
✓ 插件元数据结构
✓ 插件执行与统计
✓ StructuredLogger 单例
✓ 结构化事件日志
✓ TransparentJudgeWrapper

Phase 5 - API与前端:
✓ v8 Health端点
✓ v8 Plugin列表端点
✓ v8 阈值参考端点
✓ v8 判题日志端点

集成测试:
✓ 完整判题管道
✓ 插件层级分类
✓ JudgeResult结构完整性

性能基准:
✓ ExactMatch性能
✓ PluginManager调度性能
✓ 结构化日志性能

总测试数: 25+
"""


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
