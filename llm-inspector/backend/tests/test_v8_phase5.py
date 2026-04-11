"""
Test Suite for v8.0 Phase 5 — Frontend Upgrade API

Tests:
- v8 API endpoints
- Provenance data retrieval
- Judgment logs
- Plugin statistics

Reference: V8_IMPLEMENTATION_GUIDE.md Phase 5
"""
import pytest
import json
from unittest.mock import patch, MagicMock

# Test v8 handlers
from app.handlers.v8_handlers import (
    handle_v8_health,
    handle_v8_judgment_logs,
    handle_v8_case_provenance,
    handle_v8_data_lineage,
    handle_v8_plugin_stats,
    handle_v8_plugin_metadata,
    handle_v8_threshold_references,
    handle_v8_list_plugins,
)


class TestV8Health:
    """Test v8 health endpoint."""
    
    def test_v8_health_returns_ok(self):
        """v8 health should return healthy status."""
        status, body, content_type = handle_v8_health("/api/v8/health", {}, {})
        
        assert status == 200
        data = json.loads(body)
        assert data["status"] == "healthy"
        assert data["version"] == "8.0.0"
        assert data["phase"] == "5"
        assert "components" in data
        assert "features_enabled" in data
    
    def test_v8_health_has_required_fields(self):
        """v8 health should have all required fields."""
        status, body, content_type = handle_v8_health("/api/v8/health", {}, {})
        
        data = json.loads(body)
        assert "plugin_manager" in data["components"]
        assert "structured_logger" in data["components"]
        assert "available_plugins" in data["components"]
        
        features = data["features_enabled"]
        assert "plugin_architecture" in features
        assert "structured_logging" in features
        assert "provenance_tracking" in features
        assert "confidence_intervals" in features


class TestV8Plugins:
    """Test v8 plugin endpoints."""
    
    def setup_method(self):
        """Register built-in plugins before each test."""
        from app.judge import get_plugin_manager, register_builtin_plugins
        manager = get_plugin_manager()
        register_builtin_plugins(manager)
    
    def test_list_plugins(self):
        """Should list all available plugins."""
        status, body, content_type = handle_v8_list_plugins("/api/v8/plugins", {}, {})
        
        assert status == 200
        data = json.loads(body)
        assert "total" in data
        assert "plugins" in data
        assert isinstance(data["plugins"], list)
        # Should have at least the 6 built-in plugins
        assert data["total"] >= 6
    
    def test_plugin_metadata_found(self):
        """Should return metadata for existing plugin."""
        status, body, content_type = handle_v8_plugin_metadata(
            "/api/v8/plugins/exact_match/metadata", {}, {}
        )
        
        assert status == 200
        data = json.loads(body)
        assert data["name"] == "exact_match"
        assert "version" in data
        assert "tier" in data
        assert data["tier"] == "local"
        assert "description" in data
        assert "deterministic" in data
        assert data["deterministic"] is True
    
    def test_plugin_metadata_not_found(self):
        """Should return 404 for non-existent plugin."""
        status, body, content_type = handle_v8_plugin_metadata(
            "/api/v8/plugins/nonexistent_xyz/metadata", {}, {}
        )
        
        assert status == 404


class TestV8PluginStats:
    """Test v8 plugin statistics endpoint."""
    
    def test_plugin_stats_structure(self):
        """Plugin stats should have correct structure."""
        status, body, content_type = handle_v8_plugin_stats("/api/v8/plugin-stats", {}, {})
        
        assert status == 200
        data = json.loads(body)
        assert "plugins" in data
        assert "total_calls" in data
        assert "total_errors" in data
    
    def test_plugin_stats_fields(self):
        """Individual plugin stats should have required fields."""
        status, body, content_type = handle_v8_plugin_stats("/api/v8/plugin-stats", {}, {})
        
        data = json.loads(body)
        for plugin_name, stats in data["plugins"].items():
            assert "call_count" in stats
            assert "error_count" in stats
            assert "avg_latency_ms" in stats
            assert "total_tokens_used" in stats
            assert "error_rate" in stats


class TestV8ThresholdReferences:
    """Test v8 threshold references endpoint."""
    
    def test_threshold_references_list(self):
        """Should return list of threshold references."""
        status, body, content_type = handle_v8_threshold_references(
            "/api/v8/references/thresholds", {}, {}
        )
        
        assert status == 200
        data = json.loads(body)
        assert isinstance(data, list)
        assert len(data) > 0
    
    def test_threshold_reference_fields(self):
        """Each threshold reference should have required fields."""
        status, body, content_type = handle_v8_threshold_references(
            "/api/v8/references/thresholds", {}, {}
        )
        
        data = json.loads(body)
        for ref in data:
            assert "threshold_name" in ref
            assert "default_value" in ref
            assert "description" in ref
            assert "source" in ref
            assert "calibration_date" in ref
            assert "sample_size" in ref


class TestV8JudgmentLogs:
    """Test v8 judgment logs endpoint."""
    
    def test_judgment_logs_structure(self):
        """Should return judgment logs with correct structure."""
        status, body, content_type = handle_v8_judgment_logs(
            "/api/v8/runs/test_run_123/judgment-logs", {}, {}
        )
        
        assert status == 200
        data = json.loads(body)
        assert data["run_id"] == "test_run_123"
        assert "total_logs" in data
        assert "logs" in data
        assert isinstance(data["logs"], list)
    
    def test_judgment_logs_with_query_params(self):
        """Should handle query parameters."""
        qs = {
            "event_type": ["judge_complete"],
            "limit": ["50"]
        }
        status, body, content_type = handle_v8_judgment_logs(
            "/api/v8/runs/test_run_123/judgment-logs", qs, {}
        )
        
        assert status == 200


class TestV8CaseProvenance:
    """Test v8 case provenance endpoint."""
    
    def test_case_provenance_structure(self):
        """Should return case provenance with correct structure."""
        status, body, content_type = handle_v8_case_provenance(
            "/api/v8/runs/test_run_123/case/case_001/provenance", {}, {}
        )
        
        assert status == 200
        data = json.loads(body)
        assert data["run_id"] == "test_run_123"
        assert data["case_id"] == "case_001"
        assert "has_provenance" in data
        assert "weight_provenance" in data
        assert "threshold_provenance" in data
        assert "lineage" in data
    
    def test_case_provenance_has_lineage(self):
        """Case provenance should include data lineage."""
        status, body, content_type = handle_v8_case_provenance(
            "/api/v8/runs/test_run_123/case/case_001/provenance", {}, {}
        )
        
        data = json.loads(body)
        lineage = data.get("lineage", [])
        assert isinstance(lineage, list)
        
        for step in lineage:
            assert "operation" in step
            assert "description" in step
            assert "provenance" in step


class TestV8DataLineage:
    """Test v8 data lineage endpoint."""
    
    def test_data_lineage_structure(self):
        """Should return data lineage with correct structure."""
        status, body, content_type = handle_v8_data_lineage(
            "/api/v8/runs/test_run_123/data-lineage", {}, {}
        )
        
        assert status == 200
        data = json.loads(body)
        assert data["run_id"] == "test_run_123"
        assert "steps" in data
        assert isinstance(data["steps"], list)
    
    def test_data_lineage_steps(self):
        """Data lineage should have multiple steps."""
        status, body, content_type = handle_v8_data_lineage(
            "/api/v8/runs/test_run_123/data-lineage", {}, {}
        )
        
        data = json.loads(body)
        steps = data["steps"]
        assert len(steps) >= 4
        
        # Check for expected steps
        operations = [step["operation"] for step in steps]
        assert "test_case_loading" in operations
        assert "weight_assignment" in operations
        assert "judgment_execution" in operations
        assert "score_aggregation" in operations
    
    def test_data_lineage_step_fields(self):
        """Each lineage step should have required fields."""
        status, body, content_type = handle_v8_data_lineage(
            "/api/v8/runs/test_run_123/data-lineage", {}, {}
        )
        
        data = json.loads(body)
        for step in data["steps"]:
            assert "operation" in step
            assert "description" in step
            assert "timestamp" in step
            assert "provenance" in step
            
            prov = step["provenance"]
            assert "source" in prov
            assert "confidence" in prov


class TestV8Integration:
    """Integration tests for v8 API."""
    
    def test_v8_api_consistency(self):
        """All v8 endpoints should return consistent data formats."""
        endpoints = [
            ("/api/v8/health", handle_v8_health),
            ("/api/v8/plugins", handle_v8_list_plugins),
            ("/api/v8/plugin-stats", handle_v8_plugin_stats),
            ("/api/v8/references/thresholds", handle_v8_threshold_references),
        ]
        
        for path, handler in endpoints:
            status, body, content_type = handler(path, {}, {})
            assert status == 200, f"Endpoint {path} failed with status {status}"
            assert content_type == "application/json"
            
            # Verify valid JSON
            data = json.loads(body)
            assert data is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
