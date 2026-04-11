"""
Plugin Manager — v8.0 Centralized Judge Management

Manages all judge plugins, providing:
- Plugin registration and discovery
- Dynamic loading and caching
- Statistics and health monitoring

Reference: V8_UPGRADE_PLAN.md Section 7.2
"""
from typing import Dict, List, Optional, Type, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
import time
import importlib
import pkgutil

from app.judge.plugin_interface import JudgePlugin, JudgeResult, JudgeMetadata, JudgeTier


@dataclass
class PluginStats:
    """Runtime statistics for a plugin."""
    call_count: int = 0
    error_count: int = 0
    total_latency_ms: int = 0
    total_tokens_used: int = 0
    last_used: Optional[str] = None
    
    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(self.call_count, 1)
    
    @property
    def error_rate(self) -> float:
        return self.error_count / max(self.call_count, 1)


@dataclass
class RegisteredPlugin:
    """A registered plugin with metadata."""
    name: str
    plugin_class: Type[JudgePlugin]
    metadata: JudgeMetadata
    instance: Optional[JudgePlugin] = None
    stats: PluginStats = field(default_factory=PluginStats)
    enabled: bool = True


class PluginManager:
    """
    Central manager for all judge plugins.
    
    Usage:
        manager = PluginManager()
        manager.register_builtin_plugins()
        
        # Execute a judgment
        result = manager.judge("exact_match", response, params)
    """
    
    _instance: Optional['PluginManager'] = None
    
    def __new__(cls):
        """Singleton pattern for global access."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._plugins: Dict[str, RegisteredPlugin] = {}
        self._initialized = True
        self._judge_log: List[Dict] = []  # v8: Structured logging
    
    def register(self, plugin_class: Type[JudgePlugin]) -> bool:
        """
        Register a plugin class.
        
        Returns:
            True if registration succeeded
        """
        try:
            # Create temporary instance to get metadata
            temp_instance = plugin_class()
            metadata = temp_instance.metadata
            
            plugin = RegisteredPlugin(
                name=metadata.name,
                plugin_class=plugin_class,
                metadata=metadata
            )
            
            self._plugins[metadata.name] = plugin
            return True
            
        except Exception as e:
            self._log_event("register_failed", {"error": str(e), "class": plugin_class.__name__})
            return False
    
    def get_plugin(self, name: str) -> Optional[JudgePlugin]:
        """
        Get or create plugin instance.
        
        Returns:
            Plugin instance or None if not found
        """
        registered = self._plugins.get(name)
        if not registered:
            return None
        
        if not registered.enabled:
            return None
        
        # Lazy initialization
        if registered.instance is None:
            registered.instance = registered.plugin_class()
        
        return registered.instance
    
    def judge(self, method: str, response: str, params: Dict[str, Any]) -> JudgeResult:
        """
        Execute judgment using specified method.
        
        This is the primary entry point for all judgments.
        
        Args:
            method: Judge method name
            response: Model response text
            params: Judge parameters
            
        Returns:
            JudgeResult with full provenance
        """
        start_time = time.time()
        
        plugin = self.get_plugin(method)
        if not plugin:
            # Return error result for unknown method
            return JudgeResult(
                passed=None,
                detail={"error": f"Unknown judge method: {method}"},
                method=method,
                version="error"
            )
        
        # Validate parameters
        is_valid, errors = plugin.validate_params(params)
        if not is_valid:
            return JudgeResult(
                passed=None,
                detail={"error": "Parameter validation failed", "errors": errors},
                method=method,
                version=plugin.metadata.version
            )
        
        # Execute judgment with error handling
        try:
            result = plugin.judge(response, params)
            result.method = method  # Ensure method name is set
            
            # Update statistics
            registered = self._plugins[method]
            registered.stats.call_count += 1
            registered.stats.total_latency_ms += result.latency_ms
            registered.stats.total_tokens_used += result.tokens_used
            registered.stats.last_used = datetime.now(timezone.utc).isoformat()
            
        except Exception as e:
            # Log error and return failure result
            registered = self._plugins[method]
            registered.stats.call_count += 1
            registered.stats.error_count += 1
            
            result = JudgeResult(
                passed=None,
                detail={"error": str(e), "error_type": type(e).__name__},
                method=method,
                version=plugin.metadata.version
            )
        
        # Log the judgment
        self._log_judgment(method, response[:200], params, result)
        
        return result
    
    def list_plugins(self, tier: Optional[JudgeTier] = None) -> List[str]:
        """List available plugin names."""
        plugins = self._plugins.values()
        if tier:
            plugins = [p for p in plugins if p.metadata.tier == tier]
        return [p.name for p in plugins if p.enabled]
    
    def get_metadata(self, method: str) -> Optional[JudgeMetadata]:
        """Get metadata for a plugin."""
        registered = self._plugins.get(method)
        return registered.metadata if registered else None
    
    def get_stats(self, method: str) -> Optional[PluginStats]:
        """Get runtime statistics for a plugin."""
        registered = self._plugins.get(method)
        return registered.stats if registered else None
    
    def get_all_stats(self) -> Dict[str, PluginStats]:
        """Get statistics for all plugins."""
        return {name: p.stats for name, p in self._plugins.items()}
    
    def disable_plugin(self, method: str) -> bool:
        """Disable a plugin."""
        if method in self._plugins:
            self._plugins[method].enabled = False
            return True
        return False
    
    def enable_plugin(self, method: str) -> bool:
        """Enable a plugin."""
        if method in self._plugins:
            self._plugins[method].enabled = True
            return True
        return False
    
    def _log_judgment(self, method: str, response: str, params: Dict, result: JudgeResult):
        """Log judgment for transparency."""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "judgment",
            "method": method,
            "response_preview": response[:100],
            "params_keys": list(params.keys()),
            "result": result.to_dict()
        }
        self._judge_log.append(log_entry)
        
        # Keep log size bounded
        if len(self._judge_log) > 1000:
            self._judge_log = self._judge_log[-1000:]
    
    def _log_event(self, event_type: str, data: Dict):
        """Log system events."""
        self._judge_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            **data
        })
    
    def get_recent_logs(self, n: int = 100) -> List[Dict]:
        """Get recent judgment logs."""
        return self._judge_log[-n:]
    
    def clear_logs(self):
        """Clear judgment logs."""
        self._judge_log = []


# Global accessor
def get_plugin_manager() -> PluginManager:
    """Get the global plugin manager instance."""
    return PluginManager()
