"""
Judge Plugin Interface — v8.0 Plugin Architecture

Provides a standard interface for all judge methods to enable:
- Pluggable architecture for easy extension
- Standardized metadata and token estimation
- Consistent error handling and logging

Reference: V8_UPGRADE_PLAN.md Section 7.2.1
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum


class JudgeTier(Enum):
    """Judging tier indicating cost/complexity level."""
    LOCAL = "local"           # Local rules only (0 tokens)
    EMBEDDING = "embedding"    # Local embedding model (0 API tokens)
    LLM = "llm"               # External LLM judge (consumes API tokens)


@dataclass
class JudgeResult:
    """
    Standardized judge result with full transparency.
    
    All judges must return this structure for consistent
    logging and analysis.
    """
    passed: bool | None  # None = no judgment (observation only)
    detail: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0  # 0-1, confidence in the judgment
    tokens_used: int = 0     # API tokens consumed
    latency_ms: int = 0      # Execution time
    method: str = ""         # Judge method name
    version: str = "1.0"     # Judge version
    
    # v8: Enhanced provenance tracking
    threshold_source: Optional[str] = None  # Source of any thresholds used
    threshold_value: Optional[float] = None  # Threshold value applied
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "passed": self.passed,
            "detail": self.detail,
            "confidence": round(self.confidence, 3),
            "tokens_used": self.tokens_used,
            "latency_ms": self.latency_ms,
            "method": self.method,
            "version": self.version,
            "threshold_source": self.threshold_source,
            "threshold_value": self.threshold_value,
        }


@dataclass
class JudgeMetadata:
    """Metadata about a judge method."""
    name: str
    version: str
    tier: JudgeTier
    supported_languages: List[str]
    description: str
    deterministic: bool  # True = same input always gives same output
    
    # v8: Parameter schema for validation
    required_params: List[str] = field(default_factory=list)
    optional_params: List[str] = field(default_factory=list)
    
    # v8: Reference information
    reference_doi: Optional[str] = None
    reference_url: Optional[str] = None


class JudgePlugin(ABC):
    """
    Abstract base class for all judge methods.
    
    All new judges must implement this interface for consistency
    and interoperability with the v8 architecture.
    
    Example:
        class ExactMatchPlugin(JudgePlugin):
            @property
            def metadata(self) -> JudgeMetadata:
                return JudgeMetadata(
                    name="exact_match",
                    version="1.0",
                    tier=JudgeTier.LOCAL,
                    supported_languages=["*"],
                    description="Exact string matching",
                    deterministic=True
                )
            
            def judge(self, response: str, params: Dict) -> JudgeResult:
                target = params.get("target", "")
                passed = response.strip() == target.strip()
                return JudgeResult(
                    passed=passed,
                    detail={"expected": target, "got": response[:200]},
                    method=self.metadata.name,
                    version=self.metadata.version
                )
    """
    
    @property
    @abstractmethod
    def metadata(self) -> JudgeMetadata:
        """Return metadata about this judge."""
        pass
    
    @abstractmethod
    def judge(self, response: str, params: Dict[str, Any]) -> JudgeResult:
        """
        Execute the judgment.
        
        Args:
            response: The model's response text
            params: Judge-specific parameters
            
        Returns:
            JudgeResult with standardized output
        """
        pass
    
    def estimate_tokens(self, params: Dict[str, Any]) -> int:
        """
        Estimate token cost for this judgment.
        
        Override for judges that consume API tokens.
        Default returns 0 for local judges.
        """
        return 0
    
    def validate_params(self, params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate parameters before judging.
        
        Returns:
            (is_valid, list_of_error_messages)
        """
        errors = []
        for required in self.metadata.required_params:
            if required not in params:
                errors.append(f"Missing required parameter: {required}")
        return len(errors) == 0, errors
    
    def get_threshold(self, params: Dict[str, Any], param_name: str, 
                      default: float, source: str = "default") -> Tuple[float, str]:
        """
        Get threshold value with provenance tracking.
        
        Args:
            params: Parameter dict
            param_name: Name of threshold parameter
            default: Default value if not specified
            source: Source identifier for the threshold
            
        Returns:
            Tuple of (threshold_value, source_info)
        """
        value = params.get(param_name, default)
        source_info = params.get(f"{param_name}_source", source)
        return value, source_info


class TieredJudgePlugin(JudgePlugin):
    """
    Judge that can operate at multiple tiers (local/embedding/LLM).
    
    Automatically selects the appropriate tier based on available
    resources and configuration.
    """
    
    @abstractmethod
    def judge_local(self, response: str, params: Dict) -> JudgeResult:
        """Local rule-based judgment (fastest, no tokens)."""
        pass
    
    @abstractmethod
    def judge_embedding(self, response: str, params: Dict) -> JudgeResult:
        """Local embedding-based judgment (moderate speed, no API tokens)."""
        pass
    
    @abstractmethod
    def judge_llm(self, response: str, params: Dict) -> JudgeResult:
        """LLM-based judgment (slowest, consumes API tokens)."""
        pass
    
    def judge(self, response: str, params: Dict[str, Any]) -> JudgeResult:
        """
        Automatically select and execute appropriate tier.
        
        Selection order based on params:
        - "tier": explicit tier selection
        - "force_llm": True to always use LLM
        - Default: tries local → embedding → LLM
        """
        tier = params.get("tier", "auto")
        
        if tier == "local":
            return self.judge_local(response, params)
        elif tier == "embedding":
            return self.judge_embedding(response, params)
        elif tier == "llm":
            return self.judge_llm(response, params)
        
        # Auto mode: cascade through tiers
        try:
            result = self.judge_local(response, params)
            if result.confidence >= 0.9:  # High confidence from local
                return result
        except Exception:
            pass
        
        try:
            result = self.judge_embedding(response, params)
            if result.confidence >= 0.8:  # Good confidence from embedding
                return result
        except Exception:
            pass
        
        # Fall back to LLM
        return self.judge_llm(response, params)
