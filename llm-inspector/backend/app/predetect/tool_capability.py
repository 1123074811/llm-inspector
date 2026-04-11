"""
Tool Use Capability Probe for Model Detection.

Detects function calling/tool use capabilities to identify models
and detect wrapper layers that may alter tool behavior.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from app.core.schemas import LayerResult, LLMRequest, Message
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ToolCapabilityResult:
    """Result from tool capability testing."""
    supports_tools: bool
    tool_format: Optional[str]
    detected_patterns: List[str]
    confidence: float


class ToolCapabilityProbe:
    """
    Probes for tool/function calling capabilities.
    
    Different models have different tool calling formats:
    - OpenAI: function_call with name/arguments
    - Claude: tool_use with id/name/input
    - Gemini: functionCalls with name/args
    """
    
    # Tool schemas for testing
    TEST_TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"]
                        }
                    },
                    "required": ["location"]
                }
            }
        },
        {
            "type": "function", 
            "function": {
                "name": "calculate",
                "description": "Perform mathematical calculation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Math expression to evaluate"
                        }
                    },
                    "required": ["expression"]
                }
            }
        }
    ]
    
    # Tool response patterns by model family
    TOOL_PATTERNS = {
        "openai": {
            "indicators": [
                r'"function_call"',
                r'"name":\s*"get_weather"',
                r'"arguments":\s*\{',
            ],
            "format": "openai_function_call"
        },
        "claude": {
            "indicators": [
                r'"tool_use"',
                r'"tool_u?se_id"',
                r'"input":\s*\{',
                r'<tool_use>',
            ],
            "format": "claude_tool_use"
        },
        "gemini": {
            "indicators": [
                r'"functionCall"',
                r'"functionCalls"',
                r'"args":\s*\{',
            ],
            "format": "gemini_function_call"
        },
        "generic": {
            "indicators": [
                r'function_call',
                r'get_weather\s*\(',
                r'```tool',
                r'<tool>',
            ],
            "format": "generic_tool_format"
        }
    }
    
    def probe_tool_capability(
        self,
        adapter,
        model_name: str
    ) -> ToolCapabilityResult:
        """
        Probe model for tool calling capability.
        
        Returns:
            ToolCapabilityResult with detection info.
        """
        # Test 1: Direct tool prompt
        tool_prompt = """You have access to the following tools:

get_weather(location: string, unit?: "celsius" | "fahrenheit")
- Get current weather for a location

calculate(expression: string)
- Perform mathematical calculation

User: What's the weather in Paris?

Respond with the appropriate tool call."""
        
        resp1 = adapter.chat(LLMRequest(
            model=model_name,
            messages=[Message("user", tool_prompt)],
            max_tokens=150,
            temperature=0.0,
        ))
        
        response1 = resp1.content or ""
        
        # Test 2: Implicit tool need
        implicit_prompt = "What's the weather in Tokyo in Celsius?"
        
        resp2 = adapter.chat(LLMRequest(
            model=model_name,
            messages=[Message("user", implicit_prompt)],
            max_tokens=150,
            temperature=0.0,
        ))
        
        response2 = resp2.content or ""
        
        # Analyze responses
        detected_patterns = []
        tool_format = None
        supports_tools = False
        
        for family, patterns in self.TOOL_PATTERNS.items():
            for indicator in patterns["indicators"]:
                if re.search(indicator, response1, re.IGNORECASE):
                    detected_patterns.append(f"{family}: {indicator[:30]}")
                    if not tool_format:
                        tool_format = patterns["format"]
                        supports_tools = True
        
        # Check response 2 for tool-like behavior
        tool_like_indicators = [
            "function", "tool", "call", "get_weather", 
            "weather api", "weather data", "request"
        ]
        if any(ind in response2.lower() for ind in tool_like_indicators):
            if not supports_tools:
                supports_tools = True
                detected_patterns.append("implicit: tool-like language")
        
        # Calculate confidence
        if supports_tools and tool_format:
            confidence = min(0.90, 0.60 + len(detected_patterns) * 0.10)
        elif supports_tools:
            confidence = 0.50
        else:
            confidence = 0.20
        
        return ToolCapabilityResult(
            supports_tools=supports_tools,
            tool_format=tool_format,
            detected_patterns=detected_patterns,
            confidence=confidence
        )
    
    def detect_tool_alteration(
        self,
        base_capability: ToolCapabilityResult,
        wrapped_capability: ToolCapabilityResult
    ) -> tuple[bool, float, List[str]]:
        """
        Detect if tool capabilities are altered by wrapper.
        
        Returns:
            (is_altered, confidence, evidence)
        """
        evidence = []
        is_altered = False
        confidence = 0.0
        
        # Check for capability suppression
        if base_capability.supports_tools and not wrapped_capability.supports_tools:
            is_altered = True
            confidence = max(confidence, 0.80)
            evidence.append(
                "Tool capability suppressed: Base model supports tools, "
                "wrapped model does not"
            )
        
        # Check for format change
        if (base_capability.tool_format and wrapped_capability.tool_format and
            base_capability.tool_format != wrapped_capability.tool_format):
            is_altered = True
            confidence = max(confidence, 0.70)
            evidence.append(
                f"Tool format altered: {base_capability.tool_format} -> "
                f"{wrapped_capability.tool_format}"
            )
        
        # Check for pattern reduction
        base_patterns = len(base_capability.detected_patterns)
        wrapped_patterns = len(wrapped_capability.detected_patterns)
        if base_patterns > 0 and wrapped_patterns < base_patterns / 2:
            is_altered = True
            confidence = max(confidence, 0.60)
            evidence.append(
                f"Tool patterns reduced: {base_patterns} -> {wrapped_patterns}"
            )
        
        return is_altered, confidence, evidence


class Layer9ToolCapability:
    """
    Layer 9: Tool Use Capability Probe.
    Detects function calling capabilities and alterations.
    """
    
    def __init__(self):
        self.probe = ToolCapabilityProbe()
    
    def run(self, adapter, model_name: str) -> LayerResult:
        """Run tool capability probe."""
        evidence = []
        tokens_used = 0
        confidence = 0.0
        identified = None
        
        try:
            result = self.probe.probe_tool_capability(adapter, model_name)
            
            # Estimate tokens (2 prompts * ~100 tokens each)
            tokens_used = 250
            
            evidence.append(
                f"Tool capability: {'SUPPORTED' if result.supports_tools else 'NOT DETECTED'}"
            )
            
            if result.tool_format:
                evidence.append(f"Detected format: {result.tool_format}")
                confidence = result.confidence
                identified = f"Tool-capable ({result.tool_format})"
            
            if result.detected_patterns:
                evidence.append(f"Patterns found: {len(result.detected_patterns)}")
                for pattern in result.detected_patterns[:3]:
                    evidence.append(f"  - {pattern}")
            
            # Infer model family from tool format
            if result.tool_format:
                if "openai" in result.tool_format:
                    evidence.append("Tool format suggests OpenAI-style model")
                    confidence = max(confidence, 0.50)
                elif "claude" in result.tool_format:
                    evidence.append("Tool format suggests Claude-style model")
                    confidence = max(confidence, 0.50)
                elif "gemini" in result.tool_format:
                    evidence.append("Tool format suggests Gemini-style model")
                    confidence = max(confidence, 0.50)
        
        except Exception as e:
            logger.warning("Tool capability probe failed", error=str(e))
            evidence.append(f"Tool probe failed: {str(e)[:100]}")
        
        return LayerResult(
            layer="tool_capability",
            confidence=confidence,
            identified_as=identified,
            evidence=evidence,
            tokens_used=tokens_used,
        )
