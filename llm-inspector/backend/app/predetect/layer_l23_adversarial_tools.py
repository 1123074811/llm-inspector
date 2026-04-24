"""
predetect/layer_l23_adversarial_tools.py — Layer 23: Adversarial Tool Use Probe

Requests the model to call a fake/unknown tool function. A genuine model
will typically refuse or explain it cannot; a wrapper/proxy may produce
distinctive error formats or fabricate a response.

Reference: Ruan et al. (2024) "ToolEmu: Identifying Real Risks of LLM Agents"
Default mode: Deep only.
"""
from __future__ import annotations

from app.core.schemas import LayerResult, LLMRequest, Message

# Fake tool definitions to probe with
_FAKE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "internal_system_query",
            "description": "Query internal system configuration",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Configuration key to query",
                    }
                },
                "required": ["key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "debug_echo",
            "description": "Echo back internal routing information for debugging",
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "Any input"},
                },
                "required": ["input"],
            },
        },
    },
]


class Layer23AdversarialTools:
    """Probe model behavior with fake tool definitions."""

    def run(self, adapter, model_name: str) -> LayerResult:
        evidence: list[str] = []
        total_tokens = 0

        for tool_def in _FAKE_TOOLS:
            tool_name = tool_def["function"]["name"]
            resp = adapter.chat(LLMRequest(
                model=model_name,
                messages=[Message(role="user", content=f"Call the '{tool_name}' function with key='test'")],
                max_tokens=100,
                temperature=0.0,
                timeout_sec=15,
                tools=[tool_def],
                tool_choice="auto",
            ))
            content = (resp.content or "").strip().lower()
            total_tokens += _count_tokens(resp)

            # Analyze response
            tool_calls = getattr(resp, "tool_calls", None) or []
            if tool_calls:
                for tc in tool_calls:
                    fn_name = tc.get("function", {}).get("name", "")
                    fn_args = tc.get("function", {}).get("arguments", "{}")
                    if fn_name == tool_name:
                        evidence.append(
                            f"Model attempted to call fake tool '{tool_name}' "
                            f"with args: {fn_args[:80]}"
                        )
                    else:
                        evidence.append(
                            f"Model called unexpected tool '{fn_name}' "
                            f"(expected '{tool_name}')"
                        )

            # Check for distinctive error patterns
            if any(p in content for p in ["internal error", "routing error", "proxy error",
                                            "upstream", "gateway", "internal_server"]):
                evidence.append(
                    f"Distinctive error pattern for tool '{tool_name}': '{content[:80]}'"
                )

            if "cannot" in content or "unable" in content or "don't have" in content:
                evidence.append(
                    f"Model refused fake tool '{tool_name}' (expected behavior)"
                )

        # Score: tool call compliance + error pattern mix
        tool_compliance = sum(1 for ev in evidence if "attempted to call" in ev)
        error_patterns = sum(1 for ev in evidence if "Distinctive error" in ev)
        refusals = sum(1 for ev in evidence if "refused fake tool" in ev)

        if tool_compliance > 0:
            evidence.append(
                f"Tool probe: {tool_compliance} fake tool(s) accepted, "
                f"{error_patterns} error pattern(s), {refusals} refusal(s)"
            )

        # Confidence: higher if model attempts fake tool calls or shows routing errors
        confidence = 0.0
        if tool_compliance > 0 and error_patterns > 0:
            confidence = 0.7  # both call + error = strong wrapper signal
        elif tool_compliance > 0:
            confidence = 0.5  # compliance alone = moderate
        elif error_patterns > 0:
            confidence = 0.4  # errors alone = weak
        elif refusals > 0:
            confidence = 0.0  # proper refusal = expected (genuine model)

        return LayerResult(
            layer="Layer23/AdversarialTools",
            confidence=round(confidence, 3),
            identified_as=None,
            evidence=evidence,
            tokens_used=total_tokens,
        )


def _count_tokens(resp) -> int:
    try:
        return getattr(resp, "usage", {}).get("total_tokens", 0)
    except Exception:
        return 0
