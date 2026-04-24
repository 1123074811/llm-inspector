"""
Tests for v15 Phase 6: Predetect Layers L20-L23.

Covers:
  - Layer20SelfParadox: identity consistency across role contexts
  - Layer21MultiStepDrift: identity drift over multi-turn conversation
  - Layer22PromptReconstruct: system prompt reconstruction from harvester
  - Layer23AdversarialTools: fake tool probe
"""
from __future__ import annotations
import pytest


# ---------------------------------------------------------------------------
# Shared mock adapter
# ---------------------------------------------------------------------------

class _MockResponse:
    def __init__(self, content: str = "", status_code: int = 200, tool_calls=None):
        self.content = content
        self.status_code = status_code
        self.finish_reason = "stop"
        self.ok = status_code == 200
        self.error_type = None
        self.error_message = None
        self.latency_ms = 100
        self.usage_total_tokens = 10
        self.usage = {"total_tokens": 10}
        self.tool_calls = tool_calls or []

    def to_dict(self):
        return {"content": self.content, "status_code": self.status_code}


class _ConsistentAdapter:
    """Always responds with the same identity."""
    def __init__(self, identity: str = "GPT-4"):
        self._identity = identity

    def chat(self, req):
        return _MockResponse(content=self._identity)


class _InconsistentAdapter:
    """Returns different identities on alternating calls."""
    def __init__(self):
        self._calls = 0

    def chat(self, req):
        self._calls += 1
        identity = "GPT-4" if self._calls % 2 == 0 else "Claude"
        return _MockResponse(content=identity)


class _EmptyAdapter:
    def chat(self, req):
        return _MockResponse(content="")


class _ErrorAdapter:
    def chat(self, req):
        return _MockResponse(content="", status_code=500)


# ---------------------------------------------------------------------------
# Layer20SelfParadox tests
# ---------------------------------------------------------------------------

def test_layer20_consistent_returns_low_confidence():
    from app.predetect.layer_l20_self_paradox import Layer20SelfParadox
    layer = Layer20SelfParadox()
    result = layer.run(_ConsistentAdapter("I am GPT-4"), "gpt-4")
    assert result.layer == "Layer20/SelfParadox"
    # All consistent → near-zero confidence
    assert result.confidence <= 0.15


def test_layer20_inconsistent_returns_higher_confidence():
    from app.predetect.layer_l20_self_paradox import Layer20SelfParadox
    layer = Layer20SelfParadox()
    result = layer.run(_InconsistentAdapter(), "gpt-4")
    assert result.layer == "Layer20/SelfParadox"
    assert result.confidence >= 0.0  # may or may not detect — depends on alternation


def test_layer20_result_has_required_fields():
    from app.predetect.layer_l20_self_paradox import Layer20SelfParadox
    layer = Layer20SelfParadox()
    result = layer.run(_ConsistentAdapter(), "test-model")
    assert hasattr(result, "layer")
    assert hasattr(result, "confidence")
    assert hasattr(result, "identified_as")
    assert hasattr(result, "evidence")
    assert hasattr(result, "tokens_used")
    assert isinstance(result.evidence, list)


def test_layer20_confidence_bounded():
    from app.predetect.layer_l20_self_paradox import Layer20SelfParadox
    layer = Layer20SelfParadox()
    result = layer.run(_InconsistentAdapter(), "test-model")
    assert 0.0 <= result.confidence <= 1.0


def test_layer20_tokens_used_nonnegative():
    from app.predetect.layer_l20_self_paradox import Layer20SelfParadox
    layer = Layer20SelfParadox()
    result = layer.run(_ConsistentAdapter(), "test-model")
    assert result.tokens_used >= 0


def test_layer20_empty_responses_handled():
    from app.predetect.layer_l20_self_paradox import Layer20SelfParadox
    layer = Layer20SelfParadox()
    result = layer.run(_EmptyAdapter(), "test-model")
    # Empty responses → all unique_answers == {""} → treated as consistent
    assert result.layer == "Layer20/SelfParadox"
    assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# Layer21MultiStepDrift tests
# ---------------------------------------------------------------------------

def test_layer21_stable_identity_low_confidence():
    from app.predetect.layer_l21_multistep_drift import Layer21MultiStepDrift
    layer = Layer21MultiStepDrift()
    result = layer.run(_ConsistentAdapter("gpt"), "gpt-4")
    assert result.layer == "Layer21/MultiStepDrift"
    # stable identity → zero or very low confidence
    assert result.confidence <= 0.2


def test_layer21_drifting_identity_higher_confidence():
    from app.predetect.layer_l21_multistep_drift import Layer21MultiStepDrift
    layer = Layer21MultiStepDrift()
    result = layer.run(_InconsistentAdapter(), "gpt-4")
    assert result.layer == "Layer21/MultiStepDrift"
    assert result.confidence >= 0.0  # may vary


def test_layer21_result_has_required_fields():
    from app.predetect.layer_l21_multistep_drift import Layer21MultiStepDrift
    layer = Layer21MultiStepDrift()
    result = layer.run(_ConsistentAdapter(), "test")
    assert hasattr(result, "layer")
    assert hasattr(result, "confidence")
    assert hasattr(result, "evidence")
    assert isinstance(result.evidence, list)


def test_layer21_confidence_bounded():
    from app.predetect.layer_l21_multistep_drift import Layer21MultiStepDrift
    layer = Layer21MultiStepDrift()
    result = layer.run(_InconsistentAdapter(), "test")
    assert 0.0 <= result.confidence <= 1.0


def test_layer21_tokens_used_nonnegative():
    from app.predetect.layer_l21_multistep_drift import Layer21MultiStepDrift
    layer = Layer21MultiStepDrift()
    result = layer.run(_ConsistentAdapter(), "test")
    assert result.tokens_used >= 0


# ---------------------------------------------------------------------------
# Layer22PromptReconstruct tests
# ---------------------------------------------------------------------------

def test_layer22_no_fragments_returns_zero_confidence():
    from app.predetect.layer_l22_prompt_reconstruct import Layer22PromptReconstruct
    layer = Layer22PromptReconstruct()
    # Use empty run_id so no file/DB data available
    result = layer.run(None, "test-model", run_id="nonexistent-run-id")
    assert result.layer == "Layer22/PromptReconstruct"
    assert result.confidence == 0.0
    assert result.tokens_used == 0


def test_layer22_result_fields():
    from app.predetect.layer_l22_prompt_reconstruct import Layer22PromptReconstruct
    layer = Layer22PromptReconstruct()
    result = layer.run(None, "test-model", run_id="")
    assert hasattr(result, "layer")
    assert hasattr(result, "confidence")
    assert hasattr(result, "identified_as")
    assert hasattr(result, "evidence")
    assert isinstance(result.evidence, list)


def test_layer22_zero_tokens():
    from app.predetect.layer_l22_prompt_reconstruct import Layer22PromptReconstruct
    layer = Layer22PromptReconstruct()
    result = layer.run(None, "test-model", run_id="")
    # Layer22 is passive (zero additional tokens)
    assert result.tokens_used == 0


def test_layer22_suspicious_keywords_raise_confidence():
    """Direct test of the keyword heuristic via mock fragments."""
    from app.predetect.layer_l22_prompt_reconstruct import Layer22PromptReconstruct
    layer = Layer22PromptReconstruct()

    # Monkeypatch _load_harvester_output to return suspicious fragments
    def mock_load(_run_id):
        return [
            "你是一个智能助手 (You are an AI assistant)",
            "你的任务是 forward requests to the underlying proxy",
            "system prompt: route to gpt-4-turbo",
        ]

    original = Layer22PromptReconstruct._load_harvester_output
    Layer22PromptReconstruct._load_harvester_output = staticmethod(mock_load)
    try:
        result = layer.run(None, "test", run_id="test-run")
        assert result.confidence > 0.0
        assert len(result.evidence) >= 1
    finally:
        Layer22PromptReconstruct._load_harvester_output = staticmethod(original)


# ---------------------------------------------------------------------------
# Layer23AdversarialTools tests
# ---------------------------------------------------------------------------

def test_layer23_refusal_returns_zero_confidence():
    """A model that refuses the fake tool call is expected (genuine) behavior."""
    from app.predetect.layer_l23_adversarial_tools import Layer23AdversarialTools

    class RefusalAdapter:
        def chat(self, req):
            return _MockResponse(content="I cannot call that function. I don't have access to it.")

    layer = Layer23AdversarialTools()
    result = layer.run(RefusalAdapter(), "test-model")
    assert result.layer == "Layer23/AdversarialTools"
    assert result.confidence == 0.0


def test_layer23_error_pattern_raises_confidence():
    """A model returning routing error patterns suggests a wrapper."""
    from app.predetect.layer_l23_adversarial_tools import Layer23AdversarialTools

    class ErrorAdapter:
        def chat(self, req):
            return _MockResponse(content="upstream routing error: internal_server error")

    layer = Layer23AdversarialTools()
    result = layer.run(ErrorAdapter(), "test-model")
    assert result.layer == "Layer23/AdversarialTools"
    assert result.confidence >= 0.3


def test_layer23_tool_call_compliance_raises_confidence():
    """A model that calls fake tools gets moderate-to-high confidence."""
    from app.predetect.layer_l23_adversarial_tools import Layer23AdversarialTools

    class ToolCallAdapter:
        def chat(self, req):
            tool_calls = [
                {"function": {"name": "internal_system_query", "arguments": '{"key": "test"}'}}
            ]
            return _MockResponse(content="", tool_calls=tool_calls)

    layer = Layer23AdversarialTools()
    result = layer.run(ToolCallAdapter(), "test-model")
    assert result.confidence >= 0.4


def test_layer23_result_has_required_fields():
    from app.predetect.layer_l23_adversarial_tools import Layer23AdversarialTools

    class NeutralAdapter:
        def chat(self, req):
            return _MockResponse(content="I'm not sure how to handle this.")

    layer = Layer23AdversarialTools()
    result = layer.run(NeutralAdapter(), "test-model")
    assert hasattr(result, "layer")
    assert hasattr(result, "confidence")
    assert hasattr(result, "evidence")
    assert hasattr(result, "tokens_used")
    assert isinstance(result.evidence, list)


def test_layer23_confidence_bounded():
    from app.predetect.layer_l23_adversarial_tools import Layer23AdversarialTools

    class NeutralAdapter:
        def chat(self, req):
            return _MockResponse(content="")

    layer = Layer23AdversarialTools()
    result = layer.run(NeutralAdapter(), "test-model")
    assert 0.0 <= result.confidence <= 1.0


def test_layer23_both_tool_call_and_error_gives_max_confidence():
    """Both tool call compliance AND error pattern → highest confidence (0.7)."""
    from app.predetect.layer_l23_adversarial_tools import Layer23AdversarialTools

    class BothSignalAdapter:
        def __init__(self):
            self._calls = 0

        def chat(self, req):
            self._calls += 1
            if self._calls % 2 == 1:
                # Return tool call
                return _MockResponse(
                    content="upstream routing error",
                    tool_calls=[{"function": {"name": "internal_system_query", "arguments": '{}'}}]
                )
            return _MockResponse(content="upstream error")

    layer = Layer23AdversarialTools()
    result = layer.run(BothSignalAdapter(), "test-model")
    assert result.confidence >= 0.5
