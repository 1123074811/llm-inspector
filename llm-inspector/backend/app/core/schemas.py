"""
Shared data structures — plain dataclasses, no Pydantic needed.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


# ── LLM Adapter IO ────────────────────────────────────────────────────────────

@dataclass
class Message:
    role: str    # "system" | "user" | "assistant"
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class LLMRequest:
    model: str
    messages: list[Message]
    temperature: float = 0.0
    max_tokens: int = 5
    stream: bool = False
    response_format: dict | None = None
    tools: list[dict] | None = None
    timeout_sec: int = 60
    extra_params: dict = field(default_factory=dict)
    logprobs: bool = False
    top_logprobs: int = 0  # 0 = disabled; 1-20 = number of top tokens per position

    def to_payload(self) -> dict:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_dict() for m in self.messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.stream:
            payload["stream"] = True
        if self.response_format:
            payload["response_format"] = self.response_format
        if self.tools:
            payload["tools"] = self.tools
        if self.logprobs:
            payload["logprobs"] = True
            if self.top_logprobs > 0:
                payload["top_logprobs"] = self.top_logprobs
        payload.update(self.extra_params)
        return payload


@dataclass
class LLMResponse:
    content: str | None = None
    raw_json: dict | None = None
    status_code: int | None = None
    headers: dict = field(default_factory=dict)
    latency_ms: int | None = None
    first_token_ms: int | None = None
    finish_reason: str | None = None
    usage_prompt_tokens: int | None = None
    usage_completion_tokens: int | None = None
    usage_total_tokens: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    logprobs: list | None = None  # token-level logprobs from API (list of {token, logprob, top_logprobs})

    @property
    def ok(self) -> bool:
        return self.error_type is None and self.status_code == 200


@dataclass
class StreamChunk:
    index: int
    arrived_at_ms: int
    raw_line: str
    delta_text: str | None = None
    finish_reason: str | None = None


@dataclass
class StreamCaptureResult:
    chunks: list[StreamChunk] = field(default_factory=list)
    combined_text: str = ""
    latency_ms: int = 0
    first_token_ms: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    truncated: bool = False


# ── Test Case ─────────────────────────────────────────────────────────────────

@dataclass
class TestCase:
    id: str
    category: str
    name: str
    user_prompt: str
    expected_type: str
    judge_method: str
    system_prompt: str | None = None
    dimension: str | None = None
    tags: list[str] = field(default_factory=list)
    judge_rubric: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    max_tokens: int = 5
    n_samples: int = 1
    temperature: float = 0.0
    weight: float = 1.0
    enabled: bool = True
    suite_version: str = "v1"
    note: str = ""
    difficulty: float | None = None


@dataclass
class SampleResult:
    sample_index: int
    response: LLMResponse
    judge_passed: bool | None = None
    judge_detail: dict = field(default_factory=dict)


@dataclass
class CaseResult:
    case: TestCase
    samples: list[SampleResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        judged = [s for s in self.samples if s.judge_passed is not None]
        if not judged:
            return 0.0
        return sum(1 for s in judged if s.judge_passed) / len(judged)

    @property
    def mean_latency_ms(self) -> float | None:
        lats = [s.response.latency_ms for s in self.samples if s.response.latency_ms]
        return sum(lats) / len(lats) if lats else None


# ── Pre-Detection ─────────────────────────────────────────────────────────────

@dataclass
class LayerResult:
    layer: str
    confidence: float
    identified_as: str | None
    evidence: list[str] = field(default_factory=list)
    tokens_used: int = 0


@dataclass
class PreDetectionResult:
    success: bool
    identified_as: str | None
    confidence: float
    layer_stopped: str | None
    layer_results: list[LayerResult] = field(default_factory=list)
    total_tokens_used: int = 0
    should_proceed_to_testing: bool = True
    routing_info: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "identified_as": self.identified_as,
            "confidence": round(self.confidence, 3),
            "layer_stopped": self.layer_stopped,
            "total_tokens_used": self.total_tokens_used,
            "should_proceed_to_testing": self.should_proceed_to_testing,
            "routing_info": self.routing_info,
            "layer_results": [
                {
                    "layer": r.layer,
                    "confidence": round(r.confidence, 3),
                    "identified_as": r.identified_as,
                    "evidence": r.evidence,
                    "tokens_used": r.tokens_used,
                }
                for r in self.layer_results
            ],
        }


# ── Report ────────────────────────────────────────────────────────────────────

@dataclass
class Scores:
    protocol_score: float = 0.0
    instruction_score: float = 0.0
    system_obedience_score: float = 0.0
    param_compliance_score: float = 0.0


@dataclass
class ScoreCard:
    """v2 scoring: three-dimensional scorecard."""
    # Top-level scores (0-100)
    total_score: float = 0.0
    capability_score: float = 0.0
    authenticity_score: float = 0.0
    performance_score: float = 0.0
    # Capability sub-scores
    reasoning_score: float = 0.0
    adversarial_reasoning_score: float = 0.0
    instruction_score: float = 0.0
    coding_score: float = 0.0
    safety_score: float = 0.0
    protocol_score: float = 0.0
    # Authenticity sub-scores
    similarity_to_claimed: float = 0.0
    predetect_confidence: float = 0.0
    consistency_score: float = 0.0
    temperature_effectiveness: float = 0.0
    usage_fingerprint_match: float = 0.0
    behavioral_invariant_score: float = 0.0
    # Performance sub-scores
    speed_score: float = 0.0
    stability_score: float = 0.0
    cost_efficiency: float = 0.0

    def to_dict(self) -> dict:
        return {
            "total_score": round(self.total_score * 100),
            "capability_score": round(self.capability_score * 100),
            "authenticity_score": round(self.authenticity_score * 100),
            "performance_score": round(self.performance_score * 100),
            "breakdown": {
                "reasoning": round(self.reasoning_score * 100),
                "adversarial_reasoning": round(self.adversarial_reasoning_score * 100),
                "instruction": round(self.instruction_score * 100),
                "coding": round(self.coding_score * 100),
                "safety": round(self.safety_score * 100),
                "protocol": round(self.protocol_score * 100),
                "consistency": round(self.consistency_score * 100),
                "speed": round(self.speed_score * 100),
                "stability": round(self.stability_score * 100),
                "cost_efficiency": round(self.cost_efficiency * 100),
                "behavioral_invariant": round(self.behavioral_invariant_score * 100),
                **{k: round(v * 100) for k, v in getattr(self, "breakdown", {}).items() if k not in ["knowledge_score", "tool_use_score", "extraction_resistance", "fingerprint_match", "ttft_plausibility"]},
                "knowledge_score": round(getattr(self, "breakdown", {}).get("knowledge_score", 0.0) * 100),
                "tool_use_score": round(getattr(self, "breakdown", {}).get("tool_use_score", 0.0) * 100),
                "extraction_resistance": round(getattr(self, "breakdown", {}).get("extraction_resistance", 0.0) * 100),
                "fingerprint_match": round(getattr(self, "breakdown", {}).get("fingerprint_match", 0.0) * 100),
                "ttft_plausibility": round(getattr(self, "breakdown", {}).get("ttft_plausibility", 0.0) * 100),
            },
        }


@dataclass
class TrustVerdict:
    """v2 verdict replacing RiskAssessment."""
    level: str          # "trusted" | "suspicious" | "high_risk" | "fake"
    label: str
    total_score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    disclaimer: str = (
        "跑分结果基于行为特征对比，仅供参考，不构成确定性证明。"
        " / Benchmark scores are based on behavioural comparison and "
        "do not constitute definitive proof."
    )
    confidence_real: float = 0.0
    signal_details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "label": self.label,
            "total_score": round(self.total_score, 1),
            "confidence_real": self.confidence_real,
            "signal_details": self.signal_details,
            "reasons": self.reasons,
            "disclaimer": self.disclaimer,
        }


@dataclass
class SimilarityResult:
    benchmark_name: str
    similarity_score: float
    ci_95_low: float | None
    ci_95_high: float | None
    rank: int
    confidence_level: str = "unknown"  # "high" / "medium" / "low" / "insufficient"
    valid_feature_count: int = 0


@dataclass
class ThetaDimensionEstimate:
    dimension: str
    theta: float
    ci_low: float
    ci_high: float
    percentile: float | None = None
    n_items: int = 0

    def to_dict(self) -> dict:
        return {
            "dimension": self.dimension,
            "theta": round(self.theta, 4),
            "ci_low": round(self.ci_low, 4),
            "ci_high": round(self.ci_high, 4),
            "percentile": round(self.percentile, 2) if self.percentile is not None else None,
            "n_items": self.n_items,
        }


@dataclass
class ThetaReport:
    global_theta: float
    global_ci_low: float
    global_ci_high: float
    dimensions: list[ThetaDimensionEstimate] = field(default_factory=list)
    global_percentile: float | None = None
    calibration_version: str = "v1"
    method: str = "rasch_1pl"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "global_theta": round(self.global_theta, 4),
            "global_ci_low": round(self.global_ci_low, 4),
            "global_ci_high": round(self.global_ci_high, 4),
            "global_percentile": round(self.global_percentile, 2) if self.global_percentile is not None else None,
            "calibration_version": self.calibration_version,
            "method": self.method,
            "notes": self.notes,
            "dimensions": [d.to_dict() for d in self.dimensions],
        }


@dataclass
class ItemStat:
    item_id: str
    dimension: str
    a: float = 1.0
    b: float = 0.0
    c: float | None = None
    info_score: float = 0.0
    sample_size: int = 0
    anchor: bool = False


@dataclass
class PairwiseResult:
    model_a: str
    model_b: str
    delta_theta: float
    win_prob_a: float
    method: str = "bradley_terry"


@dataclass
class ABSignificance:
    metric: str
    golden_mean: float
    candidate_mean: float
    delta: float
    ci_95_low: float
    ci_95_high: float
    p_value: float
    significant: bool

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "golden_mean": round(self.golden_mean, 4),
            "candidate_mean": round(self.candidate_mean, 4),
            "delta": round(self.delta, 4),
            "ci_95_low": round(self.ci_95_low, 4),
            "ci_95_high": round(self.ci_95_high, 4),
            "p_value": round(self.p_value, 6),
            "significant": self.significant,
        }


@dataclass
class RiskAssessment:
    level: str          # "low" | "medium" | "high" | "very_high"
    label: str
    reasons: list[str] = field(default_factory=list)
    disclaimer: str = (
        "风险评级仅反映行为相似程度，不构成底层来源的确定性证明。"
        " / Risk level reflects behavioural similarity only and does not constitute "
        "definitive proof of the underlying model's origin."
    )
