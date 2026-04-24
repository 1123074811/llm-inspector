"""
Shared data structures — plain dataclasses, no Pydantic needed.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

# v8.0: Import provenance for data lineage support
from app.core.provenance import DataProvenance


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

    def to_dict(self) -> dict:
        """v6: Serialize to dict for caching/persistence."""
        return {
            "content": self.content,
            "raw_json": self.raw_json,
            "status_code": self.status_code,
            "headers": self.headers,
            "latency_ms": self.latency_ms,
            "first_token_ms": self.first_token_ms,
            "finish_reason": self.finish_reason,
            "usage_prompt_tokens": self.usage_prompt_tokens,
            "usage_completion_tokens": self.usage_completion_tokens,
            "usage_total_tokens": self.usage_total_tokens,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "logprobs": self.logprobs,
        }


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

    # v8.0 / v11.0: Extended attributes for data provenance and IRT/CDM
    weight_provenance: Optional[DataProvenance] = None
    difficulty_provenance: Optional[DataProvenance] = None
    
    # IRT parameters (a=discrimination, b=difficulty, c=guessing)
    irt_a: Optional[float] = None
    irt_b: Optional[float] = None
    irt_c: float = 0.25
    irt_valid: bool = True
    irt_fit_rmse: float = 0.0
    
    @property
    def has_valid_provenance(self) -> bool:
        """Check if weight has valid provenance."""
        if not self.weight_provenance:
            return False
        return self.weight_provenance.confidence > 0.5
    
    @property
    def has_irt_params(self) -> bool:
        """Check if IRT parameters are available."""
        return self.irt_a is not None and self.irt_b is not None

    def get_weight_with_fallback(self) -> float:
        """v8: Data-driven fallback with provenance check."""
        if self.has_valid_provenance:
            return self.weight
        return self.weight

    def to_dict(self) -> dict:
        """v8: Serialize case to dict including provenance."""
        d = {
            "id": self.id,
            "category": self.category,
            "name": self.name,
            "user_prompt": self.user_prompt,
            "expected_type": self.expected_type,
            "judge_method": self.judge_method,
            "system_prompt": self.system_prompt,
            "dimension": self.dimension,
            "tags": self.tags,
            "judge_rubric": self.judge_rubric,
            "params": self.params,
            "max_tokens": self.max_tokens,
            "n_samples": self.n_samples,
            "temperature": self.temperature,
            "weight": self.weight,
            "enabled": self.enabled,
            "suite_version": self.suite_version,
            "note": self.note,
            "difficulty": self.difficulty,
        }
        
        # Add v8 properties if present
        if self.weight_provenance:
            d["weight_provenance"] = self.weight_provenance.to_dict()
        if self.difficulty_provenance:
            d["difficulty_provenance"] = self.difficulty_provenance.to_dict()
        if self.has_irt_params:
            d["irt_a"] = self.irt_a
            d["irt_b"] = self.irt_b
            d["irt_c"] = self.irt_c
            d["irt_valid"] = self.irt_valid
            d["irt_fit_rmse"] = self.irt_fit_rmse
            
        return d


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
    # v15 Phase 2: failure layer for error categorization
    failure_layer: str | None = None  # None / preflight / network / upstream_http / parse / judge / analysis

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

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "confidence": round(self.confidence, 3),
            "identified_as": self.identified_as,
            "evidence": self.evidence,
            "tokens_used": self.tokens_used,
        }


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
    current_layer: str | None = None  # Real-time progress tracking

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "identified_as": self.identified_as,
            "confidence": round(self.confidence, 3),
            "layer_stopped": self.layer_stopped,
            "total_tokens_used": self.total_tokens_used,
            "should_proceed_to_testing": self.should_proceed_to_testing,
            "routing_info": self.routing_info,
            "current_layer": self.current_layer,
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


# ── Identity Exposure (v14 Phase 3) ──────────────────────────────────────────

@dataclass
class IdentityExposureReport:
    """v14 Phase 3: Result of identity collision analysis.

    Populated by predetect.identity_exposure.analyze_case_results().
    Stored as JSON in test_runs.identity_exposure_result.
    """
    claimed_model: str | None = None
    claimed_family: str | None = None
    identity_collision: bool = False
    collision_confidence: float = 0.0
    top_families: list[dict] = field(default_factory=list)  # serialised FamilyHit dicts
    extracted_system_prompt: str | None = None
    total_responses_scanned: int = 0

    def to_dict(self) -> dict:
        return {
            "claimed_model": self.claimed_model,
            "claimed_family": self.claimed_family,
            "identity_collision": self.identity_collision,
            "collision_confidence": round(self.collision_confidence, 3),
            "top_families": self.top_families,
            "extracted_system_prompt": self.extracted_system_prompt,
            "total_responses_scanned": self.total_responses_scanned,
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
    reasoning_score: float | None = None
    adversarial_reasoning_score: float | None = None
    instruction_score: float = 0.0
    coding_score: float | None = None
    safety_score: float = 0.0
    protocol_score: float = 0.0
    # Authenticity sub-scores
    similarity_to_claimed: float | None = None
    predetect_confidence: float = 0.0
    consistency_score: float = 0.0
    temperature_effectiveness: float = 0.0
    usage_fingerprint_match: float = 0.0
    behavioral_invariant_score: float = 0.0
    # Performance sub-scores
    speed_score: float | None = None
    stability_score: float | None = None
    cost_efficiency: float = 0.0
    # Confidence level (0-1)
    confidence_level: float = 0.0
    # v13: Multi-scale display (Stanine-9 + percentile + raw logit theta)
    stanine: int | None = None           # 1-9 Stanine scale (Canfield 1951)
    percentile: float | None = None      # 0-100 percentile vs reference distribution
    theta: float | None = None           # Raw IRT theta estimate (logit scale)
    theta_ci95: tuple[float, float] | None = None  # 95% CI for theta
    judge_kappa: float | None = None     # v13: Cohen's κ inter-judge agreement (Phase 2.3)
    completeness: float | None = None    # v14: fraction of non-None capability dims
    # v14 Phase 6: Token efficiency tracking
    prompt_optimizer_used: bool = False
    tokens_saved_estimate: int | None = None
    token_counting_method: str = "fallback-estimate"
    # v14 Phase 7: Skipped case tracking
    skipped_cases: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        # v15 Phase 3: helper — return None for unmeasured, integer percentage for measured
        def pct(v: float | None) -> int | None:
            return round(v * 100) if v is not None else None

        _bd = getattr(self, "breakdown", {}) or {}

        return {
            "total_score": pct(self.total_score),
            "capability_score": pct(self.capability_score),
            "authenticity_score": pct(self.authenticity_score),
            "performance_score": pct(self.performance_score),
            "breakdown": {
                "reasoning": pct(self.reasoning_score),
                "adversarial_reasoning": pct(self.adversarial_reasoning_score),
                "instruction": pct(self.instruction_score),
                "coding": pct(self.coding_score),
                "safety": pct(self.safety_score),
                "protocol": pct(self.protocol_score),
                "consistency": pct(self.consistency_score),
                "speed": pct(self.speed_score),
                "stability": pct(self.stability_score),
                "cost_efficiency": pct(self.cost_efficiency),
                "behavioral_invariant": pct(self.behavioral_invariant_score),
                "knowledge_score": pct(_bd.get("knowledge_score")),
                "tool_use_score": pct(_bd.get("tool_use_score")),
                "extraction_resistance": pct(_bd.get("extraction_resistance")),
                "fingerprint_match": pct(_bd.get("fingerprint_match")),
                "ttft_plausibility": pct(_bd.get("ttft_plausibility")),
            },
            "v13": {
                "stanine": self.stanine,
                "percentile": self.percentile,
                "theta": round(self.theta, 4) if self.theta is not None else None,
                "theta_ci95": list(self.theta_ci95) if self.theta_ci95 else None,
                "judge_kappa": round(self.judge_kappa, 3) if self.judge_kappa is not None else None,
                "completeness": self.completeness,
            },
            "measurement": {
                "completeness": self.completeness,
                "measured_dims": sum(1 for v in [
                    self.reasoning_score, self.coding_score, self.speed_score,
                    self.stability_score, self.similarity_to_claimed,
                    self.adversarial_reasoning_score,
                ] if v is not None),
                "total_possible_dims": 8,
            },
            "token_analysis": {
                "prompt_optimizer_used": getattr(self, "prompt_optimizer_used", False),
                "tokens_saved_estimate": getattr(self, "tokens_saved_estimate", None),
                "counting_method": getattr(self, "token_counting_method", "fallback-estimate"),
            },
            "skipped_cases": list(getattr(self, "skipped_cases", [])),
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


# ── v15 Preflight schemas ────────────────────────────────────────────────────

@dataclass
class PreflightStepResult:
    """Result of a single preflight check step (v15)."""
    step: str
    name: str
    passed: bool
    duration_ms: float
    error_code: str | None = None
    error_message: str | None = None
    notes: str = ""


@dataclass
class PreflightReportSchema:
    """Full preflight report stored in run record (v15)."""
    passed: bool
    steps: list[dict] = field(default_factory=list)
    first_error: dict | None = None
    total_duration_ms: float = 0.0
    capabilities: dict = field(default_factory=dict)
    checked_at: str = ""
