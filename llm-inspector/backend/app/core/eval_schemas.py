"""
EvalTestCase — v11 unified evaluation test case structure.

Extends TestCase with:
- CDM (Cognitive Diagnostic Model) skill vector: fine-grained micro-skill
  mastery probabilities for DINA model input (Phase 2)
- Bayesian prior: prior distribution parameters for θ estimation
- Telemetry hooks: structured tracing metadata for OpenTelemetry integration
- Eval metadata: norming sample, calibration version, validity flags

Backward-compatible: EvalTestCase inherits from TestCase, so all existing
code using TestCase works unchanged. The orchestrator's _load_suite() is
updated to construct EvalTestCase instances instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from app.core.schemas import TestCase
from app.core.provenance import DataProvenance


# ── Skill vector for CDM ────────────────────────────────────────────────────

@dataclass
class SkillVector:
    """
    Cognitive Diagnostic Model skill mapping.

    Each test case maps to one or more micro-skills with binary mastery
    requirements. The DINA model uses these to compute mastery probabilities.

    Example:
        SkillVector(required={"counterfactual": 1, "syllogism": 1})
        → this case requires BOTH counterfactual reasoning AND syllogism
    """
    required: dict[str, int] = field(default_factory=dict)
    # 1 = skill required for this item, 0 = not required
    # The Q-matrix row for this item

    @property
    def skill_names(self) -> list[str]:
        return [k for k, v in self.required.items() if v == 1]

    @property
    def n_skills(self) -> int:
        return sum(1 for v in self.required.values() if v == 1)

    def to_dict(self) -> dict:
        return {"required": self.required, "skill_names": self.skill_names}

    @staticmethod
    def from_dict(d: dict | None) -> SkillVector | None:
        if not d:
            return None
        return SkillVector(required=d.get("required", {}))


@dataclass
class BayesianPrior:
    """
    Bayesian prior for θ estimation.

    Instead of a point estimate, carries a prior distribution that
    gets updated with observed responses to produce a posterior with
    credible intervals.
    """
    distribution: str = "normal"  # "normal" | "beta" | "uniform"
    mu: float = 0.0               # prior mean (for normal)
    sigma: float = 1.0            # prior std (for normal)
    alpha: float = 1.0            # shape param (for beta)
    beta: float = 1.0             # shape param (for beta)

    def to_dict(self) -> dict:
        return {
            "distribution": self.distribution,
            "mu": self.mu,
            "sigma": self.sigma,
            "alpha": self.alpha,
            "beta": self.beta,
        }

    @staticmethod
    def from_dict(d: dict | None) -> BayesianPrior | None:
        if not d:
            return None
        return BayesianPrior(
            distribution=d.get("distribution", "normal"),
            mu=d.get("mu", 0.0),
            sigma=d.get("sigma", 1.0),
            alpha=d.get("alpha", 1.0),
            beta=d.get("beta", 1.0),
        )


# ── Eval metadata ───────────────────────────────────────────────────────────

@dataclass
class EvalMeta:
    """
    Evaluation metadata for test case quality and validity tracking.

    Tracks:
    - norming_sample_size: how many model responses were used to calibrate
    - calibration_version: which calibration run produced the IRT params
    - discriminative_valid: whether this case actually discriminates between
      models (IIF-based, to be computed in Phase 3 data pruning)
    - validity_flags: human/machine review flags
    """
    norming_sample_size: int = 0
    calibration_version: str = ""
    discriminative_valid: bool = True  # False if IIF ≈ 0
    validity_flags: list[str] = field(default_factory=list)
    last_calibrated_at: str = ""

    def to_dict(self) -> dict:
        return {
            "norming_sample_size": self.norming_sample_size,
            "calibration_version": self.calibration_version,
            "discriminative_valid": self.discriminative_valid,
            "validity_flags": self.validity_flags,
            "last_calibrated_at": self.last_calibrated_at,
        }

    @staticmethod
    def from_dict(d: dict | None) -> EvalMeta | None:
        if not d:
            return None
        return EvalMeta(
            norming_sample_size=d.get("norming_sample_size", 0),
            calibration_version=d.get("calibration_version", ""),
            discriminative_valid=d.get("discriminative_valid", True),
            validity_flags=d.get("validity_flags", []),
            last_calibrated_at=d.get("last_calibrated_at", ""),
        )


# ── EvalTestCase ────────────────────────────────────────────────────────────

@dataclass
class EvalTestCase(TestCase):
    """
    v11 unified evaluation test case.

    Extends TestCase with CDM skill vectors, Bayesian priors, eval metadata,
    and telemetry support. Fully backward-compatible: all existing TestCase
    fields are inherited.

    Usage:
        # From existing TestCase (upgrade path)
        eval_case = EvalTestCase.from_test_case(test_case)

        # From DB dict (in orchestrator _load_suite)
        eval_case = EvalTestCase.from_db_dict(row_dict)
    """
    # v11: CDM skill mapping
    skill_vector: Optional[SkillVector] = None

    # v11: Bayesian prior for θ estimation
    bayesian_prior: Optional[BayesianPrior] = None

    # v11: Evaluation metadata
    eval_meta: Optional[EvalMeta] = None

    # v11: Telemetry span name for OpenTelemetry (set at runtime)
    telemetry_span: str = ""

    @classmethod
    def from_test_case(cls, tc: TestCase, **overrides) -> EvalTestCase:
        """Upgrade a TestCase to EvalTestCase, preserving all fields."""
        base = {
            "id": tc.id,
            "category": tc.category,
            "name": tc.name,
            "user_prompt": tc.user_prompt,
            "expected_type": tc.expected_type,
            "judge_method": tc.judge_method,
            "system_prompt": tc.system_prompt,
            "dimension": tc.dimension,
            "tags": tc.tags,
            "judge_rubric": tc.judge_rubric,
            "params": tc.params,
            "max_tokens": tc.max_tokens,
            "n_samples": tc.n_samples,
            "temperature": tc.temperature,
            "weight": tc.weight,
            "enabled": tc.enabled,
            "suite_version": tc.suite_version,
            "note": tc.note,
            "difficulty": tc.difficulty,
            "weight_provenance": tc.weight_provenance,
            "difficulty_provenance": tc.difficulty_provenance,
            "irt_a": tc.irt_a,
            "irt_b": tc.irt_b,
            "irt_c": tc.irt_c,
            "irt_valid": tc.irt_valid,
            "irt_fit_rmse": tc.irt_fit_rmse,
        }
        base.update(overrides)
        return cls(**base)

    @classmethod
    def from_db_dict(cls, d: dict) -> EvalTestCase:
        """Construct from a DB row dict (as returned by repo.load_cases)."""
        params = d.get("params", {}) or {}
        meta = params.get("_meta", {}) or {}

        # Extract skill vector from params if present
        skill_vec = SkillVector.from_dict(meta.get("skill_vector"))
        bayesian_prior = BayesianPrior.from_dict(meta.get("bayesian_prior"))
        eval_meta = EvalMeta.from_dict(meta.get("eval_meta"))

        return cls(
            id=d["id"],
            category=d["category"],
            name=d["name"],
            user_prompt=d["user_prompt"],
            expected_type=d["expected_type"],
            judge_method=d["judge_method"],
            system_prompt=d.get("system_prompt"),
            dimension=meta.get("dimension") or d.get("dimension"),
            tags=meta.get("tags") or d.get("tags", []),
            judge_rubric=meta.get("judge_rubric") or d.get("judge_rubric", {}),
            params=params,
            max_tokens=d.get("max_tokens", 5),
            n_samples=d.get("n_samples", 1),
            temperature=d.get("temperature", 0.0),
            weight=d.get("weight", 1.0),
            enabled=bool(d.get("enabled", 1)),
            suite_version=d.get("suite_version", "v1"),
            note=d.get("note", ""),
            difficulty=d.get("difficulty"),
            weight_provenance=None,
            difficulty_provenance=None,
            irt_a=d.get("irt_a"),
            irt_b=d.get("irt_b"),
            irt_c=d.get("irt_c", 0.25),
            irt_valid=d.get("irt_valid", True),
            irt_fit_rmse=d.get("irt_fit_rmse", 0.0),
            skill_vector=skill_vec,
            bayesian_prior=bayesian_prior,
            eval_meta=eval_meta,
            telemetry_span=f"case.{d['id']}",
        )

    def to_dict(self) -> dict:
        """Serialize including v11 extensions."""
        d = super().to_dict()
        if self.skill_vector:
            d["skill_vector"] = self.skill_vector.to_dict()
        if self.bayesian_prior:
            d["bayesian_prior"] = self.bayesian_prior.to_dict()
        if self.eval_meta:
            d["eval_meta"] = self.eval_meta.to_dict()
        return d

    @property
    def has_skills(self) -> bool:
        return self.skill_vector is not None and self.skill_vector.n_skills > 0

    @property
    def has_prior(self) -> bool:
        return self.bayesian_prior is not None
