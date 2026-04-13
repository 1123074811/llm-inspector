"""
CDM Engine — v11 Cognitive Diagnostic Model (DINA) implementation.

Implements the DINA (Deterministic Input, Noisy "And" gate) model for
fine-grained micro-skill mastery diagnosis, as specified in v11_upgrade_plan.md.

Reference: de la Torre (2011) "The generalized DINA model framework"

The DINA model provides:
- Per-skill mastery probabilities (vs. MDIRT's single θ estimate)
- Q-matrix mapping: which skills each test case requires
- EM algorithm for parameter estimation (slip s, guess g)
- Attribute pattern classification (mastery profiles)

Key differences from MDIRT/IRT:
- IRT gives θ (single ability dimension) — DINA gives P(mastery_k) per skill
- IRT assumes unidimensional latent trait — DINA assumes multidimensional
  binary skill mastery
- DINA is more interpretable: "模型掌握了反事实推理(92%)但缺乏三段论(31%)"
  vs IRT: "θ = 0.85"

Integration:
- Called from orchestrator's analysis pipeline after ThetaEstimator
- Uses EvalTestCase.skill_vector from eval_schemas.py for Q-matrix
- Output feeds into ReportBuilder and frontend skill radar chart
"""
from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from app.core.eval_schemas import SkillVector, EvalTestCase
from app.core.schemas import CaseResult, ThetaReport
from app.core.logging import get_logger

logger = get_logger(__name__)


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class DINAParams:
    """DINA model parameters for a single test case (item).

    The DINA model defines:
        P(X=1 | α, s, g) = (1-s) * ∏(1 - α_k * q_jk) + g * (1 - ∏(1 - α_k * q_jk))
    
    Simplified:
        If mastered ALL required skills (α_k=1 for all k where q_jk=1):
            P(X=1) = 1 - s  (slip probability)
        If missing ANY required skill:
            P(X=1) = g      (guess probability)
    """
    item_id: str
    slip: float = 0.1       # P(fail | mastered all) — slip rate
    guess: float = 0.2      # P(pass | missing some) — guess rate
    required_skills: list[str] = field(default_factory=list)


@dataclass
class SkillMastery:
    """Per-skill mastery diagnosis result."""
    skill_name: str
    mastery_probability: float  # P(α_k = 1) — 0.0 to 1.0
    confidence: str = "medium"  # "high" | "medium" | "low"
    evidence_count: int = 0     # Number of items requiring this skill
    pass_count: int = 0         # Items requiring this skill that passed

    def to_dict(self) -> dict:
        return {
            "skill_name": self.skill_name,
            "mastery_probability": round(self.mastery_probability, 3),
            "confidence": self.confidence,
            "evidence_count": self.evidence_count,
            "pass_count": self.pass_count,
        }


@dataclass
class CDMReport:
    """Complete CDM diagnosis report for a single run."""
    mastery_profile: list[SkillMastery] = field(default_factory=list)
    attribute_pattern: list[int] = field(default_factory=list)  # Binary mastery vector
    skill_names: list[str] = field(default_factory=list)
    overall_mastery_rate: float = 0.0
    strongest_skills: list[str] = field(default_factory=list)
    weakest_skills: list[str] = field(default_factory=list)
    dina_params: list[DINAParams] = field(default_factory=list)
    model_version: str = "dina_v1"
    n_items: int = 0
    n_skills: int = 0

    def to_dict(self) -> dict:
        return {
            "mastery_profile": [m.to_dict() for m in self.mastery_profile],
            "attribute_pattern": self.attribute_pattern,
            "skill_names": self.skill_names,
            "overall_mastery_rate": round(self.overall_mastery_rate, 3),
            "strongest_skills": self.strongest_skills,
            "weakest_skills": self.weakest_skills,
            "model_version": self.model_version,
            "n_items": self.n_items,
            "n_skills": self.n_skills,
        }


# ── Q-Matrix Builder ────────────────────────────────────────────────────────

# Default skill taxonomy for LLM capability assessment.
# Maps category/dimension to required micro-skills.
# This is the Q-matrix definition — each test case maps to a subset of these.

SKILL_TAXONOMY: dict[str, list[str]] = {
    # Core reasoning skills
    "reasoning": [
        "logical_deduction",
        "counterfactual_reasoning",
        "syllogism",
        "mathematical_reasoning",
        "spatial_reasoning",
        "analogical_reasoning",
    ],
    # Adversarial / anti-deception skills
    "adversarial_reasoning": [
        "constraint_detection",
        "anti_pattern_recognition",
        "base_encoding_awareness",
        "trap_identification",
    ],
    # Instruction following skills
    "instruction": [
        "exact_format_compliance",
        "json_schema_generation",
        "line_count_control",
        "multi_constraint_integration",
        "system_prompt_obedience",
    ],
    # Coding skills
    "coding": [
        "code_generation",
        "algorithm_implementation",
        "debugging",
        "code_execution_accuracy",
    ],
    # Safety skills
    "safety": [
        "harmful_request_refusal",
        "alternative_suggestion",
        "over_refusal_resistance",
    ],
    # Knowledge & hallucination resistance
    "knowledge": [
        "factual_accuracy",
        "hallucination_resistance",
        "uncertainty_acknowledgment",
    ],
    # Protocol compliance
    "protocol": [
        "api_protocol_compliance",
        "token_accounting_accuracy",
        "parameter_transparency",
    ],
    # Consistency & authenticity
    "consistency": [
        "deterministic_consistency",
        "identity_consistency",
        "behavioral_invariance",
    ],
    # Extraction resistance
    "prompt_extraction": [
        "prompt_leak_resistance",
        "identity_protection",
        "jailbreak_resistance",
    ],
    # Identity
    "identity": [
        "identity_persistence",
        "fingerprint_consistency",
    ],
    # Tool use
    "capability": [
        "tool_invocation",
        "structured_output_generation",
        "multi_step_planning",
    ],
    # Behavioral fingerprint
    "behavioral_fingerprint": [
        "style_consistency",
        "response_pattern_stability",
    ],
}

# Flatten to unique skill list
ALL_SKILLS: list[str] = []
for _skills in SKILL_TAXONOMY.values():
    for _s in _skills:
        if _s not in ALL_SKILLS:
            ALL_SKILLS.append(_s)

SKILL_INDEX: dict[str, int] = {s: i for i, s in enumerate(ALL_SKILLS)}


def build_q_matrix(case_results: list[CaseResult]) -> np.ndarray:
    """
    Build the Q-matrix (J × K) from case results.

    Q[j, k] = 1 if item j requires skill k, 0 otherwise.

    Uses the SkillVector from EvalTestCase if available, otherwise
    infers from dimension → SKILL_TAXONOMY mapping.
    """
    n_items = len(case_results)
    n_skills = len(ALL_SKILLS)
    Q = np.zeros((n_items, n_skills), dtype=np.int8)

    for j, cr in enumerate(case_results):
        case = cr.case

        # If case has explicit SkillVector (v11 EvalTestCase), use it
        if isinstance(case, EvalTestCase) and case.skill_vector is not None:
            for skill_name, required in case.skill_vector.required.items():
                if skill_name in SKILL_INDEX and required == 1:
                    Q[j, SKILL_INDEX[skill_name]] = 1
        else:
            # Fallback: infer from dimension/category
            dim = (case.dimension or case.category or "").lower()
            if dim in SKILL_TAXONOMY:
                for skill in SKILL_TAXONOMY[dim]:
                    if skill in SKILL_INDEX:
                        Q[j, SKILL_INDEX[skill]] = 1
            else:
                # Unknown dimension — mark all skills in same category
                cat = case.category.lower() if case.category else ""
                if cat in SKILL_TAXONOMY:
                    for skill in SKILL_TAXONOMY[cat]:
                        if skill in SKILL_INDEX:
                            Q[j, SKILL_INDEX[skill]] = 1

    return Q


# ── DINA Model ───────────────────────────────────────────────────────────────

class DINAEngine:
    """
    DINA (Deterministic Input, Noisy "And" gate) Cognitive Diagnostic Model.

    Implements:
    1. Q-matrix construction from test cases
    2. EM algorithm for slip/guess parameter estimation
    3. E-step: posterior probability of each attribute pattern given responses
    4. M-step: update slip/guess to maximize expected log-likelihood
    5. Skill mastery probability: marginal of posterior over attribute patterns

    The key output is per-skill mastery probability:
        P(α_k = 1) = Σ P(α | X) over all patterns where α_k = 1

    This is more interpretable than IRT's single θ because it tells
    exactly WHICH skills the model lacks.
    """

    def __init__(self, max_em_iter: int = 50, em_tol: float = 1e-4):
        self._max_em_iter = max_em_iter
        self._em_tol = em_tol

    def diagnose(
        self,
        case_results: list[CaseResult],
        theta_report: ThetaReport | None = None,
    ) -> CDMReport:
        """
        Run DINA diagnosis on case results.

        Args:
            case_results: Test results with pass/fail outcomes
            theta_report: Optional IRT theta report for prior initialization

        Returns:
            CDMReport with per-skill mastery probabilities
        """
        if not case_results:
            return CDMReport()

        # 1. Build Q-matrix
        Q = build_q_matrix(case_results)
        n_items, n_skills = Q.shape

        # 2. Extract response vector (0/1 for each item)
        Y = np.array([
            1.0 if cr.pass_rate > 0.5 else 0.0
            for cr in case_results
        ])

        # 3. Reduce to active skills (those actually required by any item)
        active_cols = np.where(Q.sum(axis=0) > 0)[0]
        if len(active_cols) == 0:
            return CDMReport(n_items=n_items)

        Q_active = Q[:, active_cols]
        n_active = len(active_cols)
        active_skill_names = [ALL_SKILLS[c] for c in active_cols]

        # 4. If too many active skills, group by dimension for tractability
        # DINA is exponential in K, so cap at ~10 skills
        if n_active > 10:
            Q_active, active_skill_names, active_cols = self._reduce_skills(
                Q_active, active_skill_names, active_cols, Y
            )
            n_active = len(active_skill_names)

        # 5. Estimate slip/guess via EM
        slip, guess = self._em_estimate(Q_active, Y, n_active)

        # 6. Compute skill mastery probabilities
        mastery_probs = self._compute_mastery(Q_active, Y, slip, guess, n_active)

        # 7. Build report
        return self._build_report(
            Q_active, Y, slip, guess,
            active_skill_names, active_cols,
            mastery_probs, n_items,
        )

    def _reduce_skills(
        self,
        Q: np.ndarray,
        skill_names: list[str],
        active_cols: np.ndarray,
        Y: np.ndarray,
    ) -> tuple[np.ndarray, list[str], np.ndarray]:
        """
        Reduce skill count by merging correlated skills.
        Keep the top 10 skills by discrimination (information gain).
        """
        # Score each skill by how well it discriminates pass vs fail
        scores = []
        for k in range(Q.shape[1]):
            # Items requiring this skill
            req_items = Q[:, k] == 1
            if req_items.sum() == 0:
                scores.append(0.0)
                continue
            pass_rate_req = Y[req_items].mean() if req_items.sum() > 0 else 0.0
            # Items NOT requiring this skill
            nonreq_items = Q[:, k] == 0
            pass_rate_nonreq = Y[nonreq_items].mean() if nonreq_items.sum() > 0 else 0.0
            # Discrimination: how much does having this skill matter
            discrimination = abs(pass_rate_req - pass_rate_nonreq)
            scores.append(discrimination)

        # Select top-10 skills
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]
        top_indices.sort()  # Maintain order

        return (
            Q[:, top_indices],
            [skill_names[i] for i in top_indices],
            active_cols[top_indices],
        )

    def _em_estimate(
        self,
        Q: np.ndarray,
        Y: np.ndarray,
        n_skills: int,
    ) -> tuple[float, float]:
        """
        EM algorithm for DINA slip/guess estimation.

        E-step: compute posterior P(α_l | Y_i) for each attribute pattern α_l
        M-step: update slip and guess to maximize expected log-likelihood

        Simplified: use a single global slip and guess (no per-item params)
        for robustness with limited data.
        """
        n_items = len(Y)
        n_patterns = 2 ** n_skills

        # Generate all possible attribute patterns (2^K binary vectors)
        patterns = np.array([
            [(p >> k) & 1 for k in range(n_skills)]
            for p in range(n_patterns)
        ])

        # Initialize slip and guess
        slip = 0.15
        guess = 0.20

        # Prior: uniform over attribute patterns
        # (could be informed by theta_report in future)
        prior = np.ones(n_patterns) / n_patterns

        for iteration in range(self._max_em_iter):
            # ── E-step: compute posterior P(α | Y) ──
            log_likelihoods = np.zeros(n_patterns)

            for l in range(n_patterns):
                alpha = patterns[l]
                # For each item, compute P(Y_j | α_l)
                for j in range(n_items):
                    # Does this item require all skills mastered?
                    # η_jl = ∏(α_lk ^ Q_jk) = 1 iff mastered ALL required skills
                    required_skills = Q[j]
                    # Check if mastered all required: AND of (α_k for required k)
                    mastered_all = True
                    for k in range(n_skills):
                        if required_skills[k] == 1 and alpha[k] == 0:
                            mastered_all = False
                            break

                    if mastered_all:
                        p_correct = 1.0 - slip
                    else:
                        p_correct = guess

                    # Clamp to avoid log(0)
                    p_correct = max(1e-10, min(1.0 - 1e-10, p_correct))

                    if Y[j] == 1.0:
                        log_likelihoods[l] += math.log(p_correct)
                    else:
                        log_likelihoods[l] += math.log(1.0 - p_correct)

            # Posterior (unnormalized)
            log_posterior = log_likelihoods + np.log(prior + 1e-300)
            # Normalize in log space for numerical stability
            log_posterior -= log_posterior.max()
            posterior = np.exp(log_posterior)
            posterior_sum = posterior.sum()
            if posterior_sum > 0:
                posterior /= posterior_sum
            else:
                posterior = prior.copy()

            # ── M-step: update slip and guess ──
            # Expected counts
            n_mastered_pass = 0.0
            n_mastered_fail = 0.0
            n_unmastered_pass = 0.0
            n_unmastered_fail = 0.0

            for j in range(n_items):
                required_skills = Q[j]
                # P(mastered all skills for item j) = Σ P(α_l) where α masters item j
                p_mastered = 0.0
                for l in range(n_patterns):
                    alpha = patterns[l]
                    mastered = True
                    for k in range(n_skills):
                        if required_skills[k] == 1 and alpha[k] == 0:
                            mastered = False
                            break
                    if mastered:
                        p_mastered += posterior[l]

                p_mastered = max(0.0, min(1.0, p_mastered))

                if Y[j] == 1.0:
                    n_mastered_pass += p_mastered
                    n_unmastered_pass += (1.0 - p_mastered)
                else:
                    n_mastered_fail += p_mastered
                    n_unmastered_fail += (1.0 - p_mastered)

            # Update slip = P(fail | mastered)
            new_slip = n_mastered_fail / max(n_mastered_pass + n_mastered_fail, 1e-10)
            # Update guess = P(pass | unmastered)
            new_guess = n_unmastered_pass / max(n_unmastered_pass + n_unmastered_fail, 1e-10)

            # Clamp to [0.01, 0.5] for stability
            new_slip = max(0.01, min(0.50, new_slip))
            new_guess = max(0.01, min(0.50, new_guess))

            # Check convergence
            if abs(new_slip - slip) < self._em_tol and abs(new_guess - guess) < self._em_tol:
                slip, guess = new_slip, new_guess
                break

            slip, guess = new_slip, new_guess

        return slip, guess

    def _compute_mastery(
        self,
        Q: np.ndarray,
        Y: np.ndarray,
        slip: float,
        guess: float,
        n_skills: int,
    ) -> np.ndarray:
        """
        Compute per-skill mastery probability P(α_k = 1).

        Uses the estimated slip/guess to compute posterior over attribute
        patterns, then marginalizes to get per-skill mastery.
        """
        n_items = len(Y)
        n_patterns = 2 ** n_skills

        # Generate attribute patterns
        patterns = np.array([
            [(p >> k) & 1 for k in range(n_skills)]
            for p in range(n_patterns)
        ])

        # Compute posterior P(α | Y) using final slip/guess
        log_likelihoods = np.zeros(n_patterns)
        for l in range(n_patterns):
            alpha = patterns[l]
            for j in range(n_items):
                required_skills = Q[j]
                mastered_all = True
                for k in range(n_skills):
                    if required_skills[k] == 1 and alpha[k] == 0:
                        mastered_all = False
                        break

                p_correct = (1.0 - slip) if mastered_all else guess
                p_correct = max(1e-10, min(1.0 - 1e-10, p_correct))

                if Y[j] == 1.0:
                    log_likelihoods[l] += math.log(p_correct)
                else:
                    log_likelihoods[l] += math.log(1.0 - p_correct)

        # Posterior
        log_posterior = log_likelihoods
        log_posterior -= log_posterior.max()
        posterior = np.exp(log_posterior)
        posterior_sum = posterior.sum()
        if posterior_sum > 0:
            posterior /= posterior_sum

        # Marginal mastery: P(α_k = 1) = Σ P(α_l) where α_lk = 1
        mastery = np.zeros(n_skills)
        for k in range(n_skills):
            for l in range(n_patterns):
                if patterns[l][k] == 1:
                    mastery[k] += posterior[l]

        return mastery

    def _build_report(
        self,
        Q: np.ndarray,
        Y: np.ndarray,
        slip: float,
        guess: float,
        skill_names: list[str],
        active_cols: np.ndarray,
        mastery_probs: np.ndarray,
        n_items: int,
    ) -> CDMReport:
        """Build the final CDMReport from computed data."""
        n_skills = len(skill_names)

        # Build per-skill mastery results
        mastery_profile = []
        for k in range(n_skills):
            prob = float(mastery_probs[k])
            # Evidence count: how many items require this skill
            evidence = int(Q[:, k].sum())
            # Pass count: among items requiring this skill, how many passed
            req_items = Q[:, k] == 1
            passes = int(Y[req_items].sum()) if req_items.sum() > 0 else 0

            # Confidence based on evidence count
            if evidence >= 8:
                confidence = "high"
            elif evidence >= 4:
                confidence = "medium"
            else:
                confidence = "low"

            mastery_profile.append(SkillMastery(
                skill_name=skill_names[k],
                mastery_probability=prob,
                confidence=confidence,
                evidence_count=evidence,
                pass_count=passes,
            ))

        # Attribute pattern: binary mastery (threshold at 0.5)
        attribute_pattern = [
            1 if m.mastery_probability >= 0.5 else 0
            for m in mastery_profile
        ]

        # Sort by mastery for strongest/weakest
        sorted_profile = sorted(mastery_profile, key=lambda m: m.mastery_probability, reverse=True)

        # Overall mastery rate
        overall = sum(m.mastery_probability for m in mastery_profile) / max(n_skills, 1)

        # Strongest (>= 0.7) and weakest (< 0.4)
        strongest = [m.skill_name for m in sorted_profile if m.mastery_probability >= 0.7][:5]
        weakest = [m.skill_name for m in reversed(sorted_profile) if m.mastery_probability < 0.4][:5]

        # Build per-item DINA params (simplified: use global slip/guess)
        dina_params = []
        for j in range(n_items):
            required = [skill_names[k] for k in range(n_skills) if Q[j, k] == 1]
            dina_params.append(DINAParams(
                item_id=f"item_{j}",
                slip=slip,
                guess=guess,
                required_skills=required,
            ))

        return CDMReport(
            mastery_profile=mastery_profile,
            attribute_pattern=attribute_pattern,
            skill_names=skill_names,
            overall_mastery_rate=overall,
            strongest_skills=strongest,
            weakest_skills=weakest,
            dina_params=dina_params,
            model_version="dina_v1",
            n_items=n_items,
            n_skills=n_skills,
        )


# ── Global singleton ────────────────────────────────────────────────────────

cdm_engine = DINAEngine()
