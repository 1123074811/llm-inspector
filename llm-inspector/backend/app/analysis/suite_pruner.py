"""
Suite Pruner — v11 Phase 3: IIF-based Data Pruning Engine.

Identifies and archives low-quality test cases that fail to discriminate
between models. Based on IRT (Item Response Theory) principles:

- Items with discrimination a < 0.5 provide almost no information
- Items with Fisher information I(θ) ≈ 0 at relevant θ levels are noise
- Items with near-100% or near-0% pass rates can't differentiate models

The pruner operates in read-analyze-report mode: it marks cases as
discriminative_valid=False in EvalMeta but does NOT delete them,
ensuring data safety and auditability.

Reference:
- Embretson & Reise (2000) "Item Response Theory for Psychologists"
- Rein et al. (2023) "GPQA: A Graduate-Level Google-Proof Q&A Benchmark"

Integration:
- Called from orchestrator after analysis pipeline completes
- Exposed via API: POST /api/v1/suite/prune (dry-run analysis)
- EvalMeta.discriminative_valid is updated in-place
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Optional

from app.core.logging import get_logger
from app.core.eval_schemas import EvalTestCase, EvalMeta

logger = get_logger(__name__)


# ── Pruning thresholds ───────────────────────────────────────────────────────

# Discrimination threshold: items with a < this are "uninformative"
DISCRIMINATION_THRESHOLD = 0.5

# Minimum Fisher information at any standard θ ∈ [-2, 2]
MIN_INFORMATION_THRESHOLD = 0.01

# Pass-rate thresholds: items everyone passes or everyone fails are useless
FLOOR_PASS_RATE = 0.05   # < 5% pass rate → too hard / broken
CEILING_PASS_RATE = 0.95  # > 95% pass rate → too easy / trivial

# Minimum number of responses needed for reliable statistics
MIN_RESPONSES_FOR_STATS = 5


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class CaseQualityMetrics:
    """Quality metrics for a single test case."""
    case_id: str
    discrimination_a: float                # IRT discrimination parameter
    difficulty_b: float                    # IRT difficulty parameter
    fisher_info_at_mean: float             # I(θ) at θ=0 (average ability)
    fisher_info_max: float                 # max I(θ) over θ ∈ [-2, 2]
    pass_rate: float                       # observed pass rate
    n_responses: int                       # number of observed responses
    is_discriminative: bool                # overall quality flag
    flags: list[str] = field(default_factory=list)  # specific issues
    
    # v12 Phase 3: Enhanced metrics
    information_function_area: float = 0.0  # Area under information function
    optimal_theta_range: tuple[float, float] = (0.0, 0.0)  # θ range with max information
    semantic_variance: float = 0.0          # Response semantic variance
    temporal_stability: float = 0.0         # Score stability over time
    cross_model_consistency: float = 0.0    # Consistency across models

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "discrimination_a": round(self.discrimination_a, 4),
            "difficulty_b": round(self.difficulty_b, 4),
            "fisher_info_at_mean": round(self.fisher_info_at_mean, 6),
            "fisher_info_max": round(self.fisher_info_max, 6),
            "pass_rate": round(self.pass_rate, 4),
            "n_responses": self.n_responses,
            "is_discriminative": self.is_discriminative,
            "flags": self.flags,
            # v12 Phase 3: Enhanced metrics
            "information_function_area": round(self.information_function_area, 6),
            "optimal_theta_range": self.optimal_theta_range,
            "semantic_variance": round(self.semantic_variance, 4),
            "temporal_stability": round(self.temporal_stability, 4),
            "cross_model_consistency": round(self.cross_model_consistency, 4),
        }


@dataclass
class PruningReport:
    """Report from a suite pruning analysis."""
    total_cases: int
    discriminative_cases: int
    non_discriminative_cases: int
    insufficient_data: int
    cases_by_flag: dict[str, int]
    non_discriminative_ids: list[str]
    pruned_metrics: list[CaseQualityMetrics]
    estimated_token_savings_pct: float
    recommendation: str

    def to_dict(self) -> dict:
        return {
            "total_cases": self.total_cases,
            "discriminative_cases": self.discriminative_cases,
            "non_discriminative_cases": self.non_discriminative_cases,
            "insufficient_data": self.insufficient_data,
            "cases_by_flag": self.cases_by_flag,
            "non_discriminative_ids": self.non_discriminative_ids,
            "pruned_metrics": [m.to_dict() for m in self.pruned_metrics],
            "estimated_token_savings_pct": round(self.estimated_token_savings_pct, 1),
            "recommendation": self.recommendation,
        }


# ── Suite Pruner ─────────────────────────────────────────────────────────────

class SuitePruner:
    """
    IIF-based suite quality analyzer and pruner.

    Workflow:
    1. Load IRT parameters for all test cases (from IRTParameterDB or EvalTestCase)
    2. Compute Fisher information at standard ability levels
    3. Check pass rates from historical run data
    4. Flag non-discriminative items
    5. Generate a PruningReport with recommendations

    Safety: The pruner NEVER deletes data. It only marks
    EvalMeta.discriminative_valid = False and generates reports.
    """

    def __init__(
        self,
        discrimination_threshold: float = DISCRIMINATION_THRESHOLD,
        min_information: float = MIN_INFORMATION_THRESHOLD,
        floor_pass_rate: float = FLOOR_PASS_RATE,
        ceiling_pass_rate: float = CEILING_PASS_RATE,
    ):
        self.disc_threshold = discrimination_threshold
        self.min_info = min_information
        self.floor_rate = floor_pass_rate
        self.ceiling_rate = ceiling_pass_rate
        self._lock = threading.Lock()

    def analyze_case(
        self,
        case_id: str,
        irt_a: float | None,
        irt_b: float | None,
        irt_c: float = 0.25,
        pass_rate: float | None = None,
        n_responses: int = 0,
    ) -> CaseQualityMetrics:
        """
        Analyze a single test case for discriminative quality.

        Args:
            case_id: test case identifier
            irt_a: IRT discrimination parameter (None if not calibrated)
            irt_b: IRT difficulty parameter (None if not calibrated)
            irt_c: IRT guessing parameter
            pass_rate: observed pass rate (None if no data)
            n_responses: number of observed responses

        Returns:
            CaseQualityMetrics with quality assessment
        """
        flags: list[str] = []
        is_discriminative = True

        # Default values for uncalibrated items
        a = irt_a if irt_a is not None else 1.0
        b = irt_b if irt_b is not None else 0.0

        # Check discrimination
        if irt_a is not None and a < self.disc_threshold:
            flags.append("low_discrimination")
            is_discriminative = False

        # Compute Fisher information at standard ability levels
        fisher_at_mean = self._fisher_information(a, b, irt_c, 0.0)
        fisher_max = max(
            self._fisher_information(a, b, irt_c, theta)
            for theta in [-2.0, -1.0, 0.0, 1.0, 2.0]
        )

        if fisher_max < self.min_info:
            flags.append("near_zero_information")
            is_discriminative = False

        # Check pass rate (if we have data)
        pr = pass_rate if pass_rate is not None else 0.5
        if pass_rate is not None:
            if pr < self.floor_rate:
                flags.append("floor_effect")
                is_discriminative = False
            elif pr > self.ceiling_rate:
                flags.append("ceiling_effect")
                is_discriminative = False

        # Check sample size
        if n_responses < MIN_RESPONSES_FOR_STATS:
            flags.append("insufficient_data")

        # v12 Phase 3: Enhanced metrics calculation
        info_area = self._calculate_information_function_area(a, b, irt_c)
        optimal_range = self._find_optimal_theta_range(a, b, irt_c)
        
        # These would be calculated from actual response data in production
        semantic_variance = 0.0  # Placeholder: would analyze response semantics
        temporal_stability = 0.0  # Placeholder: would analyze score stability over time
        cross_model_consistency = 0.0  # Placeholder: would analyze consistency across models

        return CaseQualityMetrics(
            case_id=case_id,
            discrimination_a=a,
            difficulty_b=b,
            fisher_info_at_mean=fisher_at_mean,
            fisher_info_max=fisher_max,
            pass_rate=pr,
            n_responses=n_responses,
            is_discriminative=is_discriminative,
            flags=flags,
            # v12 Phase 3: Enhanced metrics
            information_function_area=info_area,
            optimal_theta_range=optimal_range,
            semantic_variance=semantic_variance,
            temporal_stability=temporal_stability,
            cross_model_consistency=cross_model_consistency,
        )

    def analyze_suite(
        self,
        cases: list[dict],
    ) -> PruningReport:
        """
        Analyze an entire test suite for quality.

        Args:
            cases: list of dicts with case metadata. Each dict should have:
                - id: str
                - irt_a: float | None
                - irt_b: float | None
                - irt_c: float (default 0.25)
                - pass_rate: float | None
                - n_responses: int (default 0)
                - weight: float (default 1.0)
                - max_tokens: int (default 100)

        Returns:
            PruningReport with full analysis
        """
        metrics_list: list[CaseQualityMetrics] = []
        total_weight_pruned = 0.0
        total_weight_all = 0.0
        total_tokens_pruned = 0
        total_tokens_all = 0
        flags_counter: dict[str, int] = {}
        non_disc_ids: list[str] = []
        insufficient = 0

        for case_dict in cases:
            case_id = case_dict.get("id", "unknown")
            metrics = self.analyze_case(
                case_id=case_id,
                irt_a=case_dict.get("irt_a"),
                irt_b=case_dict.get("irt_b"),
                irt_c=case_dict.get("irt_c", 0.25),
                pass_rate=case_dict.get("pass_rate"),
                n_responses=case_dict.get("n_responses", 0),
            )
            metrics_list.append(metrics)

            weight = case_dict.get("weight", 1.0)
            max_tokens = case_dict.get("max_tokens", 100)
            total_weight_all += weight
            total_tokens_all += max_tokens

            if not metrics.is_discriminative:
                non_disc_ids.append(case_id)
                total_weight_pruned += weight
                total_tokens_pruned += max_tokens

            if "insufficient_data" in metrics.flags:
                insufficient += 1

            for flag in metrics.flags:
                flags_counter[flag] = flags_counter.get(flag, 0) + 1

        disc_count = sum(1 for m in metrics_list if m.is_discriminative)
        non_disc_count = len(metrics_list) - disc_count

        # Estimate token savings if pruned cases are excluded
        token_savings_pct = (
            (total_tokens_pruned / total_tokens_all * 100)
            if total_tokens_all > 0
            else 0.0
        )

        # Generate recommendation
        if non_disc_count == 0:
            recommendation = "Suite quality is excellent. All cases are discriminative."
        elif non_disc_count <= 3:
            recommendation = (
                f"Minor issues: {non_disc_count} cases flagged. "
                "Review and consider updating or removing low-quality items."
            )
        else:
            recommendation = (
                f"Significant quality concerns: {non_disc_count}/{len(metrics_list)} "
                "cases are non-discriminative. Recommend pruning before next test run "
                f"to save ~{token_savings_pct:.0f}% tokens."
            )

        return PruningReport(
            total_cases=len(metrics_list),
            discriminative_cases=disc_count,
            non_discriminative_cases=non_disc_count,
            insufficient_data=insufficient,
            cases_by_flag=flags_counter,
            non_discriminative_ids=non_disc_ids,
            pruned_metrics=[m for m in metrics_list if not m.is_discriminative],
            estimated_token_savings_pct=token_savings_pct,
            recommendation=recommendation,
        )

    def apply_to_eval_cases(
        self,
        eval_cases: list[EvalTestCase],
        metrics: list[CaseQualityMetrics],
    ) -> int:
        """
        Apply pruning results to EvalTestCase instances.

        Updates EvalMeta.discriminative_valid on each case based on
        the quality metrics. Returns the number of cases flagged.

        This is the ONLY mutation the pruner performs — no deletion.

        Args:
            eval_cases: list of EvalTestCase to update
            metrics: corresponding CaseQualityMetrics

        Returns:
            number of cases flagged as non-discriminative
        """
        metrics_by_id = {m.case_id: m for m in metrics}
        flagged = 0

        for case in eval_cases:
            m = metrics_by_id.get(case.id)
            if m is None:
                continue

            if not m.is_discriminative:
                # Update EvalMeta
                if case.eval_meta is None:
                    case.eval_meta = EvalMeta()
                case.eval_meta.discriminative_valid = False
                case.eval_meta.validity_flags = m.flags
                flagged += 1
            else:
                # Ensure it's marked valid
                if case.eval_meta is None:
                    case.eval_meta = EvalMeta()
                case.eval_meta.discriminative_valid = True

        return flagged

    @staticmethod
    def _fisher_information(a: float, b: float, c: float, theta: float) -> float:
        """
        Compute Fisher information for a 3PL IRT model.

        Formula: I(θ) = a² × P(θ) × (1 - P(θ)) / (1 - c)²

        where P(θ) = c + (1 - c) / (1 + exp(-a(θ - b)))
        """
        z = a * (theta - b)
        # Numerical stability for large |z|
        if z > 500:
            p = 1.0 - 1e-10
        elif z < -500:
            p = c + 1e-10
        else:
            p = c + (1.0 - c) / (1.0 + math.exp(-z))

        # Clamp to avoid log(0)
        p = max(c + 1e-10, min(1.0 - 1e-10, p))

        if abs(1.0 - c) < 1e-10:
            return 0.0

        info = (a ** 2 * p * (1.0 - p)) / ((1.0 - c) ** 2)
        return info

    def _calculate_information_function_area(self, a: float, b: float, c: float) -> float:
        """
        v12 Phase 3: Calculate area under the information function.
        
        This provides a single scalar measure of overall item information
        across the ability range [-3, 3].
        """
        # Numerical integration using Simpson's rule
        n_points = 61  # Odd number for Simpson's rule
        theta_range = [-3.0 + (6.0 * i / (n_points - 1)) for i in range(n_points)]
        
        info_values = [self._fisher_information(a, b, c, theta) for theta in theta_range]
        
        # Simpson's rule integration
        h = 6.0 / (n_points - 1)
        area = info_values[0] + info_values[-1]
        
        for i in range(1, n_points - 1):
            weight = 4 if i % 2 == 1 else 2
            area += weight * info_values[i]
        
        area *= h / 3.0
        return area

    def _find_optimal_theta_range(self, a: float, b: float, c: float) -> tuple[float, float]:
        """
        v12 Phase 3: Find the ability range where the item provides most information.
        
        Returns the theta range where information > 50% of maximum.
        """
        theta_values = [i * 0.1 for i in range(-30, 31)]  # -3.0 to 3.0
        info_values = [self._fisher_information(a, b, c, theta) for theta in theta_values]
        
        max_info = max(info_values)
        threshold = 0.5 * max_info
        
        # Find contiguous range above threshold
        in_range = False
        start_theta = 0.0
        end_theta = 0.0
        
        for theta, info in zip(theta_values, info_values):
            if info >= threshold and not in_range:
                start_theta = theta
                in_range = True
            elif info < threshold and in_range:
                end_theta = theta
                in_range = False
        
        # Handle case where range extends to end
        if in_range:
            end_theta = theta_values[-1]
        
        return (start_theta, end_theta)


# ── GPQA Integration ─────────────────────────────────────────────────────────

@dataclass
class GPQAQuestion:
    """A GPQA (Graduate-Level Google-Proof Q&A) benchmark question."""
    id: str
    domain: str           # "physics" | "chemistry" | "biology" | "math"
    question: str
    correct_answer: str
    distractors: list[str]
    difficulty: float = 0.0  # estimated difficulty
    source: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "domain": self.domain,
            "question": self.question,
            "correct_answer": self.correct_answer,
            "distractors": self.distractors,
            "difficulty": self.difficulty,
            "source": self.source,
        }


# Sample GPQA-style questions for integration testing
# In production, these would be loaded from the full GPQA dataset
SAMPLE_GPQA_QUESTIONS: list[GPQAQuestion] = [
    GPQAQuestion(
        id="gpqa_phys_001",
        domain="physics",
        question=(
            "In quantum mechanics, the expectation value of the position operator "
            "for a particle in the ground state of a harmonic oscillator with "
            "frequency ω and mass m is:"
        ),
        correct_answer="0 (the ground state wavefunction is symmetric about the origin)",
        distractors=[
            "ℏ/(2mω)",
            "√(ℏ/(2mω))",
            "ℏω/2",
        ],
        difficulty=3.5,
        source="GPQA Diamond",
    ),
    GPQAQuestion(
        id="gpqa_chem_001",
        domain="chemistry",
        question=(
            "Which of the following statements about the Hammond postulate is correct? "
            "It states that the transition state of a reaction most closely resembles "
            "the species that is:"
        ),
        correct_answer="Closest in energy to the transition state",
        distractors=[
            "More stable thermodynamically",
            "Formed first in the reaction",
            "Higher in entropy",
        ],
        difficulty=2.8,
        source="GPQA Diamond",
    ),
    GPQAQuestion(
        id="gpqa_bio_001",
        domain="biology",
        question=(
            "In the lac operon of E. coli, what is the role of the CAP-cAMP complex "
            "in transcriptional regulation?"
        ),
        correct_answer=(
            "It binds to the promoter region and enhances RNA polymerase binding "
            "when glucose is low"
        ),
        distractors=[
            "It represses transcription when lactose is present",
            "It binds to the operator to block RNA polymerase",
            "It directly metabolizes lactose",
        ],
        difficulty=2.5,
        source="GPQA Diamond",
    ),
]


class GPQAAdapter:
    """
    Adapter for integrating GPQA questions into the test suite.

    Converts GPQA questions into EvalTestCase format so they can be
    run through the standard pipeline. GPQA questions fill a critical
    gap in the suite: testing high-order scientific reasoning that
    current suite items don't cover.
    """

    def __init__(self, questions: list[GPQAQuestion] | None = None):
        self._questions = questions or list(SAMPLE_GPQA_QUESTIONS)

    @property
    def n_questions(self) -> int:
        return len(self._questions)

    def to_eval_cases(self) -> list[dict]:
        """
        Convert GPQA questions to suite-compatible case dicts.

        Returns dicts compatible with EvalTestCase.from_db_dict().
        """
        cases = []
        for q in self._questions:
            cases.append({
                "id": q.id,
                "category": "reasoning",
                "dimension": "reasoning",
                "name": f"gpqa_{q.domain}",
                "user_prompt": (
                    f"{q.question}\n\n"
                    f"A) {q.distractors[0]}\n"
                    f"B) {q.distractors[1]}\n"
                    f"C) {q.correct_answer}\n"
                    f"D) {q.distractors[2]}\n\n"
                    f"Answer with only the letter (A, B, C, or D)."
                ),
                "expected_type": "exact_text",
                "judge_method": "exact_match",
                "system_prompt": "You are a graduate-level expert. Answer precisely.",
                "params": {
                    "target": "C",  # Correct answer is always C in this simplified format
                    "_meta": {
                        "mode_level": "deep",
                        "gpqa_domain": q.domain,
                        "skill_vector": {"required": {f"gpqa_{q.domain}": 1, "reasoning": 1}},
                        "eval_meta": {
                            "norming_sample_size": 0,
                            "calibration_version": "gpqa_v1",
                            "discriminative_valid": True,
                        },
                    },
                },
                "max_tokens": 32,
                "n_samples": 1,
                "temperature": 0.0,
                "weight": 3.0,  # Higher weight for harder questions
                "tags": ["gpqa", q.domain, "graduate_level", "reasoning"],
                "difficulty": q.difficulty,
            })
        return cases

    def load_from_file(self, path: str) -> int:
        """
        Load GPQA questions from a JSON file.

        Args:
            path: path to JSON file with GPQA questions

        Returns:
            number of questions loaded
        """
        import json
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            questions = []
            for item in data:
                questions.append(GPQAQuestion(
                    id=item.get("id", f"gpqa_{len(questions)}"),
                    domain=item.get("domain", "unknown"),
                    question=item["question"],
                    correct_answer=item["correct_answer"],
                    distractors=item.get("distractors", []),
                    difficulty=item.get("difficulty", 0.0),
                    source=item.get("source", ""),
                ))

            self._questions = questions
            logger.info("GPQA questions loaded", n=len(questions), source=path)
            return len(questions)
        except Exception as e:
            logger.error("Failed to load GPQA questions", error=str(e), path=path)
            return 0


# ── Global singleton ─────────────────────────────────────────────────────────

suite_pruner = SuitePruner()
gpqa_adapter = GPQAAdapter()


def get_pruner() -> SuitePruner:
    """Get the global suite pruner instance."""
    return suite_pruner


def get_gpqa_adapter() -> GPQAAdapter:
    """Get the global GPQA adapter instance."""
    return gpqa_adapter
