"""
judge/dual_judge.py — Dual-blind judge with Cohen's κ inter-rater agreement

Runs rule-judge AND semantic-judge concurrently on capability cases.
Computes Cohen's κ as a consistency signal.

Cohen's κ formula:
    κ = (P_o - P_e) / (1 - P_e)
    where:
        P_o = observed agreement proportion
        P_e = expected agreement by chance (= p_pass_rule * p_pass_sem +
                                              p_fail_rule * p_fail_sem)

References:
    Cohen (1960): https://doi.org/10.1177/001316446002000104
    Landis & Koch (1977): κ interpretation bands
        <0.20 slight | 0.21-0.40 fair | 0.41-0.60 moderate
        0.61-0.80 substantial | 0.81-1.00 almost perfect

Upgrade threshold: SRC["judge.kappa_upgrade_threshold"].value (default 0.60)
→ transparent_judge is invoked as tiebreaker.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Categories where semantic judging is meaningful. Protocol / fingerprint /
# tokenizer cases are excluded because they are metadata checks, not
# capability signals.
_SEMANTIC_CATEGORIES: set[str] = {
    "semantic_judge",
    "semantic_judge_v2",
    "constraint_reasoning",
    "text_constraints",
    "multi_step_verify",
    "hallucination_detect",
    "hallucination_detect_v2",
    "code_execution",
    "response_quality_basic",
    "json_schema",
    "line_count",
    "refusal_policy",
    "refusal_detect",
    "should_not_refuse",
    "refusal_check",
}


def _get_kappa_threshold() -> float:
    """Read Cohen's κ upgrade threshold from SRC, with safe fallback."""
    try:
        from app._data import SRC  # type: ignore
        return float(SRC["judge.kappa_upgrade_threshold"].value)
    except Exception:
        return 0.60


@dataclass
class DualJudgeResult:
    rule_passed: bool | None
    rule_detail: dict
    semantic_passed: bool | None
    semantic_detail: dict
    consensus_passed: bool | None      # final verdict
    kappa: float | None                # Cohen's κ (None if < 2 judges ran)
    agreement: bool                    # True if both judges agree
    tiebreak_used: bool                # True if transparent_judge was invoked
    tiebreak_detail: dict              # detail from transparent_judge if used
    method: str                        # "rule_only" | "semantic_only" | "dual_agree" | "dual_tiebreak"

    def to_dict(self) -> dict:
        return {
            "rule_passed": self.rule_passed,
            "rule_detail": self.rule_detail,
            "semantic_passed": self.semantic_passed,
            "semantic_detail": self.semantic_detail,
            "consensus_passed": self.consensus_passed,
            "kappa": self.kappa,
            "agreement": self.agreement,
            "tiebreak_used": self.tiebreak_used,
            "tiebreak_detail": self.tiebreak_detail,
            "method": self.method,
        }


def compute_kappa(votes_a: list[bool], votes_b: list[bool]) -> float:
    """
    Cohen's κ (1960) for two raters with binary votes.

    Edge cases:
      - Mismatched lengths → raises ValueError
      - Empty input → 1.0 (vacuous agreement)
      - Single pair → 1.0 if agree, -1.0 if disagree (approximation; κ is
        not well-defined for n=1)
      - No variance in either rater → 1.0 if fully agree, 0.0 otherwise
    """
    if len(votes_a) != len(votes_b):
        raise ValueError("votes_a and votes_b must have the same length")

    n = len(votes_a)
    if n == 0:
        return 1.0
    if n == 1:
        return 1.0 if votes_a[0] == votes_b[0] else -1.0

    # Observed agreement
    agree = sum(1 for a, b in zip(votes_a, votes_b) if a == b)
    p_o = agree / n

    # Expected agreement by chance
    p_a_true = sum(1 for v in votes_a if v) / n
    p_b_true = sum(1 for v in votes_b if v) / n
    p_a_false = 1.0 - p_a_true
    p_b_false = 1.0 - p_b_true
    p_e = p_a_true * p_b_true + p_a_false * p_b_false

    # No-variance edge case
    if abs(1.0 - p_e) < 1e-12:
        return 1.0 if abs(p_o - 1.0) < 1e-12 else 0.0

    return (p_o - p_e) / (1.0 - p_e)


def _should_run_semantic(judge_method: str, run_semantic: bool) -> bool:
    """Heuristic: only run semantic judge on capability-class methods."""
    if not run_semantic:
        return False
    return judge_method in _SEMANTIC_CATEGORIES


def _run_rule(judge_method: str, text: str | None, params: dict) -> tuple[bool | None, dict]:
    from app.judge.methods import judge as _judge
    return _judge(judge_method, text, params)


def _run_semantic(text: str | None, params: dict) -> tuple[bool | None, dict]:
    try:
        from app.judge.semantic_v2 import semantic_judge_v2
        return semantic_judge_v2(text or "", params)
    except Exception as exc:  # pragma: no cover - defensive
        return None, {"error": f"semantic_judge_v2 failed: {exc}"}


def _run_tiebreak(judge_method: str, text: str | None, params: dict) -> tuple[bool | None, dict]:
    """Invoke transparent_judge (Chain-of-Verification) as third tiebreaker."""
    try:
        from app.judge.transparent_judge import judge_with_transparency
        result = judge_with_transparency(judge_method, text, params)
        if isinstance(result, tuple) and len(result) == 2:
            return result  # type: ignore[return-value]
        # Some implementations return an object with .passed / .detail
        passed = getattr(result, "passed", None)
        detail = getattr(result, "detail", {}) or {}
        return passed, detail
    except Exception as exc:
        return None, {"error": f"transparent_judge failed: {exc}"}


def dual_judge(
    judge_method: str,
    response_text: str | None,
    params: dict,
    run_semantic: bool = True,
) -> DualJudgeResult:
    """
    Run rule-judge + optional semantic-judge and compute Cohen's κ.

    Args:
        judge_method: the rule-based judge method name.
        response_text: model response.
        params: judge params dict.
        run_semantic: whether to run semantic judge (default True for
                      capability-class cases; automatically skipped for
                      protocol/fingerprint methods).
    """
    params = params or {}
    rule_passed, rule_detail = _run_rule(judge_method, response_text, params)

    do_semantic = _should_run_semantic(judge_method, run_semantic)
    if not do_semantic:
        return DualJudgeResult(
            rule_passed=rule_passed,
            rule_detail=rule_detail,
            semantic_passed=None,
            semantic_detail={},
            consensus_passed=rule_passed,
            kappa=None,
            agreement=True,
            tiebreak_used=False,
            tiebreak_detail={},
            method="rule_only",
        )

    sem_passed, sem_detail = _run_semantic(response_text, params)

    # If one side produced no verdict (None), fall back to the other.
    if rule_passed is None and sem_passed is None:
        return DualJudgeResult(
            rule_passed=None,
            rule_detail=rule_detail,
            semantic_passed=None,
            semantic_detail=sem_detail,
            consensus_passed=None,
            kappa=None,
            agreement=True,
            tiebreak_used=False,
            tiebreak_detail={},
            method="rule_only",
        )
    if sem_passed is None:
        return DualJudgeResult(
            rule_passed=rule_passed,
            rule_detail=rule_detail,
            semantic_passed=None,
            semantic_detail=sem_detail,
            consensus_passed=rule_passed,
            kappa=None,
            agreement=True,
            tiebreak_used=False,
            tiebreak_detail={},
            method="rule_only",
        )
    if rule_passed is None:
        return DualJudgeResult(
            rule_passed=None,
            rule_detail=rule_detail,
            semantic_passed=sem_passed,
            semantic_detail=sem_detail,
            consensus_passed=sem_passed,
            kappa=None,
            agreement=True,
            tiebreak_used=False,
            tiebreak_detail={},
            method="semantic_only",
        )

    # Both sides are bools — compute per-item κ (approximation).
    kappa = compute_kappa([rule_passed], [sem_passed])
    agreement = rule_passed == sem_passed
    threshold = _get_kappa_threshold()

    if agreement:
        return DualJudgeResult(
            rule_passed=rule_passed,
            rule_detail=rule_detail,
            semantic_passed=sem_passed,
            semantic_detail=sem_detail,
            consensus_passed=rule_passed,
            kappa=kappa,
            agreement=True,
            tiebreak_used=False,
            tiebreak_detail={},
            method="dual_agree",
        )

    # Disagreement: if κ below threshold, invoke transparent_judge tiebreaker.
    tiebreak_detail: dict = {}
    consensus: bool | None = rule_passed
    tiebreak_used = False
    if kappa is None or kappa < threshold:
        tb_passed, tb_detail = _run_tiebreak(judge_method, response_text, params)
        tiebreak_used = True
        tiebreak_detail = tb_detail or {}
        if tb_passed is not None:
            consensus = tb_passed
        else:
            # Tiebreaker declined; fall back to rule judge.
            consensus = rule_passed

    return DualJudgeResult(
        rule_passed=rule_passed,
        rule_detail=rule_detail,
        semantic_passed=sem_passed,
        semantic_detail=sem_detail,
        consensus_passed=consensus,
        kappa=kappa,
        agreement=False,
        tiebreak_used=tiebreak_used,
        tiebreak_detail=tiebreak_detail,
        method="dual_tiebreak" if tiebreak_used else "dual_agree",
    )


@dataclass
class KappaAccumulator:
    """Accumulates judge votes across a run for overall κ computation."""
    rule_votes: list[bool] = field(default_factory=list)
    semantic_votes: list[bool] = field(default_factory=list)

    def add(self, rule: bool, semantic: bool) -> None:
        self.rule_votes.append(bool(rule))
        self.semantic_votes.append(bool(semantic))

    def add_result(self, result: DualJudgeResult) -> None:
        """Convenience: add from a DualJudgeResult if both votes are present."""
        if isinstance(result.rule_passed, bool) and isinstance(result.semantic_passed, bool):
            self.add(result.rule_passed, result.semantic_passed)

    def kappa(self) -> float | None:
        """Return κ if ≥4 paired samples recorded, else None."""
        if len(self.rule_votes) < 4:
            return None
        return compute_kappa(self.rule_votes, self.semantic_votes)

    def __len__(self) -> int:
        return len(self.rule_votes)

    def summary(self) -> dict[str, Any]:
        return {
            "n": len(self.rule_votes),
            "kappa": self.kappa(),
            "rule_pass_rate": (
                sum(self.rule_votes) / len(self.rule_votes)
                if self.rule_votes else None
            ),
            "semantic_pass_rate": (
                sum(self.semantic_votes) / len(self.semantic_votes)
                if self.semantic_votes else None
            ),
        }


__all__ = [
    "DualJudgeResult",
    "KappaAccumulator",
    "dual_judge",
    "compute_kappa",
]
