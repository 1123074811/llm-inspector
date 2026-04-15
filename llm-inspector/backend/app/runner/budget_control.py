"""
runner/budget_control.py — SmartBudget, SmartModeStrategy, TokenBudgetGuard

Token budget allocation and consumption tracking.
Extracted from orchestrator.py to keep individual files under ~250 lines.
"""
from __future__ import annotations

import threading

from app.core.schemas import TestCase, CaseResult
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Model family discriminators for smart mode
FAMILY_DISCRIMINATORS: dict[str, list[str]] = {
    "openai": ["reason_001", "code_001"],
    "anthropic": ["refusal_001", "reason_001"],
    "google": ["code_001", "instr_004"],
    "deepseek": ["reason_001", "reason_candy_001"],
    "alibaba": ["instr_token_001", "style_004"],
    "zhipu": ["instr_token_001", "antispoof_001"],
    "meta": ["refusal_001", "code_001"],
    "mistral": ["instr_004", "reason_001"],
}


class SmartBudget:
    def __init__(
        self,
        token_budget: int,
        phase1_size: int,
        phase2_size: int,
        phase3_size: int,
        case_filter: list[str] | None,
        description: str,
    ):
        self.token_budget = token_budget
        self.phase1_size = phase1_size
        self.phase2_size = phase2_size
        self.phase3_size = phase3_size
        self.case_filter = case_filter
        self.description = description


class SmartModeStrategy:
    def decide_budget(self, pre_result) -> SmartBudget:
        conf = pre_result.confidence
        identified = pre_result.identified_as or ""

        if conf >= 0.90:
            return SmartBudget(
                token_budget=8_000,
                phase1_size=6,
                phase2_size=0,
                phase3_size=0,
                case_filter=self._confirmation_cases(identified),
                description="High confidence verification mode",
            )
        elif conf >= 0.70:
            return SmartBudget(
                token_budget=15_000,
                phase1_size=10,
                phase2_size=4,
                phase3_size=0,
                case_filter=self._discriminative_cases(identified),
                description="Targeted discrimination mode",
            )
        elif conf >= 0.50:
            return SmartBudget(
                token_budget=25_000,
                phase1_size=12,
                phase2_size=8,
                phase3_size=4,
                case_filter=None,
                description="Standard detection mode",
            )
        else:
            return SmartBudget(
                token_budget=35_000,
                phase1_size=14,
                phase2_size=10,
                phase3_size=6,
                case_filter=None,
                description="Full detection mode",
            )

    def _confirmation_cases(self, model_family: str) -> list[str]:
        common = [
            "antispoof_001", "antispoof_002",
            "instr_001", "refusal_001",
        ]
        family_specific = FAMILY_DISCRIMINATORS.get(model_family.lower(), [])[:2]
        return common + family_specific

    def _discriminative_cases(self, model_family: str) -> list[str]:
        common = [
            "antispoof_001", "antispoof_002",
            "instr_001", "instr_004",
            "refusal_001", "reason_001",
            "sys_001", "consist_001",
        ]
        family_specific = FAMILY_DISCRIMINATORS.get(model_family.lower(), [])[:2]
        return common + family_specific


# ── Token Budget Guard ─────────────────────────────────────────────────────────

class TokenBudgetGuard:
    """
    Tracks cumulative token consumption during a run and gates
    low-value cases when the budget is exhausted.

    v6: Estimates remaining budget using historical consumption from past runs
    with the same model/base_url combination.
    """

    def __init__(self, budget: int, model_name: str = "", base_url: str = ""):
        self._budget = budget
        self._used = 0
        self._lock = threading.Lock()
        self._model_name = model_name
        self._base_url = base_url
        self._historical_median = self._load_historical_median()

    def _load_historical_median(self) -> int | None:
        """v6: Load median token consumption from historical runs."""
        if not self._model_name or not self._base_url:
            return None
        try:
            from app.repository.repo import get_repository
            repo = get_repository()
            # Get last 20 runs with same model/base_url
            history = repo.get_runs_by_model_base(
                self._model_name, self._base_url, limit=20, status="completed"
            )
            if not history:
                return None
            # Extract total token consumption
            consumptions = []
            for run in history:
                if isinstance(run, dict) and run.get("total_tokens"):
                    consumptions.append(run["total_tokens"])
            if len(consumptions) >= 3:
                sorted_consumptions = sorted(consumptions)
                return sorted_consumptions[len(sorted_consumptions) // 2]  # median
        except Exception:
            pass
        return None

    def estimate_tokens_needed(self, cases_count: int) -> int:
        """v6: Estimate tokens needed based on historical median or conservative default."""
        if self._historical_median:
            # Use historical median per case, with 20% buffer
            return int((self._historical_median / max(1, cases_count)) * cases_count * 1.2)
        # Conservative default: 1000 tokens per case
        return cases_count * 1000

    def consume(self, tokens: int) -> bool:
        """Returns False when budget is exhausted."""
        with self._lock:
            self._used += tokens
            return self._used <= self._budget

    def should_run_case(self, case: TestCase) -> bool:
        """v6: Decide whether to run a case based on remaining budget and case priority."""
        # Always run high-priority/anchor cases
        meta = case.params.get("_meta", {}) if case.params else {}
        if meta.get("anchor") or case.weight >= 2.0:
            return True
        # Check if we have enough budget for this case (estimated)
        est_cost = case.max_tokens * max(1, case.n_samples) * 2  # rough estimate
        return self.remaining >= est_cost

    def record_result(self, result: CaseResult) -> int:
        """Extract tokens from a CaseResult and add to running total. Returns tokens consumed."""
        tokens = 0
        for sample in result.samples:
            u = sample.response.usage_total_tokens
            if u:
                tokens += u
        self.consume(tokens)
        return tokens

    @property
    def remaining(self) -> int:
        return max(0, self._budget - self._used)

    @property
    def budget(self) -> int:
        return self._budget

    @property
    def used(self) -> int:
        return self._used
