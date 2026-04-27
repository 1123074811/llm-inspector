"""
validation/discrimination_audit.py — v16 Phase 11: Discrimination Validity Audit

Ensures the scoring system actually discriminates between different models.
Prevents systematic downward bias (asymmetric snap-down).

References:
    Cronbach & Meehl 1955, Psychological Bulletin 52(4)
    Cohen 1988, Statistical Power Analysis for the Behavioral Sciences
    Landis & Koch 1977, Biometrics 33(1)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.core.logging import get_logger

logger = get_logger(__name__)

# Validation thresholds (from SOURCES.yaml)
SPEARMAN_MIN = 0.70
DISCRIMINATION_INDEX_MIN = 2.0
TEST_RETEST_KAPPA_MIN = 0.75
HIGH_RISK_RATE_ON_OFFICIAL_MAX = 0.10


@dataclass
class DiscriminationReport:
    """Discrimination validity audit report."""
    spearman_rho_vs_arena: float = 0.0
    kendall_tau_vs_arena: float = 0.0
    spearman_rho_vs_mmlu_pro: float = 0.0
    discrimination_index: float = 0.0  # σ_between / σ_within
    test_retest_kappa: float = 0.0
    verdict_distribution: dict[str, float] = field(default_factory=dict)
    cohort_size: int = 0
    last_audited_at: str = ""
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "spearman_rho_vs_arena": round(self.spearman_rho_vs_arena, 4),
            "kendall_tau_vs_arena": round(self.kendall_tau_vs_arena, 4),
            "spearman_rho_vs_mmlu_pro": round(self.spearman_rho_vs_mmlu_pro, 4),
            "discrimination_index": round(self.discrimination_index, 4),
            "test_retest_kappa": round(self.test_retest_kappa, 4),
            "verdict_distribution": self.verdict_distribution,
            "cohort_size": self.cohort_size,
            "last_audited_at": self.last_audited_at,
            "failures": self.failures,
        }


def compute_spearman(x: list[float], y: list[float]) -> float:
    """Compute Spearman rank correlation (no scipy dependency)."""
    if len(x) != len(y) or len(x) < 3:
        return 0.0

    def rank(vals: list[float]) -> list[float]:
        n = len(vals)
        indexed = sorted(enumerate(vals), key=lambda p: p[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and indexed[j + 1][1] == indexed[j][1]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[indexed[k][0]] = avg_rank
            i = j + 1
        return ranks

    rx = rank(x)
    ry = rank(y)
    n = len(x)
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    rho = 1.0 - 6.0 * d2 / (n * (n * n - 1))
    return max(-1.0, min(1.0, rho))


def compute_kappa(verdicts_a: list[str], verdicts_b: list[str]) -> float:
    """Compute Cohen's kappa for test-retest consistency."""
    if len(verdicts_a) != len(verdicts_b) or not verdicts_a:
        return 0.0

    categories = sorted(set(verdicts_a + verdicts_b))
    n = len(verdicts_a)
    cat_idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)

    # Observed agreement
    observed = sum(1 for a, b in zip(verdicts_a, verdicts_b) if a == b)
    po = observed / n

    # Expected agreement
    pa = [0] * k
    pb = [0] * k
    for a, b in zip(verdicts_a, verdicts_b):
        pa[cat_idx[a]] += 1
        pb[cat_idx[b]] += 1
    pe = sum(pa[i] * pb[i] for i in range(k)) / (n * n)

    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def compute_discrimination_index(
    model_scores: dict[str, list[float]],
) -> float:
    """
    Compute discrimination index: σ_between / σ_within.

    Higher values mean the system better distinguishes different models.
    """
    if len(model_scores) < 2:
        return 0.0

    all_scores = []
    for scores in model_scores.values():
        all_scores.extend(scores)

    if not all_scores:
        return 0.0

    grand_mean = sum(all_scores) / len(all_scores)

    # Between-model variance
    model_means = {}
    for model, scores in model_scores.items():
        if scores:
            model_means[model] = sum(scores) / len(scores)

    n_models = len(model_means)
    if n_models < 2:
        return 0.0

    ss_between = sum(
        len(model_scores[m]) * (model_means[m] - grand_mean) ** 2
        for m in model_means
    )
    var_between = ss_between / (n_models - 1) if n_models > 1 else 0.0

    # Within-model variance
    ss_within = 0.0
    n_within = 0
    for model, scores in model_scores.items():
        mean = model_means.get(model, 0)
        for s in scores:
            ss_within += (s - mean) ** 2
            n_within += 1

    var_within = ss_within / max(1, n_within - n_models)

    if var_within <= 0:
        return float('inf') if var_between > 0 else 0.0

    return var_between / var_within


def run_discrimination_audit(
    model_scores: dict[str, list[float]],
    arena_elo: dict[str, float] | None = None,
    mmlu_pro_scores: dict[str, float] | None = None,
    verdicts_by_run: dict[str, list[str]] | None = None,
) -> DiscriminationReport:
    """
    Run discrimination validity audit.

    Args:
        model_scores: {model_name: [score1, score2, ...]} per run.
        arena_elo: {model_name: arena_elo} ground truth.
        mmlu_pro_scores: {model_name: mmlu_pro_score} ground truth.
        verdicts_by_run: {model_name: [verdict1, verdict2, ...]} for kappa.

    Returns:
        DiscriminationReport with all metrics and failure list.
    """
    failures: list[str] = []

    # Spearman vs Arena ELO
    rho_arena = 0.0
    tau_arena = 0.0
    if arena_elo:
        common = [m for m in model_scores if m in arena_elo]
        if len(common) >= 3:
            inspector_means = [sum(model_scores[m]) / len(model_scores[m]) for m in common]
            elo_vals = [arena_elo[m] for m in common]
            rho_arena = compute_spearman(inspector_means, elo_vals)
            if rho_arena < SPEARMAN_MIN:
                failures.append(f"spearman_rho_vs_arena={rho_arena:.3f} < {SPEARMAN_MIN}")

    # Spearman vs MMLU-Pro
    rho_mmlu = 0.0
    if mmlu_pro_scores:
        common = [m for m in model_scores if m in mmlu_pro_scores]
        if len(common) >= 3:
            inspector_means = [sum(model_scores[m]) / len(model_scores[m]) for m in common]
            mmlu_vals = [mmlu_pro_scores[m] for m in common]
            rho_mmlu = compute_spearman(inspector_means, mmlu_vals)

    # Discrimination index
    disc_idx = compute_discrimination_index(model_scores)
    if disc_idx < DISCRIMINATION_INDEX_MIN:
        failures.append(f"discrimination_index={disc_idx:.2f} < {DISCRIMINATION_INDEX_MIN}")

    # Test-retest kappa
    kappa = 0.0
    if verdicts_by_run:
        models_with_3plus = {m: v for m, v in verdicts_by_run.items() if len(v) >= 3}
        if len(models_with_3plus) >= 2:
            kappas = []
            for m, v in models_with_3plus.items():
                # Compare first half vs second half
                mid = len(v) // 2
                a = v[:mid]
                b = v[mid:mid + len(a)]
                if len(a) == len(b) and len(a) >= 2:
                    kappas.append(compute_kappa(a, b))
            if kappas:
                kappa = sum(kappas) / len(kappas)
                if kappa < TEST_RETEST_KAPPA_MIN:
                    failures.append(f"test_retest_kappa={kappa:.3f} < {TEST_RETEST_KAPPA_MIN}")

    # Verdict distribution
    verdict_dist: dict[str, float] = {}
    if verdicts_by_run:
        all_verdicts = []
        for v in verdicts_by_run.values():
            all_verdicts.extend(v)
        if all_verdicts:
            for cat in ["trusted", "suspicious", "high_risk", "fake", "inconclusive"]:
                count = all_verdicts.count(cat)
                verdict_dist[cat] = count / len(all_verdicts)

    return DiscriminationReport(
        spearman_rho_vs_arena=rho_arena,
        kendall_tau_vs_arena=tau_arena,
        spearman_rho_vs_mmlu_pro=rho_mmlu,
        discrimination_index=disc_idx,
        test_retest_kappa=kappa,
        verdict_distribution=verdict_dist,
        cohort_size=len(model_scores),
        last_audited_at=datetime.now(timezone.utc).isoformat(),
        failures=failures,
    )
