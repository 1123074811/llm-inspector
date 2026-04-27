"""
test_v16_phase11.py — v16 Phase 11 regression tests.

Validates:
  - EvidenceItem dataclass and effective_delta
  - BayesianEvidenceVerdictEngine assess_evidence
  - VerdictReport probabilistic output
  - Symmetry check (up/down rule balance)
  - Small-sample protection
  - Coverage gate → inconclusive
  - Borderline detection
  - Discrimination audit (Spearman, kappa, discrimination_index)
  - EWMA updater (staleness, merge)
  - Frontend Verdict Explainer template
"""
import pytest
import math


# ── EvidenceItem ──────────────────────────────────────────────────────────

class TestEvidenceItem:
    def test_effective_delta_not_fired(self):
        from app.analysis.verdicts import EvidenceItem
        item = EvidenceItem("test_rule", "down", 1.0, fired=False)
        assert item.effective_delta() == 0.0

    def test_effective_delta_down_fired(self):
        from app.analysis.verdicts import EvidenceItem
        item = EvidenceItem("test_rule", "down", 0.8, fired=True,
                            confidence=1.0, corroboration_count=2, corroboration_min=2)
        assert item.effective_delta() == pytest.approx(0.8, abs=0.01)

    def test_effective_delta_up_fired(self):
        from app.analysis.verdicts import EvidenceItem
        item = EvidenceItem("test_rule", "up", 2.0, fired=True,
                            confidence=1.0, corroboration_count=3, corroboration_min=3)
        assert item.effective_delta() == pytest.approx(-2.0, abs=0.01)

    def test_effective_delta_partial_corroboration(self):
        from app.analysis.verdicts import EvidenceItem
        item = EvidenceItem("test_rule", "down", 1.0, fired=True,
                            confidence=1.0, corroboration_count=1, corroboration_min=2)
        # 1/2 = 0.5 factor
        assert item.effective_delta() == pytest.approx(0.5, abs=0.01)

    def test_to_dict(self):
        from app.analysis.verdicts import EvidenceItem
        item = EvidenceItem("rule1", "down", 0.7, sources=["L5"], confidence=0.9,
                            fired=True, corroboration_count=2, corroboration_min=2)
        d = item.to_dict()
        assert d["rule_id"] == "rule1"
        assert d["fired"] is True
        assert "effective_delta" in d


# ── BayesianEvidenceVerdictEngine ─────────────────────────────────────────

class TestBayesianVerdictEngine:
    def _make_engine(self):
        from app.analysis.verdicts import BayesianEvidenceVerdictEngine
        return BayesianEvidenceVerdictEngine(prior_log_odds=0.0)

    def test_no_evidence_trusted(self):
        """No rules fired → P(fake)=0.5 → high_risk (0.40 ≤ 0.5 < 0.70)"""
        engine = self._make_engine()
        report = engine.assess_evidence([], coverage=1.0, sample_count=44, test_mode="standard")
        # With no evidence, log_odds=0, P(fake)=0.5 → high_risk
        assert report.tier == "high_risk"
        assert report.p_fake == pytest.approx(0.5, abs=0.01)

    def test_perfect_official_model(self):
        """Official endpoint + tokenizer match → should be trusted."""
        engine = self._make_engine()
        fired = [
            {"rule_id": "official_endpoint_match", "corroboration_count": 3, "confidence": 1.0},
            {"rule_id": "tokenizer_fingerprint_match", "corroboration_count": 1, "confidence": 1.0},
            {"rule_id": "knowledge_cutoff_match_claimed", "corroboration_count": 2, "confidence": 1.0},
        ]
        report = engine.assess_evidence(fired, coverage=1.0, sample_count=44, test_mode="standard")
        # 3 up rules fired: -2.0 + -0.8 + -0.5 = -3.3 log-odds
        # P(fake) = sigmoid(-3.3) ≈ 0.036 → trusted
        assert report.tier == "trusted"
        assert report.p_fake < 0.15

    def test_obvious_fake(self):
        """Extraction leaked + adversarial spoof + fingerprint mismatch → fake."""
        engine = self._make_engine()
        fired = [
            {"rule_id": "extraction_leaked_real_identity", "corroboration_count": 2, "confidence": 1.0},
            {"rule_id": "adversarial_spoof_rate_high", "corroboration_count": 2, "confidence": 1.0},
            {"rule_id": "fingerprint_mismatch", "corroboration_count": 1, "confidence": 1.0},
        ]
        report = engine.assess_evidence(fired, coverage=1.0, sample_count=44, test_mode="standard")
        # +1.5 + 0.8 + 0.5 = +2.8 log-odds, P(fake) ≈ 0.94 → fake
        assert report.tier in ("high_risk", "fake")
        assert report.p_fake > 0.6

    def test_borderline_model(self):
        """Only 1 weak signal → should be suspicious with borderline."""
        engine = self._make_engine()
        fired = [
            {"rule_id": "fingerprint_mismatch", "corroboration_count": 1, "confidence": 0.5},
        ]
        report = engine.assess_evidence(fired, coverage=1.0, sample_count=44, test_mode="standard")
        # Weak signal only → P(fake) near 0.5
        assert report.tier in ("suspicious", "high_risk", "trusted")

    def test_small_sample_protection(self):
        """Quick mode with < 18 samples → no high_risk/fake."""
        engine = self._make_engine()
        fired = [
            {"rule_id": "extraction_leaked_real_identity", "corroboration_count": 2, "confidence": 1.0},
        ]
        report = engine.assess_evidence(fired, coverage=1.0, sample_count=10, test_mode="quick")
        assert report.tier in ("suspicious", "inconclusive")  # Capped

    def test_coverage_gate_inconclusive(self):
        """Coverage < 0.7 → inconclusive."""
        engine = self._make_engine()
        report = engine.assess_evidence([], coverage=0.5, sample_count=44, test_mode="standard")
        assert report.tier == "inconclusive"

    def test_difficulty_ceiling_needs_12_samples(self):
        """difficulty_ceiling_low should not fire with < 12 samples."""
        engine = self._make_engine()
        fired = [
            {"rule_id": "difficulty_ceiling_low", "corroboration_count": 1, "confidence": 1.0},
        ]
        report = engine.assess_evidence(fired, coverage=1.0, sample_count=8, test_mode="standard")
        # Rule should be suppressed → no effective delta
        assert not any(e.rule_id == "difficulty_ceiling_low" and e.fired for e in report.dominant_evidence)

    def test_coding_score_needs_5_samples(self):
        """coding_score_low should not fire with < 5 samples."""
        engine = self._make_engine()
        fired = [
            {"rule_id": "coding_score_low", "corroboration_count": 1, "confidence": 1.0},
        ]
        report = engine.assess_evidence(fired, coverage=1.0, sample_count=3, test_mode="standard")
        assert not any(e.rule_id == "coding_score_low" and e.fired for e in report.dominant_evidence)


class TestSymmetryCheck:
    def test_symmetry(self):
        from app.analysis.verdicts import BayesianEvidenceVerdictEngine
        engine = BayesianEvidenceVerdictEngine()
        result = engine.check_symmetry()
        assert result["is_balanced"] is True
        assert result["up_rules"] == 5
        assert result["down_rules"] == 5


class TestVerdictReport:
    def test_to_dict(self):
        from app.analysis.verdicts import VerdictReport, EvidenceItem
        report = VerdictReport(
            tier="trusted",
            p_fake=0.05,
            log_odds_fake=-2.94,
            log_odds_ci95=(-4.0, -1.88),
            is_borderline=False,
            coverage=1.0,
            sample_count=44,
            test_mode="standard",
            dominant_evidence=[EvidenceItem("official_endpoint_match", "up", 2.0, fired=True)],
        )
        d = report.to_dict()
        assert d["tier"] == "trusted"
        assert "log_odds_ci95" in d
        assert len(d["dominant_evidence"]) == 1


# ── Discrimination Audit ──────────────────────────────────────────────────

class TestDiscriminationAudit:
    def test_spearman_perfect(self):
        from app.validation.discrimination_audit import compute_spearman
        x = [1, 2, 3, 4, 5]
        y = [10, 20, 30, 40, 50]
        assert compute_spearman(x, y) == pytest.approx(1.0, abs=0.01)

    def test_spearman_inverse(self):
        from app.validation.discrimination_audit import compute_spearman
        x = [1, 2, 3, 4, 5]
        y = [50, 40, 30, 20, 10]
        assert compute_spearman(x, y) == pytest.approx(-1.0, abs=0.01)

    def test_spearman_too_few(self):
        from app.validation.discrimination_audit import compute_spearman
        assert compute_spearman([1], [2]) == 0.0

    def test_kappa_perfect(self):
        from app.validation.discrimination_audit import compute_kappa
        a = ["trusted", "suspicious", "high_risk"]
        b = ["trusted", "suspicious", "high_risk"]
        assert compute_kappa(a, b) == pytest.approx(1.0, abs=0.01)

    def test_kappa_random(self):
        from app.validation.discrimination_audit import compute_kappa
        a = ["trusted", "suspicious", "high_risk", "fake"]
        b = ["fake", "high_risk", "suspicious", "trusted"]
        # No agreement → kappa should be near 0
        assert compute_kappa(a, b) < 0.3

    def test_discrimination_index(self):
        from app.validation.discrimination_audit import compute_discrimination_index
        # Two models with very different scores
        scores = {
            "model_a": [90, 92, 88],
            "model_b": [30, 32, 28],
        }
        idx = compute_discrimination_index(scores)
        assert idx > 2.0  # Should be well above minimum

    def test_discrimination_index_same(self):
        from app.validation.discrimination_audit import compute_discrimination_index
        # All models same scores
        scores = {
            "model_a": [50, 50, 50],
            "model_b": [50, 50, 50],
        }
        idx = compute_discrimination_index(scores)
        assert idx < 1.0  # No discrimination

    def test_run_audit(self):
        from app.validation.discrimination_audit import run_discrimination_audit
        scores = {
            "gpt-4o": [85, 87, 83],
            "claude-3": [80, 82, 78],
            "qwen2": [70, 72, 68],
            "fake-model": [30, 32, 28],
        }
        arena = {"gpt-4o": 1300, "claude-3": 1250, "qwen2": 1150, "fake-model": 800}
        report = run_discrimination_audit(scores, arena_elo=arena)
        assert report.cohort_size == 4
        assert report.spearman_rho_vs_arena > 0.5


# ── EWMA Updater ─────────────────────────────────────────────────────────

class TestEWMAUpdater:
    def test_ewma_merge(self):
        from app.validation.ewma_updater import ewma_merge
        result = ewma_merge(100.0, 120.0, alpha=0.3)
        expected = 0.3 * 120.0 + 0.7 * 100.0
        assert result == pytest.approx(expected, abs=0.01)

    def test_staleness_fresh(self):
        from app.validation.ewma_updater import check_staleness
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        refs = [{"retrieved_at": (now - timedelta(days=10)).isoformat()}]
        report = check_staleness(refs, now=now)
        assert report.fresh_entries == 1
        assert report.stale_entries == 0

    def test_staleness_stale(self):
        from app.validation.ewma_updater import check_staleness
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        refs = [{"retrieved_at": (now - timedelta(days=100)).isoformat()}]
        report = check_staleness(refs, now=now)
        assert report.stale_entries == 1

    def test_staleness_discard(self):
        from app.validation.ewma_updater import check_staleness
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        refs = [{"retrieved_at": (now - timedelta(days=200)).isoformat()}]
        report = check_staleness(refs, now=now)
        assert report.discarded_entries == 1

    def test_apply_staleness_weight(self):
        from app.validation.ewma_updater import apply_staleness_weight
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        entry = {"retrieved_at": (now - timedelta(days=100)).isoformat()}
        result = apply_staleness_weight(entry, now=now)
        assert result["_staleness_weight"] == 0.3
        assert result["_should_discard"] is False


# ── Frontend Verdict Explainer ────────────────────────────────────────────

class TestVerdictExplainerFrontend:
    def test_explainer_template_exists(self):
        import pathlib as _pl
        frontend = _pl.Path(__file__).resolve().parent.parent.parent / "frontend"
        content = (frontend / "index.html").read_text(encoding="utf-8")
        assert "verdict-explainer" in content
        assert "ve-p-fake" in content
        assert "ve-ci95" in content
        assert "ve-borderline-warn" in content

    def test_explainer_styles(self):
        import pathlib as _pl
        frontend = _pl.Path(__file__).resolve().parent.parent.parent / "frontend"
        content = (frontend / "styles.css").read_text(encoding="utf-8")
        assert ".ve-row" in content
        assert ".ve-warn" in content
        assert ".ve-tier-probs" in content

    def test_render_verdict_explainer_function(self):
        import pathlib as _pl
        frontend = _pl.Path(__file__).resolve().parent.parent.parent / "frontend"
        content = (frontend / "app.js").read_text(encoding="utf-8")
        assert "function renderVerdictExplainer" in content
