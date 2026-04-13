"""
Phase 1 Integration Tests — v11 Circuit Breaker + Tracing + EvalTestCase.

Tests the three core modules and their integration with the orchestrator.
"""
import pytest
import time
import threading

# ── Circuit Breaker Tests ────────────────────────────────────────────────────

from app.core.circuit_breaker import CircuitBreaker, CircuitState, circuit_breaker


class TestCircuitBreaker:
    """Unit tests for the CircuitBreaker."""

    def setup_method(self):
        """Create a fresh breaker for each test."""
        self.cb = CircuitBreaker(failure_threshold=3, recovery_timeout_sec=0.5, success_threshold=2)
        self.url = "https://api.test.example.com/v1"

    def test_initial_state_is_closed(self):
        assert not self.cb.is_open(self.url)
        metrics = self.cb.get_metrics(self.url)
        assert metrics["state"] == "closed"

    def test_opens_after_threshold_failures(self):
        for i in range(3):
            self.cb.record_failure(self.url, f"error_{i}")
        assert self.cb.is_open(self.url)
        metrics = self.cb.get_metrics(self.url)
        assert metrics["state"] == "open"
        assert metrics["consecutive_failures"] == 3

    def test_does_not_open_below_threshold(self):
        self.cb.record_failure(self.url, "error_1")
        self.cb.record_failure(self.url, "error_2")
        assert not self.cb.is_open(self.url)

    def test_success_resets_consecutive_failures(self):
        self.cb.record_failure(self.url, "error_1")
        self.cb.record_failure(self.url, "error_2")
        self.cb.record_success(self.url)
        # Consecutive failures should be reset
        metrics = self.cb.get_metrics(self.url)
        assert metrics["consecutive_failures"] == 0
        assert metrics["consecutive_successes"] == 1

    def test_half_open_after_recovery_timeout(self):
        # Open the circuit
        for i in range(3):
            self.cb.record_failure(self.url, f"error_{i}")
        assert self.cb.is_open(self.url)

        # Wait for recovery timeout
        time.sleep(0.6)

        # Should transition to half-open and allow a probe
        assert not self.cb.is_open(self.url)
        metrics = self.cb.get_metrics(self.url)
        assert metrics["state"] == "half_open"

    def test_half_open_probe_failure_reopens(self):
        # Open the circuit
        for i in range(3):
            self.cb.record_failure(self.url, f"error_{i}")

        # Wait for recovery
        time.sleep(0.6)
        self.cb.is_open(self.url)  # Triggers half-open

        # Probe fails
        self.cb.record_failure(self.url, "probe_failed")
        metrics = self.cb.get_metrics(self.url)
        assert metrics["state"] == "open"

    def test_half_open_success_closes_after_threshold(self):
        # Open the circuit
        for i in range(3):
            self.cb.record_failure(self.url, f"error_{i}")

        # Wait for recovery
        time.sleep(0.6)
        self.cb.is_open(self.url)  # Triggers half-open

        # Enough successes to close
        self.cb.record_success(self.url)
        self.cb.record_success(self.url)
        metrics = self.cb.get_metrics(self.url)
        assert metrics["state"] == "closed"

    def test_reset_specific_endpoint(self):
        self.cb.record_failure("https://a.com/v1", "err")
        self.cb.record_failure("https://b.com/v1", "err")

        self.cb.reset("https://a.com/v1")
        assert self.cb.get_metrics("https://a.com/v1") == {}
        assert self.cb.get_metrics("https://b.com/v1") != {}

    def test_reset_all(self):
        self.cb.record_failure("https://a.com/v1", "err")
        self.cb.record_failure("https://b.com/v1", "err")
        self.cb.reset()
        assert self.cb.get_metrics("https://a.com/v1") == {}
        assert self.cb.get_metrics("https://b.com/v1") == {}

    def test_stats_property(self):
        self.cb.record_failure("https://a.com/v1", "err")
        for i in range(3):
            self.cb.record_failure("https://b.com/v1", f"err_{i}")

        stats = self.cb.stats
        assert stats["total_endpoints"] == 2
        assert stats["open"] == 1
        assert stats["closed"] == 1

    def test_failure_rate(self):
        self.cb.record_success(self.url)
        self.cb.record_success(self.url)
        self.cb.record_failure(self.url, "err")
        metrics = self.cb.get_metrics(self.url)
        assert metrics["failure_rate"] == pytest.approx(1/3, abs=0.01)

    def test_per_endpoint_isolation(self):
        url_a = "https://a.com/v1"
        url_b = "https://b.com/v1"

        # Open circuit for A only
        for i in range(3):
            self.cb.record_failure(url_a, f"err_{i}")

        assert self.cb.is_open(url_a)
        assert not self.cb.is_open(url_b)

    def test_thread_safety(self):
        """Rapid concurrent failures from multiple threads."""
        errors = []

        def spam_failures(url, count):
            try:
                for _ in range(count):
                    self.cb.record_failure(url, "concurrent_err")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=spam_failures, args=(f"https://t{i}.com/v1", 50))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread safety errors: {errors}"
        stats = self.cb.stats
        assert stats["total_endpoints"] == 5


# ── Tracer Tests ─────────────────────────────────────────────────────────────

from app.core.tracer import PipelineTracer, get_tracer, remove_tracer


class TestPipelineTracer:
    """Unit tests for the PipelineTracer."""

    def test_basic_span_timing(self):
        tracer = PipelineTracer("test-run-1")
        tracer.start()

        with tracer.span("predetect") as span:
            span.set_attribute("confidence", 0.85)
            time.sleep(0.05)

        trace = tracer.finish()
        assert len(trace.spans) == 1
        assert trace.spans[0].name == "predetect"
        assert trace.spans[0].duration_ms > 40
        assert trace.spans[0].status == "ok"
        assert trace.spans[0].attributes["confidence"] == 0.85

    def test_multiple_spans(self):
        tracer = PipelineTracer("test-run-2")
        tracer.start()

        with tracer.span("predetect"):
            pass
        with tracer.span("connectivity"):
            pass
        with tracer.span("phase1"):
            pass

        trace = tracer.finish()
        assert len(trace.spans) == 3
        assert [s.name for s in trace.spans] == ["predetect", "connectivity", "phase1"]

    def test_span_error_status(self):
        tracer = PipelineTracer("test-run-3")

        with pytest.raises(ValueError):
            with tracer.span("failing") as span:
                raise ValueError("test error")

        trace = tracer.finish()
        assert trace.spans[0].status == "error"
        assert "test error" in trace.spans[0].attributes["error"]

    def test_token_accounting(self):
        tracer = PipelineTracer("test-run-4")
        tracer.start()

        with tracer.span("predetect"):
            tracer.record_tokens("predetect", 500)

        with tracer.span("phase1"):
            tracer.record_tokens("phase1", 2000)

        trace = tracer.finish()
        assert trace.total_tokens == 2500
        assert trace.spans[0].attributes.get("tokens_used") == 500

    def test_span_events(self):
        tracer = PipelineTracer("test-run-5")

        with tracer.span("phase1") as span:
            span.add_event("case_completed", case_id="c1", pass_rate=0.9)
            span.add_event("case_completed", case_id="c2", pass_rate=0.7)

        trace = tracer.finish()
        assert len(trace.spans[0].events) == 2
        assert trace.spans[0].events[0].name == "case_completed"

    def test_get_progress(self):
        tracer = PipelineTracer("test-run-6")
        tracer.start()

        with tracer.span("predetect"):
            progress = tracer.get_progress()

        assert "predetect" in progress["active_spans"]
        assert len(progress["completed_spans"]) == 0

    def test_trace_serialization(self):
        tracer = PipelineTracer("test-run-7")
        tracer.start()

        with tracer.span("predetect") as span:
            span.set_attribute("identified_as", "gpt-4")
            tracer.record_tokens("predetect", 100)

        trace = tracer.finish()
        d = trace.to_dict()

        assert d["run_id"] == "test-run-7"
        assert len(d["spans"]) == 1
        assert "span_summary" in d
        assert "predetect" in d["span_summary"]

    def test_tracer_registry(self):
        tracer = get_tracer("registry-test")
        assert tracer.run_id == "registry-test"

        # Getting same ID returns same tracer
        tracer2 = get_tracer("registry-test")
        assert tracer2 is tracer

        # Remove returns the trace
        result = remove_tracer("registry-test")
        assert result is not None
        assert result.run_id == "registry-test"

        # After removal, getting again creates a new tracer
        tracer3 = get_tracer("registry-test")
        assert tracer3 is not tracer


# ── EvalTestCase Tests ───────────────────────────────────────────────────────

from app.core.eval_schemas import EvalTestCase, SkillVector, BayesianPrior, EvalMeta
from app.core.schemas import TestCase


class TestSkillVector:
    def test_basic_construction(self):
        sv = SkillVector(required={"counterfactual": 1, "syllogism": 0, "analogy": 1})
        assert sv.skill_names == ["counterfactual", "analogy"]
        assert sv.n_skills == 2

    def test_empty(self):
        sv = SkillVector()
        assert sv.skill_names == []
        assert sv.n_skills == 0

    def test_serialization(self):
        sv = SkillVector(required={"reasoning": 1})
        d = sv.to_dict()
        sv2 = SkillVector.from_dict(d)
        assert sv2.required == sv.required

    def test_from_dict_none(self):
        assert SkillVector.from_dict(None) is None
        assert SkillVector.from_dict({}) is None


class TestBayesianPrior:
    def test_default_normal(self):
        bp = BayesianPrior()
        assert bp.distribution == "normal"
        assert bp.mu == 0.0
        assert bp.sigma == 1.0

    def test_beta_prior(self):
        bp = BayesianPrior(distribution="beta", alpha=2.0, beta=5.0)
        d = bp.to_dict()
        bp2 = BayesianPrior.from_dict(d)
        assert bp2.distribution == "beta"
        assert bp2.alpha == 2.0

    def test_from_dict_none(self):
        assert BayesianPrior.from_dict(None) is None


class TestEvalMeta:
    def test_default(self):
        em = EvalMeta()
        assert em.norming_sample_size == 0
        assert em.discriminative_valid is True

    def test_serialization_roundtrip(self):
        em = EvalMeta(
            norming_sample_size=500,
            calibration_version="v11.0",
            discriminative_valid=True,
            validity_flags=["reviewed"],
        )
        d = em.to_dict()
        em2 = EvalMeta.from_dict(d)
        assert em2.norming_sample_size == 500
        assert em2.validity_flags == ["reviewed"]


class TestEvalTestCase:
    def _make_base_test_case(self):
        return TestCase(
            id="tc-001",
            category="reasoning",
            name="Counterfactual Test",
            user_prompt="If it rained, would the ground be wet?",
            expected_type="open_ended",
            judge_method="llm_judge",
        )

    def test_from_test_case(self):
        tc = self._make_base_test_case()
        evc = EvalTestCase.from_test_case(tc, skill_vector=SkillVector(required={"reasoning": 1}))
        assert evc.id == "tc-001"
        assert evc.skill_vector is not None
        assert evc.skill_vector.skill_names == ["reasoning"]

    def test_backward_compatibility(self):
        """EvalTestCase should work anywhere TestCase is expected."""
        evc = EvalTestCase(
            id="tc-002",
            category="logic",
            name="Syllogism",
            user_prompt="All men are mortal...",
            expected_type="open_ended",
            judge_method="keyword",
            skill_vector=SkillVector(required={"syllogism": 1}),
        )
        # Can access base TestCase fields
        assert evc.category == "logic"
        assert evc.user_prompt == "All men are mortal..."

    def test_from_db_dict(self):
        d = {
            "id": "tc-003",
            "category": "reasoning",
            "name": "DB Test Case",
            "user_prompt": "What is 2+2?",
            "expected_type": "exact_match",
            "judge_method": "exact",
            "params": {
                "_meta": {
                    "skill_vector": {"required": {"math": 1}},
                    "bayesian_prior": {"distribution": "normal", "mu": 0.5, "sigma": 0.3},
                    "eval_meta": {"norming_sample_size": 100, "calibration_version": "v11.0"},
                }
            },
        }
        evc = EvalTestCase.from_db_dict(d)
        assert evc.skill_vector is not None
        assert evc.skill_vector.skill_names == ["math"]
        assert evc.bayesian_prior is not None
        assert evc.bayesian_prior.mu == 0.5
        assert evc.eval_meta is not None
        assert evc.eval_meta.norming_sample_size == 100
        assert evc.telemetry_span == "case.tc-003"

    def test_to_dict_includes_extensions(self):
        evc = EvalTestCase(
            id="tc-004",
            category="test",
            name="Serialize Test",
            user_prompt="hello",
            expected_type="open_ended",
            judge_method="keyword",
            skill_vector=SkillVector(required={"reasoning": 1}),
            bayesian_prior=BayesianPrior(mu=0.3),
        )
        d = evc.to_dict()
        assert "skill_vector" in d
        assert "bayesian_prior" in d
        assert d["skill_vector"]["required"] == {"reasoning": 1}
        assert d["bayesian_prior"]["mu"] == 0.3

    def test_has_skills_property(self):
        evc_no = EvalTestCase(id="x", category="t", name="t", user_prompt="t",
                              expected_type="t", judge_method="t")
        assert not evc_no.has_skills

        evc_yes = EvalTestCase(id="x", category="t", name="t", user_prompt="t",
                               expected_type="t", judge_method="t",
                               skill_vector=SkillVector(required={"r": 1}))
        assert evc_yes.has_skills

    def test_has_prior_property(self):
        evc_no = EvalTestCase(id="x", category="t", name="t", user_prompt="t",
                              expected_type="t", judge_method="t")
        assert not evc_no.has_prior

        evc_yes = EvalTestCase(id="x", category="t", name="t", user_prompt="t",
                               expected_type="t", judge_method="t",
                               bayesian_prior=BayesianPrior())
        assert evc_yes.has_prior


# ── Integration: Orchestrator imports work ────────────────────────────────────

class TestOrchestratorV11Imports:
    """Verify that the orchestrator module imports v11 components correctly."""

    def test_orchestrator_imports_eval_schemas(self):
        from app.runner.orchestrator import EvalTestCase as _
        assert _ is not None

    def test_orchestrator_imports_circuit_breaker(self):
        from app.runner.orchestrator import circuit_breaker as cb
        assert cb is not None

    def test_orchestrator_imports_tracer(self):
        from app.runner.orchestrator import get_tracer, remove_tracer
        assert callable(get_tracer)
        assert callable(remove_tracer)


# ── Integration: v11 API Handlers ─────────────────────────────────────────────

class TestV11Handlers:
    """Test v11 API handler functions."""

    def test_circuit_breaker_status_handler(self):
        from app.handlers.v11_handlers import handle_circuit_breaker_status
        # Should return stats for all endpoints
        result = handle_circuit_breaker_status("/api/v1/circuit-breaker", {}, {})
        assert result is not None

    def test_circuit_breaker_reset_handler(self):
        from app.handlers.v11_handlers import handle_circuit_breaker_reset
        result = handle_circuit_breaker_reset("/api/v1/circuit-breaker/reset", {}, {})
        assert result is not None

    def test_run_trace_handler_not_found(self):
        from app.handlers.v11_handlers import handle_run_trace
        result = handle_run_trace("/api/v1/runs/nonexistent/trace", {}, {})
        assert result is not None

    def test_tracer_progress_handler(self):
        from app.handlers.v11_handlers import handle_tracer_progress_all
        result = handle_tracer_progress_all("/api/v1/tracers/progress", {}, {})
        assert result is not None
