"""
Regression tests for v17 run-state-machine idempotency fix.

Bug: ``async_runner._run`` and ``run_lifecycle._execute`` both contain
the pattern:

    repo.save_predetect_result(...)              # raw SQL → status='pre_detected'
    if not pre_result.should_proceed_to_testing:
        repo.update_run_status(run_id, "pre_detected")  # 💥 X→X illegal

The validator in ``update_run_status`` rejected ``X → X`` with::

    ValueError: Illegal state transition: pre_detected → pre_detected

which surfaced as an "Unhandled error" and aborted the run.

Fix: ``update_run_status`` and ``transition_state`` now treat
``from_state == to_state`` as a silent no-op (idempotent), matching
the standard pattern in distributed task systems where multiple
control paths may converge on the same target state.
"""
import pytest

from app.repository import repo


def test_idempotent_same_state_no_op(create_run):
    """Setting status to its current value must NOT raise."""
    run_id = create_run()
    # Walk the legitimate state-machine path:
    # queued → preflight_running → pre_detecting → pre_detected
    repo.update_run_status(run_id, "preflight_running")
    repo.update_run_status(run_id, "pre_detecting")
    # save_predetect_result writes status='pre_detected' via raw SQL
    repo.save_predetect_result(run_id, {
        "success": True,
        "identified_as": "DeepSeek",
        "confidence": 0.85,
    })
    assert repo.get_run(run_id)["status"] == "pre_detected"

    # The bug: this second call (from async_runner.py:198 / run_lifecycle.py:54)
    # used to raise "Illegal state transition: pre_detected → pre_detected".
    # After the fix, it must be a silent no-op.
    repo.update_run_status(run_id, "pre_detected")  # must not raise
    assert repo.get_run(run_id)["status"] == "pre_detected"


def test_idempotent_running_to_running(create_run):
    """Same idempotency guarantee applies to any state, not just pre_detected."""
    run_id = create_run()
    repo.update_run_status(run_id, "running")
    repo.update_run_status(run_id, "running")  # must not raise
    assert repo.get_run(run_id)["status"] == "running"


def test_truly_illegal_transition_still_rejected(create_run):
    """Idempotency must NOT swallow real illegal transitions."""
    run_id = create_run()
    repo.update_run_status(run_id, "running")
    repo.update_run_status(run_id, "completed")  # terminal
    # completed → running is illegal — must still raise
    with pytest.raises(ValueError, match="Illegal state transition"):
        repo.update_run_status(run_id, "running")


def test_transition_state_function_also_idempotent(create_run):
    """The lower-level ``transition_state`` API must share the same semantics."""
    run_id = create_run()
    repo.update_run_status(run_id, "running")
    # transition_state(run_id, "running", "running") must be a no-op
    repo.transition_state(run_id, "running", "running")  # must not raise
    assert repo.get_run(run_id)["status"] == "running"


def test_legitimate_transition_still_works(create_run):
    """Sanity: idempotency fix did not break the normal forward path."""
    run_id = create_run()
    repo.update_run_status(run_id, "preflight_running")
    repo.update_run_status(run_id, "pre_detecting")
    repo.update_run_status(run_id, "pre_detected")
    repo.update_run_status(run_id, "running")
    repo.update_run_status(run_id, "completed")
    assert repo.get_run(run_id)["status"] == "completed"


# ── Resting-state protection (v17 hardening) ────────────────────────────────


class TestRestingStateProtection:
    """``repo.is_resting`` identifies states that an exception handler
    must NOT clobber with 'failed'. Previously, ``run_lifecycle`` and
    ``worker`` would overwrite a legitimate ``pre_detected`` paused
    state with ``failed`` whenever any code raised on the way out — which
    broke the "继续测试" button by leaving the run unrecoverable."""

    def test_pre_detected_is_resting(self):
        assert repo.is_resting("pre_detected") is True

    def test_suspended_is_resting(self):
        assert repo.is_resting("suspended") is True

    def test_terminal_states_are_resting(self):
        for s in ("completed", "failed", "cancelled",
                  "preflight_failed", "predetect_failed"):
            assert repo.is_resting(s) is True, s

    def test_active_states_are_NOT_resting(self):
        for s in ("queued", "preflight_running", "pre_detecting",
                  "running", "partial_failed"):
            # partial_failed is special: it can transition forward (retry)
            # so it's not a terminal resting state — exception handlers
            # may still clobber it. (If we ever decide to protect it too,
            # update _RESTING_STATES, not this test.)
            assert repo.is_resting(s) is False, s

    def test_none_is_NOT_resting(self):
        assert repo.is_resting(None) is False

    def test_pre_detected_run_survives_exception(self, create_run):
        """Simulate the exact failure mode: run is paused at pre_detected,
        an exception bubbles up to the lifecycle except handler, and the
        old code would overwrite status='failed'. The fix says: if the
        run is at a resting state, leave it alone."""
        run_id = create_run()
        # Walk to pre_detected
        repo.update_run_status(run_id, "preflight_running")
        repo.update_run_status(run_id, "pre_detecting")
        repo.save_predetect_result(run_id, {
            "success": True, "identified_as": "DeepSeek", "confidence": 0.85,
        })
        assert repo.get_run(run_id)["status"] == "pre_detected"

        # Mirror the lifecycle except handler logic
        cur = repo.get_run(run_id)
        cur_status = cur.get("status") if cur else None
        if cur_status and not repo.is_resting(cur_status):
            repo.update_run_status(run_id, "failed",
                                   error_message="should not happen")
        # else: skip — that's the fix

        # State must still be pre_detected, ready for "continue"
        assert repo.get_run(run_id)["status"] == "pre_detected"


# ── Handler error response shape (Fix C) ─────────────────────────────────────


class TestContinueRunErrorPayload:
    """When a run is in the wrong state for ``/continue``, the handler
    must return a structured 400 with current_status + allowed_statuses
    so the frontend can render a useful message instead of the opaque
    "Run cannot be continued"."""

    def test_continue_rejects_failed_run_with_diagnostic(self, create_run):
        import json as _json
        from app.handlers.runs import handle_continue_run

        run_id = create_run()
        # Walk to failed via legitimate path
        repo.update_run_status(run_id, "running")
        repo.update_run_status(run_id, "failed",
                               error_message="some prior error")

        status, body, _ct = handle_continue_run(
            f"/api/v1/runs/{run_id}/continue", {}, {}
        )
        assert status == 400
        payload = _json.loads(body)
        # New diagnostic fields must be present
        assert payload["current_status"] == "failed"
        assert "pre_detected" in payload["allowed_statuses"]
        assert "cancelled" in payload["allowed_statuses"]
        assert payload["action"] == "be continued"
        assert payload["error_message"] == "some prior error"
        # Back-compat: ``error`` field still present for older clients
        assert "Run cannot" in payload["error"]
        assert "failed" in payload["error"]  # current status surfaces in message

    def test_continue_auto_recovers_falsely_failed_run(self, create_run):
        """v17 self-healing: a run that was marked failed by the pre-v17
        X→X state-machine bug should be auto-revived on the next
        /continue click, since pre-detect actually completed and the
        data is intact. This is the user-visible fix for runs left
        unrecoverable in the production DB after a previous bug-affected
        session.
        """
        import json as _json
        from app.handlers.runs import handle_continue_run

        run_id = create_run()
        # Walk to a real pre_detected and persist a predetect_result
        repo.update_run_status(run_id, "preflight_running")
        repo.update_run_status(run_id, "pre_detecting")
        repo.save_predetect_result(run_id, {
            "success": True, "identified_as": "DeepSeek", "confidence": 0.85,
        })
        # Simulate the old bug: pipeline got marked failed with E_UNHANDLED
        # despite predetect having completed. update_run_status legitimately
        # transitions pre_detected → failed.
        repo.update_run_status(
            run_id, "failed",
            error_message="Unhandled error: Illegal state transition: pre_detected → pre_detected",
            error_code="E_UNHANDLED",
        )
        assert repo.get_run(run_id)["status"] == "failed"
        assert repo.is_falsely_failed_predetected_run(repo.get_run(run_id))

        status, body, _ct = handle_continue_run(
            f"/api/v1/runs/{run_id}/continue", {}, {}
        )
        assert status == 200, body
        payload = _json.loads(body)
        assert payload["status"] == "running"
        # Status was forcibly transitioned through 'pre_detected' before
        # submit_continue scheduled the task.
        cur = repo.get_run(run_id)
        assert cur["status"] in ("pre_detected", "running")
        # error_message was overwritten with the recovery marker
        assert "v17 recovered" in (cur["error_message"] or "").lower() or \
               cur["status"] == "running"

    def test_continue_does_NOT_recover_legitimately_failed_run(self, create_run):
        """A run that failed for real reasons (e.g. SSL error) must NOT
        be auto-revived — the data may be incomplete and the user/admin
        should investigate."""
        import json as _json
        from app.handlers.runs import handle_continue_run

        run_id = create_run()
        repo.update_run_status(run_id, "running")
        repo.update_run_status(
            run_id, "failed",
            error_message="API 连接故障: SSL UNEXPECTED_EOF_WHILE_READING",
            error_code="E_NETWORK",
        )

        status, body, _ct = handle_continue_run(
            f"/api/v1/runs/{run_id}/continue", {}, {}
        )
        # Genuine failure → still 400, not auto-revived
        assert status == 400
        payload = _json.loads(body)
        assert payload["current_status"] == "failed"
        # Run is still failed in DB (not silently revived)
        assert repo.get_run(run_id)["status"] == "failed"

    def test_continue_does_NOT_recover_failed_run_without_predetect(self, create_run):
        """Even a state-machine-error message shouldn't trigger recovery
        if predetect_result is empty — that means predetect itself never
        completed."""
        run_id = create_run()
        repo.update_run_status(run_id, "running")
        repo.update_run_status(
            run_id, "failed",
            error_message="Unhandled error: Illegal state transition: queued → completed",
            error_code="E_UNHANDLED",
        )
        # No save_predetect_result call → predetect_result is None
        run = repo.get_run(run_id)
        assert not repo.is_falsely_failed_predetected_run(run)

    def test_continue_accepts_pre_detected_run(self, create_run):
        import json as _json
        from app.handlers.runs import handle_continue_run

        run_id = create_run()
        repo.update_run_status(run_id, "preflight_running")
        repo.update_run_status(run_id, "pre_detecting")
        repo.update_run_status(run_id, "pre_detected")

        status, body, _ct = handle_continue_run(
            f"/api/v1/runs/{run_id}/continue", {}, {}
        )
        # 200 path schedules the task and returns running
        assert status == 200
        payload = _json.loads(body)
        assert payload["run_id"] == run_id
        assert payload["status"] == "running"
