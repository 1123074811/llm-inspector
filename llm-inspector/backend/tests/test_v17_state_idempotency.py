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
