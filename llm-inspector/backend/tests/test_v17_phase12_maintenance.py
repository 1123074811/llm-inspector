"""
v17 Phase 12 — maintenance_jobs daemon tests.

Covers:
  * MaintenanceJob: due()/run_and_reschedule() + failure counting
  * is_enabled() reads MAINTENANCE_JOBS_ENABLED truthy values
  * start_maintenance_jobs respects opt-in env var, supports `force=True`
  * background daemon actually invokes registered jobs and reschedules
  * default_jobs() lists the four production jobs in correct order
  * one job's failure does NOT block the other jobs in the same loop
"""
from __future__ import annotations

import os
import threading
import time

import pytest


@pytest.fixture(autouse=True)
def _stop_maintenance(monkeypatch):
    """Always stop the daemon thread after each test."""
    monkeypatch.delenv("MAINTENANCE_JOBS_ENABLED", raising=False)
    yield
    from app.tasks import maintenance_jobs as mj
    mj.stop_maintenance_jobs()
    # Reset module-level thread reference so subsequent tests can start a new one
    mj._thread = None
    time.sleep(0.05)


# ── MaintenanceJob unit tests ───────────────────────────────────────────────


def test_job_due_and_reschedule_after_success():
    from app.tasks.maintenance_jobs import MaintenanceJob

    runs = []
    job = MaintenanceJob(
        name="t", interval_sec=60, run_fn=lambda: runs.append(1) or "ok",
    )
    now = 1000.0
    job.schedule_initial(now)
    assert job.due(now) is True

    job.run_and_reschedule(now)
    assert job.successes == 1
    assert job.failures == 0
    assert job.next_run_at == now + 60
    assert job.due(now + 30) is False
    assert job.due(now + 60) is True
    assert runs == [1]


def test_job_failure_is_counted_and_rescheduled():
    from app.tasks.maintenance_jobs import MaintenanceJob

    def boom():
        raise RuntimeError("nope")

    job = MaintenanceJob(name="boom", interval_sec=42, run_fn=boom)
    now = 100.0
    job.schedule_initial(now)
    job.run_and_reschedule(now)
    assert job.successes == 0
    assert job.failures == 1
    assert "nope" in (job.last_error or "")
    # Even after failure, the job must be rescheduled normally
    assert job.next_run_at == now + 42


def test_job_initial_delay_postpones_first_run():
    from app.tasks.maintenance_jobs import MaintenanceJob
    job = MaintenanceJob(
        name="d", interval_sec=10, run_fn=lambda: None, initial_delay_sec=120,
    )
    now = 500.0
    job.schedule_initial(now)
    assert job.due(now) is False
    assert job.due(now + 119) is False
    assert job.due(now + 120) is True


# ── is_enabled / opt-in semantics ──────────────────────────────────────────


def test_is_enabled_reads_env_truthy_values(monkeypatch):
    from app.tasks.maintenance_jobs import is_enabled
    for v in ("1", "true", "TRUE", "yes", "on", "True"):
        monkeypatch.setenv("MAINTENANCE_JOBS_ENABLED", v)
        assert is_enabled() is True, f"expected True for env={v!r}"
    for v in ("0", "false", "no", "off", "", "anything-else"):
        monkeypatch.setenv("MAINTENANCE_JOBS_ENABLED", v)
        assert is_enabled() is False, f"expected False for env={v!r}"


def test_start_maintenance_jobs_opts_out_by_default(monkeypatch):
    from app.tasks.maintenance_jobs import start_maintenance_jobs
    monkeypatch.delenv("MAINTENANCE_JOBS_ENABLED", raising=False)
    started = start_maintenance_jobs(jobs=[])
    assert started is False


# ── Daemon thread integration ──────────────────────────────────────────────


def test_daemon_thread_invokes_jobs_periodically(monkeypatch):
    from app.tasks.maintenance_jobs import (
        MaintenanceJob, start_maintenance_jobs, stop_maintenance_jobs,
    )
    monkeypatch.setenv("MAINTENANCE_JOBS_ENABLED", "1")

    counter = {"n": 0}

    def job():
        counter["n"] += 1
        return "ok"

    j = MaintenanceJob(
        name="fast", interval_sec=1, run_fn=job, initial_delay_sec=0,
    )
    started = start_maintenance_jobs(jobs=[j], check_interval_sec=1)
    assert started is True
    # Wait for at least 2 invocations (initial + 1 reschedule)
    deadline = time.time() + 6
    while time.time() < deadline and counter["n"] < 2:
        time.sleep(0.1)
    stop_maintenance_jobs()
    assert counter["n"] >= 2, f"expected >=2 invocations, got {counter['n']}"


def test_one_failing_job_does_not_block_others(monkeypatch):
    from app.tasks.maintenance_jobs import (
        MaintenanceJob, start_maintenance_jobs, stop_maintenance_jobs,
    )
    monkeypatch.setenv("MAINTENANCE_JOBS_ENABLED", "1")

    counters = {"a": 0, "b": 0}

    def fail_job():
        counters["a"] += 1
        raise RuntimeError("explode")

    def ok_job():
        counters["b"] += 1
        return "ok"

    a = MaintenanceJob(name="a", interval_sec=1, run_fn=fail_job)
    b = MaintenanceJob(name="b", interval_sec=1, run_fn=ok_job)
    start_maintenance_jobs(jobs=[a, b], check_interval_sec=1)

    deadline = time.time() + 6
    while time.time() < deadline and counters["b"] < 2:
        time.sleep(0.1)
    stop_maintenance_jobs()

    assert counters["a"] >= 1
    assert counters["b"] >= 2     # ok_job ran multiple times despite a failing
    assert a.failures >= 1
    assert b.successes >= 2


def test_start_idempotent_when_already_running(monkeypatch):
    from app.tasks.maintenance_jobs import (
        MaintenanceJob, start_maintenance_jobs, stop_maintenance_jobs,
    )
    monkeypatch.setenv("MAINTENANCE_JOBS_ENABLED", "1")
    j = MaintenanceJob(name="x", interval_sec=10, run_fn=lambda: None)
    assert start_maintenance_jobs(jobs=[j], check_interval_sec=1) is True
    # Second call must not start a 2nd thread
    assert start_maintenance_jobs(jobs=[j], check_interval_sec=1) is False
    stop_maintenance_jobs()


def test_force_starts_even_when_env_unset(monkeypatch):
    from app.tasks.maintenance_jobs import (
        MaintenanceJob, start_maintenance_jobs, stop_maintenance_jobs,
    )
    monkeypatch.delenv("MAINTENANCE_JOBS_ENABLED", raising=False)
    j = MaintenanceJob(name="forced", interval_sec=10, run_fn=lambda: None)
    assert start_maintenance_jobs(jobs=[j], check_interval_sec=1, force=True) is True
    stop_maintenance_jobs()


# ── default_jobs() ──────────────────────────────────────────────────────────


def test_get_status_when_daemon_not_running(monkeypatch):
    from app.tasks.maintenance_jobs import get_status
    monkeypatch.delenv("MAINTENANCE_JOBS_ENABLED", raising=False)
    s = get_status()
    assert s["enabled"] is False
    assert s["running"] is False
    assert isinstance(s["jobs"], list)


def test_get_status_returns_per_job_view_when_running(monkeypatch):
    from app.tasks.maintenance_jobs import (
        MaintenanceJob, get_status, start_maintenance_jobs, stop_maintenance_jobs,
    )
    monkeypatch.setenv("MAINTENANCE_JOBS_ENABLED", "1")
    j = MaintenanceJob(name="probe", interval_sec=600, run_fn=lambda: None,
                       initial_delay_sec=300)
    start_maintenance_jobs(jobs=[j], check_interval_sec=5)
    try:
        s = get_status()
        assert s["enabled"] is True
        assert s["running"] is True
        names = [item["name"] for item in s["jobs"]]
        assert names == ["probe"]
        item = s["jobs"][0]
        assert item["interval_sec"] == 600
        assert item["successes"] == 0
        assert item["failures"] == 0
        assert item["last_error"] is None
        assert item["next_run_in_sec"] >= 250  # ~300s initial delay
    finally:
        stop_maintenance_jobs()


def test_v17_maintenance_status_endpoint(monkeypatch):
    """The /api/v17/maintenance/status route returns a stable JSON shape."""
    import json
    from app.main import _handle_v17_maintenance_status
    monkeypatch.delenv("MAINTENANCE_JOBS_ENABLED", raising=False)
    status_code, body, _ = _handle_v17_maintenance_status("", {}, {})
    assert status_code == 200
    data = json.loads(body)
    assert data["status"] == "ok"
    assert data["api_version"] == "v17"
    assert "enabled" in data
    assert "running" in data
    assert isinstance(data["jobs"], list)


def test_default_jobs_lists_four_production_tasks():
    from app.tasks.maintenance_jobs import default_jobs
    jobs = default_jobs()
    names = [j.name for j in jobs]
    assert names == [
        "model_registry_sync",
        "changelog_harvester",
        "dataset_sync",
        "pruner_job",
    ]
    # Sanity: intervals are positive and ordered (registry < changelog < dataset)
    by_name = {j.name: j for j in jobs}
    assert by_name["model_registry_sync"].interval_sec == 6 * 3600
    assert by_name["changelog_harvester"].interval_sec == 24 * 3600
    assert by_name["dataset_sync"].interval_sec == 7 * 24 * 3600
    assert by_name["pruner_job"].interval_sec == 3600
    # Initial delays prevent thundering-herd at startup
    for j in jobs:
        assert j.initial_delay_sec >= 0
