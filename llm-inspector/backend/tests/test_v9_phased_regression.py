"""v9 Phase D regression tests.

Covers async pipeline routing and fallback policy in worker.submit_run.
"""

from __future__ import annotations


class _DummyExecutor:
    def __init__(self):
        self.last_fn = None

    def submit(self, fn):
        self.last_fn = fn
        fn()


def test_submit_run_prefers_async_when_enabled(monkeypatch):
    from app.tasks import worker

    monkeypatch.setattr(worker.settings, "ASYNC_PIPELINE_ENABLED", True)
    monkeypatch.setattr(worker.settings, "ASYNC_PIPELINE_FALLBACK_SYNC", False)

    ex = _DummyExecutor()
    monkeypatch.setattr(worker, "_local_executor", ex)

    called = {"async": 0, "sync": 0}

    async def fake_run_pipeline_async(_run_id: str):
        called["async"] += 1

    def fake_run_pipeline(_run_id: str):
        called["sync"] += 1

    monkeypatch.setattr("app.runner.orchestrator.run_pipeline_async", fake_run_pipeline_async)
    monkeypatch.setattr("app.runner.orchestrator.run_pipeline", fake_run_pipeline)

    worker.submit_run("run-async")

    assert called["async"] == 1
    assert called["sync"] == 0


def test_submit_run_fallback_to_sync_on_async_error(monkeypatch):
    from app.tasks import worker

    monkeypatch.setattr(worker.settings, "ASYNC_PIPELINE_ENABLED", True)
    monkeypatch.setattr(worker.settings, "ASYNC_PIPELINE_FALLBACK_SYNC", True)

    ex = _DummyExecutor()
    monkeypatch.setattr(worker, "_local_executor", ex)

    called = {"async": 0, "sync": 0}

    async def fake_run_pipeline_async(_run_id: str):
        called["async"] += 1
        raise RuntimeError("boom")

    def fake_run_pipeline(_run_id: str):
        called["sync"] += 1

    monkeypatch.setattr("app.runner.orchestrator.run_pipeline_async", fake_run_pipeline_async)
    monkeypatch.setattr("app.runner.orchestrator.run_pipeline", fake_run_pipeline)

    worker.submit_run("run-fallback")

    assert called["async"] == 1
    assert called["sync"] == 1


def test_submit_run_sync_when_async_disabled(monkeypatch):
    from app.tasks import worker

    monkeypatch.setattr(worker.settings, "ASYNC_PIPELINE_ENABLED", False)

    ex = _DummyExecutor()
    monkeypatch.setattr(worker, "_local_executor", ex)

    called = {"async": 0, "sync": 0}

    async def fake_run_pipeline_async(_run_id: str):
        called["async"] += 1

    def fake_run_pipeline(_run_id: str):
        called["sync"] += 1

    monkeypatch.setattr("app.runner.orchestrator.run_pipeline_async", fake_run_pipeline_async)
    monkeypatch.setattr("app.runner.orchestrator.run_pipeline", fake_run_pipeline)

    worker.submit_run("run-sync")

    assert called["async"] == 0
    assert called["sync"] == 1
