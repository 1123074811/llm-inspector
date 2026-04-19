"""v9 Phase C regression tests (performance/concurrency optimizations)."""

from __future__ import annotations


def test_mode_concurrency_uses_settings(monkeypatch):
    from app.runner import orchestrator

    monkeypatch.setattr(orchestrator.settings, "CONCURRENCY_QUICK", 21)
    monkeypatch.setattr(orchestrator.settings, "CONCURRENCY_STANDARD", 13)
    monkeypatch.setattr(orchestrator.settings, "CONCURRENCY_DEEP", 5)

    assert orchestrator._mode_concurrency("quick") == 21
    assert orchestrator._mode_concurrency("standard") == 13
    assert orchestrator._mode_concurrency("deep") == 5


def test_load_benchmarks_uses_ttl_cache(monkeypatch):
    from app.runner import orchestrator

    calls = {"n": 0}

    def fake_get_benchmarks(suite_version: str):
        calls["n"] += 1
        return [{"suite": suite_version, "idx": calls["n"]}]

    monkeypatch.setattr(orchestrator.repo, "get_benchmarks", fake_get_benchmarks)
    monkeypatch.setattr(orchestrator.settings, "BENCHMARK_CACHE_TTL_SEC", 120)
    monkeypatch.setattr(orchestrator, "_benchmark_cache", {})

    a = orchestrator._load_benchmarks("v1")
    b = orchestrator._load_benchmarks("v1")

    assert calls["n"] == 1
    assert a == b


def test_load_benchmarks_cache_expires(monkeypatch):
    from app.runner import orchestrator

    now = {"t": 1000.0}
    calls = {"n": 0}

    def fake_time():
        return now["t"]

    def fake_get_benchmarks(suite_version: str):
        calls["n"] += 1
        return [{"suite": suite_version, "idx": calls["n"]}]

    monkeypatch.setattr(orchestrator, "time", type("T", (), {"time": staticmethod(fake_time)}))
    monkeypatch.setattr(orchestrator.repo, "get_benchmarks", fake_get_benchmarks)
    monkeypatch.setattr(orchestrator.settings, "BENCHMARK_CACHE_TTL_SEC", 10)
    monkeypatch.setattr(orchestrator, "_benchmark_cache", {})

    first = orchestrator._load_benchmarks("v1")
    now["t"] += 11.0
    second = orchestrator._load_benchmarks("v1")

    assert calls["n"] == 2
    assert first != second
