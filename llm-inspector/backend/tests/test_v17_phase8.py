"""
v17 Phase 8 — self-probe register tests (no real API calls).

All four probe functions are injected; assertions cover:

  * cutoff binary search picks the correct anchor (and short-circuits on
    edge cases)
  * tokenizer matcher selects the closest known signature
  * timing yields p50 over real samples and skips invalid ones
  * self-report banner-grab normalises the response
  * full pipeline persists into model_registry with data_source='self_probed'
    and confidence=0.85, and produces a stable fingerprint hash
  * is_known_model integrates with registry
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    db_path = tmp_path / "v17p8.sqlite"
    from app.core import db as _db_mod

    monkeypatch.setattr(_db_mod.settings, "DATABASE_URL", f"sqlite:///{db_path.as_posix()}")
    monkeypatch.setattr(_db_mod, "_DB_PATH", db_path)
    if hasattr(_db_mod._local, "conn") and _db_mod._local.conn is not None:
        try:
            _db_mod._local.conn.close()
        except Exception:
            pass
        _db_mod._local.conn = None

    _db_mod.init_db()
    from app.core.db_migrations import migrate
    migrate(_db_mod.get_conn())

    yield

    if hasattr(_db_mod._local, "conn") and _db_mod._local.conn is not None:
        try:
            _db_mod._local.conn.close()
        except Exception:
            pass
        _db_mod._local.conn = None


# ── probe_cutoff ────────────────────────────────────────────────────────────


def test_probe_cutoff_knows_latest_short_circuits():
    from app.runner.self_probe_register import probe_cutoff

    def ask(_p):
        return True

    res = probe_cutoff(ask)
    assert res.cutoff_date.startswith(">= ")
    assert res.total_calls == 1


def test_probe_cutoff_knows_nothing_short_circuits():
    from app.runner.self_probe_register import probe_cutoff

    def ask(_p):
        return False

    res = probe_cutoff(ask)
    assert res.cutoff_date.startswith("< ")
    assert res.total_calls == 2  # latest + earliest


def test_probe_cutoff_locates_middle_anchor():
    """Model that knows up to 2024-09-30 but not 2025-03-31 → cutoff = 2024-09-30."""
    from app.runner.self_probe_register import probe_cutoff

    def ask(prompt: str) -> bool:
        # Extract the date in the prompt (format: "Event (YYYY-MM-DD): ...")
        import re
        m = re.search(r"Event \((\d{4}-\d{2}-\d{2})\)", prompt)
        date = m.group(1) if m else ""
        return date <= "2024-09-30"

    res = probe_cutoff(ask)
    assert res.cutoff_date == "2024-09-30"


# ── probe_tokenizer ─────────────────────────────────────────────────────────


def test_probe_tokenizer_matches_cl100k_signature():
    from app.runner.self_probe_register import probe_tokenizer, _TOKENIZER_REFERENCE

    cl100k_ref = _TOKENIZER_REFERENCE["cl100k"]

    def tokenize(text: str) -> int:
        # Return the cl100k expected count by reverse lookup against probe text
        from app.runner.self_probe_register import _TOKENIZER_PROBE_PROMPTS
        for probe_id, probe_text in _TOKENIZER_PROBE_PROMPTS:
            if probe_text == text:
                return cl100k_ref[probe_id]
        return 0

    res = probe_tokenizer(tokenize)
    assert res.tokenizer_id == "cl100k"
    assert res.distance == 0.0


def test_probe_tokenizer_matches_claude_signature_with_noise():
    from app.runner.self_probe_register import probe_tokenizer, _TOKENIZER_REFERENCE

    claude_ref = _TOKENIZER_REFERENCE["claude"]

    def tokenize(text: str) -> int:
        from app.runner.self_probe_register import _TOKENIZER_PROBE_PROMPTS
        for probe_id, probe_text in _TOKENIZER_PROBE_PROMPTS:
            if probe_text == text:
                # Add ±1 noise to simulate real tokenizers
                return claude_ref[probe_id] + (1 if probe_id == "claude_marker" else 0)
        return 0

    res = probe_tokenizer(tokenize)
    assert res.tokenizer_id == "claude"
    assert res.distance < 1.0


def test_probe_tokenizer_handles_failures_gracefully():
    from app.runner.self_probe_register import probe_tokenizer

    def tokenize(_text: str) -> int:
        raise RuntimeError("tokenize failed")

    res = probe_tokenizer(tokenize)
    # All probes fail → tokenizer_id is None
    assert res.tokenizer_id is None


# ── probe_timing ────────────────────────────────────────────────────────────


def test_probe_timing_computes_p50_and_skips_invalid():
    from app.runner.self_probe_register import probe_timing

    samples = iter([
        (300.0, 50.0),
        (400.0, 60.0),
        (350.0, 55.0),
        None,                     # skipped
        (-1.0, 50.0),             # skipped
        (380.0, 58.0),
        (320.0, 52.0),
    ])

    def time_fn():
        try:
            return next(samples)
        except StopIteration:
            return None

    res = probe_timing(time_fn, n=10)
    assert res.n == 5
    assert res.ttft_p50_ms == 350.0
    assert res.tps_p50 == 55.0


def test_probe_timing_zero_n_returns_none():
    from app.runner.self_probe_register import probe_timing
    res = probe_timing(lambda: (100.0, 50.0), n=0)
    assert res.n == 0
    assert res.ttft_p50_ms is None and res.tps_p50 is None


def test_probe_timing_handles_exceptions():
    from app.runner.self_probe_register import probe_timing

    def explode():
        raise RuntimeError("boom")

    res = probe_timing(explode, n=3)
    assert res.n == 0


# ── probe_identity ──────────────────────────────────────────────────────────


def test_probe_identity_normalises_response():
    from app.runner.self_probe_register import probe_identity
    res = probe_identity(lambda: "  I  am   ChatGPT, a large language model.\n  ")
    assert res.self_report_id == "i am chatgpt, a large language model."
    assert "ChatGPT" in res.raw_text


def test_probe_identity_handles_exceptions():
    from app.runner.self_probe_register import probe_identity

    def explode():
        raise RuntimeError("boom")

    res = probe_identity(explode)
    assert res.self_report_id == ""


# ── run_self_probe (full pipeline) ──────────────────────────────────────────


def _make_clean_probes():
    """Return a tuple of probe callables that match the cl100k signature."""
    from app.runner.self_probe_register import (
        _TOKENIZER_PROBE_PROMPTS, _TOKENIZER_REFERENCE,
    )
    cl100k_ref = _TOKENIZER_REFERENCE["cl100k"]

    def ask(prompt: str) -> bool:
        import re
        m = re.search(r"Event \((\d{4}-\d{2}-\d{2})\)", prompt)
        date = m.group(1) if m else ""
        return date <= "2024-09-30"

    def tokenize(text: str) -> int:
        for probe_id, probe_text in _TOKENIZER_PROBE_PROMPTS:
            if probe_text == text:
                return cl100k_ref[probe_id]
        return 0

    timing_iter = iter([(300.0, 50.0)] * 30)

    def time_fn():
        return next(timing_iter)

    def identity_fn():
        return "I am ChatGPT, a large language model trained by OpenAI."

    return ask, tokenize, time_fn, identity_fn


def test_run_self_probe_persists_to_registry():
    from app.runner.self_probe_register import run_self_probe
    from app.repository import registry_repo as rr

    ask, tokenize, time_fn, identity_fn = _make_clean_probes()
    report = run_self_probe(
        model_id="some-unknown-model",
        ask_fn=ask,
        tokenize_fn=tokenize,
        time_fn=time_fn,
        identity_fn=identity_fn,
        vendor_hint="OpenAI",
        timing_samples=5,
    )

    assert report.persisted is True
    assert report.model_id == "some-unknown-model"
    assert report.tokenizer.tokenizer_id == "cl100k"
    assert report.cutoff.cutoff_date == "2024-09-30"
    assert report.timing.ttft_p50_ms == 300.0
    assert report.fingerprint_sha256 and len(report.fingerprint_sha256) == 64

    card = rr.get_model_card("some-unknown-model")
    assert card["data_source"] == "self_probed"
    assert card["confidence"] == 0.85
    assert card["status"] == "self_probed"
    assert card["tokenizer_id"] == "cl100k"
    assert card["cutoff_date"] == "2024-09-30"
    assert card["ttft_p50_ms"] == 300.0
    assert "ChatGPT" in card["self_report_id"] or "chatgpt" in card["self_report_id"]


def test_run_self_probe_does_not_persist_when_persist_false():
    from app.runner.self_probe_register import run_self_probe
    from app.repository import registry_repo as rr

    ask, tokenize, time_fn, identity_fn = _make_clean_probes()
    report = run_self_probe(
        model_id="dryrun-model",
        ask_fn=ask, tokenize_fn=tokenize, time_fn=time_fn, identity_fn=identity_fn,
        timing_samples=3,
        persist=False,
    )
    assert report.persisted is False
    assert rr.get_model_card("dryrun-model") is None


def test_run_self_probe_does_not_overwrite_official_data():
    """A self-probe row must not clobber a previously stored official row."""
    from app.runner.self_probe_register import run_self_probe
    from app.repository import registry_repo as rr

    rr.upsert_model({
        "model_id": "gpt-4o",
        "vendor": "openai",
        "context_window": 128000,
        "data_source": "openai_api",
    })
    ask, tokenize, time_fn, identity_fn = _make_clean_probes()
    run_self_probe(
        model_id="gpt-4o",
        ask_fn=ask, tokenize_fn=tokenize, time_fn=time_fn, identity_fn=identity_fn,
        timing_samples=3,
    )
    card = rr.get_model_card("gpt-4o")
    assert card["data_source"] == "openai_api"  # preserved
    # Self-probed only filled in fields the official source did not supply
    assert card["tokenizer_id"] == "cl100k"


def test_run_self_probe_fingerprint_is_stable_across_runs():
    from app.runner.self_probe_register import run_self_probe

    ask, tokenize, time_fn, identity_fn = _make_clean_probes()
    r1 = run_self_probe(
        model_id="reproducible-model",
        ask_fn=ask, tokenize_fn=tokenize, time_fn=time_fn, identity_fn=identity_fn,
        timing_samples=5, persist=False,
    )
    ask, tokenize, time_fn, identity_fn = _make_clean_probes()
    r2 = run_self_probe(
        model_id="reproducible-model",
        ask_fn=ask, tokenize_fn=tokenize, time_fn=time_fn, identity_fn=identity_fn,
        timing_samples=5, persist=False,
    )
    assert r1.fingerprint_sha256 == r2.fingerprint_sha256


def test_is_known_model():
    from app.runner.self_probe_register import is_known_model
    from app.repository import registry_repo as rr
    assert is_known_model("nonexistent-x") is False
    rr.upsert_model({
        "model_id": "known-x", "vendor": "openai", "data_source": "openai_api",
    })
    assert is_known_model("known-x") is True


def test_run_self_probe_requires_model_id():
    from app.runner.self_probe_register import run_self_probe
    ask, tokenize, time_fn, identity_fn = _make_clean_probes()
    with pytest.raises(ValueError):
        run_self_probe(
            model_id="",
            ask_fn=ask, tokenize_fn=tokenize, time_fn=time_fn, identity_fn=identity_fn,
        )
