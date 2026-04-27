"""
v17 Phase 7 — changelog_harvester tests (no real network, no real LLM).

Covers:
  * strip_html: scripts/styles dropped, whitespace collapsed, max_chars enforced
  * validate_extracted_record:
      - valid record passes
      - invalid model_id rejected
      - missing/short evidence_quote rejected
      - evidence_quote that does NOT appear in source rejected (anti-hallucination)
      - bad date format rejected
      - context_window non-int rejected
      - prices: out of range, non-numeric
  * apply_extracted_records: upserts surviving records as 'changelog' source
    with confidence=0.85, lower-priority than official sources
  * run_harvest orchestrates fetch → strip → extract → validate → upsert
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    db_path = tmp_path / "v17p7.sqlite"
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


# ── strip_html ───────────────────────────────────────────────────────────────


def test_strip_html_drops_scripts_and_styles():
    from app.runner.changelog_harvester import strip_html
    html = """
    <html><head>
      <style>body{color:red}</style>
      <script>console.log('x')</script>
    </head><body>
      <h1>GPT-5 launches today</h1>
      <p>Available with 2M context window.</p>
    </body></html>
    """
    text = strip_html(html)
    assert "GPT-5 launches today" in text
    assert "2M context window" in text
    assert "console.log" not in text
    assert "color:red" not in text


def test_strip_html_collapses_whitespace_and_truncates():
    from app.runner.changelog_harvester import strip_html
    html = "<p>" + ("hello world  " * 1000) + "</p>"
    text = strip_html(html, max_chars=100)
    assert len(text) <= 110
    assert "  " not in text   # collapsed


def test_strip_html_handles_empty_and_malformed():
    from app.runner.changelog_harvester import strip_html
    assert strip_html("") == ""
    # Malformed: opening tag never closed; helper should not raise.
    assert "fallback" in strip_html("<not closed <p>fallback")


# ── validate_extracted_record ───────────────────────────────────────────────


_SRC_TEXT = (
    "Anthropic launches Claude 3.5 Sonnet on October 22, 2024. "
    "The model has a 200,000 token context window and is priced at "
    "$3.00 per million input tokens and $15.00 per million output tokens. "
    "Knowledge cutoff is April 2024."
)


def test_validate_record_passes_with_verbatim_quote():
    from app.runner.changelog_harvester import validate_extracted_record
    item = {
        "model_id": "claude-3-5-sonnet-20241022",
        "release_date": "2024-10-22",
        "context_window": 200000,
        "input_price_per_mtok": 3.0,
        "output_price_per_mtok": 15.0,
        "evidence_quote": "Claude 3.5 Sonnet on October 22, 2024",
    }
    rec, reason = validate_extracted_record(item, _SRC_TEXT)
    assert rec is not None and reason == ""
    assert rec.context_window == 200000


def test_validate_record_rejects_hallucinated_quote():
    from app.runner.changelog_harvester import validate_extracted_record
    item = {
        "model_id": "claude-3-5-sonnet-20241022",
        "context_window": 200000,
        "evidence_quote": "Claude 3.5 launched in 2099 with infinite context",
    }
    rec, reason = validate_extracted_record(item, _SRC_TEXT)
    assert rec is None
    assert reason == "evidence_quote_not_in_source"


def test_validate_record_rejects_invalid_model_id():
    from app.runner.changelog_harvester import validate_extracted_record
    item = {
        "model_id": "not a valid id with spaces!!",
        "evidence_quote": "Claude 3.5 Sonnet on October 22, 2024",
    }
    rec, reason = validate_extracted_record(item, _SRC_TEXT)
    assert rec is None
    assert reason == "invalid_model_id"


def test_validate_record_rejects_short_quote():
    from app.runner.changelog_harvester import validate_extracted_record
    item = {"model_id": "claude-3-5-sonnet", "evidence_quote": "short"}
    rec, reason = validate_extracted_record(item, _SRC_TEXT)
    assert rec is None
    assert reason == "evidence_quote_missing_or_short"


def test_validate_record_rejects_bad_date_format():
    from app.runner.changelog_harvester import validate_extracted_record
    item = {
        "model_id": "claude-3-5-sonnet-20241022",
        "release_date": "October 22 2024",        # not YYYY-MM-DD
        "evidence_quote": "Claude 3.5 Sonnet on October 22, 2024",
    }
    rec, reason = validate_extracted_record(item, _SRC_TEXT)
    assert rec is None
    assert reason == "invalid_release_date"


def test_validate_record_rejects_non_numeric_price():
    from app.runner.changelog_harvester import validate_extracted_record
    item = {
        "model_id": "claude-3-5-sonnet-20241022",
        "input_price_per_mtok": "free",
        "evidence_quote": "Claude 3.5 Sonnet on October 22, 2024",
    }
    rec, reason = validate_extracted_record(item, _SRC_TEXT)
    assert rec is None
    assert reason == "input_price_per_mtok_not_numeric"


def test_validate_record_rejects_out_of_range_context_window():
    from app.runner.changelog_harvester import validate_extracted_record
    item = {
        "model_id": "claude-3-5-sonnet-20241022",
        "context_window": -1,
        "evidence_quote": "Claude 3.5 Sonnet on October 22, 2024",
    }
    rec, reason = validate_extracted_record(item, _SRC_TEXT)
    assert rec is None
    assert reason == "context_window_out_of_range"


def test_validate_record_handles_non_dict_input():
    from app.runner.changelog_harvester import validate_extracted_record
    rec, reason = validate_extracted_record("not a dict", _SRC_TEXT)
    assert rec is None and reason == "not_a_dict"


# ── apply_extracted_records: upsert behaviour ──────────────────────────────


def test_apply_records_upserts_with_changelog_source():
    from app.runner.changelog_harvester import apply_extracted_records
    from app.repository import registry_repo as rr
    items = [
        {
            "model_id": "claude-3-5-sonnet-20241022",
            "context_window": 200000,
            "input_price_per_mtok": 3.0,
            "output_price_per_mtok": 15.0,
            "evidence_quote": "Claude 3.5 Sonnet on October 22, 2024",
        },
        {
            "model_id": "bogus model id with spaces",   # invalid → rejected
            "evidence_quote": "Claude 3.5 Sonnet on October 22, 2024",
        },
    ]
    out = apply_extracted_records(items, _SRC_TEXT, vendor="anthropic",
                                  source_url="https://anthropic.example/blog")
    assert out.raw_records == 2
    assert out.accepted_records == 1
    assert out.rejected_records == 1
    assert "invalid_model_id" in out.rejection_reasons

    card = rr.get_model_card("claude-3-5-sonnet-20241022")
    assert card["data_source"] == "changelog"
    assert card["confidence"] == 0.85
    assert card["context_window"] == 200000
    assert card["input_price_usd"] == 3.0


def test_apply_records_does_not_overwrite_official_source():
    from app.runner.changelog_harvester import apply_extracted_records
    from app.repository import registry_repo as rr

    # Pre-populate with an authoritative anthropic_api row
    rr.upsert_model({
        "model_id": "claude-3-5-sonnet-20241022",
        "vendor": "anthropic",
        "context_window": 200000,
        "data_source": "anthropic_api",
    })
    items = [{
        "model_id": "claude-3-5-sonnet-20241022",
        "context_window": 50000,        # bogus, must NOT clobber
        "evidence_quote": "Claude 3.5 Sonnet on October 22, 2024",
    }]
    apply_extracted_records(items, _SRC_TEXT, vendor="anthropic",
                            source_url="https://anthropic.example/blog")
    card = rr.get_model_card("claude-3-5-sonnet-20241022")
    assert card["context_window"] == 200000        # preserved
    assert card["data_source"] == "anthropic_api"  # preserved


# ── run_harvest orchestration ───────────────────────────────────────────────


def test_run_harvest_pipeline_with_injected_components(monkeypatch):
    from app.runner import changelog_harvester as ch
    from app.repository import registry_repo as rr

    fake_html = (
        "<html><body><article>"
        "<h1>Anthropic launches Claude 4 Opus today</h1>"
        "<p>Claude 4 Opus on April 27, 2026 with 500,000 token context window.</p>"
        "</article></body></html>"
    )

    def _fake_fetch(url: str):
        return fake_html

    def _fake_extract(prompt: str):
        return [
            {
                "model_id": "claude-4-opus-20260427",
                "release_date": "2026-04-27",
                "context_window": 500000,
                "evidence_quote": "Claude 4 Opus on April 27, 2026",
            },
            {
                # Hallucinated record — must be filtered
                "model_id": "phantom-model-x",
                "evidence_quote": "this sentence does not appear in the source",
            },
        ]

    sources = (ch.ChangelogSource(
        name="anthropic_news",
        url="https://anthropic.example/news",
        vendor="anthropic", kind="html",
    ),)
    report = ch.run_harvest(sources=sources, fetcher=_fake_fetch, extractor=_fake_extract)
    assert report["total_accepted"] == 1
    assert report["total_rejected"] == 1
    assert report["per_source"][0]["fetched"] is True
    assert report["per_source"][0]["text_chars"] > 0
    assert "claude-4-opus-20260427" in report["per_source"][0]["upserted_model_ids"]
    # Confirm registry side-effect
    assert rr.get_model_card("claude-4-opus-20260427")["data_source"] == "changelog"


def test_run_harvest_handles_fetch_failure(monkeypatch):
    from app.runner import changelog_harvester as ch

    def _fail_fetch(url: str):
        return None

    sources = (ch.ChangelogSource(
        name="dead_source", url="https://dead.example", vendor="openai", kind="html",
    ),)
    report = ch.run_harvest(sources=sources, fetcher=_fail_fetch,
                            extractor=lambda _t: [])
    assert report["per_source"][0]["fetched"] is False
    assert report["total_accepted"] == 0


def test_run_harvest_handles_extractor_exception():
    from app.runner import changelog_harvester as ch

    def _fetch(url: str):
        return "<html><body>Some text</body></html>"

    def _bad_extract(_text: str):
        raise RuntimeError("simulated LLM crash")

    sources = (ch.ChangelogSource(
        name="exploding", url="https://x.example", vendor="openai", kind="html",
    ),)
    report = ch.run_harvest(sources=sources, fetcher=_fetch, extractor=_bad_extract)
    # The extractor crash must be swallowed; no records accepted, but the
    # source is still recorded as fetched.
    assert report["per_source"][0]["fetched"] is True
    assert report["total_accepted"] == 0
