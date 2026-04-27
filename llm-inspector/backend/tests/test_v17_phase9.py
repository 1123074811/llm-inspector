"""
v17 Phase 9 — dataset_sync tests (offline, mocked fetcher).

Covers:
  * transformers: livebench / swebench / hle return well-formed cases
  * transformer rejects malformed records with None
  * sync_one_source: insert / skip existing / skip invalid / fetch_failed
  * run_dataset_sync aggregates per-source counts
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    db_path = tmp_path / "v17p9.sqlite"
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


# ── Transformers ────────────────────────────────────────────────────────────


def test_livebench_transformer_accepts_well_formed_record():
    from app.runner.dataset_sync import _livebench_transformer, DEFAULT_SOURCES
    src = DEFAULT_SOURCES[0]
    rec = {"question_id": "lb_001", "question": "What is 2+2?", "ground_truth": "4"}
    case = _livebench_transformer(rec, src)
    assert case is not None
    assert case["id"].startswith("livebench_")
    assert case["category"] == "reasoning"
    assert case["user_prompt"] == "What is 2+2?"
    assert case["params"]["expected_output"] == "4"
    assert case["params"]["_meta"]["source_dataset"] == "LiveBench"
    assert case["params"]["_meta"]["source_id"] == "lb_001"
    assert case["judge_method"] == "exact_match"


def test_livebench_transformer_handles_missing_question():
    from app.runner.dataset_sync import _livebench_transformer, DEFAULT_SOURCES
    src = DEFAULT_SOURCES[0]
    assert _livebench_transformer({"question_id": "lb_x"}, src) is None


def test_swebench_transformer_accepts_well_formed_record():
    from app.runner.dataset_sync import _swebench_transformer, DEFAULT_SOURCES
    src = DEFAULT_SOURCES[1]
    rec = {
        "instance_id": "django__django-12345",
        "problem_statement": "fix bug X in Y",
        "repo": "django/django",
        "base_commit": "abcd1234",
    }
    case = _swebench_transformer(rec, src)
    assert case is not None
    assert case["category"] == "coding"
    assert case["judge_method"] == "code_execution"
    assert case["params"]["repo"] == "django/django"
    assert case["params"]["_meta"]["source_id"] == "django__django-12345"


def test_swebench_transformer_handles_missing_required_fields():
    from app.runner.dataset_sync import _swebench_transformer, DEFAULT_SOURCES
    src = DEFAULT_SOURCES[1]
    assert _swebench_transformer({"instance_id": "x"}, src) is None
    assert _swebench_transformer({"problem_statement": "x"}, src) is None


def test_hle_transformer_accepts_well_formed_record():
    from app.runner.dataset_sync import _hle_transformer, DEFAULT_SOURCES
    src = DEFAULT_SOURCES[2]
    rec = {"id": "hle-42", "question": "Riemann hypothesis?", "answer": "Open problem"}
    case = _hle_transformer(rec, src)
    assert case is not None
    assert case["category"] == "reasoning"
    assert case["weight"] == 2.0


# ── _make_case_id ───────────────────────────────────────────────────────────


def test_make_case_id_slugifies_safely():
    from app.runner.dataset_sync import _make_case_id
    assert _make_case_id("livebench", "Q with spaces!") == "livebench_q-with-spaces"
    assert _make_case_id("swebench", "django__django-12345") == "swebench_django__django-12345"
    # Idempotent for plain alphanumeric
    assert _make_case_id("hle", "abc123") == "hle_abc123"


# ── sync_one_source ────────────────────────────────────────────────────────


def test_sync_one_source_inserts_new_cases():
    from app.runner.dataset_sync import sync_one_source, DEFAULT_SOURCES
    from app.core.db import get_conn
    src = DEFAULT_SOURCES[0]    # LiveBench

    def fake_fetcher(_src):
        return [
            {"question_id": f"lb_{i}", "question": f"Q{i}?", "ground_truth": str(i)}
            for i in range(5)
        ]

    res = sync_one_source(src, fetcher=fake_fetcher)
    assert res.fetched_rows == 5
    assert res.transformed == 5
    assert res.inserted == 5
    assert res.skipped_existing == 0
    assert res.error is None

    n = get_conn().execute("SELECT COUNT(*) AS n FROM test_cases").fetchone()["n"]
    assert n == 5


def test_sync_one_source_skips_existing_cases_on_second_pass():
    from app.runner.dataset_sync import sync_one_source, DEFAULT_SOURCES
    src = DEFAULT_SOURCES[0]

    def fake_fetcher(_src):
        return [{"question_id": "lb_1", "question": "Q?", "ground_truth": "A"}]

    r1 = sync_one_source(src, fetcher=fake_fetcher)
    r2 = sync_one_source(src, fetcher=fake_fetcher)
    assert r1.inserted == 1
    assert r2.inserted == 0
    assert r2.skipped_existing == 1


def test_sync_one_source_records_invalid_rows():
    from app.runner.dataset_sync import sync_one_source, DEFAULT_SOURCES
    src = DEFAULT_SOURCES[0]

    def fake_fetcher(_src):
        return [
            {"question_id": "lb_ok", "question": "Q?", "ground_truth": "A"},
            {"question_id": "lb_bad"},                  # missing question
            {"unrelated": "shape"},
        ]

    res = sync_one_source(src, fetcher=fake_fetcher)
    assert res.transformed == 1
    assert res.inserted == 1
    assert res.skipped_invalid == 2


def test_sync_one_source_handles_fetch_failure():
    from app.runner.dataset_sync import sync_one_source, DEFAULT_SOURCES
    src = DEFAULT_SOURCES[0]

    def fake_fetcher(_src):
        return None     # transport failure

    res = sync_one_source(src, fetcher=fake_fetcher)
    assert res.error == "fetch_failed"
    assert res.inserted == 0


def test_sync_one_source_handles_fetcher_exception():
    from app.runner.dataset_sync import sync_one_source, DEFAULT_SOURCES
    src = DEFAULT_SOURCES[0]

    def boom(_src):
        raise RuntimeError("network lost")

    res = sync_one_source(src, fetcher=boom)
    assert res.error and res.error.startswith("fetcher_raised")
    assert res.inserted == 0


def test_sync_one_source_handles_transformer_exception():
    from app.runner.dataset_sync import DatasetSource, sync_one_source

    def bad_transformer(_record, _src):
        raise RuntimeError("transformer crash")

    src = DatasetSource(
        name="X", hf_repo="x/x", config="x", split="x",
        category="reasoning", transformer=bad_transformer,
    )
    res = sync_one_source(src, fetcher=lambda s: [{"a": 1}, {"b": 2}])
    assert res.fetched_rows == 2
    assert res.skipped_invalid == 2
    assert res.inserted == 0


# ── run_dataset_sync orchestration ─────────────────────────────────────────


def test_run_dataset_sync_aggregates_counts():
    from app.runner.dataset_sync import run_dataset_sync, DEFAULT_SOURCES

    payloads = {
        "LiveBench": [
            {"question_id": f"lb_{i}", "question": f"Q{i}?", "ground_truth": str(i)}
            for i in range(3)
        ],
        "SWE-bench-Verified": [
            {"instance_id": f"swe_{i}", "problem_statement": f"problem {i}"} for i in range(2)
        ],
        "HLE": [
            {"id": f"hle_{i}", "question": f"hle Q{i}?", "answer": str(i)} for i in range(4)
        ],
    }

    def fetcher(src):
        return payloads.get(src.name, [])

    report = run_dataset_sync(sources=DEFAULT_SOURCES, fetcher=fetcher)
    d = report.to_dict()
    assert d["total_inserted"] == 9
    assert d["total_skipped_existing"] == 0
    by_src = {r["source_name"]: r for r in d["per_source"]}
    assert by_src["LiveBench"]["inserted"] == 3
    assert by_src["SWE-bench-Verified"]["inserted"] == 2
    assert by_src["HLE"]["inserted"] == 4


def test_dataset_sync_persists_meta_keys_in_params():
    from app.runner.dataset_sync import sync_one_source, DEFAULT_SOURCES
    from app.core.db import get_conn
    import json as _json
    src = DEFAULT_SOURCES[0]

    def fetcher(_src):
        return [{"question_id": "lb_meta", "question": "Q?", "ground_truth": "A"}]

    sync_one_source(src, fetcher=fetcher)
    row = get_conn().execute(
        "SELECT params FROM test_cases WHERE id=?", ("livebench_lb_meta",)
    ).fetchone()
    params = _json.loads(row["params"])
    meta = params["_meta"]
    assert meta["source_dataset"] == "LiveBench"
    assert meta["source_id"] == "lb_meta"
    assert "ingested_at" in meta
