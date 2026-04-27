"""
v17 Phase 10 — pruner_job + case_quality_flags tests.

Covers:
  * Migration 009 creates the table & indexes
  * pass_rate computation respects the eligible-baseline filter
  * upsert_quality_flag persists ceiling/floor/discrimination flags
  * run_pruner_job marks ceiling-effect cases as discriminative_valid=0
  * exhaustion warning triggers when many cases exceed 0.95 pass-rate
  * list_discriminative_case_ids / list_non_discriminative_case_ids selectors
"""
from __future__ import annotations

import json
import time

import pytest


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    db_path = tmp_path / "v17p10.sqlite"
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


# ── Helpers ────────────────────────────────────────────────────────────────


def _seed_run(run_id: str, model_name: str, base_url: str = "https://example.test"):
    from app.core.db import get_conn
    get_conn().execute(
        """INSERT INTO test_runs
           (id, base_url, api_key_encrypted, api_key_hash, model_name,
            test_mode, status, created_at)
           VALUES (?, ?, '', '', ?, 'standard', 'completed', '2026-04-27T00:00:00Z')""",
        (run_id, base_url, model_name),
    )
    get_conn().commit()


def _seed_case(case_id: str, *, params: dict | None = None,
               max_tokens: int = 100, weight: float = 1.0):
    from app.core.db import get_conn
    get_conn().execute(
        """INSERT INTO test_cases
           (id, category, name, system_prompt, user_prompt, expected_type,
            judge_method, params, max_tokens, n_samples, temperature, weight,
            enabled, suite_version)
           VALUES (?, 'reasoning', ?, NULL, 'p?', 'text', 'exact_match',
                   ?, ?, 1, 0.0, ?, 1, 'v17_test')""",
        (case_id, case_id, json.dumps(params or {}), max_tokens, weight),
    )
    get_conn().commit()


def _seed_response(resp_id: str, run_id: str, case_id: str, judge_passed: bool):
    from app.core.db import get_conn
    get_conn().execute(
        """INSERT INTO test_responses
           (id, run_id, case_id, sample_index, request_payload, response_text,
            judge_passed, created_at)
           VALUES (?, ?, ?, 0, '{}', 'r', ?, '2026-04-27T00:00:00Z')""",
        (resp_id, run_id, case_id, 1 if judge_passed else 0),
    )
    get_conn().commit()


# ── Migration ──────────────────────────────────────────────────────────────


def test_migration_creates_case_quality_flags_table():
    from app.core.db import get_conn
    tables = {r["name"] for r in get_conn().execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    assert "case_quality_flags" in tables


# ── Pass-rate aggregation ──────────────────────────────────────────────────


def test_pass_rate_only_counts_eligible_baselines():
    from app.tasks.pruner_job import _compute_case_pass_rates
    _seed_run("r1", "gpt-4o")
    _seed_run("r2", "rogue-model")
    _seed_case("c_a")
    # gpt-4o: 8 pass, 2 fail
    for i in range(10):
        _seed_response(f"a{i}", "r1", "c_a", judge_passed=(i < 8))
    # rogue: 0 pass, 10 fail (must be ignored by eligibility filter below)
    for i in range(10):
        _seed_response(f"b{i}", "r2", "c_a", judge_passed=False)

    out = _compute_case_pass_rates({"gpt-4o"})
    assert out["c_a"]["pass_rate"] == 0.8
    assert out["c_a"]["n_responses"] == 10

    # When the rogue is included, average drops to 0.4 — confirms filter works
    out_all = _compute_case_pass_rates({"gpt-4o", "rogue-model"})
    assert out_all["c_a"]["pass_rate"] == 0.4
    assert out_all["c_a"]["n_responses"] == 20


def test_pass_rate_ignores_unjudged_responses():
    from app.tasks.pruner_job import _compute_case_pass_rates
    from app.core.db import get_conn
    _seed_run("r1", "gpt-4o")
    _seed_case("c_a")
    _seed_response("y1", "r1", "c_a", judge_passed=True)
    # Insert an explicitly NULL judged row
    get_conn().execute(
        """INSERT INTO test_responses
           (id, run_id, case_id, sample_index, request_payload, judge_passed, created_at)
           VALUES ('y2', 'r1', 'c_a', 1, '{}', NULL, '2026-04-27T00:00:00Z')""",
    )
    get_conn().commit()
    out = _compute_case_pass_rates({"gpt-4o"})
    assert out["c_a"]["n_responses"] == 1
    assert out["c_a"]["pass_rate"] == 1.0


# ── Upsert + selectors ─────────────────────────────────────────────────────


def test_upsert_and_get_quality_flag():
    from app.tasks.pruner_job import _upsert_quality_flag, get_quality_flag
    from app.analysis.suite_pruner import CaseQualityMetrics

    metric = CaseQualityMetrics(
        case_id="c_x", discrimination_a=1.0, difficulty_b=0.0,
        fisher_info_at_mean=0.25, fisher_info_max=0.25,
        pass_rate=0.97, n_responses=50, is_discriminative=False,
        flags=["ceiling_effect"],
    )
    _upsert_quality_flag(metric, now=int(time.time()))
    out = get_quality_flag("c_x")
    assert out is not None
    assert out["discriminative_valid"] == 0
    assert out["ceiling_effect"] == 1
    assert out["floor_effect"] == 0
    assert out["pass_rate"] == 0.97
    assert "ceiling_effect" in out["flags"]


def test_list_selectors_partition_by_validity():
    from app.tasks.pruner_job import (
        _upsert_quality_flag, list_discriminative_case_ids,
        list_non_discriminative_case_ids,
    )
    from app.analysis.suite_pruner import CaseQualityMetrics

    now = int(time.time())
    _upsert_quality_flag(CaseQualityMetrics(
        case_id="ok", discrimination_a=1.5, difficulty_b=0.0,
        fisher_info_at_mean=0.5, fisher_info_max=0.5,
        pass_rate=0.5, n_responses=100, is_discriminative=True,
    ), now)
    _upsert_quality_flag(CaseQualityMetrics(
        case_id="ceil", discrimination_a=1.0, difficulty_b=0.0,
        fisher_info_at_mean=0.25, fisher_info_max=0.25,
        pass_rate=0.99, n_responses=100, is_discriminative=False,
        flags=["ceiling_effect"],
    ), now)
    assert list_discriminative_case_ids() == {"ok"}
    assert list_non_discriminative_case_ids() == {"ceil"}


# ── run_pruner_job end-to-end ──────────────────────────────────────────────


def _seed_eligible_registry_models(model_ids):
    """Mark each model as a Phase 11-eligible baseline."""
    from app.repository import registry_repo as rr
    now = int(time.time())
    aged = now - 60 * 86400
    for mid in model_ids:
        rr.upsert_model({
            "model_id": mid,
            "vendor": "openai",
            "data_source": "openai_api",
            "first_seen_at": aged,
            "last_seen_at": now,
            "last_synced_at": now,
        })


def test_run_pruner_job_marks_ceiling_cases_invalid():
    from app.tasks.pruner_job import run_pruner_job, get_quality_flag

    _seed_eligible_registry_models(["gpt-4o"])
    _seed_run("r_ceil", "gpt-4o")
    _seed_case("case_ceil")
    _seed_case("case_balanced")
    # Ceiling: 100/100 pass
    for i in range(100):
        _seed_response(f"c{i}", "r_ceil", "case_ceil", judge_passed=True)
    # Balanced: 50/100 pass
    for i in range(100):
        _seed_response(f"b{i}", "r_ceil", "case_balanced", judge_passed=(i < 50))

    report = run_pruner_job()
    assert report.cases_evaluated == 2
    assert report.pass_rate_window_models >= 1

    ceil_flag = get_quality_flag("case_ceil")
    assert ceil_flag["ceiling_effect"] == 1
    assert ceil_flag["discriminative_valid"] == 0
    assert ceil_flag["pass_rate"] == 1.0

    balanced_flag = get_quality_flag("case_balanced")
    assert balanced_flag["ceiling_effect"] == 0
    assert balanced_flag["discriminative_valid"] == 1
    assert balanced_flag["pass_rate"] == 0.5


def test_run_pruner_job_emits_exhaustion_warning_when_saturated():
    from app.tasks.pruner_job import run_pruner_job

    _seed_eligible_registry_models(["gpt-4o"])
    _seed_run("r_exh", "gpt-4o")
    # 5 ceiling cases, each with 30 responses (above EXHAUSTION_MIN_SAMPLES=20)
    for i in range(5):
        _seed_case(f"case_exh_{i}")
        for j in range(30):
            _seed_response(f"e{i}_{j}", "r_exh", f"case_exh_{i}", judge_passed=True)

    report = run_pruner_job()
    assert report.exhaustion_warning is not None
    assert report.exhaustion_warning["event"] == "suite_exhaustion_warning"
    assert report.exhaustion_warning["high_pass_rate_cases"] == 5


def test_run_pruner_job_no_warning_below_min_samples():
    from app.tasks.pruner_job import run_pruner_job

    _seed_eligible_registry_models(["gpt-4o"])
    _seed_run("r_thin", "gpt-4o")
    # Only 5 responses: below EXHAUSTION_MIN_SAMPLES → no warning even at 100% pass
    _seed_case("case_thin")
    for i in range(5):
        _seed_response(f"t{i}", "r_thin", "case_thin", judge_passed=True)

    report = run_pruner_job()
    assert report.exhaustion_warning is None


def test_run_pruner_job_handles_empty_db():
    from app.tasks.pruner_job import run_pruner_job
    report = run_pruner_job()
    assert report.cases_seen == 0
    assert report.cases_evaluated == 0
    assert report.exhaustion_warning is None
