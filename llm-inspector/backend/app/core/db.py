"""
Database layer — SQLite via stdlib sqlite3.
Schema mirrors the spec (PostgreSQL-compatible column names).
Swap DATABASE_URL to postgresql+asyncpg://... when Postgres is available.
"""
import sqlite3
import threading
import pathlib
import json
import uuid
from datetime import datetime, timezone

from app.core.config import settings

# ── connection pool (thread-local) ────────────────────────────────────────────

_local = threading.local()
_DB_PATH = pathlib.Path(settings.DATABASE_URL.replace("sqlite:///", "").replace("./", ""))


def get_conn() -> sqlite3.Connection:
    """Return the thread-local SQLite connection, creating it if needed."""
    if not hasattr(_local, "conn") or _local.conn is None:
        conn = sqlite3.connect(
            str(_DB_PATH),
            check_same_thread=False,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        _local.conn = conn
    return _local.conn


# ── schema ────────────────────────────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS test_runs (
    id                    TEXT PRIMARY KEY,
    base_url              TEXT NOT NULL,
    api_key_encrypted     TEXT,
    api_key_hash          TEXT NOT NULL,
    model_name            TEXT NOT NULL,
    protocol              TEXT NOT NULL DEFAULT 'openai_compatible',
    test_mode             TEXT NOT NULL DEFAULT 'standard',
    suite_version         TEXT NOT NULL DEFAULT 'v1',
    status                TEXT NOT NULL DEFAULT 'queued',
    predetect_result      TEXT,
    predetect_confidence  REAL,
    predetect_identified  INTEGER DEFAULT 0,
    evaluation_mode       TEXT NOT NULL DEFAULT 'normal',
    calibration_case_id   TEXT,
    scoring_profile_version TEXT NOT NULL DEFAULT 'v1',
    calibration_tag       TEXT,
    cancel_requested      INTEGER NOT NULL DEFAULT 0,
    resume_from_existing  INTEGER NOT NULL DEFAULT 0,
    created_at            TEXT NOT NULL,
    started_at            TEXT,
    completed_at          TEXT,
    error_message         TEXT,
    error_code            TEXT,
    metadata              TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_runs_status  ON test_runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_created ON test_runs(created_at);

CREATE TABLE IF NOT EXISTS test_cases (
    id              TEXT PRIMARY KEY,
    category        TEXT NOT NULL,
    name            TEXT NOT NULL,
    system_prompt   TEXT,
    user_prompt     TEXT NOT NULL,
    expected_type   TEXT NOT NULL,
    judge_method    TEXT NOT NULL,
    params          TEXT NOT NULL DEFAULT '{}',
    max_tokens      INTEGER NOT NULL DEFAULT 5,
    n_samples       INTEGER NOT NULL DEFAULT 1,
    temperature     REAL NOT NULL DEFAULT 0.0,
    weight          REAL NOT NULL DEFAULT 1.0,
    enabled         INTEGER NOT NULL DEFAULT 1,
    suite_version   TEXT NOT NULL DEFAULT 'v1'
);

CREATE TABLE IF NOT EXISTS test_responses (
    id                        TEXT PRIMARY KEY,
    run_id                    TEXT NOT NULL REFERENCES test_runs(id) ON DELETE CASCADE,
    case_id                   TEXT NOT NULL REFERENCES test_cases(id),
    sample_index              INTEGER NOT NULL DEFAULT 0,
    request_payload           TEXT NOT NULL,
    response_text             TEXT,
    raw_response              TEXT,
    raw_headers               TEXT,
    status_code               INTEGER,
    latency_ms                INTEGER,
    first_token_ms            INTEGER,
    finish_reason             TEXT,
    usage_prompt_tokens       INTEGER,
    usage_completion_tokens   INTEGER,
    usage_total_tokens        INTEGER,
    error_type                TEXT,
    error_message             TEXT,
    judge_passed              INTEGER,
    judge_detail              TEXT,
    created_at                TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_responses_run  ON test_responses(run_id);
CREATE INDEX IF NOT EXISTS idx_responses_case ON test_responses(case_id);

CREATE TABLE IF NOT EXISTS stream_chunks (
    id              TEXT PRIMARY KEY,
    response_id     TEXT NOT NULL REFERENCES test_responses(id) ON DELETE CASCADE,
    chunk_index     INTEGER NOT NULL,
    arrived_at_ms   INTEGER NOT NULL,
    raw_line        TEXT NOT NULL,
    delta_text      TEXT,
    finish_reason   TEXT
);

CREATE INDEX IF NOT EXISTS idx_chunks_response ON stream_chunks(response_id);

CREATE TABLE IF NOT EXISTS extracted_features (
    id              TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES test_runs(id) ON DELETE CASCADE,
    feature_name    TEXT NOT NULL,
    feature_value   REAL,
    feature_text    TEXT,
    created_at      TEXT NOT NULL,
    UNIQUE(run_id, feature_name)
);

CREATE TABLE IF NOT EXISTS similarity_results (
    id                TEXT PRIMARY KEY,
    run_id            TEXT NOT NULL REFERENCES test_runs(id) ON DELETE CASCADE,
    benchmark_name    TEXT NOT NULL,
    similarity_score  REAL NOT NULL,
    ci_95_low         REAL,
    ci_95_high        REAL,
    rank_pos          INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS reports (
    id          TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL UNIQUE REFERENCES test_runs(id) ON DELETE CASCADE,
    summary     TEXT NOT NULL,
    details     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS data_retention_schedule (
    table_name          TEXT NOT NULL,
    field_name          TEXT NOT NULL,
    run_id              TEXT NOT NULL REFERENCES test_runs(id) ON DELETE CASCADE,
    scheduled_purge_at  TEXT NOT NULL,
    purged_at           TEXT,
    PRIMARY KEY(table_name, field_name, run_id)
);

CREATE TABLE IF NOT EXISTS score_breakdown (
    id          TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL REFERENCES test_runs(id) ON DELETE CASCADE,
    dimension   TEXT NOT NULL,
    score       REAL NOT NULL,
    max_score   REAL NOT NULL DEFAULT 10000.0,
    details     TEXT,
    created_at  TEXT NOT NULL,
    UNIQUE(run_id, dimension)
);

CREATE TABLE IF NOT EXISTS compare_runs (
    id                  TEXT PRIMARY KEY,
    golden_run_id       TEXT NOT NULL REFERENCES test_runs(id),
    candidate_run_id    TEXT NOT NULL REFERENCES test_runs(id),
    status              TEXT NOT NULL DEFAULT 'queued',
    delta_capability    REAL,
    delta_authenticity  REAL,
    delta_performance   REAL,
    verdict             TEXT,
    details             TEXT,
    created_at          TEXT NOT NULL,
    completed_at        TEXT
);

CREATE INDEX IF NOT EXISTS idx_compare_status ON compare_runs(status);

CREATE TABLE IF NOT EXISTS model_scores_history (
    id              TEXT PRIMARY KEY,
    model_name      TEXT NOT NULL,
    base_url        TEXT NOT NULL,
    run_id          TEXT NOT NULL REFERENCES test_runs(id) ON DELETE CASCADE,
    total_score     REAL,
    capability      REAL,
    authenticity    REAL,
    performance     REAL,
    recorded_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_scores_model ON model_scores_history(model_name, recorded_at);

CREATE TABLE IF NOT EXISTS golden_baselines (
    id                  TEXT PRIMARY KEY,
    model_name          TEXT NOT NULL,
    display_name        TEXT NOT NULL,
    source_run_id       TEXT NOT NULL REFERENCES test_runs(id) ON DELETE RESTRICT,
    suite_version       TEXT NOT NULL,
    feature_vector      TEXT NOT NULL,
    total_score         REAL NOT NULL,
    capability_score    REAL NOT NULL,
    authenticity_score  REAL NOT NULL,
    performance_score   REAL NOT NULL,
    score_breakdown     TEXT NOT NULL,
    theta               REAL,
    sample_count        INTEGER NOT NULL DEFAULT 1,
    notes               TEXT DEFAULT '',
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL,
    is_active           INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_golden_baselines_model ON golden_baselines(model_name, is_active);

CREATE TABLE IF NOT EXISTS baseline_comparisons (
    id                        TEXT PRIMARY KEY,
    run_id                    TEXT NOT NULL REFERENCES test_runs(id) ON DELETE CASCADE,
    baseline_id               TEXT NOT NULL REFERENCES golden_baselines(id),
    cosine_similarity         REAL NOT NULL,
    score_delta_total         REAL NOT NULL,
    score_delta_capability    REAL NOT NULL,
    score_delta_authenticity  REAL NOT NULL,
    score_delta_performance   REAL NOT NULL,
    verdict                   TEXT NOT NULL,
    p_value                   REAL,
    details                   TEXT NOT NULL,
    created_at                TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_baseline_comparisons_run ON baseline_comparisons(run_id);

CREATE TABLE IF NOT EXISTS item_bank (
    item_id          TEXT PRIMARY KEY,
    dimension        TEXT NOT NULL,
    anchor_flag      INTEGER NOT NULL DEFAULT 0,
    active           INTEGER NOT NULL DEFAULT 1,
    version          TEXT NOT NULL DEFAULT 'v1',
    created_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS item_stats (
    id                  TEXT PRIMARY KEY,
    item_id             TEXT NOT NULL UNIQUE,
    dimension           TEXT NOT NULL,
    irt_a               REAL NOT NULL DEFAULT 1.0,
    irt_b               REAL NOT NULL DEFAULT 0.0,
    irt_c               REAL,
    info_score          REAL NOT NULL DEFAULT 0.0,
    sample_size         INTEGER NOT NULL DEFAULT 0,
    last_calibrated_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_item_stats_dimension ON item_stats(dimension);

CREATE TABLE IF NOT EXISTS model_theta_history (
    id                    TEXT PRIMARY KEY,
    run_id                TEXT NOT NULL REFERENCES test_runs(id) ON DELETE CASCADE,
    model_name            TEXT NOT NULL,
    base_url              TEXT NOT NULL,
    theta_global          REAL NOT NULL,
    theta_global_ci_low   REAL NOT NULL,
    theta_global_ci_high  REAL NOT NULL,
    theta_dims_json       TEXT NOT NULL,
    percentile_global     REAL,
    percentile_dims_json  TEXT,
    calibration_version   TEXT NOT NULL,
    method                TEXT NOT NULL,
    created_at            TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_theta_history_model_time ON model_theta_history(model_name, created_at);

CREATE TABLE IF NOT EXISTS pairwise_results (
    id              TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES test_runs(id) ON DELETE CASCADE,
    model_a         TEXT NOT NULL,
    model_b         TEXT NOT NULL,
    delta_theta     REAL NOT NULL,
    win_prob_a      REAL NOT NULL,
    method          TEXT NOT NULL,
    details         TEXT,
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pairwise_run ON pairwise_results(run_id, created_at);

CREATE TABLE IF NOT EXISTS calibration_snapshots (
    id                TEXT PRIMARY KEY,
    version           TEXT NOT NULL UNIQUE,
    item_params_json  TEXT NOT NULL,
    notes             TEXT,
    created_at        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS calibration_replays (
    id                TEXT PRIMARY KEY,
    status            TEXT NOT NULL DEFAULT 'queued',
    cases_json        TEXT NOT NULL,
    result_json       TEXT,
    error_message     TEXT,
    created_at        TEXT NOT NULL,
    started_at        TEXT,
    completed_at      TEXT
);

CREATE INDEX IF NOT EXISTS idx_calibration_replays_status ON calibration_replays(status);

CREATE TABLE IF NOT EXISTS llm_response_cache (
    cache_key         TEXT PRIMARY KEY,
    response_json     TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    expires_at        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_cache_expires ON llm_response_cache(expires_at);

CREATE TABLE IF NOT EXISTS model_elo (
    model_name      TEXT PRIMARY KEY,
    display_name    TEXT NOT NULL,
    elo_rating      REAL NOT NULL DEFAULT 1500.0,
    games_played    INTEGER NOT NULL DEFAULT 0,
    wins            INTEGER NOT NULL DEFAULT 0,
    losses          INTEGER NOT NULL DEFAULT 0,
    draws           INTEGER NOT NULL DEFAULT 0,
    peak_elo        REAL NOT NULL DEFAULT 1500.0,
    last_run_id     TEXT,
    updated_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_elo_rating ON model_elo(elo_rating DESC);
"""




def init_db() -> None:
    """Create all tables (idempotent)."""
    conn = get_conn()
    conn.executescript(SCHEMA_SQL)
    conn.executescript("""
        CREATE INDEX IF NOT EXISTS idx_features_run ON extracted_features(run_id);
        CREATE INDEX IF NOT EXISTS idx_scores_run ON score_breakdown(run_id);
        CREATE INDEX IF NOT EXISTS idx_similarity_run ON similarity_results(run_id);
    """)
    conn.commit()


# ── helpers ───────────────────────────────────────────────────────────────────

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return str(uuid.uuid4())


def row_to_dict(row: sqlite3.Row) -> dict:
    return dict(row)


def json_col(value) -> str:
    """Serialize value to JSON string for TEXT columns. Robustly handles numpy types."""
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        # Fallback for numpy types and other non-standard objects
        import numpy as np
        class RobustEncoder(json.JSONEncoder):
            def default(self, o):
                if isinstance(o, (np.integer, np.int64, np.int32)):
                    return int(o)
                if isinstance(o, (np.floating, np.float64, np.float32)):
                    return float(o)
                if isinstance(o, (np.bool_, np.bool)):
                    return bool(o)
                if isinstance(o, np.ndarray):
                    return o.tolist()
                if hasattr(o, "to_dict"):
                    return o.to_dict()
                return str(o)
        return json.dumps(value, ensure_ascii=False, cls=RobustEncoder)


def from_json_col(value: str | None):
    if value is None:
        return None
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value
