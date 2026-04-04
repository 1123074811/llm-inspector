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
    created_at            TEXT NOT NULL,
    started_at            TEXT,
    completed_at          TEXT,
    error_message         TEXT,
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

CREATE TABLE IF NOT EXISTS benchmark_profiles (
    id              TEXT PRIMARY KEY,
    benchmark_name  TEXT NOT NULL,
    suite_version   TEXT NOT NULL,
    feature_vector  TEXT NOT NULL,
    sample_count    INTEGER NOT NULL DEFAULT 3,
    generated_at    TEXT NOT NULL,
    UNIQUE(benchmark_name, suite_version)
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
"""


def init_db() -> None:
    """Create all tables (idempotent)."""
    conn = get_conn()
    conn.executescript(SCHEMA_SQL)
    conn.commit()


# ── helpers ───────────────────────────────────────────────────────────────────

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return str(uuid.uuid4())


def row_to_dict(row: sqlite3.Row) -> dict:
    return dict(row)


def json_col(value) -> str:
    """Serialize value to JSON string for TEXT columns."""
    return json.dumps(value, ensure_ascii=False)


def from_json_col(value: str | None):
    if value is None:
        return None
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value
