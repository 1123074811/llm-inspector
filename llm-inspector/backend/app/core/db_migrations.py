"""
Database migration system with version tracking.
Migrations are applied sequentially and stored in the schema_migrations table.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from app.core.logging import get_logger

logger = get_logger(__name__)


class Migration:
    """Base class for migrations."""

    version: int
    description: str

    def apply(self, conn: sqlite3.Connection) -> None:
        raise NotImplementedError


_migrations: dict[int, Migration] = {}


def register_migration(migr: Migration) -> None:
    """Register a migration by version number."""
    _migrations[migr.version] = migr


def get_schema_version(conn: sqlite3.Connection) -> int:
    """Get current schema version from migrations table."""
    try:
        row = conn.execute(
            "SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1"
        ).fetchone()
        return row["version"] if row else 0
    except sqlite3.OperationalError:
        return 0


def _ensure_migrations_table(conn: sqlite3.Connection) -> None:
    """Create migrations table if it doesn't exist."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            description TEXT NOT NULL,
            applied_at TEXT NOT NULL
        )
        """
    )


def migrate(conn: sqlite3.Connection) -> list[int]:
    """Apply all pending migrations. Returns list of applied versions."""
    _ensure_migrations_table(conn)
    current = get_schema_version(conn)
    pending = sorted(v for v in _migrations if v > current)

    if not pending:
        logger.info("Database schema is up to date", version=current)
        return []

    applied = []
    for version in pending:
        migr = _migrations[version]
        logger.info("Applying migration", version=version, description=migr.description)
        migr.apply(conn)
        conn.execute(
            "INSERT INTO schema_migrations (version, description, applied_at) VALUES (?, ?, ?)",
            (version, migr.description, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        logger.info("Migration applied", version=version)
        applied.append(version)

    return applied


class Migration001InitialSchema(Migration):
    """Initial schema - create all tables."""

    version = 1
    description = "Initial schema - create all tables"

    def apply(self, conn: sqlite3.Connection) -> None:
        from app.core.db import SCHEMA_SQL
        conn.executescript(SCHEMA_SQL)


class Migration002JsonColumnsToColumns(Migration):
    """Migrate JSON metadata columns to proper database columns."""

    version = 2
    description = "Migrate JSON metadata columns to proper columns"

    def apply(self, conn: sqlite3.Connection) -> None:
        cursor = conn.execute("PRAGMA table_info(test_runs)")
        columns = {row[1] for row in cursor.fetchall()}

        new_columns = {
            "evaluation_mode", "calibration_case_id", "scoring_profile_version",
            "calibration_tag", "cancel_requested", "resume_from_existing"
        }
        missing = new_columns - columns
        if not missing:
            return

        for col in missing:
            if col in ("evaluation_mode", "calibration_case_id", "scoring_profile_version",
                       "calibration_tag"):
                conn.execute(f"ALTER TABLE test_runs ADD COLUMN {col} TEXT")
            elif col in ("cancel_requested", "resume_from_existing"):
                conn.execute(f"ALTER TABLE test_runs ADD COLUMN {col} INTEGER NOT NULL DEFAULT 0")
        conn.commit()

        with conn:
            rows = conn.execute("SELECT id, metadata FROM test_runs WHERE metadata IS NOT NULL").fetchall()
            for row in rows:
                run_id = row["id"]
                meta = json.loads(row["metadata"]) if row["metadata"] else {}
                updates = []
                vals = []
                if "evaluation_mode" in missing:
                    updates.append("evaluation_mode=?")
                    vals.append(meta.get("evaluation_mode", "normal"))
                if "calibration_case_id" in missing:
                    updates.append("calibration_case_id=?")
                    vals.append(meta.get("calibration_case_id"))
                if "scoring_profile_version" in missing:
                    updates.append("scoring_profile_version=?")
                    vals.append(meta.get("scoring_profile_version", "v1"))
                if "calibration_tag" in missing:
                    updates.append("calibration_tag=?")
                    vals.append(meta.get("calibration_tag"))
                if "cancel_requested" in missing:
                    updates.append("cancel_requested=?")
                    vals.append(1 if meta.get("cancel_requested") else 0)
                if "resume_from_existing" in missing:
                    updates.append("resume_from_existing=?")
                    vals.append(1 if meta.get("resume_from_existing") else 0)
                if updates:
                    vals.append(run_id)
                    conn.execute(f"UPDATE test_runs SET {','.join(updates)} WHERE id=?", vals)


class Migration003V14DropBenchmarkProfiles(Migration):
    """v14 Phase 1: drop deprecated benchmark_profiles table if it exists.

    benchmark_profiles was superseded by golden_baselines (real user-marked
    baselines) in v12. This migration safely removes the stale table from any
    database that was created before the table was removed from SCHEMA_SQL.
    Reference: UPGRADE_PLAN_V14.md §B13.
    """

    version = 3
    description = "v14-phase1: drop deprecated benchmark_profiles table"

    def apply(self, conn: sqlite3.Connection) -> None:
        conn.execute("DROP TABLE IF EXISTS benchmark_profiles")
        conn.commit()
        logger.info("Dropped benchmark_profiles table (if existed)")


class Migration004V14IdentityExposureColumn(Migration):
    """v14 Phase 3: add identity_exposure_result column to test_runs.

    Stores the serialised IdentityExposureReport JSON for each run.
    Column is nullable TEXT (JSON); absent in runs completed before v14 Phase 3.
    """
    version = 4
    description = "v14-phase3: add identity_exposure_result column"

    def apply(self, conn: sqlite3.Connection) -> None:
        cursor = conn.execute("PRAGMA table_info(test_runs)")
        columns = {row[1] for row in cursor.fetchall()}
        if "identity_exposure_result" not in columns:
            conn.execute("ALTER TABLE test_runs ADD COLUMN identity_exposure_result TEXT")
            conn.commit()
        logger.info("Added identity_exposure_result column (if not existed)")


class Migration005V15PreflightReportColumn(Migration):
    """v15 Phase 1: add preflight_report column to test_runs.

    Stores the serialised PreflightReport JSON for each run.
    Column is nullable TEXT (JSON); absent in runs completed before v15 Phase 1.
    """
    version = 5
    description = "v15-phase1: add preflight_report column"

    def apply(self, conn: sqlite3.Connection) -> None:
        cursor = conn.execute("PRAGMA table_info(test_runs)")
        columns = {row[1] for row in cursor.fetchall()}
        if "preflight_report" not in columns:
            conn.execute("ALTER TABLE test_runs ADD COLUMN preflight_report TEXT")
            conn.commit()
        logger.info("Added preflight_report column (if not existed)")


class Migration006IdentityExposureColumnGuard(Migration):
    """Guard migration: ensure identity_exposure_result column exists.

    Versions 3 and 4 in the migrations table may refer to different historical
    migrations depending on when the DB was first created (before or after
    UPGRADE_PLAN_V14.md was merged). This migration unconditionally adds the
    column when absent, regardless of what happened at v4.
    """
    version = 6
    description = "v14-phase3-guard: ensure identity_exposure_result column exists"

    def apply(self, conn: sqlite3.Connection) -> None:
        cursor = conn.execute("PRAGMA table_info(test_runs)")
        columns = {row[1] for row in cursor.fetchall()}
        if "identity_exposure_result" not in columns:
            conn.execute("ALTER TABLE test_runs ADD COLUMN identity_exposure_result TEXT")
            conn.commit()
            logger.info("Added identity_exposure_result column")
        else:
            logger.info("identity_exposure_result column already present, skipping")


class Migration007V15CacheTable(Migration):
    """v15 Phase 10: ensure llm_response_cache table exists.

    The CacheStrategy module (runner/cache_strategy.py) uses an SQLite table
    for persistent caching. This migration creates the table idempotently so
    that the table is available even on databases created before Phase 10.
    """
    version = 7
    description = "v15-phase10: create llm_response_cache table if not exists"

    def apply(self, conn: sqlite3.Connection) -> None:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS llm_response_cache (
                cache_key TEXT PRIMARY KEY,
                response_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL
            )
        """)
        conn.commit()
        logger.info("Ensured llm_response_cache table exists")


class Migration008V17ModelRegistry(Migration):
    """v17 Phase 5: model_registry + model_registry_audit tables.

    Phase 5 introduces a single-source-of-truth registry for the models the
    inspector knows about (and their canonical metadata).  Phase 6/7 sync
    daemons keep the table fresh from upstream ``/v1/models`` endpoints,
    OpenRouter, and changelog-harvested LLM extractions.  Phase 11 refers
    to ``model_registry`` to compute the eligible-baseline pool used by
    ``analysis/similarity_engine.py``.

    The audit table records per-field changes for traceability across syncs
    (eg. when an official_api source overrides a self_probed entry).
    """

    version = 8
    description = "v17-phase5: create model_registry + model_registry_audit tables"

    def apply(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS model_registry (
                model_id              TEXT PRIMARY KEY,
                vendor                TEXT NOT NULL,
                family                TEXT,
                status                TEXT NOT NULL,
                first_seen_at         INTEGER NOT NULL,
                last_seen_at          INTEGER NOT NULL,
                deprecated_at         INTEGER,
                sunset_at             INTEGER,
                cutoff_date           TEXT,
                context_window        INTEGER,
                max_output_tokens     INTEGER,
                modality              TEXT,
                supports_thinking     INTEGER,
                supports_tools        INTEGER,
                input_price_usd       REAL,
                output_price_usd      REAL,
                cache_read_price_usd  REAL,
                ttft_p50_ms           REAL,
                tps_p50               REAL,
                tokenizer_id          TEXT,
                self_report_id        TEXT,
                fingerprint_sha256    TEXT,
                data_source           TEXT NOT NULL,
                confidence            REAL NOT NULL DEFAULT 1.0,
                last_synced_at        INTEGER NOT NULL,
                raw_metadata_json     TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_registry_vendor_status
                ON model_registry(vendor, status);
            CREATE INDEX IF NOT EXISTS idx_registry_last_seen
                ON model_registry(last_seen_at);

            CREATE TABLE IF NOT EXISTS model_registry_audit (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id      TEXT NOT NULL,
                field         TEXT NOT NULL,
                old_value     TEXT,
                new_value     TEXT,
                data_source   TEXT NOT NULL,
                changed_at    INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_registry_audit_model
                ON model_registry_audit(model_id, changed_at);
            """
        )
        conn.commit()
        logger.info("Created model_registry + model_registry_audit tables (v17 Phase 5)")


class Migration009V17CaseQualityFlags(Migration):
    """v17 Phase 10: case_quality_flags table.

    Stores per-case ceiling/floor/discrimination flags computed by
    ``tasks.pruner_job`` and consulted by CAT-style selection so that
    items with no discriminative power can be skipped.  Designed as a
    side-table (not a mutation of ``test_cases``) to keep suite
    immutability and audit history simple.
    """

    version = 9
    description = "v17-phase10: create case_quality_flags table"

    def apply(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS case_quality_flags (
                case_id                TEXT PRIMARY KEY,
                discriminative_valid   INTEGER NOT NULL DEFAULT 1,
                ceiling_effect         INTEGER NOT NULL DEFAULT 0,
                floor_effect           INTEGER NOT NULL DEFAULT 0,
                low_discrimination     INTEGER NOT NULL DEFAULT 0,
                pass_rate              REAL,
                discrimination_a       REAL,
                n_responses            INTEGER NOT NULL DEFAULT 0,
                flags_json             TEXT,
                last_evaluated_at      INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_quality_valid
                ON case_quality_flags(discriminative_valid);
            """
        )
        conn.commit()
        logger.info("Created case_quality_flags table (v17 Phase 10)")


register_migration(Migration001InitialSchema())
register_migration(Migration002JsonColumnsToColumns())
register_migration(Migration003V14DropBenchmarkProfiles())
register_migration(Migration004V14IdentityExposureColumn())
register_migration(Migration005V15PreflightReportColumn())
register_migration(Migration006IdentityExposureColumnGuard())
register_migration(Migration007V15CacheTable())
register_migration(Migration008V17ModelRegistry())
register_migration(Migration009V17CaseQualityFlags())
