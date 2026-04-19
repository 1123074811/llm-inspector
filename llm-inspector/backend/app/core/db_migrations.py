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


register_migration(Migration001InitialSchema())
register_migration(Migration002JsonColumnsToColumns())
register_migration(Migration003V14DropBenchmarkProfiles())
