"""AuthNZ admin-performance indexes (perf plan A, stage 3).

F3/F4 from the 2026-10-06 admin WebUI performance review: time-windowed
session activity queries (``sessions.created_at``) and org-scoped joins
(``org_members(org_id, user_id)``) previously full-scanned. These tests pin
both indexes through the real migration paths:

* SQLite: versioned migrations via ``ensure_authnz_tables`` (fresh DBs) and
  the upgrade path for databases already at version 100 (the new versioned
  migration 101, mirroring the migration-099 precedent).
* PostgreSQL: the always-run idempotent bootstrap list executed by
  ``ensure_authnz_core_tables_pg`` at startup, which covers both fresh and
  existing databases (the PG module has no versioned migration runner).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

# Bare-column, sargable activity filter mirroring the PostgreSQL branch of
# admin_system_service.get_activity_summary (SQLite branch wraps the column
# in datetime(); the index must serve the bare range form).
_ACTIVITY_SQL = (
    "SELECT date(created_at) as bucket, COUNT(DISTINCT user_id) as active_users "
    "FROM sessions WHERE created_at >= ? GROUP BY bucket ORDER BY bucket"
)


def _index_names(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA index_list({table})").fetchall()
    return {str(row[1]) for row in rows}


def _explain_detail(conn: sqlite3.Connection, sql: str, params: tuple) -> str:
    plan = conn.execute("EXPLAIN QUERY PLAN " + sql, params).fetchall()
    return " | ".join(str(row[-1]) for row in plan)


def _utc_iso(delta_days: int = 0) -> str:
    moment = datetime.now(timezone.utc) + timedelta(days=delta_days)
    return moment.strftime("%Y-%m-%d %H:%M:%S")


@pytest.mark.unit
def test_sessions_created_at_index_exists(tmp_path: Path) -> None:
    """Fresh SQLite DBs must carry both admin perf indexes after migrations."""
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables

    db_path = tmp_path / "admin_perf_indexes.db"
    ensure_authnz_tables(db_path)

    with sqlite3.connect(db_path) as conn:
        session_indexes = _index_names(conn, "sessions")
        assert "idx_sessions_created_at" in session_indexes, (
            f"missing idx_sessions_created_at, got: {sorted(session_indexes)}"
        )
        org_member_indexes = _index_names(conn, "org_members")
        assert "idx_org_members_org_user" in org_member_indexes, (
            f"missing idx_org_members_org_user, got: {sorted(org_member_indexes)}"
        )


@pytest.mark.unit
def test_activity_query_uses_index(tmp_path: Path) -> None:
    """The windowed sessions activity query must range-scan, not SCAN."""
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables

    db_path = tmp_path / "admin_perf_activity.db"
    ensure_authnz_tables(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            ("activity-user", "activity@example.test", "x"),
        )
        conn.executemany(
            "INSERT INTO sessions (user_id, token_hash, expires_at, created_at)"
            " VALUES (?, ?, ?, ?)",
            [
                (1, "hash-1", _utc_iso(1), _utc_iso()),
                (1, "hash-2", _utc_iso(1), _utc_iso(-30)),
            ],
        )
        conn.commit()

        detail = _explain_detail(conn, _ACTIVITY_SQL, (_utc_iso(-7),))
        assert "SCAN sessions" not in detail, f"full table scan, plan: {detail}"
        assert "idx_sessions_created_at" in detail, (
            f"created_at range not served by idx_sessions_created_at, plan: {detail}"
        )


@pytest.mark.unit
def test_upgrade_path_existing_db_gets_admin_perf_indexes(tmp_path: Path) -> None:
    """Databases already at the previous latest version must gain the indexes on upgrade.

    Versioned SQLite migrations never re-run applied steps, so appending to
    migrations 002/016 alone cannot reach databases in the wild; a versioned
    migration above 100 must deliver both indexes (cf. migration 099).
    """
    from tldw_Server_API.app.core.AuthNZ.migrations import (
        apply_authnz_migrations,
        ensure_authnz_tables,
    )

    db_path = tmp_path / "admin_perf_upgrade.db"
    apply_authnz_migrations(db_path, target_version=100)

    with sqlite3.connect(db_path) as conn:
        # Emulate a database that upgraded before stage 3: fully migrated to
        # version 100 by the pre-stage-3 migrations 002/016, which never
        # created these indexes.
        conn.execute("DROP INDEX IF EXISTS idx_sessions_created_at")
        conn.execute("DROP INDEX IF EXISTS idx_org_members_org_user")
        conn.commit()
        assert "idx_sessions_created_at" not in _index_names(conn, "sessions")
        assert "idx_org_members_org_user" not in _index_names(conn, "org_members")

    ensure_authnz_tables(db_path)

    with sqlite3.connect(db_path) as conn:
        applied_versions = {
            int(row[0])
            for row in conn.execute("SELECT version FROM schema_migrations").fetchall()
        }
        assert 101 in applied_versions, (
            f"migration 101 must be the pending upgrade step, got: {sorted(applied_versions)[-3:]}"
        )
        assert "idx_sessions_created_at" in _index_names(conn, "sessions")
        assert "idx_org_members_org_user" in _index_names(conn, "org_members")


@pytest.mark.unit
async def test_pg_bootstrap_emits_admin_perf_indexes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The PG always-run core bootstrap must emit both CREATE INDEX statements."""

    class _RecordingPostgresPool:
        def __init__(self) -> None:
            self.pool = object()
            self.executed_sql: list[str] = []

        def transaction(self):
            pool = self

            class _Transaction:
                async def __aenter__(self):
                    return pool

                async def __aexit__(self, exc_type, exc, traceback):
                    del exc_type, exc, traceback
                    return False

            return _Transaction()

        async def execute(self, query: str, *args: object) -> None:
            self.executed_sql.append(str(query))

        async def fetchval(self, query: str, *args: object) -> bool:
            del query, args
            return False

    from tldw_Server_API.app.core.AuthNZ import pg_migrations_extra

    monkeypatch.setattr(
        pg_migrations_extra,
        "ensure_postgres_profile_version_on_connection",
        AsyncMock(),
    )
    monkeypatch.setattr(
        pg_migrations_extra,
        "repair_postgres_profile_candidate_timestamps",
        AsyncMock(),
    )
    monkeypatch.setattr(
        pg_migrations_extra,
        "validate_postgres_profile_candidate_schema",
        AsyncMock(),
    )
    monkeypatch.setattr(
        pg_migrations_extra,
        "ensure_mcp_prompt_read_permission_pg",
        AsyncMock(return_value=True),
    )

    pool = _RecordingPostgresPool()
    assert await pg_migrations_extra.ensure_authnz_core_tables_pg(pool) is True

    assert any(
        "CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions(created_at)"
        in sql
        for sql in pool.executed_sql
    ), "bootstrap must create idx_sessions_created_at"
    assert any(
        "CREATE INDEX IF NOT EXISTS idx_org_members_org_user"
        " ON public.org_members(org_id, user_id)" in sql
        for sql in pool.executed_sql
    ), "bootstrap must create idx_org_members_org_user"


@pytest.mark.integration
async def test_pg_indexes_exist_after_bootstrap(isolated_test_environment) -> None:
    """The PG fixture database must expose both indexes in pg_indexes after bootstrap."""
    _client, _db_name = isolated_test_environment

    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
        ensure_authnz_core_tables_pg,
    )

    pool = await get_db_pool()
    assert await ensure_authnz_core_tables_pg(pool) is True

    rows = await pool.fetchall(
        "SELECT indexname FROM pg_indexes "
        "WHERE indexname IN ('idx_sessions_created_at', 'idx_org_members_org_user')"
    )
    index_names = {str(row["indexname"]) for row in rows}
    assert index_names == {"idx_sessions_created_at", "idx_org_members_org_user"}
