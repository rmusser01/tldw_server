from __future__ import annotations

import re

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.integration


def test_postgres_workspace_source_review_schema_backfill_and_batch_transition(
    pg_database_config: DatabaseConfig,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(db_path=":memory:", client_id="workspace-source-pg-test", backend=backend)

    try:
        db.upsert_workspace("ws-pg", "Postgres Workspace")
        backend.execute("DROP TABLE workspace_sources")
        backend.execute(
            """
            CREATE TABLE workspace_sources (
                id            TEXT NOT NULL,
                workspace_id  TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                media_id      INTEGER NOT NULL,
                title         TEXT NOT NULL,
                source_type   TEXT NOT NULL,
                url           TEXT,
                position      INTEGER NOT NULL DEFAULT 0,
                selected      BOOLEAN NOT NULL DEFAULT true,
                added_at      TEXT NOT NULL,
                version       INTEGER NOT NULL DEFAULT 1,
                PRIMARY KEY (workspace_id, id)
            )
            """
        )
        backend.execute(
            """
            INSERT INTO workspace_sources (
                id, workspace_id, media_id, title, source_type, added_at
            ) VALUES
                ('src-a', 'ws-pg', 1, 'A', 'pdf', '2026-01-02T03:04:05.000Z'),
                ('src-b', 'ws-pg', 2, 'B', 'pdf', '   '),
                ('src-c', 'ws-pg', 3, 'C', 'pdf', '2026-01-03T03:04:05.000Z')
            """
        )

        with db.transaction() as conn:
            db._ensure_workspace_subresource_schema_postgres(conn)

        column_rows = list(
            backend.execute(
                """
                SELECT column_name
                  FROM information_schema.columns
                 WHERE table_schema = current_schema()
                   AND table_name = 'workspace_sources'
                """
            )
        )
        columns = {row["column_name"] for row in column_rows}
        assert {
            "review_state",
            "review_state_updated_at",
            "reviewed_at",
            "reviewed_by_user_id",
        } <= columns

        migrated_rows = list(
            backend.execute(
                "SELECT * FROM workspace_sources WHERE workspace_id = ? ORDER BY id",
                ("ws-pg",),
            )
        )
        migrated_by_id = {row["id"]: row for row in migrated_rows}
        assert all(row["review_state"] == "unset" for row in migrated_rows)
        assert migrated_by_id["src-a"]["review_state_updated_at"] == migrated_by_id["src-a"]["added_at"]
        assert migrated_by_id["src-b"]["review_state_updated_at"].strip()
        assert migrated_by_id["src-b"]["review_state_updated_at"] != migrated_by_id["src-b"]["added_at"]
        iso_utc_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")
        assert all(
            iso_utc_pattern.fullmatch(row["review_state_updated_at"])
            for row in migrated_rows
        )

        fallback_age_rows = list(
            backend.execute(
                """
                SELECT ABS(EXTRACT(EPOCH FROM (
                    (CURRENT_TIMESTAMP AT TIME ZONE 'UTC') - CAST(? AS TIMESTAMP)
                ))) AS age_seconds
                """,
                (migrated_by_id["src-b"]["review_state_updated_at"],),
            )
        )
        assert float(fallback_age_rows[0]["age_seconds"]) < 30

        backend.execute(
            """
            UPDATE workspace_sources
               SET review_state = 'needs_review',
                   reviewed_at = '2026-01-04T03:04:05.000Z',
                   reviewed_by_user_id = 'stale-reviewer'
             WHERE workspace_id = 'ws-pg' AND id = 'src-c'
            """
        )
        with db.transaction() as conn:
            db._ensure_workspace_subresource_schema_postgres(conn)

        cleaned = list(
            backend.execute(
                """
                SELECT reviewed_at, reviewed_by_user_id
                  FROM workspace_sources
                 WHERE workspace_id = 'ws-pg' AND id = 'src-c'
                """
            )
        )[0]
        assert cleaned["reviewed_at"] is None
        assert cleaned["reviewed_by_user_id"] is None

        updated = db.update_workspace_source_review_states(
            "ws-pg",
            ["src-a", "src-a", "src-b"],
            "reviewed",
            "reviewer-pg",
        )

        assert [row["id"] for row in updated] == ["src-a", "src-b"]
        assert all(row["review_state"] == "reviewed" for row in updated)
        assert all(row["reviewed_by_user_id"] == "reviewer-pg" for row in updated)
        assert all(row["version"] == 2 for row in updated)
        unrelated = db.get_workspace_source("ws-pg", "src-c")
        assert unrelated["review_state"] == "needs_review"
        assert unrelated["version"] == 1
    finally:
        db.close_connection()
