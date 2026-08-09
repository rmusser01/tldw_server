from __future__ import annotations

import uuid

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig, DatabaseError
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.integration


def test_postgres_note_folder_schema_backfills_active_duplicates_and_uses_partial_index(
    pg_database_config: DatabaseConfig,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(db_path=":memory:", client_id="note-folder-pg-test", backend=backend)

    try:
        backend.execute("DROP INDEX IF EXISTS idx_note_folders_path_lower")
        backend.execute("ALTER TABLE note_folders DROP CONSTRAINT IF EXISTS note_folders_path_key")
        backend.execute("DELETE FROM note_folders")
        backend.execute(
            """
            INSERT INTO note_folders(sync_id, name, path, parent_id, deleted, client_id, version)
            VALUES
              (?, 'Inbox', 'Inbox', NULL, FALSE, 'test', 1),
              (?, 'inbox', 'inbox', NULL, FALSE, 'test', 1),
              (?, 'INBOX', 'INBOX', NULL, TRUE, 'test', 1)
            """,
            (str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())),
        )

        with db.transaction() as conn:
            db._ensure_note_folder_schema_postgres(conn)

        active_rows = list(
            backend.execute(
                "SELECT id, path FROM note_folders WHERE LOWER(path) = LOWER(?) AND deleted = FALSE",
                ("inbox",),
            )
        )
        all_rows = list(
            backend.execute(
                "SELECT id, path, deleted FROM note_folders WHERE LOWER(path) = LOWER(?)",
                ("inbox",),
            )
        )
        index_rows = list(
            backend.execute(
                """
                SELECT indexdef
                  FROM pg_indexes
                 WHERE tablename = 'note_folders'
                   AND indexname = 'idx_note_folders_path_lower'
                """
            )
        )
        sync_column_rows = list(
            backend.execute(
                """
                SELECT is_nullable
                  FROM information_schema.columns
                 WHERE table_schema = current_schema()
                   AND table_name = 'note_folders'
                   AND column_name = 'sync_id'
                """
            )
        )
        sync_index_rows = list(
            backend.execute(
                """
                SELECT indexdef
                  FROM pg_indexes
                 WHERE schemaname = current_schema()
                   AND tablename = 'note_folders'
                   AND indexname = 'idx_note_folders_sync_id_unique'
                """
            )
        )

        assert len(active_rows) == 1
        assert len(all_rows) == 2
        assert index_rows
        assert "WHERE" in index_rows[0]["indexdef"]
        assert "deleted" in index_rows[0]["indexdef"]
        assert sync_column_rows == [{"is_nullable": "NO"}]
        assert sync_index_rows
        assert "UNIQUE INDEX" in sync_index_rows[0]["indexdef"]

        with pytest.raises(DatabaseError):
            backend.execute(
                """
                INSERT INTO note_folders(sync_id, name, path, parent_id, deleted, client_id, version)
                VALUES (?, 'duplicate active', 'INBOX', NULL, FALSE, 'test', 1)
                """,
                (str(uuid.uuid4()),),
            )

        backend.execute(
            """
            INSERT INTO note_folders(sync_id, name, path, parent_id, deleted, client_id, version)
            VALUES (?, 'duplicate deleted', 'INBOX', NULL, TRUE, 'test', 1)
            """,
            (str(uuid.uuid4()),),
        )
    finally:
        db.close_connection()
