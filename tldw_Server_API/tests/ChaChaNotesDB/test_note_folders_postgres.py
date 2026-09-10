from __future__ import annotations

import uuid

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig, DatabaseError
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.integration


def test_postgres_organization_uniqueness_is_owner_scoped(
    pg_database_config: DatabaseConfig,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(db_path=":memory:", client_id="910001", backend=backend)
    owners = ("910001", "910002")
    keyword_sync_id = str(uuid.uuid4())
    collection_sync_id = str(uuid.uuid4())
    folder_sync_id = str(uuid.uuid4())

    try:
        backend.execute(
            """
            INSERT INTO chacha_keywords(sync_id, keyword, client_id)
            VALUES (?, 'Shared tenancy probe', ?), (?, 'shared TENANCY probe', ?)
            """,
            (keyword_sync_id, owners[0], keyword_sync_id, owners[1]),
        )
        backend.execute(
            """
            INSERT INTO keyword_collections(sync_id, name, client_id)
            VALUES (?, 'Shared tenancy probe', ?), (?, 'shared TENANCY probe', ?)
            """,
            (collection_sync_id, owners[0], collection_sync_id, owners[1]),
        )
        backend.execute(
            """
            INSERT INTO note_folders(sync_id, name, path, client_id)
            VALUES (?, 'Shared tenancy probe', 'Shared/Tenancy/Probe', ?),
                   (?, 'shared TENANCY probe', 'shared/tenancy/probe', ?)
            """,
            (folder_sync_id, owners[0], folder_sync_id, owners[1]),
        )

        assert backend.execute(
            "SELECT COUNT(*) FROM chacha_keywords WHERE sync_id = ?",
            (keyword_sync_id,),
        ).scalar == 2
        assert backend.execute(
            "SELECT COUNT(*) FROM keyword_collections WHERE sync_id = ?",
            (collection_sync_id,),
        ).scalar == 2
        assert backend.execute(
            "SELECT COUNT(*) FROM note_folders WHERE sync_id = ?",
            (folder_sync_id,),
        ).scalar == 2

        with pytest.raises(DatabaseError):
            backend.execute(
                "INSERT INTO chacha_keywords(sync_id, keyword, client_id) "
                "VALUES (?, 'same owner collision', ?)",
                (keyword_sync_id, owners[0]),
            )
    finally:
        backend.execute(
            "DELETE FROM note_folders WHERE sync_id = ?",
            (folder_sync_id,),
        )
        backend.execute(
            "DELETE FROM keyword_collections WHERE sync_id = ?",
            (collection_sync_id,),
        )
        backend.execute(
            "DELETE FROM chacha_keywords WHERE sync_id = ?",
            (keyword_sync_id,),
        )
        db.close_connection()


def test_postgres_note_folder_schema_backfills_active_duplicates_and_uses_partial_index(
    pg_database_config: DatabaseConfig,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(db_path=":memory:", client_id="1", backend=backend)

    try:
        backend.execute("DROP INDEX IF EXISTS idx_note_folders_path_lower")
        backend.execute("ALTER TABLE note_folders DROP CONSTRAINT IF EXISTS note_folders_path_key")
        backend.execute("DELETE FROM note_folders")
        backend.execute(
            """
            INSERT INTO note_folders(sync_id, name, path, parent_id, deleted, client_id, version)
            VALUES
              (?, 'Inbox', 'Inbox', NULL, FALSE, '1', 1),
              (?, 'inbox', 'inbox', NULL, FALSE, '1', 1),
              (?, 'Other tenant inbox', 'INBOX', NULL, FALSE, '2', 1),
              (?, 'Deleted inbox', 'INBOX', NULL, TRUE, '1', 1)
            """,
            (
                str(uuid.uuid4()),
                str(uuid.uuid4()),
                str(uuid.uuid4()),
                str(uuid.uuid4()),
            ),
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

        assert len(active_rows) == 2
        assert len(all_rows) == 3
        assert index_rows
        assert "client_id" in index_rows[0]["indexdef"]
        assert "WHERE" in index_rows[0]["indexdef"]
        assert "deleted" in index_rows[0]["indexdef"]
        assert sync_column_rows == [{"is_nullable": "NO"}]
        assert sync_index_rows
        assert "UNIQUE INDEX" in sync_index_rows[0]["indexdef"]
        assert "client_id" in sync_index_rows[0]["indexdef"]

        with pytest.raises(DatabaseError):
            backend.execute(
                """
                INSERT INTO note_folders(sync_id, name, path, parent_id, deleted, client_id, version)
                VALUES (?, 'duplicate active', 'INBOX', NULL, FALSE, '1', 1)
                """,
                (str(uuid.uuid4()),),
            )

        backend.execute(
            """
            INSERT INTO note_folders(sync_id, name, path, parent_id, deleted, client_id, version)
            VALUES (?, 'duplicate deleted', 'INBOX', NULL, TRUE, '1', 1)
            """,
            (str(uuid.uuid4()),),
        )
    finally:
        db.close_connection()
