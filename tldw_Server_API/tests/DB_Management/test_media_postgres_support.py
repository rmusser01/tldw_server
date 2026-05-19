from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys
import threading
import types
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock
import uuid

import pytest

from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseError as BackendDatabaseError,
)
from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.media_db.repositories.media_repository import (
    MediaRepository,
)


def _insert_claim_record(db: MediaDatabase) -> None:
    """Helper to seed a minimal media + claim pair."""
    timestamp = db._get_current_utc_timestamp_str()
    media_uuid = str(uuid.uuid4())
    claim_uuid = str(uuid.uuid4())

    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO Media (uuid, title, type, content_hash, last_modified, version, client_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                media_uuid,
                "Test media",
                "text",
                f"media-hash-{media_uuid}",
                timestamp,
                1,
                db.client_id,
            ),
        )
        media_id = conn.execute(
            "SELECT id FROM Media WHERE uuid = ?",
            (media_uuid,),
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO Claims (media_id, chunk_index, claim_text, chunk_hash, extractor, extractor_version, created_at, uuid, last_modified, version, client_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                media_id,
                0,
                "Confidence in backend migrations",
                f"chunk-hash-{claim_uuid}",
                "pytest",
                "v1",
                timestamp,
                claim_uuid,
                timestamp,
                1,
                db.client_id,
            ),
        )


def test_rebuild_claims_fts_sqlite_populates_index(tmp_path: Path) -> None:
    db_path = tmp_path / "media.sqlite"
    db = MediaDatabase(str(db_path), client_id="test-client")
    try:
        _insert_claim_record(db)

        indexed = db.rebuild_claims_fts()
        assert indexed == 1

        cursor = db.execute_query("SELECT claim_text FROM claims_fts")
        rows = cursor.fetchall()
        assert rows
        first = rows[0]
        if isinstance(first, tuple):
            claim_text = first[0]
        else:
            mapping = first if isinstance(first, dict) else dict(first)
            claim_text = mapping.get("claim_text")
        assert claim_text == "Confidence in backend migrations"
    finally:
        db.close_connection()


def test_rebuild_claims_fts_postgres_uses_backend() -> None:

    db = MediaDatabase.__new__(MediaDatabase)
    db.backend_type = BackendType.POSTGRESQL
    db.backend = MagicMock()
    executed: List[Tuple[str, Any]] = []

    class _Conn:
        pass

    conn = _Conn()

    def fake_transaction():

        @contextmanager
        def _ctx():
            yield conn

        return _ctx()

    db.transaction = fake_transaction  # type: ignore[assignment]

    def record(conn_arg: Any, query: str, params: Any = None):
        assert conn_arg is conn
        executed.append((query, params))
        return MagicMock()

    db._execute_with_connection = MagicMock(side_effect=record)  # type: ignore[attr-defined]
    db._fetchone_with_connection = MagicMock(return_value={"total": 4})  # type: ignore[attr-defined]

    result = db.rebuild_claims_fts()

    assert result == 4
    db.backend.create_fts_table.assert_called_once_with(
        table_name="claims_fts",
        source_table="claims",
        columns=["claim_text"],
        connection=conn,
    )
    assert executed[0][0].startswith("UPDATE claims ")
    db._fetchone_with_connection.assert_called_once_with(
        conn,
        "SELECT COUNT(*) AS total FROM claims WHERE deleted = 0",
    )


def test_postgres_migrations_include_v6() -> None:

    migrations = MediaDatabase._get_postgres_migrations(MediaDatabase.__new__(MediaDatabase))
    assert 6 in migrations


def test_postgres_migrations_include_current_schema_version() -> None:

    migrations = MediaDatabase._get_postgres_migrations(MediaDatabase.__new__(MediaDatabase))
    assert MediaDatabase._CURRENT_SCHEMA_VERSION in migrations


def test_postgres_migrations_include_v23() -> None:

    migrations = MediaDatabase._get_postgres_migrations(MediaDatabase.__new__(MediaDatabase))
    assert 23 in migrations


def test_run_postgres_migrate_to_v23_executes_transcript_run_history_updates() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema.migration_bodies import (
        postgres_transcript_run_history as postgres_transcript_run_history_module,
    )

    conn = object()
    calls: list[tuple[str, tuple[object, ...] | None, object]] = []

    class FakeBackend:
        def escape_identifier(self, name: str) -> str:
            return f'"{name}"'

        def execute(
            self,
            query: str,
            params: tuple[object, ...] | None = None,
            *,
            connection: object,
        ) -> None:
            calls.append((query, params, connection))

    db = MediaDatabase.__new__(MediaDatabase)
    db.backend = FakeBackend()

    postgres_transcript_run_history_module.run_postgres_migrate_to_v23(db, conn)

    executed_sql = "\n".join(query for query, _params, _connection in calls)
    assert 'ALTER TABLE "media" ADD COLUMN IF NOT EXISTS "latest_transcription_run_id" BIGINT' in executed_sql
    assert 'ALTER TABLE "transcripts" ADD COLUMN IF NOT EXISTS "transcription_run_id" BIGINT' in executed_sql
    assert 'CREATE UNIQUE INDEX IF NOT EXISTS "idx_transcripts_media_run_id"' in executed_sql
    assert 'WHERE "transcription_run_id" IS NOT NULL' in executed_sql
    assert 'CREATE UNIQUE INDEX IF NOT EXISTS "idx_transcripts_media_idempotency_key"' in executed_sql
    assert 'WHERE "idempotency_key" IS NOT NULL' in executed_sql
    assert 'DROP CONSTRAINT IF EXISTS transcripts_media_id_whisper_model_key' in executed_sql
    assert "ROW_NUMBER() OVER (" in executed_sql
    assert "PARTITION BY media_id" in executed_sql
    assert "ORDER BY created_at NULLS FIRST, id ASC" in executed_sql
    assert "WHERE deleted = FALSE" in executed_sql


@pytest.mark.integration
def test_fresh_postgres_schema_enforces_transcript_run_history_uniqueness(
    pg_database_config: DatabaseConfig,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = MediaDatabase(db_path=":memory:", client_id="pg-bootstrap-v23", backend=backend)

    try:
        media_uuid = str(uuid.uuid4())
        now = db._get_current_utc_timestamp_str()

        with backend.transaction() as conn:
            backend.execute(
                """
                INSERT INTO Media (uuid, title, type, content_hash, last_modified, version, client_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    media_uuid,
                    "Bootstrap uniqueness",
                    "audio",
                    f"hash-{media_uuid}",
                    now,
                    1,
                    db.client_id,
                ),
                connection=conn,
            )
            media_id = int(
                backend.execute(
                    "SELECT id FROM media WHERE uuid = %s",
                    (media_uuid,),
                    connection=conn,
                ).scalar
            )
            backend.execute(
                """
                INSERT INTO Transcripts (
                    media_id, whisper_model, transcription, created_at, transcription_run_id,
                    idempotency_key, uuid, last_modified, version, client_id, deleted
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    media_id,
                    "small",
                    "baseline transcript",
                    now,
                    1,
                    "job-1",
                    str(uuid.uuid4()),
                    now,
                    1,
                    db.client_id,
                    False,
                ),
                connection=conn,
            )
            backend.execute(
                """
                INSERT INTO Transcripts (
                    media_id, whisper_model, transcription, created_at, transcription_run_id,
                    idempotency_key, uuid, last_modified, version, client_id, deleted
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    media_id,
                    "medium",
                    "nullable key transcript",
                    now,
                    2,
                    None,
                    str(uuid.uuid4()),
                    now,
                    1,
                    db.client_id,
                    False,
                ),
                connection=conn,
            )
            backend.execute(
                """
                INSERT INTO Transcripts (
                    media_id, whisper_model, transcription, created_at, transcription_run_id,
                    idempotency_key, uuid, last_modified, version, client_id, deleted
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    media_id,
                    "large",
                    "second nullable key transcript",
                    now,
                    3,
                    None,
                    str(uuid.uuid4()),
                    now,
                    1,
                    db.client_id,
                    False,
                ),
                connection=conn,
            )

        with pytest.raises(BackendDatabaseError, match="unique|duplicate key"):
            with backend.transaction() as conn:
                backend.execute(
                    """
                    INSERT INTO Transcripts (
                        media_id, whisper_model, transcription, created_at, transcription_run_id,
                        idempotency_key, uuid, last_modified, version, client_id, deleted
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        media_id,
                        "duplicate-run",
                        "duplicate run transcript",
                        now,
                        1,
                        "job-2",
                        str(uuid.uuid4()),
                        now,
                        1,
                        db.client_id,
                        False,
                    ),
                    connection=conn,
                )

        with pytest.raises(BackendDatabaseError, match="unique|duplicate key"):
            with backend.transaction() as conn:
                backend.execute(
                    """
                    INSERT INTO Transcripts (
                        media_id, whisper_model, transcription, created_at, transcription_run_id,
                        idempotency_key, uuid, last_modified, version, client_id, deleted
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        media_id,
                        "duplicate-key",
                        "duplicate idempotency transcript",
                        now,
                        4,
                        "job-1",
                        str(uuid.uuid4()),
                        now,
                        1,
                        db.client_id,
                        False,
                    ),
                    connection=conn,
                )
    finally:
        db.close_connection()


def test_postgres_migrate_to_v6_creates_identifier_table() -> None:

    db = MediaDatabase.__new__(MediaDatabase)
    db.backend_type = BackendType.POSTGRESQL
    db.backend = MagicMock()

    conn = object()
    db.backend.escape_identifier.side_effect = lambda value: f'"{value}"'

    db._postgres_migrate_to_v6(conn)

    create_call = db.backend.execute.call_args_list[0]
    sql_text = create_call.args[0]
    # Accept either CamelCase (legacy) or lowercase (standardized) identifiers
    assert (
        'CREATE TABLE IF NOT EXISTS "DocumentVersionIdentifiers"' in sql_text
        or 'CREATE TABLE IF NOT EXISTS "documentversionidentifiers"' in sql_text
    )
    assert 'REFERENCES "DocumentVersions"' in sql_text or 'REFERENCES "documentversions"' in sql_text

    index_calls = [call.args[0] for call in db.backend.execute.call_args_list[1:]]
    expected_indices = {
        "idx_dvi_doi",
        "idx_dvi_pmid",
        "idx_dvi_pmcid",
        "idx_dvi_arxiv",
        "idx_dvi_s2",
    }
    assert all(any(name in sql for sql in index_calls) for name in expected_indices)


def test_update_fts_media_postgres_updates_vector() -> None:

    db = MediaDatabase.__new__(MediaDatabase)
    db.backend_type = BackendType.POSTGRESQL
    db._execute_with_connection = MagicMock()  # type: ignore[attr-defined]

    conn = object()
    db._update_fts_media(conn, 9, "Title", "Body")

    db._execute_with_connection.assert_called_once()
    sql, params = db._execute_with_connection.call_args.args[1:]
    assert "UPDATE media SET media_fts_tsv" in sql
    assert params == (9,)


def test_delete_fts_media_postgres_nulls_vector() -> None:

    db = MediaDatabase.__new__(MediaDatabase)
    db.backend_type = BackendType.POSTGRESQL
    db._execute_with_connection = MagicMock()  # type: ignore[attr-defined]

    conn = object()
    db._delete_fts_media(conn, 11)

    db._execute_with_connection.assert_called_once()
    sql, params = db._execute_with_connection.call_args.args[1:]
    assert sql.strip().startswith("UPDATE media SET media_fts_tsv = NULL")
    assert params == (11,)


def test_update_fts_keyword_postgres_updates_vector() -> None:

    db = MediaDatabase.__new__(MediaDatabase)
    db.backend_type = BackendType.POSTGRESQL
    db._execute_with_connection = MagicMock()  # type: ignore[attr-defined]

    conn = object()
    db._update_fts_keyword(conn, 5, "science")

    db._execute_with_connection.assert_called_once()
    sql, params = db._execute_with_connection.call_args.args[1:]
    assert "UPDATE keywords SET keyword_fts_tsv" in sql
    assert params == (5,)


def test_delete_fts_keyword_postgres_nulls_vector() -> None:

    db = MediaDatabase.__new__(MediaDatabase)
    db.backend_type = BackendType.POSTGRESQL
    db._execute_with_connection = MagicMock()  # type: ignore[attr-defined]

    conn = object()
    db._delete_fts_keyword(conn, 7)

    db._execute_with_connection.assert_called_once()
    sql, params = db._execute_with_connection.call_args.args[1:]
    assert sql.strip().startswith("UPDATE keywords SET keyword_fts_tsv = NULL")
    assert params == (7,)


def test_runtime_fts_ops_update_fts_media_sqlite_preserves_synonym_expansion_and_fallback(
    monkeypatch,
) -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.runtime import fts_ops

    class _Conn:
        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple[object, ...]]] = []

        def execute(self, sql: str, params: tuple[object, ...]):
            self.calls.append((sql, params))
            return None

    class _Db:
        backend_type = BackendType.SQLITE

    module_name = "tldw_Server_API.app.core.RAG.rag_service.synonyms_registry"
    synonym_module = types.ModuleType(module_name)

    def fake_get_corpus_synonyms(_corpus):
        return {"title": ["alias-one"], "body": ["alias-two"]}

    synonym_module.get_corpus_synonyms = fake_get_corpus_synonyms  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module_name, synonym_module)
    monkeypatch.setenv("DEFAULT_FTS_CORPUS", "test-corpus")
    monkeypatch.setenv("FTS_SYNONYM_EXPANSION_LIMIT", "1")

    conn = _Conn()
    fts_ops._update_fts_media(_Db(), conn, 9, "Title", "Body")

    assert len(conn.calls) == 1
    sql, params = conn.calls[0]
    assert sql.startswith("INSERT OR REPLACE INTO media_fts")
    assert params[0:2] == (9, "Title")
    assert params[2] in {"Body alias-one", "Body alias-two"}

    def raising_get_corpus_synonyms(_corpus):
        raise RuntimeError("boom")

    synonym_module.get_corpus_synonyms = raising_get_corpus_synonyms  # type: ignore[attr-defined]
    conn_fallback = _Conn()
    fts_ops._update_fts_media(_Db(), conn_fallback, 10, "Title", "Body")

    assert conn_fallback.calls[0][1] == (10, "Title", "Body")


def test_runtime_fts_ops_sync_refresh_noops_when_update_payload_has_no_relevant_fields() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.runtime import fts_ops

    class _Db:
        def __init__(self) -> None:
            self.deleted_media: list[int] = []
            self.updated_media: list[tuple[int, str, object]] = []
            self.deleted_keywords: list[int] = []
            self.updated_keywords: list[tuple[int, str]] = []

        def _fetchone_with_connection(self, _conn, query: str, _params=None):
            if "FROM Media" in query:
                return {"id": 7, "title": "Title", "content": "Body", "deleted": 0}
            if "FROM Keywords" in query:
                return {"id": 5, "keyword": "science", "deleted": 0}
            raise AssertionError(f"unexpected query: {query}")

        def _delete_fts_media(self, _conn, media_id: int) -> None:
            self.deleted_media.append(media_id)

        def _update_fts_media(self, _conn, media_id: int, title: str, content: object) -> None:
            self.updated_media.append((media_id, title, content))

        def _delete_fts_keyword(self, _conn, keyword_id: int) -> None:
            self.deleted_keywords.append(keyword_id)

        def _update_fts_keyword(self, _conn, keyword_id: int, keyword: str) -> None:
            self.updated_keywords.append((keyword_id, keyword))

    db = _Db()
    conn = object()

    fts_ops.sync_refresh_fts_for_entity(
        db,
        conn,
        entity="Media",
        entity_uuid="media-uuid",
        operation="update",
        payload={"author": "ignored"},
    )
    fts_ops.sync_refresh_fts_for_entity(
        db,
        conn,
        entity="Keywords",
        entity_uuid="kw-uuid",
        operation="update",
        payload={"confidence": 0.9},
    )

    assert db.deleted_media == []
    assert db.updated_media == []
    assert db.deleted_keywords == []
    assert db.updated_keywords == []


def test_media_repository_uses_execution_helper_for_postgres_chunk_persistence(monkeypatch) -> None:
    import tldw_Server_API.app.core.config as config_module

    monkeypatch.setattr(config_module, "rag_enable_structure_index", lambda: False, raising=False)

    executed: List[Tuple[str, Any]] = []

    class _Cursor:
        def __init__(
            self,
            *,
            fetchone_result: Dict[str, Any] | None = None,
            rowcount: int = 1,
            lastrowid: int | None = None,
        ) -> None:
            self._fetchone_result = fetchone_result
            self.rowcount = rowcount
            self.lastrowid = lastrowid

        def fetchone(self):
            return self._fetchone_result

    class _Conn:
        pass

    conn = _Conn()

    class _FakeDb:
        backend_type = BackendType.POSTGRESQL
        client_id = "pg-media-repo"
        backend = object()
        _media_insert_lock = threading.Lock()

        def __init__(self) -> None:
            self._uuid_counter = 0

        def _get_current_utc_timestamp_str(self) -> str:
            return "2026-03-15T23:40:00.000Z"

        def transaction(self):
            @contextmanager
            def _ctx():
                yield conn

            return _ctx()

        def _execute_with_connection(self, conn_arg: Any, query: str, params: Any = None):
            assert conn_arg is conn
            executed.append((query, params))
            if "INSERT INTO Media (" in query:
                return _Cursor(fetchone_result={"id": 41})
            return _Cursor()

        def _fetchone_with_connection(self, conn_arg: Any, query: str, params: Any = None):
            assert conn_arg is conn
            return None

        def _generate_uuid(self) -> str:
            self._uuid_counter += 1
            return f"uuid-{self._uuid_counter}"

        def _resolve_scope_ids(self) -> tuple[None, None]:
            return None, None

        def _log_sync_event(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def _update_fts_media(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def update_keywords_for_media(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def create_document_version(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    repo = MediaRepository(session=_FakeDb())

    media_id, media_uuid, message = repo.add_media_with_keywords(
        title="Chunked PG doc",
        content="chunk source",
        media_type="text",
        keywords=[],
        overwrite=True,
        chunks=[{"text": "chunk one", "chunk_type": "text"}],
    )

    assert media_id == 41
    assert media_uuid == "uuid-1"
    assert message == "Media 'Chunked PG doc' added."
    assert any("INSERT INTO UnvectorizedMediaChunks" in sql for sql, _ in executed)


def test_backup_database_postgres_returns_false(tmp_path: Path) -> None:
    db = MediaDatabase.__new__(MediaDatabase)
    db.backend_type = BackendType.POSTGRESQL
    db.backend = MagicMock()
    db.db_path_str = "postgresql://cluster"
    db.is_memory_db = False

    backup_path = tmp_path / "pg_backup.sql"
    result = db.backup_database(str(backup_path))

    assert result is False


def test_search_media_db_postgres_uses_tsquery():

    db = MediaDatabase.__new__(MediaDatabase)
    db.backend_type = BackendType.POSTGRESQL
    db.client_id = "pg-test"
    db.db_path_str = "pg-test-db"

    calls: List[Tuple[str, Tuple[Any, ...]]] = []

    class _CountCursor:
        def fetchone(self):
            return (1,)

    class _ResultCursor:
        def fetchall(self):
            return [{"id": 1, "title": "Deep Learning", "relevance_score": 0.5}]

    def fake_execute(sql: str, params: Tuple[Any, ...] | None = None):
        captured_params = tuple(params) if params is not None else tuple()
        calls.append((sql, captured_params))
        if "COUNT" in sql:
            return _CountCursor()
        return _ResultCursor()

    # Bind helpers expected by search_media_db
    db.execute_query = fake_execute  # type: ignore[assignment]
    db._append_case_insensitive_like = MediaDatabase._append_case_insensitive_like.__get__(db, MediaDatabase)

    results, total = db.search_media_db(
        "deep learning",
        search_fields=["title"],
        sort_by="relevance",
    )

    assert total == 1
    assert results and results[0]["title"] == "Deep Learning"

    count_sql, count_params = calls[0]
    assert "m.media_fts_tsv @@ to_tsquery('english', ?)" in count_sql
    assert count_params[0] == "deep & learning"

    result_sql, result_params = calls[1]
    assert "ts_rank(m.media_fts_tsv, to_tsquery('english', ?))" in result_sql
    assert result_params[0] == "deep & learning"
    assert result_params[1] == "deep & learning"


def test_search_media_db_postgres_uses_weighted_ts_rank_when_boost_fields_set():

    db = MediaDatabase.__new__(MediaDatabase)
    db.backend_type = BackendType.POSTGRESQL
    db.client_id = "pg-test"
    db.db_path_str = "pg-test-db"

    calls: List[Tuple[str, Tuple[Any, ...]]] = []

    class _CountCursor:
        def fetchone(self):
            return (1,)

    class _ResultCursor:
        def fetchall(self):
            return [{"id": 1, "title": "Deep Learning", "relevance_score": 0.5}]

    def fake_execute(sql: str, params: Tuple[Any, ...] | None = None):
        captured_params = tuple(params) if params is not None else tuple()
        calls.append((sql, captured_params))
        if "COUNT" in sql:
            return _CountCursor()
        return _ResultCursor()

    db.execute_query = fake_execute  # type: ignore[assignment]
    db._append_case_insensitive_like = MediaDatabase._append_case_insensitive_like.__get__(db, MediaDatabase)

    results, total = db.search_media_db(
        "deep learning",
        search_fields=["title", "content"],
        sort_by="relevance",
        boost_fields={"title": 4.0, "content": 0.5},
    )

    assert total == 1
    assert results and results[0]["title"] == "Deep Learning"
    result_sql, result_params = calls[1]
    assert "ts_rank(" in result_sql
    assert "0.500000,1.000000,0.500000,4.000000" in result_sql
    assert result_params[0] == "deep & learning"
    assert result_params[1] == "deep & learning"


def test_soft_delete_keyword_postgres_uses_backend_helpers() -> None:

    db = MediaDatabase.__new__(MediaDatabase)
    db.backend_type = BackendType.POSTGRESQL
    db.client_id = "tenant-42"

    conn = object()

    def fake_transaction():

        @contextmanager
        def _ctx():
            yield conn

        return _ctx()

    db.transaction = fake_transaction  # type: ignore[assignment]

    class FakeCursor:
        def __init__(self, rows=None, rowcount: int = 0):
            self._rows = rows or []
            self.rowcount = rowcount

        def fetchall(self):

            return self._rows

    db._fetchone_with_connection = MagicMock(  # type: ignore[attr-defined]
        side_effect=[
            {"id": 7, "uuid": "kw-uuid", "version": 2},
        ]
    )
    db._execute_with_connection = MagicMock(  # type: ignore[attr-defined]
        side_effect=[
            FakeCursor(rowcount=1),
            FakeCursor(rows=[{"media_id": 3, "media_uuid": "media-uuid"}]),
            FakeCursor(rowcount=1),
        ]
    )
    db._log_sync_event = MagicMock()  # type: ignore[attr-defined]
    db._delete_fts_keyword = MagicMock()  # type: ignore[attr-defined]

    assert db.soft_delete_keyword("Science") is True

    db._fetchone_with_connection.assert_called_once()  # type: ignore[attr-defined]
    update_call = db._execute_with_connection.call_args_list[0]  # type: ignore[attr-defined]
    assert "UPDATE Keywords" in update_call.args[1]
    db._delete_fts_keyword.assert_called_once_with(conn, 7)  # type: ignore[attr-defined]
    assert db._log_sync_event.call_count >= 2  # type: ignore[attr-defined]


def test_batch_insert_chunks_postgres_handles_dict_rows() -> None:

    db = MediaDatabase.__new__(MediaDatabase)
    db.backend_type = BackendType.POSTGRESQL
    db.client_id = "tenant-42"

    conn = object()

    def fake_transaction():

        @contextmanager
        def _ctx():
            yield conn

        return _ctx()

    db.transaction = fake_transaction  # type: ignore[assignment]

    base_chunk_count = 2

    def fetch_side_effect(connection, query, params=None):

        assert connection is conn
        if "SELECT 1 FROM Media" in query:
            return {"exists": 1}
        if "SELECT COUNT(*)" in query:
            return {"chunk_count": base_chunk_count}
        raise AssertionError(f"Unexpected query: {query}")

    db._fetchone_with_connection = MagicMock(side_effect=fetch_side_effect)  # type: ignore[attr-defined]
    db._executemany_with_connection = MagicMock()  # type: ignore[attr-defined]
    db._log_sync_event = MagicMock()  # type: ignore[attr-defined]
    db._get_current_utc_timestamp_str = MagicMock(return_value="2024-01-01T00:00:00Z")  # type: ignore[attr-defined]

    db._generate_uuid = MagicMock(return_value="uuid-1")  # type: ignore[attr-defined]

    result = db.batch_insert_chunks(
        99,
        [
            {"text": "First chunk", "metadata": {"start_index": 0, "end_index": 12}},
        ],
    )

    assert result == 1
    db._executemany_with_connection.assert_called_once()  # type: ignore[attr-defined]
    insert_call = db._executemany_with_connection.call_args  # type: ignore[attr-defined]
    assert insert_call.args[0] is conn
    assert "INSERT INTO MediaChunks" in insert_call.args[1]
    params_list = insert_call.args[2]
    expected_chunk_id = f"99_chunk_{base_chunk_count + 1}"
    assert params_list[0][4] == expected_chunk_id
    assert params_list[0][5] == "uuid-1"
    db._log_sync_event.assert_called_once()  # type: ignore[attr-defined]


def test_process_chunks_postgres_avoids_direct_connection_execute() -> None:

    db = MediaDatabase.__new__(MediaDatabase)
    db.backend_type = BackendType.POSTGRESQL
    db.client_id = "tenant-42"

    class FailingConnection:
        def execute(self, *args, **kwargs):
            raise AssertionError("execute should not be called directly on backend connections")

    check_conn = FailingConnection()
    db.get_connection = MagicMock(return_value=check_conn)  # type: ignore[attr-defined]

    tx_conn = object()

    def fake_transaction():

        @contextmanager
        def _ctx():
            yield tx_conn

        return _ctx()

    db.transaction = fake_transaction  # type: ignore[assignment]

    db._fetchone_with_connection = MagicMock(return_value={"exists": 1})  # type: ignore[attr-defined]
    db.execute_many = MagicMock()  # type: ignore[attr-defined]
    db._log_sync_event = MagicMock()  # type: ignore[attr-defined]
    db._get_current_utc_timestamp_str = MagicMock(return_value="2024-01-01T00:00:00Z")  # type: ignore[attr-defined]
    db._generate_uuid = MagicMock(side_effect=["chunk-id-1", "uuid-1"])  # type: ignore[attr-defined]

    db.process_chunks(
        11,
        [{"text": "Body", "start_index": 0, "end_index": 5}],
        batch_size=1,
    )

    db._fetchone_with_connection.assert_called_once()  # type: ignore[attr-defined]
    db.execute_many.assert_called_once()  # type: ignore[attr-defined]
    args, kwargs = db.execute_many.call_args  # type: ignore[attr-defined]
    assert "INSERT INTO MediaChunks" in args[0]
    db._log_sync_event.assert_called_once()  # type: ignore[attr-defined]
    args, kwargs = db._log_sync_event.call_args  # type: ignore[attr-defined]
    if kwargs:
        payload = kwargs["payload"]
        assert kwargs["conn"] is tx_conn
        assert kwargs["entity"] == "MediaChunks"
        assert kwargs["entity_uuid"] == "uuid-1"
        assert kwargs["operation"] == "create"
        assert kwargs["version"] == 1
    else:
        payload = args[5]
        assert args[0] is tx_conn
        assert args[1] == "MediaChunks"
        assert args[2] == "uuid-1"
        assert args[3] == "create"
        assert args[4] == 1
    assert payload == {
        "media_id": 11,
        "chunk_text": "Body",
        "start_index": 0,
        "end_index": 5,
        "chunk_id": "chunk-id-1",
        "uuid": "uuid-1",
        "last_modified": "2024-01-01T00:00:00Z",
        "version": 1,
        "client_id": "tenant-42",
        "deleted": 0,
    }
