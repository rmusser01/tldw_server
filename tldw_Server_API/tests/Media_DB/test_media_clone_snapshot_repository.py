"""Repeatable source snapshot contracts for Media clone reads."""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from loguru import logger

from tldw_Server_API.app.core.DB_Management.backends.base import (
    BackendType,
    DatabaseConfig,
    QueryResult,
)
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.backends.sqlite_backend import SQLiteBackend
from tldw_Server_API.app.core.DB_Management.media_db import api as media_db_api
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context
from tldw_Server_API.app.core.Sharing.clone_models import CloneSnapshotUnavailable


@pytest.fixture
def media_db(tmp_path: Path) -> Iterator[MediaDatabase]:
    db_path = str(tmp_path / "clone-media.sqlite")
    backend = SQLiteBackend(
        DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=db_path)
    )
    database = MediaDatabase(db_path=db_path, client_id="owner-1", backend=backend)
    try:
        yield database
    finally:
        database.close_connection()
        backend.get_pool().close_all()


def _insert_transcript(
    db: MediaDatabase,
    media_id: int,
    transcription: str,
    *,
    run_id: int = 1,
    deleted: bool = False,
) -> None:
    db.execute_query(
        """
        INSERT INTO Transcripts (
            media_id, whisper_model, transcription, created_at, transcription_run_id,
            uuid, last_modified, version, client_id, deleted
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
        """,
        (
            media_id,
            "test-model",
            transcription,
            "2026-08-25T12:00:00+00:00",
            run_id,
            str(uuid.uuid4()),
            "2026-08-25T12:00:00+00:00",
            db.client_id,
            deleted,
        ),
        commit=True,
    )


def _seed_media(
    db: MediaDatabase,
    *,
    title: str,
    content: str,
    chunk_text: str,
    transcript_text: str,
    keywords: list[str] | None = None,
) -> int:
    media_id, _media_uuid, _message = db.add_media_with_keywords(
        url=f"https://example.test/{uuid.uuid4()}",
        title=title,
        media_type="document",
        content=content,
        keywords=keywords or [],
        chunks=[
            {
                "text": chunk_text,
                "start_char": 0,
                "end_char": len(chunk_text),
                "chunk_type": "text",
            }
        ],
    )
    assert media_id is not None
    _insert_transcript(db, int(media_id), transcript_text)
    return int(media_id)


@pytest.mark.unit
def test_media_clone_snapshot_preserves_requested_identity_order_and_collections(
    media_db: MediaDatabase,
) -> None:
    first_id = _seed_media(
        media_db,
        title="First",
        content="first content",
        chunk_text="first chunk",
        transcript_text="first transcript",
    )
    second_id = _seed_media(
        media_db,
        title="Second",
        content="second content",
        chunk_text="second chunk",
        transcript_text="second transcript",
    )

    snapshots = media_db.read_media_clone_snapshots([second_id, first_id])

    assert list(snapshots) == [second_id, first_id]
    assert snapshots[second_id].media["id"] == second_id
    assert snapshots[first_id].media["id"] == first_id
    assert snapshots[second_id].chunks[0]["chunk_text"] == "second chunk"
    assert snapshots[first_id].transcripts[0]["transcription"] == "first transcript"


@pytest.mark.unit
def test_media_clone_snapshot_materializes_only_active_keywords_and_children(
    media_db: MediaDatabase,
) -> None:
    media_id = _seed_media(
        media_db,
        title="Filtered",
        content="filtered content",
        chunk_text="active chunk",
        transcript_text="active transcript",
        keywords=["Zulu", "alpha", "hidden"],
    )
    media_db.execute_query(
        "UPDATE Keywords SET deleted = 1, version = version + 1, "
        "last_modified = ?, client_id = ? WHERE keyword = ?",
        ("2026-08-25T13:00:00+00:00", media_db.client_id, "hidden"),
        commit=True,
    )
    media_db.execute_query(
        "INSERT INTO UnvectorizedMediaChunks "
        "(media_id, chunk_text, chunk_index, uuid, last_modified, client_id, deleted) "
        "VALUES (?, ?, ?, ?, ?, ?, 1)",
        (
            media_id,
            "deleted chunk",
            999,
            str(uuid.uuid4()),
            "2026-08-25T12:00:00+00:00",
            media_db.client_id,
        ),
        commit=True,
    )
    _insert_transcript(
        media_db,
        media_id,
        "deleted transcript",
        run_id=2,
        deleted=True,
    )

    snapshot = media_db.read_media_clone_snapshots([media_id])[media_id]

    assert snapshot.media["keywords"] == ("alpha", "zulu")
    assert [row["chunk_text"] for row in snapshot.chunks] == ["active chunk"]
    assert [row["transcription"] for row in snapshot.transcripts] == [
        "active transcript"
    ]


@pytest.mark.unit
def test_media_clone_snapshot_package_facade_uses_narrow_repository_binding(
    media_db: MediaDatabase,
) -> None:
    media_id = _seed_media(
        media_db,
        title="Facade",
        content="facade content",
        chunk_text="facade chunk",
        transcript_text="facade transcript",
    )

    snapshots = media_db_api.read_media_clone_snapshots(media_db, [media_id])

    assert snapshots[media_id].media["title"] == "Facade"


@pytest.mark.unit
@pytest.mark.parametrize(
    "media_ids",
    [
        [True],
        [0],
        [-1],
        ["1"],
        [1, 1],
    ],
)
def test_media_clone_snapshot_rejects_invalid_or_duplicate_identities(
    media_db: MediaDatabase,
    media_ids: Sequence[Any],
) -> None:
    with pytest.raises(CloneSnapshotUnavailable):
        media_db.read_media_clone_snapshots(media_ids)


@pytest.mark.unit
def test_media_clone_snapshot_rejects_any_missing_or_inactive_reference(
    media_db: MediaDatabase,
) -> None:
    active_id = _seed_media(
        media_db,
        title="Active",
        content="active content",
        chunk_text="active chunk",
        transcript_text="active transcript",
    )
    deleted_id = _seed_media(
        media_db,
        title="Deleted",
        content="deleted content",
        chunk_text="deleted chunk",
        transcript_text="deleted transcript",
    )
    trashed_id = _seed_media(
        media_db,
        title="Trashed",
        content="trashed content",
        chunk_text="trashed chunk",
        transcript_text="trashed transcript",
    )
    media_db.execute_query(
        "UPDATE Media SET deleted = 1, version = version + 1, client_id = ? WHERE id = ?",
        (media_db.client_id, deleted_id),
        commit=True,
    )
    media_db.execute_query(
        "UPDATE Media SET is_trash = 1, version = version + 1, client_id = ? WHERE id = ?",
        (media_db.client_id, trashed_id),
        commit=True,
    )

    for media_ids in (
        [active_id, 999_999],
        [active_id, deleted_id],
        [active_id, trashed_id],
    ):
        with pytest.raises(CloneSnapshotUnavailable):
            media_db.read_media_clone_snapshots(media_ids)


@pytest.mark.unit
def test_media_clone_snapshot_does_not_mix_concurrent_collection_versions(
    media_db: MediaDatabase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media_id = _seed_media(
        media_db,
        title="Media v1",
        content="content v1",
        chunk_text="chunk v1",
        transcript_text="transcript v1",
        keywords=["alpha"],
    )
    writer_backend = SQLiteBackend(
        DatabaseConfig(
            backend_type=BackendType.SQLITE,
            sqlite_path=media_db.db_path_str,
        )
    )
    writer = MediaDatabase(
        db_path=media_db.db_path_str,
        client_id="writer-1",
        backend=writer_backend,
    )
    original_execute = media_db.backend.execute
    media_row_read = False

    def interleaved_execute(query, params=None, connection=None, **kwargs):
        nonlocal media_row_read
        result = original_execute(query, params, connection=connection, **kwargs)
        if not media_row_read and "FROM Media" in query:
            media_row_read = True
            with writer.transaction() as writer_conn:
                writer_conn.execute(
                    "UPDATE Media SET title = ?, content = ?, version = version + 1 WHERE id = ?",
                    ("Media v2", "content v2", media_id),
                )
                writer_conn.execute(
                    "UPDATE UnvectorizedMediaChunks SET chunk_text = ?, version = version + 1 "
                    "WHERE media_id = ?",
                    ("chunk v2", media_id),
                )
                writer_conn.execute(
                    "UPDATE Transcripts SET transcription = ?, version = version + 1 WHERE media_id = ?",
                    ("transcript v2", media_id),
                )
                writer_conn.execute(
                    "DELETE FROM MediaKeywords WHERE media_id = ?",
                    (media_id,),
                )
        return result

    monkeypatch.setattr(media_db.backend, "execute", interleaved_execute)
    try:
        snapshot = media_db.read_media_clone_snapshots([media_id])[media_id]
    finally:
        writer.close_connection()
        writer_backend.get_pool().close_all()

    assert media_row_read is True
    assert snapshot.media["title"] == "Media v1"
    assert snapshot.media["content"] == "content v1"
    assert snapshot.chunks[0]["chunk_text"] == "chunk v1"
    assert snapshot.transcripts[0]["transcription"] == "transcript v1"
    assert snapshot.media["keywords"] == ("alpha",)
    assert media_db.get_media_by_id(media_id)["title"] == "Media v2"
    assert media_db.execute_query(
        "SELECT k.keyword FROM MediaKeywords mk "
        "JOIN Keywords k ON k.id = mk.keyword_id WHERE mk.media_id = ?",
        (media_id,),
    ).fetchone() is None


@pytest.mark.unit
def test_media_clone_snapshot_rejects_private_memory_database() -> None:
    memory_db = MediaDatabase(db_path=":memory:", client_id="memory-source")
    try:
        with pytest.raises(CloneSnapshotUnavailable):
            memory_db.read_media_clone_snapshots([1])
    finally:
        memory_db.close_connection()
        memory_db.backend.get_pool().close_all()


@pytest.mark.unit
def test_media_clone_snapshot_reads_named_shared_cache_memory_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_uri = f"file:task3-media-{uuid.uuid4()}?mode=memory&cache=shared"
    literal_artifact = Path.cwd() / db_uri
    backend = SQLiteBackend(
        DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=db_uri)
    )
    memory_db = MediaDatabase(
        db_path=str(tmp_path / "media-facade.sqlite"),
        client_id="shared-memory",
        backend=backend,
    )
    try:
        assert not literal_artifact.exists()
        memory_db.execute_query(
            """
            CREATE TABLE IF NOT EXISTS Transcripts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                media_id INTEGER NOT NULL,
                whisper_model TEXT,
                transcription TEXT,
                created_at TEXT,
                transcription_run_id INTEGER,
                uuid TEXT NOT NULL,
                last_modified TEXT NOT NULL,
                version INTEGER NOT NULL,
                client_id TEXT NOT NULL,
                deleted INTEGER NOT NULL DEFAULT 0
            )
            """,
            commit=True,
        )
        media_id = _seed_media(
            memory_db,
            title="Shared memory",
            content="shared memory content",
            chunk_text="shared memory chunk",
            transcript_text="shared memory transcript",
            keywords=["shared"],
        )
        original_connect = sqlite3.connect
        snapshot_connects: list[tuple[str, bool]] = []

        def tracked_connect(database, *args, **kwargs):
            if database == db_uri:
                snapshot_connects.append((database, bool(kwargs.get("uri"))))
            return original_connect(database, *args, **kwargs)

        monkeypatch.setattr(sqlite3, "connect", tracked_connect)

        snapshot = memory_db.read_media_clone_snapshots([media_id])[media_id]

        assert snapshot.media["title"] == "Shared memory"
        assert snapshot.media["keywords"] == ("shared",)
        assert snapshot_connects == [(db_uri, True)]
        assert not literal_artifact.exists()
    finally:
        memory_db.close_connection()
        backend.get_pool().close_all()
    assert not literal_artifact.exists()


@pytest.mark.unit
def test_media_clone_snapshot_redacts_setup_failure(
    media_db: MediaDatabase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media_id = _seed_media(
        media_db,
        title="Failure",
        content="failure content",
        chunk_text="failure chunk",
        transcript_text="failure transcript",
    )
    messages: list[str] = []
    sink_id = logger.add(messages.append, format="{message}", level="WARNING")

    def fail_connect():
        raise sqlite3.OperationalError("sensitive media backend detail")

    monkeypatch.setattr(media_db.backend, "connect", fail_connect)

    try:
        with pytest.raises(CloneSnapshotUnavailable) as exc_info:
            media_db.read_media_clone_snapshots([media_id])
    finally:
        logger.remove(sink_id)

    assert str(exc_info.value) == "source_snapshot_unavailable"
    assert "sensitive media backend detail" not in repr(exc_info.value)
    assert "sensitive media backend detail" not in "".join(messages)
    assert "Media clone snapshot read failed" in "".join(messages)


@pytest.mark.unit
@pytest.mark.parametrize("missing", [False, True])
def test_media_clone_snapshot_closes_dedicated_handle_on_every_path(
    media_db: MediaDatabase,
    monkeypatch: pytest.MonkeyPatch,
    missing: bool,
) -> None:
    media_id = _seed_media(
        media_db,
        title="Cleanup",
        content="cleanup content",
        chunk_text="cleanup chunk",
        transcript_text="cleanup transcript",
    )
    caller_connection = media_db.get_connection()
    opened_connections = []
    release_states: list[bool] = []
    original_connect = media_db.backend.connect
    original_disconnect = media_db.backend.disconnect

    def tracked_connect():
        connection = original_connect()
        opened_connections.append(connection)
        return connection

    def tracked_disconnect(connection) -> None:
        release_states.append(bool(connection.in_transaction))
        original_disconnect(connection)

    monkeypatch.setattr(media_db.backend, "connect", tracked_connect)
    monkeypatch.setattr(media_db.backend, "disconnect", tracked_disconnect)

    if missing:
        with pytest.raises(CloneSnapshotUnavailable):
            media_db.read_media_clone_snapshots([media_id, 999_999])
    else:
        media_db.read_media_clone_snapshots([media_id])

    assert len(opened_connections) == 1
    assert opened_connections[0] is not caller_connection
    assert release_states == [False]
    with pytest.raises(sqlite3.ProgrammingError):
        opened_connections[0].execute("SELECT 1")


@pytest.mark.integration
@pytest.mark.postgres
@pytest.mark.timeout(120)
def test_postgres_media_clone_snapshot_is_read_only_repeatable_read(
    pg_database_config: DatabaseConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = MediaDatabase(db_path=":memory:", client_id="901", backend=backend)
    observed_modes: dict[str, str] = {}
    observed_scope: dict[str, str] = {}
    try:
        with scoped_context(user_id=901, org_ids=[12, 13], team_ids=[77], is_admin=True):
            media_id = _seed_media(
                db,
                title="Postgres v1",
                content="postgres content v1",
                chunk_text="postgres chunk v1",
                transcript_text="postgres transcript v1",
            )
            original_execute = backend.execute

            def interleaved_execute(query, params=None, connection=None, **kwargs):
                result = original_execute(query, params, connection=connection, **kwargs)
                if not observed_modes and "FROM Media" in query:
                    with connection.cursor() as cursor:
                        cursor.execute(
                            "SELECT current_setting('app.current_user_id', true) AS current_user_id, "
                            "current_setting('app.user_id', true) AS user_id, "
                            "current_setting('app.org_ids', true) AS org_ids, "
                            "current_setting('app.team_ids', true) AS team_ids, "
                            "current_setting('app.is_admin', true) AS is_admin, "
                            "current_role::text AS current_role, session_user::text AS session_user"
                        )
                        observed_scope.update(cursor.fetchone())
                        cursor.execute("SHOW transaction_isolation")
                        isolation_row = cursor.fetchone()
                        cursor.execute("SHOW transaction_read_only")
                        read_only_row = cursor.fetchone()
                    observed_modes["isolation"] = str(next(iter(isolation_row.values())))
                    observed_modes["read_only"] = str(next(iter(read_only_row.values())))

                    writer = backend.get_pool().get_connection()
                    try:
                        writer.commit()
                        with writer.cursor() as cursor:
                            cursor.execute(
                                "UPDATE Media SET title = %s, content = %s, version = version + 1 "
                                "WHERE id = %s",
                                ("Postgres v2", "postgres content v2", media_id),
                            )
                            cursor.execute(
                                "UPDATE UnvectorizedMediaChunks "
                                "SET chunk_text = %s, version = version + 1 WHERE media_id = %s",
                                ("postgres chunk v2", media_id),
                            )
                            cursor.execute(
                                "UPDATE Transcripts SET transcription = %s, version = version + 1 "
                                "WHERE media_id = %s",
                                ("postgres transcript v2", media_id),
                            )
                        writer.commit()
                    finally:
                        backend.get_pool().return_connection(writer)
                return result

            monkeypatch.setattr(backend, "execute", interleaved_execute)
            snapshot = db.read_media_clone_snapshots([media_id])[media_id]

        assert observed_modes == {"isolation": "repeatable read", "read_only": "on"}
        assert observed_scope["current_user_id"] == "901"
        assert observed_scope["user_id"] == "901"
        assert observed_scope["org_ids"] == "12,13"
        assert observed_scope["team_ids"] == "77"
        assert observed_scope["is_admin"] == "1"
        assert observed_scope["current_role"] == observed_scope["session_user"]
        assert snapshot.media["title"] == "Postgres v1"
        assert snapshot.chunks[0]["chunk_text"] == "postgres chunk v1"
        assert snapshot.transcripts[0]["transcription"] == "postgres transcript v1"
    finally:
        db.close_connection()
        backend.get_pool().close_all()


@pytest.mark.integration
@pytest.mark.postgres
@pytest.mark.timeout(120)
@pytest.mark.parametrize("outcome", ["success", "missing", "scope_apply_failure"])
def test_postgres_media_clone_snapshot_returns_idle_dedicated_connection_on_every_path(
    pg_database_config: DatabaseConfig,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = MediaDatabase(db_path=":memory:", client_id="901", backend=backend)
    pool = backend.get_pool()
    try:
        with scoped_context(user_id=901, org_ids=[], team_ids=[], is_admin=True):
            media_id = _seed_media(
                db,
                title="Cleanup",
                content="cleanup content",
                chunk_text="cleanup chunk",
                transcript_text="cleanup transcript",
            )
            caller_connection = db.get_connection()

        original_get_connection = pool.get_connection
        original_return_connection = pool.return_connection
        acquired_connections = []
        returned_connections: list[tuple[object, str]] = []
        table_queries: list[str] = []

        def tracked_get_connection():
            connection = original_get_connection()
            acquired_connections.append(connection)
            return connection

        def tracked_return_connection(connection) -> None:
            status = getattr(connection.info.transaction_status, "name", "")
            returned_connections.append((connection, str(status)))
            original_return_connection(connection)

        original_execute = backend.execute

        def tracked_execute(query, params=None, connection=None, **kwargs):
            if any(
                table_name in query
                for table_name in (
                    "FROM Media",
                    "FROM UnvectorizedMediaChunks",
                    "FROM Transcripts",
                    "FROM MediaKeywords",
                )
            ):
                table_queries.append(query)
            return original_execute(query, params, connection=connection, **kwargs)

        monkeypatch.setattr(pool, "get_connection", tracked_get_connection)
        monkeypatch.setattr(pool, "return_connection", tracked_return_connection)
        monkeypatch.setattr(backend, "execute", tracked_execute)

        if outcome == "scope_apply_failure":
            monkeypatch.setenv("TLDW_CONTENT_PG_ROLE_SWITCH", "1")
            monkeypatch.setenv(
                "TLDW_CONTENT_PG_ROLE_WHITELIST",
                "task3_missing_snapshot_role",
            )
            scope_kwargs = {
                "user_id": 901,
                "org_ids": [],
                "team_ids": [],
                "is_admin": True,
                "session_role": "task3_missing_snapshot_role",
            }
        else:
            scope_kwargs = {
                "user_id": 901,
                "org_ids": [],
                "team_ids": [],
                "is_admin": True,
            }

        with scoped_context(**scope_kwargs):
            if outcome == "success":
                db.read_media_clone_snapshots([media_id])
            else:
                with pytest.raises(CloneSnapshotUnavailable):
                    db.read_media_clone_snapshots(
                        [999_999 if outcome == "missing" else media_id]
                    )

        assert len(acquired_connections) == 1
        snapshot_connection = acquired_connections[0]
        assert snapshot_connection is not caller_connection
        assert returned_connections == [(snapshot_connection, "IDLE")]
        if outcome == "scope_apply_failure":
            assert table_queries == []
    finally:
        db.close_connection()
        pool.close_all()


@pytest.mark.integration
@pytest.mark.postgres
@pytest.mark.timeout(120)
@pytest.mark.parametrize("failure_mode", ["scope_verification", "transaction_mode"])
def test_postgres_media_clone_snapshot_setup_mismatch_stops_before_table_queries(
    pg_database_config: DatabaseConfig,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = MediaDatabase(db_path=":memory:", client_id="901", backend=backend)
    pool = backend.get_pool()
    try:
        with scoped_context(user_id=901, org_ids=[], team_ids=[], is_admin=True):
            media_id = _seed_media(
                db,
                title="Setup mismatch",
                content="setup mismatch content",
                chunk_text="setup mismatch chunk",
                transcript_text="setup mismatch transcript",
            )
            caller_connection = db.get_connection()

            original_get_connection = pool.get_connection
            original_return_connection = pool.return_connection
            original_execute = backend.execute
            acquired_connections = []
            returned_connections: list[tuple[object, str]] = []
            table_queries: list[str] = []

            def tracked_get_connection():
                connection = original_get_connection()
                acquired_connections.append(connection)
                return connection

            def tracked_return_connection(connection) -> None:
                status = getattr(connection.info.transaction_status, "name", "")
                returned_connections.append((connection, str(status)))
                original_return_connection(connection)

            def mismatched_execute(query, params=None, connection=None, **kwargs):
                if any(
                    table_name in query
                    for table_name in (
                        "FROM Media",
                        "FROM UnvectorizedMediaChunks",
                        "FROM Transcripts",
                        "FROM MediaKeywords",
                    )
                ):
                    table_queries.append(query)
                if (
                    failure_mode == "scope_verification"
                    and "current_setting('app.current_user_id'" in query
                ):
                    return QueryResult(
                        rows=[
                            {
                                "current_user_id": "wrong-user",
                                "user_id": "901",
                                "org_ids": "",
                                "team_ids": "",
                                "is_admin": "1",
                                "row_security": "on",
                                "current_role": "postgres",
                                "session_user": "postgres",
                            }
                        ],
                        rowcount=1,
                    )
                if failure_mode == "transaction_mode" and query.strip().upper() == (
                    "SHOW TRANSACTION_READ_ONLY"
                ):
                    return QueryResult(
                        rows=[{"transaction_read_only": "off"}],
                        rowcount=1,
                    )
                return original_execute(query, params, connection=connection, **kwargs)

            monkeypatch.setattr(pool, "get_connection", tracked_get_connection)
            monkeypatch.setattr(pool, "return_connection", tracked_return_connection)
            monkeypatch.setattr(backend, "execute", mismatched_execute)

            with pytest.raises(CloneSnapshotUnavailable):
                db.read_media_clone_snapshots([media_id])

        assert table_queries == []
        assert len(acquired_connections) == 1
        snapshot_connection = acquired_connections[0]
        assert snapshot_connection is not caller_connection
        assert returned_connections == [(snapshot_connection, "IDLE")]
    finally:
        db.close_connection()
        pool.close_all()


@pytest.mark.integration
@pytest.mark.postgres
@pytest.mark.timeout(120)
def test_postgres_pool_scope_failure_log_does_not_emit_driver_text(
    pg_database_config: DatabaseConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = MediaDatabase(db_path=":memory:", client_id="901", backend=backend)
    pool = backend.get_pool()
    messages: list[str] = []
    sink_id = logger.add(messages.append, format="{message}", level="DEBUG")
    try:
        with scoped_context(user_id=901, org_ids=[], team_ids=[], is_admin=True):
            media_id = _seed_media(
                db,
                title="Bounded logging",
                content="bounded logging content",
                chunk_text="bounded logging chunk",
                transcript_text="bounded logging transcript",
            )

            def fail_pool_scope(_connection) -> None:
                raise RuntimeError("sensitive scope driver detail")

            monkeypatch.setattr(pool, "_apply_scope_settings", fail_pool_scope)
            db.read_media_clone_snapshots([media_id])
    finally:
        logger.remove(sink_id)
        db.close_connection()
        pool.close_all()

    assert "sensitive scope driver detail" not in "".join(messages)
