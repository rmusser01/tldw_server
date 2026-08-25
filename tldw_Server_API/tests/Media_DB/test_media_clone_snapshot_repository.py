"""Repeatable source snapshot contracts for Media clone reads."""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from loguru import logger

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
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
) -> None:
    db.execute_query(
        """
        INSERT INTO Transcripts (
            media_id, whisper_model, transcription, created_at, transcription_run_id,
            uuid, last_modified, version, client_id, deleted
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, 0)
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
) -> int:
    media_id, _media_uuid, _message = db.add_media_with_keywords(
        url=f"https://example.test/{uuid.uuid4()}",
        title=title,
        media_type="document",
        content=content,
        keywords=[],
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
    media_db.execute_query(
        "UPDATE Media SET deleted = 1, version = version + 1, client_id = ? WHERE id = ?",
        (media_db.client_id, deleted_id),
        commit=True,
    )

    for media_ids in ([active_id, 999_999], [active_id, deleted_id]):
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
    assert media_db.get_media_by_id(media_id)["title"] == "Media v2"


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
    try:
        with scoped_context(user_id=901, org_ids=[], team_ids=[], is_admin=True):
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
        assert snapshot.media["title"] == "Postgres v1"
        assert snapshot.chunks[0]["chunk_text"] == "postgres chunk v1"
        assert snapshot.transcripts[0]["transcription"] == "postgres transcript v1"
    finally:
        db.close_connection()
        backend.get_pool().close_all()
