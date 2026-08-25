"""Repeatable source snapshot contracts for Media clone reads."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sqlite3
import threading
import uuid
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from decimal import Decimal
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
from tldw_Server_API.app.core.DB_Management.media_db.errors import ConflictError, InputError
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context
from tldw_Server_API.app.core.Sharing.clone_models import (
    CloneSnapshotUnavailable,
    MediaCloneSnapshot,
)


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
    url: str | None = None,
) -> int:
    media_id, _media_uuid, _message = db.add_media_with_keywords(
        url=url or f"https://example.test/{uuid.uuid4()}",
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


def _operation_snapshot(
    *,
    original_url: str = "https://source.example.test/document",
    content: str | None = "source snapshot content",
    title: str = "Source snapshot",
    keywords: tuple[str, ...] = ("alpha", "Research"),
) -> MediaCloneSnapshot:
    return MediaCloneSnapshot.from_rows(
        media={
            "id": 41,
            "url": original_url,
            "title": title,
            "type": "document",
            "content": content,
            "author": "Source Author",
            "ingestion_date": "2026-08-25T10:00:00+00:00",
            "transcription_model": "snapshot-model",
            "content_hash": hashlib.sha256((content or "").encode("utf-8")).hexdigest(),
            "source_hash": hashlib.sha256(b"source-bytes").hexdigest(),
            "chunking_status": "completed",
            "latest_transcription_run_id": 2,
            "next_transcription_run_id": 3,
            "keywords": keywords,
            "vector_embedding": b"source-vector-must-not-copy",
            "vector_processing": 1,
            "uuid": "2d608c80-e428-49b4-becd-f50fa3ce7f23",
            "version": 7,
            "last_modified": datetime(2026, 8, 25, 10, 5, tzinfo=timezone.utc),
            "deleted": False,
            "is_trash": False,
        },
        chunks=(
            {
                "id": 501,
                "media_id": 41,
                "chunk_text": "first source chunk",
                "chunk_index": 0,
                "start_char": 0,
                "end_char": 18,
                "chunk_type": "text",
                "creation_date": "2026-08-25T10:01:00+00:00",
                "last_modified_orig": "2026-08-25T10:01:00+00:00",
                "is_processed": True,
                "metadata": {"page": 1, "confidence": Decimal("0.95")},
                "uuid": "c9639597-c16d-46de-934d-0211c646a97a",
                "last_modified": "2026-08-25T10:01:00+00:00",
                "version": 4,
                "client_id": "source-owner",
                "deleted": False,
            },
            {
                "id": 502,
                "media_id": 41,
                "chunk_text": "second source chunk",
                "chunk_index": 1,
                "start_char": 19,
                "end_char": 38,
                "chunk_type": "text",
                "creation_date": "2026-08-25T10:02:00+00:00",
                "last_modified_orig": "2026-08-25T10:02:00+00:00",
                "is_processed": True,
                "metadata": {"page": 2},
                "uuid": "f540e75a-ddf5-45a3-b95d-cb5ab444d39f",
                "last_modified": "2026-08-25T10:02:00+00:00",
                "version": 3,
                "client_id": "source-owner",
                "deleted": False,
            },
        ),
        transcripts=(
            {
                "id": 601,
                "media_id": 41,
                "whisper_model": "snapshot-model",
                "transcription": "latest source transcript",
                "created_at": "2026-08-25T10:03:00+00:00",
                "transcription_run_id": 2,
                "supersedes_run_id": 1,
                "idempotency_key": "source-transcript-2",
                "uuid": "3ae72579-32a2-4f0d-af33-4953983e7512",
                "last_modified": "2026-08-25T10:03:00+00:00",
                "version": 2,
                "client_id": "source-owner",
                "deleted": False,
            },
            {
                "id": 600,
                "media_id": 41,
                "whisper_model": "snapshot-model",
                "transcription": "first source transcript",
                "created_at": "2026-08-25T10:02:30+00:00",
                "transcription_run_id": 1,
                "supersedes_run_id": None,
                "idempotency_key": "source-transcript-1",
                "uuid": "208ee7b7-ea14-42c2-aaf4-5fa8b4a72d2e",
                "last_modified": "2026-08-25T10:02:30+00:00",
                "version": 1,
                "client_id": "source-owner",
                "deleted": False,
            },
        ),
    )


def _ordinary_media_state(db: MediaDatabase, media_id: int) -> dict[str, Any]:
    media = dict(
        db.execute_query(
            "SELECT * FROM Media WHERE id = ?",
            (media_id,),
        ).fetchone()
    )
    chunks = [
        dict(row)
        for row in db.execute_query(
            "SELECT * FROM UnvectorizedMediaChunks WHERE media_id = ? ORDER BY id",
            (media_id,),
        ).fetchall()
    ]
    transcripts = [
        dict(row)
        for row in db.execute_query(
            "SELECT * FROM Transcripts WHERE media_id = ? ORDER BY id",
            (media_id,),
        ).fetchall()
    ]
    return {"media": media, "chunks": chunks, "transcripts": transcripts}


@pytest.mark.unit
def test_operation_owned_snapshot_hash_is_canonical_bounded_and_type_safe() -> None:
    snapshot = _operation_snapshot()

    digest = media_db_api.hash_media_clone_snapshot(snapshot)

    reordered_mapping = MediaCloneSnapshot.from_rows(
        media=dict(reversed(tuple(snapshot.media.items()))),
        chunks=(
            dict(reversed(tuple(snapshot.chunks[0].items()))),
            dict(reversed(tuple(snapshot.chunks[1].items()))),
        ),
        transcripts=tuple(
            dict(reversed(tuple(row.items()))) for row in snapshot.transcripts
        ),
    )
    reordered_sequence = MediaCloneSnapshot.from_rows(
        media=snapshot.media,
        chunks=reversed(snapshot.chunks),
        transcripts=snapshot.transcripts,
    )
    mutated_nested_value = MediaCloneSnapshot.from_rows(
        media=snapshot.media,
        chunks=(
            {
                **dict(snapshot.chunks[0]),
                "metadata": {"page": 1, "confidence": Decimal("0.96")},
            },
            snapshot.chunks[1],
        ),
        transcripts=snapshot.transcripts,
    )
    string_typed_value = MediaCloneSnapshot.from_rows(
        media=snapshot.media,
        chunks=(
            {
                **dict(snapshot.chunks[0]),
                "metadata": {"page": "1", "confidence": Decimal("0.95")},
            },
            snapshot.chunks[1],
        ),
        transcripts=snapshot.transcripts,
    )

    assert len(digest) == 64
    assert digest == digest.lower()
    assert set(digest) <= set("0123456789abcdef")
    assert media_db_api.hash_media_clone_snapshot(reordered_mapping) == digest
    assert media_db_api.hash_media_clone_snapshot(reordered_sequence) == digest
    assert media_db_api.hash_media_clone_snapshot(mutated_nested_value) != digest
    assert media_db_api.hash_media_clone_snapshot(string_typed_value) != digest


@pytest.mark.unit
def test_operation_owned_snapshot_hash_uses_only_the_persisted_logical_projection() -> None:
    snapshot = _operation_snapshot()
    media = dict(snapshot.media)
    media.update(
        {
            "id": 999,
            "uuid": str(uuid.uuid4()),
            "version": 91,
            "last_modified": "2099-01-01T00:00:00+00:00",
            "deleted": True,
            "is_trash": True,
            "vector_embedding": b"different-source-vector",
            "vector_processing": 99,
            "ingestion_date": datetime(2026, 8, 25, 10, 0, tzinfo=timezone.utc),
        }
    )
    chunks = []
    for index, row in enumerate(reversed(snapshot.chunks), start=1):
        changed = dict(row)
        changed.update(
            {
                "id": 900 + index,
                "media_id": 999,
                "uuid": str(uuid.uuid4()),
                "last_modified": "2099-01-01T00:00:00+00:00",
                "version": 99,
                "client_id": "different-source-owner",
                "deleted": True,
                "is_processed": False,
            }
        )
        if changed["chunk_index"] == 0:
            changed["metadata"] = {"page": 1, "confidence": "0.95"}
        chunks.append(changed)
    transcripts = []
    for index, row in enumerate(reversed(snapshot.transcripts), start=1):
        changed = dict(row)
        changed.update(
            {
                "id": 800 + index,
                "media_id": 999,
                "uuid": str(uuid.uuid4()),
                "last_modified": "2099-01-01T00:00:00+00:00",
                "version": 99,
                "client_id": "different-source-owner",
                "deleted": True,
            }
        )
        transcripts.append(changed)
    equivalent = MediaCloneSnapshot.from_rows(
        media=media,
        chunks=chunks,
        transcripts=transcripts,
    )

    assert media_db_api.hash_media_clone_snapshot(equivalent) == (
        media_db_api.hash_media_clone_snapshot(snapshot)
    )

    long_url_prefix = "https://source.example.test/" + ("a" * 5000)
    first_long = _operation_snapshot(original_url=long_url_prefix + "first")
    second_long = _operation_snapshot(original_url=long_url_prefix + "second")
    assert media_db_api.hash_media_clone_snapshot(first_long) == (
        media_db_api.hash_media_clone_snapshot(second_long)
    )


@pytest.mark.unit
def test_operation_owned_snapshot_hash_rejects_unsupported_and_overdeep_values() -> None:
    unsupported = _operation_snapshot()
    unsupported_chunks = [dict(row) for row in unsupported.chunks]
    unsupported_chunks[0]["metadata"] = {"unsupported": complex(1, 2)}
    unsupported_snapshot = MediaCloneSnapshot.from_rows(
        media=unsupported.media,
        chunks=unsupported_chunks,
        transcripts=unsupported.transcripts,
    )

    nested: Any = "leaf"
    for _ in range(66):
        nested = (nested,)
    overdeep_chunks = [dict(row) for row in unsupported.chunks]
    overdeep_chunks[0]["metadata"] = {"nested": nested}
    overdeep_snapshot = MediaCloneSnapshot.from_rows(
        media=unsupported.media,
        chunks=overdeep_chunks,
        transcripts=unsupported.transcripts,
    )

    with pytest.raises(TypeError):
        media_db_api.hash_media_clone_snapshot(unsupported_snapshot)
    with pytest.raises(ValueError, match="nesting bound"):
        media_db_api.hash_media_clone_snapshot(overdeep_snapshot)


@pytest.mark.unit
def test_operation_owned_clone_replay_normalizes_persisted_json_sequences(
    media_db: MediaDatabase,
) -> None:
    snapshot = _operation_snapshot()
    chunks = [dict(row) for row in snapshot.chunks]
    chunks[0]["metadata"] = {"labels": ["first", 2, True]}
    snapshot = MediaCloneSnapshot.from_rows(
        media=snapshot.media,
        chunks=chunks,
        transcripts=snapshot.transcripts,
    )
    expected_hash = media_db_api.hash_media_clone_snapshot(snapshot)

    created = media_db.insert_operation_owned_clone_media(
        snapshot=snapshot,
        operation_id="clone-operation-json-sequence",
        source_identity="workspace-source-json-sequence",
        expected_content_hash=expected_hash,
    )
    replayed = media_db.insert_operation_owned_clone_media(
        snapshot=snapshot,
        operation_id="clone-operation-json-sequence",
        source_identity="workspace-source-json-sequence",
        expected_content_hash=expected_hash,
    )

    assert replayed.media_id == created.media_id
    assert replayed.replayed is True


@pytest.mark.unit
def test_operation_owned_clone_stays_hidden_until_exact_confirmation(
    media_db: MediaDatabase,
) -> None:
    snapshot = _operation_snapshot()
    operation_id = "clone-operation-staged"
    source_identity = "workspace-source-staged"
    expected_hash = media_db_api.hash_media_clone_snapshot(snapshot)

    created = media_db.insert_operation_owned_clone_media(
        snapshot=snapshot,
        operation_id=operation_id,
        source_identity=source_identity,
        expected_content_hash=expected_hash,
    )

    stored = dict(
        media_db.execute_query(
            "SELECT deleted, is_trash FROM Media WHERE id = ?",
            (created.media_id,),
        ).fetchone()
    )
    active_rows, *_ = media_db.get_paginated_media_list(page=1, results_per_page=100)
    trash_rows, *_ = media_db.get_paginated_trash_list(page=1, results_per_page=100)
    search_rows, _ = media_db.search_media_db("Source snapshot")
    pending = media_db.list_operation_owned_clone_media(
        operation_id=operation_id,
        limit=100,
    )

    assert stored == {"deleted": 0, "is_trash": 1}
    assert media_db.get_media_by_id(created.media_id) is None
    assert created.media_id not in {int(row["id"]) for row in active_rows}
    assert created.media_id not in {int(row["id"]) for row in trash_rows}
    assert created.media_id not in {int(row["id"]) for row in search_rows}
    assert len(pending) == 1
    assert dataclasses.is_dataclass(pending[0])
    assert not hasattr(pending[0], "__dict__")
    assert {field.name for field in dataclasses.fields(pending[0])} == {
        "media_id",
        "media_uuid",
        "source_identity",
        "expected_content_hash",
    }
    assert pending[0].media_id == created.media_id
    assert pending[0].media_uuid == created.media_uuid
    assert pending[0].source_identity == source_identity
    assert pending[0].expected_content_hash == expected_hash
    assert not hasattr(pending[0], "url")
    assert not hasattr(pending[0], "content")

    with pytest.raises(ConflictError):
        media_db.confirm_operation_owned_clone_media(
            operation_id=operation_id,
            source_identity=source_identity,
            expected_content_hash=hashlib.sha256(b"wrong-confirm-hash").hexdigest(),
        )

    assert media_db.confirm_operation_owned_clone_media(
        operation_id=operation_id,
        source_identity=source_identity,
        expected_content_hash=expected_hash,
    ) == 1
    confirmed = media_db.get_media_by_id(created.media_id)
    assert confirmed is not None
    assert confirmed["deleted"] in {0}
    assert confirmed["is_trash"] in {0}
    assert confirmed["system_operation_id"] is None
    assert confirmed["system_operation_kind"] is None
    assert confirmed["system_source_identity"] is None
    assert confirmed["system_content_hash"] is None
    assert media_db.list_operation_owned_clone_media(
        operation_id=operation_id,
        limit=100,
    ) == []


@pytest.mark.unit
def test_operation_owned_clone_enumeration_is_bounded_and_operation_correlated(
    media_db: MediaDatabase,
) -> None:
    snapshot = _operation_snapshot()
    first_operation = "clone-operation-reconcile-a"
    second_operation = "clone-operation-reconcile-b"
    first_hash = media_db_api.hash_media_clone_snapshot(snapshot)
    first = media_db.insert_operation_owned_clone_media(
        snapshot=snapshot,
        operation_id=first_operation,
        source_identity="orphan-source-a",
        expected_content_hash=first_hash,
    )
    second_snapshot = _operation_snapshot(title="Second orphan snapshot")
    second_hash = media_db_api.hash_media_clone_snapshot(second_snapshot)
    media_db.insert_operation_owned_clone_media(
        snapshot=second_snapshot,
        operation_id=second_operation,
        source_identity="orphan-source-b",
        expected_content_hash=second_hash,
    )

    listed = media_db_api.list_operation_owned_clone_media(
        media_db,
        operation_id=first_operation,
        limit=1,
    )

    assert [item.media_id for item in listed] == [first.media_id]
    for invalid_limit in (False, 0, 101):
        with pytest.raises(InputError):
            media_db.list_operation_owned_clone_media(
                operation_id=first_operation,
                limit=invalid_limit,
            )


@pytest.mark.unit
def test_operation_owned_clone_keywords_stay_hidden_and_cleanup_is_proven(
    media_db: MediaDatabase,
) -> None:
    ordinary_id = _seed_media(
        media_db,
        title="Ordinary keyword owner",
        content="ordinary keyword content",
        chunk_text="ordinary keyword chunk",
        transcript_text="ordinary keyword transcript",
        keywords=["recipient-existing"],
    )
    existing_before = dict(
        media_db.execute_query(
            "SELECT * FROM Keywords WHERE keyword = ?",
            ("recipient-existing",),
        ).fetchone()
    )
    snapshot = _operation_snapshot(
        keywords=("recipient-existing", "clone-only-staged"),
    )
    operation_id = "clone-operation-keyword-cleanup"
    source_identity = "workspace-source-keyword-cleanup"
    expected_hash = media_db_api.hash_media_clone_snapshot(snapshot)

    created = media_db.insert_operation_owned_clone_media(
        snapshot=snapshot,
        operation_id=operation_id,
        source_identity=source_identity,
        expected_content_hash=expected_hash,
    )

    assert media_db.fetch_all_keywords() == ["recipient-existing"]
    holds = [
        dict(row)
        for row in media_db.execute_query(
            "SELECT k.keyword, h.created_by_clone "
            "FROM OperationOwnedCloneKeywords h "
            "JOIN Keywords k ON k.id = h.keyword_id "
            "WHERE h.media_id = ? ORDER BY k.keyword",
            (created.media_id,),
        ).fetchall()
    ]
    assert holds == [
        {"keyword": "clone-only-staged", "created_by_clone": 1},
        {"keyword": "recipient-existing", "created_by_clone": 0},
    ]

    assert media_db.delete_operation_owned_clone_media(
        operation_id=operation_id,
        source_identity=source_identity,
        expected_content_hash=expected_hash,
    ) == 1
    assert media_db.execute_query(
        "SELECT COUNT(*) AS count FROM Keywords WHERE keyword = ?",
        ("clone-only-staged",),
    ).fetchone()["count"] == 0
    assert dict(
        media_db.execute_query(
            "SELECT * FROM Keywords WHERE keyword = ?",
            ("recipient-existing",),
        ).fetchone()
    ) == existing_before
    assert media_db.execute_query(
        "SELECT COUNT(*) AS count FROM MediaKeywords WHERE media_id = ?",
        (ordinary_id,),
    ).fetchone()["count"] == 1


@pytest.mark.unit
def test_operation_owned_clone_keyword_holds_are_shared_and_confirmation_releases(
    media_db: MediaDatabase,
) -> None:
    snapshot = _operation_snapshot(keywords=("clone-shared-staged",))
    expected_hash = media_db_api.hash_media_clone_snapshot(snapshot)
    identities = (
        ("clone-operation-keyword-shared-a", "workspace-source-keyword-shared-a"),
        ("clone-operation-keyword-shared-a", "workspace-source-keyword-shared-b"),
        ("clone-operation-keyword-shared-b", "workspace-source-keyword-shared-c"),
    )
    created = [
        media_db.insert_operation_owned_clone_media(
            snapshot=snapshot,
            operation_id=operation_id,
            source_identity=source_identity,
            expected_content_hash=expected_hash,
        )
        for operation_id, source_identity in identities
    ]

    assert "clone-shared-staged" not in media_db.fetch_all_keywords()
    assert media_db.execute_query(
        "SELECT COUNT(*) AS count FROM OperationOwnedCloneKeywords "
        "WHERE keyword_id = (SELECT id FROM Keywords WHERE keyword = ?)",
        ("clone-shared-staged",),
    ).fetchone()["count"] == 3

    for operation_id, source_identity in identities[:2]:
        assert media_db.delete_operation_owned_clone_media(
            operation_id=operation_id,
            source_identity=source_identity,
            expected_content_hash=expected_hash,
        ) == 1
    assert media_db.execute_query(
        "SELECT COUNT(*) AS count FROM Keywords WHERE keyword = ?",
        ("clone-shared-staged",),
    ).fetchone()["count"] == 1
    assert media_db.execute_query(
        "SELECT COUNT(*) AS count FROM OperationOwnedCloneKeywords "
        "WHERE created_by_clone = 1",
    ).fetchone()["count"] == 1

    final_operation, final_source = identities[2]
    assert media_db.confirm_operation_owned_clone_media(
        operation_id=final_operation,
        source_identity=final_source,
        expected_content_hash=expected_hash,
    ) == 1
    assert media_db.fetch_all_keywords() == ["clone-shared-staged"]
    assert media_db.execute_query(
        "SELECT COUNT(*) AS count FROM OperationOwnedCloneKeywords"
    ).fetchone()["count"] == 0
    assert media_db.get_media_by_id(created[2].media_id) is not None


@pytest.mark.unit
def test_ordinary_keyword_use_releases_clone_created_keyword_hold(
    media_db: MediaDatabase,
) -> None:
    snapshot = _operation_snapshot(keywords=("clone-keyword-adopted",))
    operation_id = "clone-operation-keyword-adopted"
    source_identity = "workspace-source-keyword-adopted"
    expected_hash = media_db_api.hash_media_clone_snapshot(snapshot)
    created = media_db.insert_operation_owned_clone_media(
        snapshot=snapshot,
        operation_id=operation_id,
        source_identity=source_identity,
        expected_content_hash=expected_hash,
    )
    keyword_before = dict(
        media_db.execute_query(
            "SELECT * FROM Keywords WHERE keyword = ?",
            ("clone-keyword-adopted",),
        ).fetchone()
    )

    ordinary_id = _seed_media(
        media_db,
        title="Ordinary keyword adopter",
        content="ordinary adopter content",
        chunk_text="ordinary adopter chunk",
        transcript_text="ordinary adopter transcript",
        keywords=["clone-keyword-adopted"],
    )

    assert media_db.fetch_all_keywords() == ["clone-keyword-adopted"]
    assert dict(
        media_db.execute_query(
            "SELECT * FROM Keywords WHERE keyword = ?",
            ("clone-keyword-adopted",),
        ).fetchone()
    ) == keyword_before
    assert media_db.execute_query(
        "SELECT created_by_clone FROM OperationOwnedCloneKeywords WHERE media_id = ?",
        (created.media_id,),
    ).fetchone()["created_by_clone"] == 0

    assert media_db.delete_operation_owned_clone_media(
        operation_id=operation_id,
        source_identity=source_identity,
        expected_content_hash=expected_hash,
    ) == 1
    assert media_db.execute_query(
        "SELECT COUNT(*) AS count FROM MediaKeywords WHERE media_id = ?",
        (ordinary_id,),
    ).fetchone()["count"] == 1
    assert media_db.fetch_all_keywords() == ["clone-keyword-adopted"]


@pytest.mark.unit
def test_operation_owned_clone_insert_isolated_from_url_and_content_collisions(
    media_db: MediaDatabase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _operation_snapshot()
    source_url = str(snapshot.media["url"])
    source_content = str(snapshot.media["content"])
    content_collision_id = _seed_media(
        media_db,
        title="Ordinary content collision",
        content=source_content,
        chunk_text="ordinary content chunk",
        transcript_text="ordinary content transcript",
        keywords=["ordinary-content"],
        url="https://recipient.example.test/content-collision",
    )
    url_collision_id = _seed_media(
        media_db,
        title="Ordinary URL collision",
        content="ordinary URL content",
        chunk_text="ordinary URL chunk",
        transcript_text="ordinary URL transcript",
        keywords=["ordinary-url"],
        url=source_url,
    )
    before = {
        media_id: _ordinary_media_state(media_db, media_id)
        for media_id in (content_collision_id, url_collision_id)
    }

    def forbidden_path(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("clone persistence entered an ordinary ingest/source-read path")

    monkeypatch.setattr(media_db, "add_media_with_keywords", forbidden_path)
    monkeypatch.setattr(media_db, "read_media_clone_snapshots", forbidden_path)
    operation_id = "clone-operation-001"
    source_identity = "workspace-source-41"
    expected_hash = media_db_api.hash_media_clone_snapshot(snapshot)

    result = media_db.insert_operation_owned_clone_media(
        snapshot=snapshot,
        operation_id=operation_id,
        source_identity=source_identity,
        expected_content_hash=expected_hash,
    )

    operation_digest = hashlib.sha256(operation_id.encode("utf-8")).hexdigest()
    source_digest = hashlib.sha256(source_identity.encode("utf-8")).hexdigest()
    expected_url = f"tldw-clone://workspace/{operation_digest}/{source_digest}"
    assert dataclasses.is_dataclass(result)
    assert result.media_id not in {content_collision_id, url_collision_id}
    assert result.media_uuid
    assert result.created is True
    assert result.replayed is False
    assert not hasattr(result, "__dict__")
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.created = False  # type: ignore[misc]

    owned_row = dict(
        media_db.execute_query(
            "SELECT * FROM Media WHERE id = ?",
            (result.media_id,),
        ).fetchone()
    )
    assert owned_row["url"] == expected_url
    assert owned_row["url"] != source_url
    assert owned_row["system_operation_id"] == operation_id
    assert owned_row["system_operation_kind"] == "shared_workspace_clone"
    assert owned_row["system_source_identity"] == source_identity
    assert owned_row["system_content_hash"] == expected_hash
    assert owned_row["vector_embedding"] is None
    assert owned_row["vector_processing"] == 0

    document_row = dict(
        media_db.execute_query(
            "SELECT content, prompt, analysis_content, safe_metadata, version_number "
            "FROM DocumentVersions WHERE media_id = ? AND deleted = 0",
            (result.media_id,),
        ).fetchone()
    )
    provenance = json.loads(document_row["safe_metadata"])
    assert document_row["content"] == source_content
    assert document_row["version_number"] == 1
    assert provenance == {
        "clone_provenance": {
            "source_url": source_url,
        }
    }

    keywords = [
        row["keyword"]
        for row in media_db.execute_query(
            "SELECT k.keyword FROM MediaKeywords mk "
            "JOIN Keywords k ON k.id = mk.keyword_id "
            "WHERE mk.media_id = ? AND k.deleted = 0 ORDER BY LOWER(k.keyword), k.keyword",
            (result.media_id,),
        ).fetchall()
    ]
    chunks = [
        dict(row)
        for row in media_db.execute_query(
            "SELECT chunk_text, chunk_index, start_char, end_char, chunk_type, "
            "is_processed, metadata FROM UnvectorizedMediaChunks "
            "WHERE media_id = ? AND deleted = 0 ORDER BY chunk_index",
            (result.media_id,),
        ).fetchall()
    ]
    transcripts = [
        dict(row)
        for row in media_db.execute_query(
            "SELECT whisper_model, transcription, transcription_run_id, "
            "supersedes_run_id, idempotency_key FROM Transcripts "
            "WHERE media_id = ? AND deleted = 0 ORDER BY transcription_run_id DESC",
            (result.media_id,),
        ).fetchall()
    ]
    assert keywords == ["alpha", "research"]
    assert [row["chunk_text"] for row in chunks] == [
        "first source chunk",
        "second source chunk",
    ]
    assert [row["is_processed"] for row in chunks] == [0, 0]
    assert [row["transcription"] for row in transcripts] == [
        "latest source transcript",
        "first source transcript",
    ]
    assert media_db.execute_query(
        "SELECT COUNT(*) AS count FROM MediaChunks WHERE media_id = ?",
        (result.media_id,),
    ).fetchone()["count"] == 0
    assert {
        media_id: _ordinary_media_state(media_db, media_id)
        for media_id in (content_collision_id, url_collision_id)
    } == before

    replay = media_db.insert_operation_owned_clone_media(
        snapshot=snapshot,
        operation_id=operation_id,
        source_identity=source_identity,
        expected_content_hash=expected_hash,
    )

    assert replay.media_id == result.media_id
    assert replay.media_uuid == result.media_uuid
    assert replay.created is False
    assert replay.replayed is True
    assert media_db.execute_query(
        "SELECT COUNT(*) AS count FROM Media WHERE system_operation_id = ?",
        (operation_id,),
    ).fetchone()["count"] == 1
    assert {
        media_id: _ordinary_media_state(media_db, media_id)
        for media_id in (content_collision_id, url_collision_id)
    } == before


@pytest.mark.unit
def test_operation_owned_clone_preserves_nullable_media_content(
    media_db: MediaDatabase,
) -> None:
    snapshot = _operation_snapshot(content=None, keywords=())
    operation_id = "clone-operation-null-content"
    source_identity = "workspace-source-null-content"
    expected_hash = media_db_api.hash_media_clone_snapshot(snapshot)

    created = media_db.insert_operation_owned_clone_media(
        snapshot=snapshot,
        operation_id=operation_id,
        source_identity=source_identity,
        expected_content_hash=expected_hash,
    )
    stored = media_db.execute_query(
        "SELECT m.content AS media_content, dv.content AS document_content "
        "FROM Media m JOIN DocumentVersions dv ON dv.media_id = m.id "
        "WHERE m.id = ?",
        (created.media_id,),
    ).fetchone()
    replayed = media_db.insert_operation_owned_clone_media(
        snapshot=snapshot,
        operation_id=operation_id,
        source_identity=source_identity,
        expected_content_hash=expected_hash,
    )

    assert stored["media_content"] is None
    assert stored["document_content"] == ""
    assert replayed.media_id == created.media_id
    assert replayed.replayed is True
    assert media_db.delete_operation_owned_clone_media(
        operation_id=operation_id,
        source_identity=source_identity,
        expected_content_hash=expected_hash,
    ) == 1


@pytest.mark.unit
def test_operation_owned_clone_replay_rejects_mutated_snapshot_and_candidate(
    media_db: MediaDatabase,
) -> None:
    snapshot = _operation_snapshot()
    expected_hash = media_db_api.hash_media_clone_snapshot(snapshot)
    operation_id = "clone-operation-replay"
    source_identity = "workspace-source-replay"
    created = media_db.insert_operation_owned_clone_media(
        snapshot=snapshot,
        operation_id=operation_id,
        source_identity=source_identity,
        expected_content_hash=expected_hash,
    )
    changed_snapshot = _operation_snapshot(content="different source snapshot content")

    with pytest.raises(ConflictError):
        media_db.insert_operation_owned_clone_media(
            snapshot=changed_snapshot,
            operation_id=operation_id,
            source_identity=source_identity,
            expected_content_hash=expected_hash,
        )

    replacement_hash = hashlib.sha256(b"different-owned-snapshot").hexdigest()
    media_db.execute_query(
        "UPDATE Media SET system_content_hash = ?, version = version + 1, "
        "last_modified = ?, client_id = ? WHERE id = ?",
        (
            replacement_hash,
            "2026-08-25T14:00:00+00:00",
            media_db.client_id,
            created.media_id,
        ),
        commit=True,
    )
    with pytest.raises(ConflictError):
        media_db.insert_operation_owned_clone_media(
            snapshot=snapshot,
            operation_id=operation_id,
            source_identity=source_identity,
            expected_content_hash=expected_hash,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "graph_mutation",
    [
        "media_title",
        "media_content",
        "document_removed",
        "document_changed",
        "keyword_removed",
        "keyword_changed",
        "chunk_removed",
        "chunk_changed",
        "transcript_removed",
        "transcript_changed",
    ],
)
def test_operation_owned_clone_replay_rehydrates_and_validates_the_logical_graph(
    media_db: MediaDatabase,
    graph_mutation: str,
) -> None:
    snapshot = _operation_snapshot()
    expected_hash = media_db_api.hash_media_clone_snapshot(snapshot)
    operation_id = f"clone-operation-graph-{graph_mutation}"
    source_identity = f"workspace-source-graph-{graph_mutation}"
    created = media_db.insert_operation_owned_clone_media(
        snapshot=snapshot,
        operation_id=operation_id,
        source_identity=source_identity,
        expected_content_hash=expected_hash,
    )

    if graph_mutation == "media_title":
        query = (
            "UPDATE Media SET title = ?, version = version + 1, "
            "last_modified = ?, client_id = ? WHERE id = ?"
        )
        params = (
            "tampered title",
            "2026-08-25T15:00:00+00:00",
            media_db.client_id,
            created.media_id,
        )
    elif graph_mutation == "media_content":
        query = (
            "UPDATE Media SET content = ?, version = version + 1, "
            "last_modified = ?, client_id = ? WHERE id = ?"
        )
        params = (
            "tampered content",
            "2026-08-25T15:00:00+00:00",
            media_db.client_id,
            created.media_id,
        )
    elif graph_mutation == "document_removed":
        query = "DELETE FROM DocumentVersions WHERE media_id = ?"
        params = (created.media_id,)
    elif graph_mutation == "document_changed":
        query = (
            "UPDATE DocumentVersions SET content = ?, version = version + 1, "
            "last_modified = ?, client_id = ? WHERE media_id = ?"
        )
        params = (
            "tampered document",
            "2026-08-25T15:00:00+00:00",
            media_db.client_id,
            created.media_id,
        )
    elif graph_mutation == "keyword_removed":
        query = "DELETE FROM MediaKeywords WHERE media_id = ? AND id = (SELECT MIN(id) FROM MediaKeywords WHERE media_id = ?)"
        params = (created.media_id, created.media_id)
    elif graph_mutation == "keyword_changed":
        query = (
            "UPDATE Keywords SET keyword = ?, version = version + 1, "
            "last_modified = ?, client_id = ? WHERE id = "
            "(SELECT MIN(keyword_id) FROM MediaKeywords WHERE media_id = ?)"
        )
        params = (
            "tampered-keyword",
            "2026-08-25T15:00:00+00:00",
            media_db.client_id,
            created.media_id,
        )
    elif graph_mutation == "chunk_removed":
        query = (
            "DELETE FROM UnvectorizedMediaChunks WHERE media_id = ? AND id = "
            "(SELECT MIN(id) FROM UnvectorizedMediaChunks WHERE media_id = ?)"
        )
        params = (created.media_id, created.media_id)
    elif graph_mutation == "chunk_changed":
        query = (
            "UPDATE UnvectorizedMediaChunks SET chunk_text = ?, "
            "version = version + 1, last_modified = ?, client_id = ? "
            "WHERE media_id = ? "
            "AND id = (SELECT MIN(id) FROM UnvectorizedMediaChunks WHERE media_id = ?)"
        )
        params = (
            "tampered chunk",
            "2026-08-25T15:00:00+00:00",
            media_db.client_id,
            created.media_id,
            created.media_id,
        )
    elif graph_mutation == "transcript_removed":
        query = (
            "DELETE FROM Transcripts WHERE media_id = ? AND id = "
            "(SELECT MIN(id) FROM Transcripts WHERE media_id = ?)"
        )
        params = (created.media_id, created.media_id)
    else:
        query = (
            "UPDATE Transcripts SET transcription = ?, version = version + 1, "
            "last_modified = ?, client_id = ? WHERE media_id = ? "
            "AND id = (SELECT MIN(id) FROM Transcripts WHERE media_id = ?)"
        )
        params = (
            "tampered transcript",
            "2026-08-25T15:00:00+00:00",
            media_db.client_id,
            created.media_id,
            created.media_id,
        )
    media_db.execute_query(query, params, commit=True)

    with pytest.raises(ConflictError):
        media_db.insert_operation_owned_clone_media(
            snapshot=snapshot,
            operation_id=operation_id,
            source_identity=source_identity,
            expected_content_hash=expected_hash,
        )

    assert media_db.delete_operation_owned_clone_media(
        operation_id=operation_id,
        source_identity=source_identity,
        expected_content_hash=expected_hash,
    ) == 1


@pytest.mark.unit
def test_operation_owned_clone_replay_rejects_dual_source_and_url_tamper(
    media_db: MediaDatabase,
) -> None:
    snapshot = _operation_snapshot()
    expected_hash = media_db_api.hash_media_clone_snapshot(snapshot)
    operation_id = "clone-operation-dual-tamper"
    source_identity = "workspace-source-dual-tamper"
    created = media_db.insert_operation_owned_clone_media(
        snapshot=snapshot,
        operation_id=operation_id,
        source_identity=source_identity,
        expected_content_hash=expected_hash,
    )
    media_db.execute_query(
        "UPDATE Media SET url = ?, system_source_identity = ?, version = version + 1, "
        "last_modified = ?, client_id = ? WHERE id = ?",
        (
            "tldw-clone://workspace/tampered/tampered",
            "tampered-source",
            "2026-08-25T15:00:00+00:00",
            media_db.client_id,
            created.media_id,
        ),
        commit=True,
    )

    with pytest.raises(ConflictError):
        media_db.insert_operation_owned_clone_media(
            snapshot=snapshot,
            operation_id=operation_id,
            source_identity=source_identity,
            expected_content_hash=expected_hash,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "candidate_mutation",
    [
        "soft_deleted",
        "partial_marker",
        "wrong_kind",
    ],
)
def test_operation_owned_clone_replay_rejects_inactive_or_inconsistent_candidate(
    media_db: MediaDatabase,
    candidate_mutation: str,
) -> None:
    snapshot = _operation_snapshot()
    expected_hash = media_db_api.hash_media_clone_snapshot(snapshot)
    operation_id = f"clone-operation-{candidate_mutation}"
    source_identity = "workspace-source-inconsistent"
    created = media_db.insert_operation_owned_clone_media(
        snapshot=snapshot,
        operation_id=operation_id,
        source_identity=source_identity,
        expected_content_hash=expected_hash,
    )

    conn = media_db.get_connection()
    if candidate_mutation == "soft_deleted":
        conn.execute(
            "UPDATE Media SET deleted = 1, version = version + 1, "
            "last_modified = ?, client_id = ? WHERE id = ?",
            (
                "2026-08-25T14:00:00+00:00",
                media_db.client_id,
                created.media_id,
            ),
        )
    else:
        conn.execute("PRAGMA ignore_check_constraints = ON")
        try:
            if candidate_mutation == "partial_marker":
                conn.execute(
                    "UPDATE Media SET system_source_identity = NULL, "
                    "version = version + 1, last_modified = ?, client_id = ? "
                    "WHERE id = ?",
                    (
                        "2026-08-25T14:00:00+00:00",
                        media_db.client_id,
                        created.media_id,
                    ),
                )
            else:
                conn.execute(
                    "UPDATE Media SET system_operation_kind = ?, "
                    "version = version + 1, last_modified = ?, client_id = ? "
                    "WHERE id = ?",
                    (
                        "unrelated_operation",
                        "2026-08-25T14:00:00+00:00",
                        media_db.client_id,
                        created.media_id,
                    ),
                )
        finally:
            conn.execute("PRAGMA ignore_check_constraints = OFF")
    conn.commit()

    with pytest.raises(ConflictError):
        media_db.insert_operation_owned_clone_media(
            snapshot=snapshot,
            operation_id=operation_id,
            source_identity=source_identity,
            expected_content_hash=expected_hash,
        )


@pytest.mark.unit
def test_operation_owned_clone_cleanup_hard_deletes_only_exact_owned_graph(
    media_db: MediaDatabase,
) -> None:
    snapshot = _operation_snapshot()
    expected_hash = media_db_api.hash_media_clone_snapshot(snapshot)
    ordinary_id = _seed_media(
        media_db,
        title="Ordinary cleanup collision",
        content=str(snapshot.media["content"]),
        chunk_text="ordinary cleanup chunk",
        transcript_text="ordinary cleanup transcript",
        url=str(snapshot.media["url"]),
    )
    ordinary_before = _ordinary_media_state(media_db, ordinary_id)
    created = media_db.insert_operation_owned_clone_media(
        snapshot=snapshot,
        operation_id="clone-operation-cleanup",
        source_identity="workspace-source-cleanup",
        expected_content_hash=expected_hash,
    )

    with pytest.raises(ConflictError):
        media_db.delete_operation_owned_clone_media(
            operation_id="clone-operation-cleanup",
            source_identity="workspace-source-cleanup",
            expected_content_hash=hashlib.sha256(b"wrong-cleanup-hash").hexdigest(),
        )
    assert media_db.execute_query(
        "SELECT COUNT(*) AS count FROM Media WHERE id = ?",
        (created.media_id,),
    ).fetchone()["count"] == 1

    deleted_count = media_db.delete_operation_owned_clone_media(
        operation_id="clone-operation-cleanup",
        source_identity="workspace-source-cleanup",
        expected_content_hash=expected_hash,
    )

    assert deleted_count == 1
    assert media_db.delete_operation_owned_clone_media(
        operation_id="clone-operation-cleanup",
        source_identity="workspace-source-cleanup",
        expected_content_hash=expected_hash,
    ) == 0
    assert media_db.execute_query(
        "SELECT COUNT(*) AS count FROM Media WHERE id = ?",
        (created.media_id,),
    ).fetchone()["count"] == 0
    for table in (
        "DocumentVersions",
        "MediaKeywords",
        "UnvectorizedMediaChunks",
        "Transcripts",
    ):
        assert media_db.execute_query(
            f"SELECT COUNT(*) AS count FROM {table} WHERE media_id = ?",  # nosec B608
            (created.media_id,),
        ).fetchone()["count"] == 0
    assert _ordinary_media_state(media_db, ordinary_id) == ordinary_before


@pytest.mark.unit
def test_operation_owned_clone_package_facade_exposes_insert_and_cleanup(
    media_db: MediaDatabase,
) -> None:
    snapshot = _operation_snapshot()
    expected_hash = media_db_api.hash_media_clone_snapshot(snapshot)

    result = media_db_api.insert_operation_owned_clone_media(
        media_db,
        snapshot=snapshot,
        operation_id="clone-operation-facade",
        source_identity="workspace-source-facade",
        expected_content_hash=expected_hash,
    )

    assert result.created is True
    assert media_db_api.delete_operation_owned_clone_media(
        media_db,
        operation_id="clone-operation-facade",
        source_identity="workspace-source-facade",
        expected_content_hash=expected_hash,
    ) == 1


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


@pytest.mark.integration
@pytest.mark.postgres
@pytest.mark.timeout(120)
def test_postgres_concurrent_same_operation_clone_inserts_converge(
    pg_database_config: DatabaseConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    databases = [
        MediaDatabase(db_path=":memory:", client_id="901", backend=backend),
        MediaDatabase(db_path=":memory:", client_id="901", backend=backend),
    ]
    snapshot = _operation_snapshot()
    operation_id = f"clone-operation-pg-concurrent-{uuid.uuid4()}"
    source_identity = f"workspace-source-pg-concurrent-{uuid.uuid4()}"
    expected_hash = media_db_api.hash_media_clone_snapshot(snapshot)
    start = threading.Barrier(2)
    lock_queries: list[str] = []
    query_guard = threading.Lock()
    original_execute = backend.execute

    def tracked_execute(query, params=None, connection=None, **kwargs):
        if "pg_advisory_xact_lock" in query:
            with query_guard:
                lock_queries.append(query)
        return original_execute(query, params, connection=connection, **kwargs)

    def insert(database: MediaDatabase):
        with scoped_context(user_id=901, org_ids=[], team_ids=[], is_admin=True):
            start.wait(timeout=30)
            return database.insert_operation_owned_clone_media(
                snapshot=snapshot,
                operation_id=operation_id,
                source_identity=source_identity,
                expected_content_hash=expected_hash,
            )

    monkeypatch.setattr(backend, "execute", tracked_execute)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(insert, databases))

        assert {result.media_id for result in results} == {results[0].media_id}
        assert sorted(result.created for result in results) == [False, True]
        assert sorted(result.replayed for result in results) == [False, True]
        assert len(lock_queries) == 2
        with scoped_context(user_id=901, org_ids=[], team_ids=[], is_admin=True):
            assert databases[0].execute_query(
                "SELECT COUNT(*) AS count FROM Media WHERE system_operation_id = ? ",
                (operation_id,),
            ).fetchone()["count"] == 1
            assert databases[0].delete_operation_owned_clone_media(
                operation_id=operation_id,
                source_identity=source_identity,
                expected_content_hash=expected_hash,
            ) == 1
    finally:
        for database in databases:
            database.close_connection()
        backend.get_pool().close_all()


@pytest.mark.integration
@pytest.mark.postgres
@pytest.mark.timeout(120)
def test_postgres_non_admin_owner_can_reconcile_soft_deleted_owned_clone(
    pg_database_config: DatabaseConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    owner_db = MediaDatabase(db_path=":memory:", client_id="901", backend=backend)
    other_db = MediaDatabase(db_path=":memory:", client_id="902", backend=backend)
    role_name = f"clone_media_rls_{uuid.uuid4().hex[:12]}"
    ident = backend.escape_identifier  # type: ignore[attr-defined]
    operation_id = f"clone-operation-pg-rls-{uuid.uuid4()}"
    source_identity = f"workspace-source-pg-rls-{uuid.uuid4()}"
    snapshot = _operation_snapshot()
    expected_hash = media_db_api.hash_media_clone_snapshot(snapshot)
    role_created = False
    monkeypatch.setenv("TLDW_CONTENT_PG_ROLE_SWITCH", "1")
    monkeypatch.setenv("TLDW_CONTENT_PG_ROLE_WHITELIST", role_name)
    try:
        with backend.transaction() as connection:
            backend.execute(
                f"CREATE ROLE {ident(role_name)} NOLOGIN NOSUPERUSER NOBYPASSRLS",
                connection=connection,
            )
            backend.execute(
                f"GRANT USAGE ON SCHEMA public TO {ident(role_name)}",
                connection=connection,
            )
            backend.execute(
                "GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public "
                f"TO {ident(role_name)}",
                connection=connection,
            )
            backend.execute(
                "GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public "
                f"TO {ident(role_name)}",
                connection=connection,
            )
            backend.execute(
                f"GRANT {ident(role_name)} TO CURRENT_USER",
                connection=connection,
            )
        role_created = True

        with scoped_context(
            user_id=901,
            org_ids=[],
            team_ids=[],
            is_admin=False,
            session_role=role_name,
        ):
            assert owner_db.execute_query(
                "SELECT current_role::text AS role"
            ).fetchone()["role"] == role_name
            created = owner_db.insert_operation_owned_clone_media(
                snapshot=snapshot,
                operation_id=operation_id,
                source_identity=source_identity,
                expected_content_hash=expected_hash,
            )

        with scoped_context(user_id=901, org_ids=[], team_ids=[], is_admin=True):
            owner_db.execute_query(
                "UPDATE Media SET deleted = TRUE, version = version + 1 "
                "WHERE id = ?",
                (created.media_id,),
                commit=True,
            )

        with scoped_context(
            user_id=902,
            org_ids=[],
            team_ids=[],
            is_admin=False,
            session_role=role_name,
        ):
            assert other_db.list_operation_owned_clone_media(
                operation_id=operation_id,
            ) == []
            assert other_db.delete_operation_owned_clone_media(
                operation_id=operation_id,
                source_identity=source_identity,
                expected_content_hash=expected_hash,
            ) == 0

        with scoped_context(
            user_id=901,
            org_ids=[],
            team_ids=[],
            is_admin=False,
            session_role=role_name,
        ):
            assert [
                item.media_id
                for item in owner_db.list_operation_owned_clone_media(
                    operation_id=operation_id,
                )
            ] == [created.media_id]
            with pytest.raises(ConflictError):
                owner_db.insert_operation_owned_clone_media(
                    snapshot=snapshot,
                    operation_id=operation_id,
                    source_identity=source_identity,
                    expected_content_hash=expected_hash,
                )
            assert owner_db.delete_operation_owned_clone_media(
                operation_id=operation_id,
                source_identity=source_identity,
                expected_content_hash=expected_hash,
            ) == 1
    finally:
        owner_db.close_connection()
        other_db.close_connection()
        if role_created:
            with backend.transaction() as connection:
                backend.execute(
                    f"DROP OWNED BY {ident(role_name)}",
                    connection=connection,
                )
                backend.execute(
                    f"REVOKE {ident(role_name)} FROM CURRENT_USER",
                    connection=connection,
                )
                backend.execute(f"DROP ROLE {ident(role_name)}", connection=connection)
        backend.get_pool().close_all()


@pytest.mark.integration
@pytest.mark.postgres
@pytest.mark.timeout(120)
def test_postgres_operation_owned_clone_insert_replay_collision_and_cleanup(
    pg_database_config: DatabaseConfig,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = MediaDatabase(db_path=":memory:", client_id="901", backend=backend)
    snapshot = _operation_snapshot(
        content=None,
        keywords=("pg-existing-keyword", "pg-clone-only-keyword"),
    )
    operation_id = f"clone-operation-pg-{uuid.uuid4()}"
    source_identity = f"workspace-source-pg-{uuid.uuid4()}"
    expected_hash = media_db_api.hash_media_clone_snapshot(snapshot)
    try:
        with scoped_context(user_id=901, org_ids=[], team_ids=[], is_admin=True):
            ordinary_id = _seed_media(
                db,
                title="Postgres ordinary collision",
                content=str(snapshot.media["content"]),
                chunk_text="postgres ordinary chunk",
                transcript_text="postgres ordinary transcript",
                url=str(snapshot.media["url"]),
                keywords=["pg-existing-keyword"],
            )
            ordinary_before = _ordinary_media_state(db, ordinary_id)

            created = db.insert_operation_owned_clone_media(
                snapshot=snapshot,
                operation_id=operation_id,
                source_identity=source_identity,
                expected_content_hash=expected_hash,
            )
            replayed = db.insert_operation_owned_clone_media(
                snapshot=snapshot,
                operation_id=operation_id,
                source_identity=source_identity,
                expected_content_hash=expected_hash,
            )

            assert created.created is True
            assert replayed.replayed is True
            assert replayed.media_id == created.media_id
            stored = db.execute_query(
                "SELECT m.content AS media_content, dv.content AS document_content "
                "FROM Media m JOIN DocumentVersions dv ON dv.media_id = m.id "
                "WHERE m.id = ?",
                (created.media_id,),
            ).fetchone()
            assert stored["media_content"] is None
            assert stored["document_content"] == ""
            assert "pg-existing-keyword" in db.fetch_all_keywords()
            assert "pg-clone-only-keyword" not in db.fetch_all_keywords()
            assert _ordinary_media_state(db, ordinary_id) == ordinary_before
            assert db.delete_operation_owned_clone_media(
                operation_id=operation_id,
                source_identity=source_identity,
                expected_content_hash=expected_hash,
            ) == 1
            assert db.execute_query(
                "SELECT COUNT(*) AS count FROM Media WHERE id = ?",
                (created.media_id,),
            ).fetchone()["count"] == 0
            assert db.execute_query(
                "SELECT COUNT(*) AS count FROM Keywords WHERE keyword = ?",
                ("pg-clone-only-keyword",),
            ).fetchone()["count"] == 0
            assert db.execute_query(
                "SELECT COUNT(*) AS count FROM Keywords WHERE keyword = ?",
                ("pg-existing-keyword",),
            ).fetchone()["count"] == 1
            assert _ordinary_media_state(db, ordinary_id) == ordinary_before
    finally:
        db.close_connection()
        backend.get_pool().close_all()
