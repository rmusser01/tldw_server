"""Repeatable source snapshot contracts for Media clone reads."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sqlite3
import threading
import time
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
from tldw_Server_API.app.core.DB_Management.media_db.errors import (
    ConflictError,
    InputError,
    SchemaError,
)
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

    overlong_chunks = [dict(row) for row in unsupported.chunks]
    overlong_chunks[0]["metadata"] = "x" * 1_000_001
    overlong_snapshot = MediaCloneSnapshot.from_rows(
        media=unsupported.media,
        chunks=overlong_chunks,
        transcripts=unsupported.transcripts,
    )
    with pytest.raises(ValueError, match="string bound"):
        media_db_api.hash_media_clone_snapshot(overlong_snapshot)

    reserved_chunks = [dict(row) for row in unsupported.chunks]
    reserved_chunks[0]["metadata"] = {"$tldw_legacy_json_text_v1": "ambiguous"}
    reserved_snapshot = MediaCloneSnapshot.from_rows(
        media=unsupported.media,
        chunks=reserved_chunks,
        transcripts=unsupported.transcripts,
    )
    assert len(media_db_api.hash_media_clone_snapshot(reserved_snapshot)) == 64

    overlong_keyword = _operation_snapshot(keywords=("k" * 256,))
    with pytest.raises(InputError, match="255"):
        media_db_api.hash_media_clone_snapshot(overlong_keyword)


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
@pytest.mark.parametrize(
    ("source_metadata", "semantic_metadata"),
    [
        ('{"b":[2],"a":1}', {"a": 1, "b": [2]}),
        ("[1,true,null]", [1, True, None]),
        ("42", 42),
        ("true", True),
        ("null", None),
        ('"json scalar"', "json scalar"),
        ("invalid legacy json text", "invalid legacy json text"),
        (
            {"$tldw_legacy_json_text_v1": "ordinary mapping"},
            {"$tldw_legacy_json_text_v1": "ordinary mapping"},
        ),
    ],
)
def test_operation_owned_clone_normalizes_source_chunk_json_text_once(
    media_db: MediaDatabase,
    source_metadata: Any,
    semantic_metadata: Any,
) -> None:
    base = _operation_snapshot()
    chunks = [dict(row) for row in base.chunks]
    chunks[0]["metadata"] = source_metadata
    snapshot = MediaCloneSnapshot.from_rows(
        media=base.media,
        chunks=chunks,
        transcripts=base.transcripts,
    )
    operation_id = f"clone-operation-json-{uuid.uuid4()}"
    source_identity = f"workspace-source-json-{uuid.uuid4()}"
    expected_hash = media_db_api.hash_media_clone_snapshot(snapshot)

    semantic_chunks = [dict(row) for row in base.chunks]
    semantic_chunks[0]["metadata"] = semantic_metadata
    semantic_snapshot = MediaCloneSnapshot.from_rows(
        media=base.media,
        chunks=semantic_chunks,
        transcripts=base.transcripts,
    )
    assert media_db_api.hash_media_clone_snapshot(semantic_snapshot) == expected_hash

    created = media_db.insert_operation_owned_clone_media(
        snapshot=snapshot,
        operation_id=operation_id,
        source_identity=source_identity,
        expected_content_hash=expected_hash,
    )
    stored = media_db.execute_query(
        "SELECT metadata FROM UnvectorizedMediaChunks "
        "WHERE media_id = ? AND chunk_index = 0",
        (created.media_id,),
    ).fetchone()["metadata"]
    assert (json.loads(stored) if stored is not None else None) == semantic_metadata
    assert media_db.insert_operation_owned_clone_media(
        snapshot=snapshot,
        operation_id=operation_id,
        source_identity=source_identity,
        expected_content_hash=expected_hash,
    ).replayed is True


@pytest.mark.unit
def test_operation_owned_clone_chunk_json_string_forms_share_logical_value(
    media_db: MediaDatabase,
) -> None:
    base = _operation_snapshot()

    def with_metadata(value: Any) -> MediaCloneSnapshot:
        chunks = [dict(row) for row in base.chunks]
        chunks[0]["metadata"] = value
        return MediaCloneSnapshot.from_rows(
            media=base.media,
            chunks=chunks,
            transcripts=base.transcripts,
        )

    invalid_text = with_metadata("legacy scalar")
    valid_json_string = with_metadata('"legacy scalar"')
    assert media_db_api.hash_media_clone_snapshot(invalid_text) == (
        media_db_api.hash_media_clone_snapshot(valid_json_string)
    )

    expected_hash = media_db_api.hash_media_clone_snapshot(invalid_text)
    created = media_db.insert_operation_owned_clone_media(
        snapshot=invalid_text,
        operation_id="clone-operation-json-ambiguity",
        source_identity="workspace-source-json-ambiguity",
        expected_content_hash=expected_hash,
    )
    replayed = media_db.insert_operation_owned_clone_media(
        snapshot=valid_json_string,
        operation_id="clone-operation-json-ambiguity",
        source_identity="workspace-source-json-ambiguity",
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
            "SELECT deleted, is_trash, url, uuid, title, content_hash FROM Media WHERE id = ?",
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

    assert stored["deleted"] == 0
    assert stored["is_trash"] == 1
    assert media_db.get_media_by_id(created.media_id) is None
    assert media_db.get_media_by_id(
        created.media_id,
        include_deleted=True,
        include_trash=True,
    ) is None
    assert media_db.get_media_status_by_id(
        created.media_id,
        include_deleted=True,
        include_trash=True,
    ) is None
    assert media_db.get_media_by_uuid(
        stored["uuid"],
        include_deleted=True,
        include_trash=True,
    ) is None
    assert media_db.get_media_by_url(
        stored["url"],
        include_deleted=True,
        include_trash=True,
    ) is None
    assert media_db.get_media_by_hash(
        stored["content_hash"],
        include_deleted=True,
        include_trash=True,
    ) is None
    assert media_db.get_media_by_title(
        stored["title"],
        include_deleted=True,
        include_trash=True,
    ) is None
    assert media_db_api.check_media_exists(
        media_db,
        media_id=created.media_id,
        url=stored["url"],
        content_hash=stored["content_hash"],
    ) is None
    assert media_db_api.fetch_keywords_for_media(media_db, created.media_id) == []
    assert media_db_api.get_media_transcripts(media_db, created.media_id) == []
    assert media_db_api.list_document_versions(
        media_db,
        created.media_id,
        include_deleted=True,
    ) == []
    assert media_db_api.has_unvectorized_chunks(media_db, created.media_id) is False
    assert media_db_api.get_unvectorized_chunk_count(media_db, created.media_id) == 0
    assert media_db_api.get_unvectorized_max_chunk_index(media_db, created.media_id) is None
    assert media_db_api.get_unvectorized_chunk_by_index(
        media_db,
        created.media_id,
        0,
    ) is None
    assert media_db_api.get_unvectorized_chunks_in_range(
        media_db,
        created.media_id,
        0,
        100,
    ) == []
    assert media_db.count_chatbook_scope_category("media_records") == 0
    assert media_db.count_chatbook_scope_category("media_transcripts") == 0
    assert media_db.count_chatbook_scope_category("media_chunks") == 0
    assert media_db.count_chatbook_scope_category("media_pointers") == 0
    assert media_db.list_chatbook_scope_ids("media_records") == []
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
def test_operation_owned_clone_ordinary_mutation_contract_cannot_touch_staged_row(
    media_db: MediaDatabase,
) -> None:
    from tldw_Server_API.app.core.DB_Management.media_db import legacy_state

    snapshot = _operation_snapshot()
    operation_id = "clone-operation-mutation-fence"
    source_identity = "workspace-source-mutation-fence"
    expected_hash = media_db_api.hash_media_clone_snapshot(snapshot)
    created = media_db.insert_operation_owned_clone_media(
        snapshot=snapshot,
        operation_id=operation_id,
        source_identity=source_identity,
        expected_content_hash=expected_hash,
    )
    before = dict(media_db.execute_query(
        "SELECT version, vector_processing, chunking_status FROM Media WHERE id = ?",
        (created.media_id,),
    ).fetchone())

    legacy_state.mark_media_as_processed(media_db, created.media_id)
    assert media_db_api.permanently_delete_item(media_db, created.media_id) is False
    with pytest.raises(InputError):
        media_db_api.update_keywords_for_media(
            media_db,
            created.media_id,
            ["ordinary-mutation-must-not-link"],
        )

    after = dict(media_db.execute_query(
        "SELECT version, vector_processing, chunking_status FROM Media WHERE id = ?",
        (created.media_id,),
    ).fetchone())
    assert after == before
    assert media_db.execute_query(
        "SELECT COUNT(*) AS count FROM MediaKeywords WHERE media_id = ?",
        (created.media_id,),
    ).fetchone()["count"] == 0
    assert media_db.delete_operation_owned_clone_media(
        operation_id=operation_id,
        source_identity=source_identity,
        expected_content_hash=expected_hash,
    ) == 1


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
def test_operation_owned_clone_stages_owner_scoped_keyword_values_only(
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
    assert media_db.execute_query(
        "SELECT COUNT(*) AS count FROM Keywords WHERE keyword = ?",
        ("clone-only-staged",),
    ).fetchone()["count"] == 0
    assert media_db.execute_query(
        "SELECT COUNT(*) AS count FROM MediaKeywords WHERE media_id = ?",
        (created.media_id,),
    ).fetchone()["count"] == 0
    pending = [
        dict(row)
        for row in media_db.execute_query(
            "SELECT keyword, operation_id, source_identity, client_id "
            "FROM OperationOwnedCloneKeywords WHERE media_id = ? ORDER BY keyword",
            (created.media_id,),
        ).fetchall()
    ]
    assert pending == [
        {
            "keyword": "clone-only-staged",
            "operation_id": operation_id,
            "source_identity": source_identity,
            "client_id": media_db.client_id,
        },
        {
            "keyword": "recipient-existing",
            "operation_id": operation_id,
            "source_identity": source_identity,
            "client_id": media_db.client_id,
        },
    ]
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
def test_operation_owned_clone_cleanup_removes_pending_values_not_canonical_keywords(
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

    media_db.execute_query(
        "UPDATE OperationOwnedCloneKeywords SET operation_id = ?, source_identity = ?, "
        "client_id = ? WHERE media_id = ? AND keyword = ?",
        (
            "tampered-operation",
            "tampered-source",
            "tampered-owner",
            created.media_id,
            "clone-only-staged",
        ),
        commit=True,
    )

    assert media_db.delete_operation_owned_clone_media(
        operation_id=operation_id,
        source_identity=source_identity,
        expected_content_hash=expected_hash,
    ) == 1
    assert media_db.execute_query(
        "SELECT COUNT(*) AS count FROM OperationOwnedCloneKeywords WHERE media_id = ?",
        (created.media_id,),
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
def test_operation_owned_clone_confirmation_promotes_pending_keywords(
    media_db: MediaDatabase,
) -> None:
    ordinary_id = _seed_media(
        media_db,
        title="Canonical keyword owner",
        content="canonical keyword owner content",
        chunk_text="canonical keyword owner chunk",
        transcript_text="canonical keyword owner transcript",
        keywords=["recipient-existing"],
    )
    existing_keyword_id = media_db.execute_query(
        "SELECT id FROM Keywords WHERE keyword = ?",
        ("recipient-existing",),
    ).fetchone()["id"]
    snapshot = _operation_snapshot(
        keywords=("recipient-existing", "clone-confirmed"),
    )
    expected_hash = media_db_api.hash_media_clone_snapshot(snapshot)
    operation_id = "clone-operation-keyword-confirm"
    source_identity = "workspace-source-keyword-confirm"
    created = media_db.insert_operation_owned_clone_media(
        snapshot=snapshot,
        operation_id=operation_id,
        source_identity=source_identity,
        expected_content_hash=expected_hash,
    )

    assert media_db.confirm_operation_owned_clone_media(
        operation_id=operation_id,
        source_identity=source_identity,
        expected_content_hash=expected_hash,
    ) == 1
    assert media_db.fetch_all_keywords() == ["clone-confirmed", "recipient-existing"]
    assert media_db.execute_query(
        "SELECT COUNT(*) AS count FROM OperationOwnedCloneKeywords"
    ).fetchone()["count"] == 0
    linked = [
        dict(row)
        for row in media_db.execute_query(
            "SELECT k.id, k.keyword FROM MediaKeywords mk "
            "JOIN Keywords k ON k.id = mk.keyword_id "
            "WHERE mk.media_id = ? ORDER BY k.keyword",
            (created.media_id,),
        ).fetchall()
    ]
    assert linked == [
        {"id": linked[0]["id"], "keyword": "clone-confirmed"},
        {"id": existing_keyword_id, "keyword": "recipient-existing"},
    ]
    assert media_db.execute_query(
        "SELECT COUNT(*) AS count FROM MediaKeywords WHERE media_id = ?",
        (ordinary_id,),
    ).fetchone()["count"] == 1
    assert media_db.get_media_by_id(created.media_id) is not None


@pytest.mark.integration
def test_sqlite_v26_migration_releases_real_v25_keyword_graph_and_replays(
    tmp_path: Path,
) -> None:
    from tldw_Server_API.app.core.DB_Management.db_migration import DatabaseMigrator

    db_path = str(tmp_path / "clone-media-fix1-v25.sqlite")
    backend = SQLiteBackend(
        DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=db_path)
    )
    database = MediaDatabase(db_path=db_path, client_id="owner-1", backend=backend)
    snapshot = _operation_snapshot(
        keywords=("clone-orphan", "recipient-existing", "clone-shared"),
    )
    operation_id = "clone-operation-fix1-v25"
    source_identity = "workspace-source-fix1-v25"
    expected_hash = media_db_api.hash_media_clone_snapshot(snapshot)
    try:
        ordinary_id = _seed_media(
            database,
            title="Ordinary shared keyword owner",
            content="ordinary shared keyword content",
            chunk_text="ordinary shared keyword chunk",
            transcript_text="ordinary shared keyword transcript",
            keywords=["clone-shared"],
        )
        created = database.insert_operation_owned_clone_media(
            snapshot=snapshot,
            operation_id=operation_id,
            source_identity=source_identity,
            expected_content_hash=expected_hash,
        )
        now = database._get_current_utc_timestamp_str()
        connection = database.get_connection()
        for keyword in ("clone-orphan", "recipient-existing"):
            connection.execute(
                "INSERT INTO Keywords "
                "(keyword, uuid, last_modified, version, client_id, deleted) "
                "VALUES (?, ?, ?, 1, ?, 0)",
                (keyword, str(uuid.uuid4()), now, database.client_id),
            )
        keyword_rows = {
            row["keyword"]: int(row["id"])
            for row in connection.execute(
                "SELECT id, keyword FROM Keywords WHERE keyword IN (?, ?, ?)",
                ("clone-orphan", "recipient-existing", "clone-shared"),
            ).fetchall()
        }
        for keyword_id in keyword_rows.values():
            connection.execute(
                "INSERT INTO MediaKeywords (media_id, keyword_id) VALUES (?, ?)",
                (created.media_id, keyword_id),
            )
        connection.executescript(
            """
            DROP TABLE OperationOwnedCloneKeywords;
            CREATE TABLE OperationOwnedCloneKeywords (
                media_id INTEGER NOT NULL,
                keyword_id INTEGER NOT NULL,
                operation_id TEXT NOT NULL,
                source_identity TEXT NOT NULL,
                created_by_clone BOOLEAN NOT NULL,
                PRIMARY KEY (media_id, keyword_id),
                FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE CASCADE,
                FOREIGN KEY (keyword_id) REFERENCES Keywords(id) ON DELETE CASCADE
            );
            UPDATE schema_version SET version = 25;
            """
        )
        for keyword, created_by_clone in (
            ("clone-orphan", 1),
            ("recipient-existing", 0),
            ("clone-shared", 1),
        ):
            connection.execute(
                "INSERT INTO OperationOwnedCloneKeywords "
                "(media_id, keyword_id, operation_id, source_identity, created_by_clone) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    created.media_id,
                    keyword_rows[keyword],
                    operation_id,
                    source_identity,
                    created_by_clone,
                ),
            )
        connection.commit()
    finally:
        database.close_connection()
        backend.get_pool().close_all()

    assert DatabaseMigrator(db_path).migrate_to_version(26, create_backup=False)["status"] == "success"

    migrated_backend = SQLiteBackend(
        DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=db_path)
    )
    migrated = MediaDatabase(
        db_path=db_path,
        client_id="owner-1",
        backend=migrated_backend,
    )
    try:
        pending = [
            row["keyword"]
            for row in migrated.execute_query(
                "SELECT keyword FROM OperationOwnedCloneKeywords "
                "WHERE media_id = ? ORDER BY keyword",
                (created.media_id,),
            ).fetchall()
        ]
        assert pending == ["clone-orphan", "clone-shared", "recipient-existing"]
        assert migrated.execute_query(
            "SELECT COUNT(*) AS count FROM MediaKeywords WHERE media_id = ?",
            (created.media_id,),
        ).fetchone()["count"] == 0
        remaining_keywords = {
            row["keyword"]
            for row in migrated.execute_query(
                "SELECT keyword FROM Keywords WHERE keyword IN (?, ?, ?)",
                ("clone-orphan", "recipient-existing", "clone-shared"),
            ).fetchall()
        }
        assert remaining_keywords == {"recipient-existing", "clone-shared"}
        assert migrated.execute_query(
            "SELECT COUNT(*) AS count FROM MediaKeywords mk "
            "JOIN Keywords k ON k.id = mk.keyword_id "
            "WHERE mk.media_id = ? AND k.keyword = ?",
            (ordinary_id, "clone-shared"),
        ).fetchone()["count"] == 1

        replayed = migrated.insert_operation_owned_clone_media(
            snapshot=snapshot,
            operation_id=operation_id,
            source_identity=source_identity,
            expected_content_hash=expected_hash,
        )
        assert replayed.replayed is True
        assert replayed.media_id == created.media_id
        assert migrated.confirm_operation_owned_clone_media(
            operation_id=operation_id,
            source_identity=source_identity,
            expected_content_hash=expected_hash,
        ) == 1
        assert migrated.execute_query(
            "SELECT COUNT(*) AS count FROM MediaKeywords WHERE media_id = ?",
            (created.media_id,),
        ).fetchone()["count"] == 3
    finally:
        migrated.close_connection()
        migrated_backend.get_pool().close_all()


@pytest.mark.integration
def test_sqlite_v26_migration_harvests_original_v25_direct_keywords_and_replays(
    tmp_path: Path,
) -> None:
    from tldw_Server_API.app.core.DB_Management.db_migration import DatabaseMigrator

    db_path = str(tmp_path / "clone-media-original-v25.sqlite")
    backend = SQLiteBackend(
        DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=db_path)
    )
    database = MediaDatabase(db_path=db_path, client_id="owner-1", backend=backend)
    keywords = ("original-v25-alpha", "original-v25-research")
    snapshot = _operation_snapshot(keywords=keywords)
    operation_id = "clone-operation-original-v25"
    source_identity = "workspace-source-original-v25"
    expected_hash = media_db_api.hash_media_clone_snapshot(snapshot)
    try:
        created = database.insert_operation_owned_clone_media(
            snapshot=snapshot,
            operation_id=operation_id,
            source_identity=source_identity,
            expected_content_hash=expected_hash,
        )
        connection = database.get_connection()
        now = database._get_current_utc_timestamp_str()
        for keyword in keywords:
            keyword_id = connection.execute(
                "INSERT INTO Keywords "
                "(keyword, uuid, last_modified, version, client_id, deleted) "
                "VALUES (?, ?, ?, 1, ?, 0)",
                (keyword, str(uuid.uuid4()), now, database.client_id),
            ).lastrowid
            connection.execute(
                "INSERT INTO MediaKeywords (media_id, keyword_id) VALUES (?, ?)",
                (created.media_id, keyword_id),
            )
        connection.executescript(
            """
            DROP TABLE OperationOwnedCloneKeywords;
            UPDATE schema_version SET version = 25;
            """
        )
        connection.commit()
    finally:
        database.close_connection()
        backend.get_pool().close_all()

    assert DatabaseMigrator(db_path).migrate_to_version(26, create_backup=False)["status"] == "success"

    migrated_backend = SQLiteBackend(
        DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=db_path)
    )
    migrated = MediaDatabase(
        db_path=db_path,
        client_id="owner-1",
        backend=migrated_backend,
    )
    try:
        assert [
            row["keyword"]
            for row in migrated.execute_query(
                "SELECT keyword FROM OperationOwnedCloneKeywords "
                "WHERE media_id = ? ORDER BY keyword",
                (created.media_id,),
            ).fetchall()
        ] == list(keywords)
        assert migrated.execute_query(
            "SELECT COUNT(*) AS count FROM MediaKeywords WHERE media_id = ?",
            (created.media_id,),
        ).fetchone()["count"] == 0
        assert {
            row["keyword"]
            for row in migrated.execute_query(
                "SELECT keyword FROM Keywords WHERE keyword IN (?, ?)",
                keywords,
            ).fetchall()
        } == set(keywords)

        replayed = migrated.insert_operation_owned_clone_media(
            snapshot=snapshot,
            operation_id=operation_id,
            source_identity=source_identity,
            expected_content_hash=expected_hash,
        )
        assert replayed.media_id == created.media_id
        assert replayed.replayed is True
        assert migrated.confirm_operation_owned_clone_media(
            operation_id=operation_id,
            source_identity=source_identity,
            expected_content_hash=expected_hash,
        ) == 1
        assert migrated.execute_query(
            "SELECT COUNT(*) AS count FROM MediaKeywords WHERE media_id = ?",
            (created.media_id,),
        ).fetchone()["count"] == len(keywords)
    finally:
        migrated.close_connection()
        migrated_backend.get_pool().close_all()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("column", "replacement"),
    [
        ("keyword", "different-keyword"),
        ("operation_id", "different-operation"),
        ("source_identity", "different-source"),
        ("client_id", "different-owner"),
    ],
)
def test_operation_owned_clone_replay_rejects_pending_keyword_tamper(
    media_db: MediaDatabase,
    column: str,
    replacement: str,
) -> None:
    snapshot = _operation_snapshot(keywords=("pending-exact",))
    operation_id = f"clone-operation-pending-tamper-{column}"
    source_identity = f"workspace-source-pending-tamper-{column}"
    expected_hash = media_db_api.hash_media_clone_snapshot(snapshot)
    created = media_db.insert_operation_owned_clone_media(
        snapshot=snapshot,
        operation_id=operation_id,
        source_identity=source_identity,
        expected_content_hash=expected_hash,
    )
    media_db.execute_query(
        f"UPDATE OperationOwnedCloneKeywords SET {column} = ? WHERE media_id = ?",  # nosec B608
        (replacement, created.media_id),
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
            "SELECT keyword FROM OperationOwnedCloneKeywords "
            "WHERE media_id = ? ORDER BY keyword",
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
def test_operation_owned_clone_is_readable_for_target_readiness(
    media_db: MediaDatabase,
) -> None:
    snapshot = _operation_snapshot()
    operation_id = "clone-readiness-operation"
    source_identity = "workspace-source-readiness"
    expected_hash = media_db_api.hash_media_clone_snapshot(snapshot)
    created = media_db.insert_operation_owned_clone_media(
        snapshot=snapshot,
        operation_id=operation_id,
        source_identity=source_identity,
        expected_content_hash=expected_hash,
    )

    persisted = media_db.read_operation_owned_clone_media_readiness(
        operation_id=operation_id,
        items=((source_identity, expected_hash),),
    )

    assert set(persisted) == {source_identity}
    assert persisted[source_identity].source_identity == source_identity
    assert persisted[source_identity].media_id == created.media_id
    assert persisted[source_identity].has_chunks is True


@pytest.mark.unit
def test_operation_owned_clone_readiness_rejects_duplicate_source_identities(
    media_db: MediaDatabase,
) -> None:
    expected_hash = hashlib.sha256(b"readiness-duplicate").hexdigest()

    with pytest.raises(InputError):
        media_db.read_operation_owned_clone_media_readiness(
            operation_id="clone-readiness-duplicate",
            items=(
                ("workspace-source-duplicate", expected_hash),
                ("workspace-source-duplicate", expected_hash),
            ),
        )


@pytest.mark.unit
def test_operation_owned_clone_readiness_rejects_content_hash_mismatch(
    media_db: MediaDatabase,
) -> None:
    snapshot = _operation_snapshot()
    operation_id = "clone-readiness-hash-mismatch"
    source_identity = "workspace-source-hash-mismatch"
    expected_hash = media_db_api.hash_media_clone_snapshot(snapshot)
    media_db.insert_operation_owned_clone_media(
        snapshot=snapshot,
        operation_id=operation_id,
        source_identity=source_identity,
        expected_content_hash=expected_hash,
    )

    with pytest.raises(CloneSnapshotUnavailable):
        media_db.read_operation_owned_clone_media_readiness(
            operation_id=operation_id,
            items=((source_identity, hashlib.sha256(b"wrong").hexdigest()),),
        )


@pytest.mark.unit
def test_operation_owned_clone_readiness_rejects_foreign_row_in_operation(
    media_db: MediaDatabase,
) -> None:
    snapshot = _operation_snapshot()
    operation_id = "clone-readiness-foreign-row"
    expected_hash = media_db_api.hash_media_clone_snapshot(snapshot)
    media_db.insert_operation_owned_clone_media(
        snapshot=snapshot,
        operation_id=operation_id,
        source_identity="workspace-source-owned",
        expected_content_hash=expected_hash,
    )
    foreign = media_db.insert_operation_owned_clone_media(
        snapshot=snapshot,
        operation_id=operation_id,
        source_identity="workspace-source-foreign",
        expected_content_hash=expected_hash,
    )
    media_db.execute_query(
        "UPDATE Media SET client_id = ?, version = version + 1, last_modified = ? "
        "WHERE id = ?",
        (
            "different-owner",
            "2026-08-25T16:00:00+00:00",
            foreign.media_id,
        ),
        commit=True,
    )

    with pytest.raises(CloneSnapshotUnavailable):
        media_db.read_operation_owned_clone_media_readiness(
            operation_id=operation_id,
            items=(("workspace-source-owned", expected_hash),),
        )


@pytest.mark.unit
def test_operation_owned_clone_readiness_rejects_unrequested_row_in_operation(
    media_db: MediaDatabase,
) -> None:
    snapshot = _operation_snapshot()
    operation_id = "clone-readiness-extra-row"
    expected_hash = media_db_api.hash_media_clone_snapshot(snapshot)
    for source_identity in ("workspace-source-requested", "workspace-source-extra"):
        media_db.insert_operation_owned_clone_media(
            snapshot=snapshot,
            operation_id=operation_id,
            source_identity=source_identity,
            expected_content_hash=expected_hash,
        )

    with pytest.raises(CloneSnapshotUnavailable):
        media_db.read_operation_owned_clone_media_readiness(
            operation_id=operation_id,
            items=(("workspace-source-requested", expected_hash),),
        )


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
        query = (
            "DELETE FROM OperationOwnedCloneKeywords WHERE media_id = ? AND keyword = "
            "(SELECT MIN(keyword) FROM OperationOwnedCloneKeywords WHERE media_id = ?)"
        )
        params = (created.media_id, created.media_id)
    elif graph_mutation == "keyword_changed":
        query = (
            "UPDATE OperationOwnedCloneKeywords SET keyword = ? WHERE media_id = ? "
            "AND keyword = (SELECT MIN(keyword) FROM OperationOwnedCloneKeywords "
            "WHERE media_id = ?)"
        )
        params = (
            "tampered-keyword",
            created.media_id,
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
    readiness = media_db_api.read_operation_owned_clone_media_readiness(
        media_db,
        operation_id="clone-operation-facade",
        items=(("workspace-source-facade", expected_hash),),
    )
    assert readiness["workspace-source-facade"].media_id == result.media_id
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
def test_postgres_clone_chunk_json_text_semantics_match_sqlite(
    pg_database_config: DatabaseConfig,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = MediaDatabase(db_path=":memory:", client_id="901", backend=backend)
    cases = [
        ('{"b":[2],"a":1}', {"a": 1, "b": [2]}),
        ("[1,true,null]", [1, True, None]),
        ("42", 42),
        ("true", True),
        ("null", None),
        ('"json scalar"', "json scalar"),
        ("invalid legacy json text", "invalid legacy json text"),
    ]
    try:
        with scoped_context(user_id=901, org_ids=[], team_ids=[], is_admin=True):
            for index, (source_metadata, semantic_metadata) in enumerate(cases):
                base = _operation_snapshot()
                chunks = [dict(row) for row in base.chunks]
                chunks[0]["metadata"] = source_metadata
                snapshot = MediaCloneSnapshot.from_rows(
                    media=base.media,
                    chunks=chunks,
                    transcripts=base.transcripts,
                )
                expected_hash = media_db_api.hash_media_clone_snapshot(snapshot)
                operation_id = f"clone-operation-pg-json-{index}-{uuid.uuid4()}"
                source_identity = f"workspace-source-pg-json-{index}-{uuid.uuid4()}"
                created = db.insert_operation_owned_clone_media(
                    snapshot=snapshot,
                    operation_id=operation_id,
                    source_identity=source_identity,
                    expected_content_hash=expected_hash,
                )
                stored = db.execute_query(
                    "SELECT metadata FROM UnvectorizedMediaChunks "
                    "WHERE media_id = ? AND chunk_index = 0",
                    (created.media_id,),
                ).fetchone()["metadata"]
                assert (json.loads(stored) if isinstance(stored, str) else stored) == (
                    semantic_metadata
                )
                assert db.insert_operation_owned_clone_media(
                    snapshot=snapshot,
                    operation_id=operation_id,
                    source_identity=source_identity,
                    expected_content_hash=expected_hash,
                ).replayed is True
                assert db.delete_operation_owned_clone_media(
                    operation_id=operation_id,
                    source_identity=source_identity,
                    expected_content_hash=expected_hash,
                ) == 1
    finally:
        db.close_connection()
        backend.get_pool().close_all()


@pytest.mark.integration
@pytest.mark.postgres
@pytest.mark.timeout(120)
def test_postgres_v26_migration_harvests_original_v25_direct_keywords_and_replays(
    pg_database_config: DatabaseConfig,
) -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema.migration_bodies.postgres_staged_clone_persistence import (
        run_postgres_migrate_to_v26,
    )

    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = MediaDatabase(db_path=":memory:", client_id="901", backend=backend)
    suffix = uuid.uuid4().hex
    keywords = (f"original-pg-alpha-{suffix}", f"original-pg-research-{suffix}")
    snapshot = _operation_snapshot(keywords=keywords)
    operation_id = f"clone-operation-pg-original-{suffix}"
    source_identity = f"workspace-source-pg-original-{suffix}"
    expected_hash = media_db_api.hash_media_clone_snapshot(snapshot)
    try:
        with scoped_context(user_id=901, org_ids=[], team_ids=[], is_admin=True):
            created = db.insert_operation_owned_clone_media(
                snapshot=snapshot,
                operation_id=operation_id,
                source_identity=source_identity,
                expected_content_hash=expected_hash,
            )
            with db.transaction() as connection:
                backend.execute(
                    "DROP TABLE operationownedclonekeywords CASCADE",
                    connection=connection,
                )
                for keyword in keywords:
                    keyword_id = backend.execute(
                        "INSERT INTO Keywords "
                        "(keyword, uuid, last_modified, client_id, deleted) "
                        "VALUES (%s, %s, CURRENT_TIMESTAMP, %s, FALSE) RETURNING id",
                        (keyword, str(uuid.uuid4()), "901"),
                        connection=connection,
                    ).rows[0]["id"]
                    backend.execute(
                        "INSERT INTO MediaKeywords (media_id, keyword_id) VALUES (%s, %s)",
                        (created.media_id, keyword_id),
                        connection=connection,
                    )

                run_postgres_migrate_to_v26(db, connection)

            assert [
                row["keyword"]
                for row in db.execute_query(
                    "SELECT keyword FROM OperationOwnedCloneKeywords "
                    "WHERE media_id = ? ORDER BY keyword",
                    (created.media_id,),
                ).fetchall()
            ] == list(keywords)
            assert db.execute_query(
                "SELECT COUNT(*) AS count FROM MediaKeywords WHERE media_id = ?",
                (created.media_id,),
            ).fetchone()["count"] == 0
            assert {
                row["keyword"]
                for row in db.execute_query(
                    "SELECT keyword FROM Keywords WHERE keyword = ANY(?)",
                    (list(keywords),),
                ).fetchall()
            } == set(keywords)

            replayed = db.insert_operation_owned_clone_media(
                snapshot=snapshot,
                operation_id=operation_id,
                source_identity=source_identity,
                expected_content_hash=expected_hash,
            )
            assert replayed.media_id == created.media_id
            assert replayed.replayed is True
            assert db.confirm_operation_owned_clone_media(
                operation_id=operation_id,
                source_identity=source_identity,
                expected_content_hash=expected_hash,
            ) == 1
            assert db.execute_query(
                "SELECT COUNT(*) AS count FROM MediaKeywords WHERE media_id = ?",
                (created.media_id,),
            ).fetchone()["count"] == len(keywords)
    finally:
        db.close_connection()
        backend.get_pool().close_all()


@pytest.mark.integration
@pytest.mark.postgres
@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    ("index_name", "index_columns"),
    [
        ("idx_owned_clone_keywords_keyword", "keyword"),
        (
            "idx_owned_clone_keywords_operation",
            "operation_id, source_identity",
        ),
    ],
)
def test_postgres_v26_rejects_unrelated_pending_index_name_collision(
    pg_database_config: DatabaseConfig,
    index_name: str,
    index_columns: str,
) -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema.migration_bodies.postgres_staged_clone_persistence import (
        run_postgres_migrate_to_v26,
    )

    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = MediaDatabase(db_path=":memory:", client_id="901", backend=backend)
    ident = backend.escape_identifier  # type: ignore[attr-defined]
    suffix = uuid.uuid4().hex[:12]
    collision_table = f"clone_index_collision_{suffix}"
    snapshot = _operation_snapshot(keywords=(f"collision-keyword-{suffix}",))
    operation_id = f"clone-operation-pg-index-collision-{suffix}"
    source_identity = f"workspace-source-pg-index-collision-{suffix}"
    expected_hash = media_db_api.hash_media_clone_snapshot(snapshot)
    try:
        with scoped_context(user_id=901, org_ids=[], team_ids=[], is_admin=True):
            created = db.insert_operation_owned_clone_media(
                snapshot=snapshot,
                operation_id=operation_id,
                source_identity=source_identity,
                expected_content_hash=expected_hash,
            )
            with db.transaction() as connection:
                backend.execute(
                    "DROP TABLE operationownedclonekeywords CASCADE",
                    connection=connection,
                )
                backend.execute(
                    f"CREATE TABLE {ident(collision_table)} ("  # nosec B608
                    "id BIGINT PRIMARY KEY, keyword TEXT NOT NULL, "
                    "operation_id TEXT NOT NULL, source_identity TEXT NOT NULL)",
                    connection=connection,
                )
                backend.execute(
                    f"CREATE INDEX {ident(index_name)} "  # nosec B608
                    f"ON {ident(collision_table)} ({index_columns})",  # nosec B608
                    connection=connection,
                )
                backend.execute(
                    f"INSERT INTO {ident(collision_table)} "  # nosec B608
                    "(id, keyword, operation_id, source_identity) "
                    "VALUES (1, 'sentinel', 'sentinel-operation', 'sentinel-source')",
                    connection=connection,
                )
                keyword_id = backend.execute(
                    "INSERT INTO Keywords "
                    "(keyword, uuid, last_modified, client_id, deleted) "
                    "VALUES (%s, %s, CURRENT_TIMESTAMP, %s, FALSE) RETURNING id",
                    (snapshot.media["keywords"][0], str(uuid.uuid4()), "901"),
                    connection=connection,
                ).rows[0]["id"]
                backend.execute(
                    "INSERT INTO MediaKeywords (media_id, keyword_id) VALUES (%s, %s)",
                    (created.media_id, keyword_id),
                    connection=connection,
                )

            with pytest.raises(SchemaError, match="index ownership"):
                with db.transaction() as connection:
                    run_postgres_migrate_to_v26(db, connection)

            with db.transaction() as connection:
                assert backend.execute(
                    "SELECT indexed_table.relname AS table_name "
                    "FROM pg_class AS index_row "
                    "JOIN pg_namespace AS namespace_row "
                    "ON namespace_row.oid = index_row.relnamespace "
                    "JOIN pg_index AS index_meta "
                    "ON index_meta.indexrelid = index_row.oid "
                    "JOIN pg_class AS indexed_table "
                    "ON indexed_table.oid = index_meta.indrelid "
                    "WHERE namespace_row.nspname = current_schema() "
                    "AND index_row.relname = %s",
                    (index_name,),
                    connection=connection,
                ).rows == [{"table_name": collision_table}]
                assert backend.execute(
                    f"SELECT keyword FROM {ident(collision_table)} WHERE id = 1",  # nosec B608
                    connection=connection,
                ).rows == [{"keyword": "sentinel"}]
                assert backend.execute(
                    "SELECT to_regclass(current_schema() || "
                    "'.operationownedclonekeywords') AS pending",
                    connection=connection,
                ).rows[0]["pending"] is None
                assert backend.execute(
                    "SELECT COUNT(*) AS count FROM MediaKeywords WHERE media_id = %s",
                    (created.media_id,),
                    connection=connection,
                ).rows[0]["count"] == 1
                assert backend.execute(
                    "SELECT system_operation_id, system_operation_kind, "
                    "system_source_identity, system_content_hash "
                    "FROM Media WHERE id = %s",
                    (created.media_id,),
                    connection=connection,
                ).rows == [
                    {
                        "system_operation_id": operation_id,
                        "system_operation_kind": "shared_workspace_clone",
                        "system_source_identity": source_identity,
                        "system_content_hash": expected_hash,
                    }
                ]
    finally:
        db.close_connection()
        backend.get_pool().close_all()


def _seed_cross_tenant_v25_keyword_graph(
    db: MediaDatabase,
    backend: Any,
    *,
    suffix: str,
) -> dict[str, int]:
    media_ids: dict[str, int] = {}
    with scoped_context(user_id=901, org_ids=[], team_ids=[], is_admin=True):
        with db.transaction() as connection:
            backend.execute(
                "DROP TABLE operationownedclonekeywords CASCADE",
                connection=connection,
            )
            backend.execute(
                """
                CREATE TABLE operationownedclonekeywords (
                    media_id BIGINT NOT NULL REFERENCES media(id) ON DELETE CASCADE,
                    keyword_id BIGINT NOT NULL REFERENCES keywords(id) ON DELETE CASCADE,
                    operation_id TEXT NOT NULL,
                    source_identity TEXT NOT NULL,
                    created_by_clone BOOLEAN NOT NULL,
                    PRIMARY KEY (media_id, keyword_id)
                )
                """,
                connection=connection,
            )
            for tenant in ("tenant-a", "tenant-b"):
                operation_id = f"operation-{tenant}-{suffix}"
                source_identity = f"source-{tenant}-{suffix}"
                media_id = int(
                    backend.execute(
                        "INSERT INTO Media "
                        "(title, type, content_hash, uuid, last_modified, client_id, is_trash, "
                        "system_operation_id, system_operation_kind, system_source_identity, "
                        "system_content_hash) VALUES "
                        "(%s, %s, %s, %s, CURRENT_TIMESTAMP, %s, TRUE, %s, %s, %s, %s) "
                        "RETURNING id",
                        (
                            f"staged {tenant}",
                            "text",
                            f"content-{tenant}-{suffix}",
                            str(uuid.uuid4()),
                            tenant,
                            operation_id,
                            "shared_workspace_clone",
                            source_identity,
                            "a" * 64,
                        ),
                        connection=connection,
                    ).rows[0]["id"]
                )
                keyword_id = int(
                    backend.execute(
                        "INSERT INTO Keywords "
                        "(keyword, uuid, last_modified, client_id, deleted) "
                        "VALUES (%s, %s, CURRENT_TIMESTAMP, %s, FALSE) RETURNING id",
                        (f"keyword-{tenant}-{suffix}", str(uuid.uuid4()), tenant),
                        connection=connection,
                    ).rows[0]["id"]
                )
                backend.execute(
                    "INSERT INTO MediaKeywords (media_id, keyword_id) VALUES (%s, %s)",
                    (media_id, keyword_id),
                    connection=connection,
                )
                backend.execute(
                    "INSERT INTO operationownedclonekeywords "
                    "(media_id, keyword_id, operation_id, source_identity, created_by_clone) "
                    "VALUES (%s, %s, %s, %s, FALSE)",
                    (media_id, keyword_id, operation_id, source_identity),
                    connection=connection,
                )
                media_ids[tenant] = media_id
            backend.execute(
                "ALTER TABLE operationownedclonekeywords ENABLE ROW LEVEL SECURITY",
                connection=connection,
            )
            backend.execute(
                "ALTER TABLE operationownedclonekeywords FORCE ROW LEVEL SECURITY",
                connection=connection,
            )
            backend.execute(
                """
                CREATE POLICY operationownedclonekeywords_v25_tenant
                ON operationownedclonekeywords FOR ALL
                USING (
                    EXISTS (
                        SELECT 1 FROM Media AS owned_media
                        WHERE owned_media.id = operationownedclonekeywords.media_id
                          AND COALESCE(
                              owned_media.owner_user_id::TEXT,
                              owned_media.client_id
                          ) = current_setting('app.current_user_id', TRUE)
                    )
                )
                WITH CHECK (
                    EXISTS (
                        SELECT 1 FROM Media AS owned_media
                        WHERE owned_media.id = operationownedclonekeywords.media_id
                          AND COALESCE(
                              owned_media.owner_user_id::TEXT,
                              owned_media.client_id
                          ) = current_setting('app.current_user_id', TRUE)
                    )
                )
                """,
                connection=connection,
            )
    return media_ids


def _transfer_clone_migration_ownership(
    backend: Any,
    *,
    role_name: str,
) -> tuple[str, str]:
    ident = backend.escape_identifier
    with backend.transaction() as connection:
        original_user = str(
            backend.execute(
                "SELECT current_user::TEXT AS current_user",
                connection=connection,
            ).rows[0]["current_user"]
        )
        schema_owner = str(
            backend.execute(
                "SELECT pg_get_userbyid(nspowner) AS owner "
                "FROM pg_namespace WHERE nspname = current_schema()",
                connection=connection,
            ).rows[0]["owner"]
        )
        backend.execute(
            f"CREATE ROLE {ident(role_name)} NOLOGIN NOSUPERUSER NOBYPASSRLS",
            connection=connection,
        )
        backend.execute(
            f"GRANT {ident(role_name)} TO CURRENT_USER",
            connection=connection,
        )
        backend.execute(
            f"ALTER SCHEMA {ident('public')} OWNER TO {ident(role_name)}",
            connection=connection,
        )
        for table in (
            "media",
            "keywords",
            "mediakeywords",
            "operationownedclonekeywords",
        ):
            backend.execute(
                f"ALTER TABLE {ident(table)} OWNER TO {ident(role_name)}",
                connection=connection,
            )
    return original_user, schema_owner


def _restore_clone_migration_ownership(
    backend: Any,
    *,
    role_name: str,
    original_user: str,
    schema_owner: str,
) -> None:
    ident = backend.escape_identifier
    with backend.transaction() as connection:
        relation_names = {
            row["relname"]
            for row in backend.execute(
                "SELECT c.relname FROM pg_class AS c "
                "JOIN pg_namespace AS n ON n.oid = c.relnamespace "
                "WHERE n.nspname = current_schema() "
                "AND c.relname = ANY(%s) AND c.relkind IN ('r', 'p')",
                (
                    [
                        "media",
                        "keywords",
                        "mediakeywords",
                        "operationownedclonekeywords",
                        "operationownedclonekeywords_v25",
                    ],
                ),
                connection=connection,
            ).rows
        }
        for table in sorted(relation_names):
            backend.execute(
                f"ALTER TABLE {ident(table)} OWNER TO {ident(original_user)}",
                connection=connection,
            )
        backend.execute(
            f"ALTER SCHEMA {ident('public')} OWNER TO {ident(schema_owner)}",
            connection=connection,
        )
        backend.execute(
            f"REVOKE {ident(role_name)} FROM CURRENT_USER",
            connection=connection,
        )
        backend.execute(f"DROP ROLE {ident(role_name)}", connection=connection)


@pytest.mark.integration
@pytest.mark.postgres
@pytest.mark.timeout(120)
def test_postgres_v26_schema_owner_migrates_all_tenants_under_forced_rls(
    pg_database_config: DatabaseConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema.migration_bodies.postgres_staged_clone_persistence import (
        run_postgres_migrate_to_v26,
    )

    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = MediaDatabase(db_path=":memory:", client_id="tenant-a", backend=backend)
    suffix = uuid.uuid4().hex[:12]
    role_name = f"clone_v26_owner_{suffix}"
    media_ids = _seed_cross_tenant_v25_keyword_graph(db, backend, suffix=suffix)
    ownership: tuple[str, str] | None = None
    monkeypatch.setenv("TLDW_CONTENT_PG_ROLE_SWITCH", "1")
    monkeypatch.setenv("TLDW_CONTENT_PG_ROLE_WHITELIST", role_name)
    try:
        ownership = _transfer_clone_migration_ownership(
            backend,
            role_name=role_name,
        )
        with scoped_context(
            user_id="tenant-a",
            org_ids=[],
            team_ids=[],
            is_admin=False,
            session_role=role_name,
        ):
            with db.transaction() as connection:
                principal = backend.execute(
                    "SELECT current_role::TEXT AS role, rolsuper, rolbypassrls "
                    "FROM pg_roles WHERE rolname = current_role",
                    connection=connection,
                ).rows[0]
                assert principal == {
                    "role": role_name,
                    "rolsuper": False,
                    "rolbypassrls": False,
                }
                assert backend.execute(
                    "SELECT current_setting('app.current_user_id', TRUE) AS user_id, "
                    "current_setting('app.is_admin', TRUE) AS is_admin",
                    connection=connection,
                ).rows[0] == {"user_id": "tenant-a", "is_admin": "0"}
                run_postgres_migrate_to_v26(db, connection)

        with backend.transaction() as connection:
            assert {
                (int(row["media_id"]), row["keyword"])
                for row in backend.execute(
                    "SELECT media_id, keyword FROM operationownedclonekeywords",
                    connection=connection,
                ).rows
            } == {
                (media_ids[tenant], f"keyword-{tenant}-{suffix}")
                for tenant in ("tenant-a", "tenant-b")
            }
            assert backend.execute(
                "SELECT COUNT(*) AS count FROM MediaKeywords "
                "WHERE media_id = ANY(%s)",
                (list(media_ids.values()),),
                connection=connection,
            ).rows[0]["count"] == 0
            states = {
                row["relname"]: (row["relrowsecurity"], row["relforcerowsecurity"])
                for row in backend.execute(
                    "SELECT relname, relrowsecurity, relforcerowsecurity "
                    "FROM pg_class WHERE oid = ANY(%s::regclass[])",
                    (
                        [
                            "media",
                            "operationownedclonekeywords",
                        ],
                    ),
                    connection=connection,
                ).rows
            }
            assert states == {
                "media": (True, True),
                "operationownedclonekeywords": (True, True),
            }
            assert backend.execute(
                "SELECT to_regclass(current_schema() || "
                "'.operationownedclonekeywords_v25') AS legacy",
                connection=connection,
            ).rows[0]["legacy"] is None
    finally:
        if ownership is not None:
            _restore_clone_migration_ownership(
                backend,
                role_name=role_name,
                original_user=ownership[0],
                schema_owner=ownership[1],
            )
        db.close_connection()
        backend.get_pool().close_all()


@pytest.mark.integration
@pytest.mark.postgres
@pytest.mark.timeout(120)
def test_postgres_v26_failure_after_no_force_rolls_back_rls_and_legacy_graph(
    pg_database_config: DatabaseConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema.migration_bodies.postgres_staged_clone_persistence import (
        run_postgres_migrate_to_v26,
    )

    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = MediaDatabase(db_path=":memory:", client_id="tenant-a", backend=backend)
    suffix = uuid.uuid4().hex[:12]
    role_name = f"clone_v26_rollback_{suffix}"
    media_ids = _seed_cross_tenant_v25_keyword_graph(db, backend, suffix=suffix)
    ownership: tuple[str, str] | None = None
    original_execute = backend.execute
    saw_no_force = False
    monkeypatch.setenv("TLDW_CONTENT_PG_ROLE_SWITCH", "1")
    monkeypatch.setenv("TLDW_CONTENT_PG_ROLE_WHITELIST", role_name)

    def interrupt_after_no_force(query, params=None, connection=None, **kwargs):
        nonlocal saw_no_force
        normalized = " ".join(query.split()).lower()
        if " no force row level security" in normalized:
            saw_no_force = True
        elif saw_no_force and normalized.startswith("update \"media\""):
            raise SchemaError("simulated v26 migration interruption")
        return original_execute(query, params, connection=connection, **kwargs)

    try:
        ownership = _transfer_clone_migration_ownership(
            backend,
            role_name=role_name,
        )
        monkeypatch.setattr(backend, "execute", interrupt_after_no_force)
        with scoped_context(
            user_id="tenant-a",
            org_ids=[],
            team_ids=[],
            is_admin=False,
            session_role=role_name,
        ):
            with pytest.raises(SchemaError, match="simulated v26 migration interruption"):
                with db.transaction() as connection:
                    run_postgres_migrate_to_v26(db, connection)
        assert saw_no_force is True

        with backend.transaction() as connection:
            states = {
                row["relname"]: (row["relrowsecurity"], row["relforcerowsecurity"])
                for row in original_execute(
                    "SELECT relname, relrowsecurity, relforcerowsecurity "
                    "FROM pg_class WHERE oid = ANY(%s::regclass[])",
                    (["media", "operationownedclonekeywords"],),
                    connection=connection,
                ).rows
            }
            assert states == {
                "media": (True, True),
                "operationownedclonekeywords": (True, True),
            }
            assert original_execute(
                "SELECT COUNT(*) AS count FROM operationownedclonekeywords",
                connection=connection,
            ).rows[0]["count"] == 2
            assert original_execute(
                "SELECT COUNT(*) AS count FROM MediaKeywords "
                "WHERE media_id = ANY(%s)",
                (list(media_ids.values()),),
                connection=connection,
            ).rows[0]["count"] == 2
    finally:
        monkeypatch.setattr(backend, "execute", original_execute)
        if ownership is not None:
            _restore_clone_migration_ownership(
                backend,
                role_name=role_name,
                original_user=ownership[0],
                schema_owner=ownership[1],
            )
        db.close_connection()
        backend.get_pool().close_all()


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
def test_postgres_distinct_clone_confirmations_converge_on_absent_keyword(
    pg_database_config: DatabaseConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    databases = [
        MediaDatabase(db_path=":memory:", client_id="901", backend=backend),
        MediaDatabase(db_path=":memory:", client_id="901", backend=backend),
    ]
    keyword = f"pg-confirm-shared-{uuid.uuid4().hex}"
    snapshot = _operation_snapshot(keywords=(keyword,))
    expected_hash = media_db_api.hash_media_clone_snapshot(snapshot)
    operations = [f"clone-operation-pg-confirm-{uuid.uuid4()}" for _ in range(2)]
    sources = [f"workspace-source-pg-confirm-{uuid.uuid4()}" for _ in range(2)]
    start = threading.Barrier(2)
    original_execute = backend.execute

    def widen_absent_keyword_race(query, params=None, connection=None, **kwargs):
        result = original_execute(query, params, connection=connection, **kwargs)
        if "SELECT id, deleted FROM Keywords" in query and not result.rows:
            time.sleep(0.2)
        return result

    def confirm(index: int) -> int:
        with scoped_context(user_id=901, org_ids=[], team_ids=[], is_admin=True):
            start.wait(timeout=30)
            return databases[index].confirm_operation_owned_clone_media(
                operation_id=operations[index],
                source_identity=sources[index],
                expected_content_hash=expected_hash,
            )

    monkeypatch.setattr(backend, "execute", widen_absent_keyword_race)
    try:
        with scoped_context(user_id=901, org_ids=[], team_ids=[], is_admin=True):
            for index, database in enumerate(databases):
                database.insert_operation_owned_clone_media(
                    snapshot=snapshot,
                    operation_id=operations[index],
                    source_identity=sources[index],
                    expected_content_hash=expected_hash,
                )

        with ThreadPoolExecutor(max_workers=2) as executor:
            assert list(executor.map(confirm, range(2))) == [1, 1]

        with scoped_context(user_id=901, org_ids=[], team_ids=[], is_admin=True):
            keyword_rows = databases[0].execute_query(
                "SELECT id FROM Keywords WHERE LOWER(keyword) = ?",
                (keyword,),
            ).fetchall()
            assert len(keyword_rows) == 1
            assert databases[0].execute_query(
                "SELECT COUNT(*) AS count FROM MediaKeywords WHERE keyword_id = ?",
                (keyword_rows[0]["id"],),
            ).fetchone()["count"] == 2
            assert databases[0].execute_query(
                "SELECT COUNT(*) AS count FROM OperationOwnedCloneKeywords "
                "WHERE keyword = ?",
                (keyword,),
            ).fetchone()["count"] == 0
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
            assert owner_db.execute_query(
                "SELECT COUNT(*) AS count FROM OperationOwnedCloneKeywords "
                "WHERE media_id = ?",
                (created.media_id,),
            ).fetchone()["count"] == len(snapshot.media["keywords"])

        with scoped_context(user_id=901, org_ids=[], team_ids=[], is_admin=True):
            owner_db.execute_query(
                "UPDATE Media SET deleted = TRUE, version = version + 1 "
                "WHERE id = ?",
                (created.media_id,),
                commit=True,
            )
            owner_db.execute_query(
                "UPDATE OperationOwnedCloneKeywords SET operation_id = ?, "
                "source_identity = ?, client_id = ? WHERE media_id = ?",
                (
                    "tampered-operation",
                    "tampered-source",
                    "tampered-owner",
                    created.media_id,
                ),
                commit=True,
            )

        with scoped_context(
            user_id=902,
            org_ids=[],
            team_ids=[],
            is_admin=False,
            session_role=role_name,
        ):
            assert other_db.execute_query(
                "SELECT COUNT(*) AS count FROM OperationOwnedCloneKeywords "
                "WHERE media_id = ?",
                (created.media_id,),
            ).fetchone()["count"] == 0
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
            assert owner_db.execute_query(
                "SELECT COUNT(*) AS count FROM OperationOwnedCloneKeywords "
                "WHERE media_id = ?",
                (created.media_id,),
            ).fetchone()["count"] == len(snapshot.media["keywords"])
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
        with scoped_context(user_id=901, org_ids=[], team_ids=[], is_admin=True):
            assert owner_db.execute_query(
                "SELECT COUNT(*) AS count FROM OperationOwnedCloneKeywords "
                "WHERE media_id = ?",
                (created.media_id,),
            ).fetchone()["count"] == 0
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
            readiness = db.read_operation_owned_clone_media_readiness(
                operation_id=operation_id,
                items=((source_identity, expected_hash),),
            )
            assert readiness[source_identity].media_id == created.media_id
            assert readiness[source_identity].has_chunks is True
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
