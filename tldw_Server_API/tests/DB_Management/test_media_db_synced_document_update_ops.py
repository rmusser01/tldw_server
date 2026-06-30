from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sqlite3
import sys
import types
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.media_db.errors import (
    ConflictError,
    DatabaseError,
    InputError,
)


pytestmark = pytest.mark.unit


def _load_synced_document_update_ops_module():
    module_name = (
        "tldw_Server_API.app.core.DB_Management.media_db.runtime.synced_document_update_ops"
    )
    assert importlib.util.find_spec(module_name) is not None
    return importlib.import_module(module_name)


def test_apply_synced_document_content_update_rebinds_on_media_database() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.media_database import MediaDatabase

    synced_document_update_ops_module = _load_synced_document_update_ops_module()

    assert (
        MediaDatabase.apply_synced_document_content_update
        is synced_document_update_ops_module.apply_synced_document_content_update
    )


def test_apply_synced_document_content_update_rejects_missing_content() -> None:
    synced_document_update_ops_module = _load_synced_document_update_ops_module()

    with pytest.raises(InputError, match="Content is required for synced document updates"):
        synced_document_update_ops_module.apply_synced_document_content_update(
            SimpleNamespace(),
            media_id=9,
            content=None,
        )


def test_apply_synced_document_content_update_rejects_missing_media() -> None:
    synced_document_update_ops_module = _load_synced_document_update_ops_module()

    class _Txn:
        def __enter__(self):
            return "conn"

        def __exit__(self, exc_type, exc, tb):
            return False

    db = SimpleNamespace(
        client_id="sync-client",
        backend=object(),
        transaction=lambda: _Txn(),
        _get_current_utc_timestamp_str=lambda: "2026-03-22T20:00:00Z",
        _fetchone_with_connection=lambda conn, query, params: None,
        _execute_with_connection=lambda *_args, **_kwargs: None,
        create_document_version=lambda **_kwargs: None,
        _log_sync_event=lambda *_args, **_kwargs: None,
        _update_fts_media=lambda *_args, **_kwargs: None,
    )

    with pytest.raises(InputError, match="Media 9 not found or deleted"):
        synced_document_update_ops_module.apply_synced_document_content_update(
            db,
            media_id=9,
            content="updated body",
        )


def test_apply_synced_document_content_update_rejects_optimistic_conflict() -> None:
    synced_document_update_ops_module = _load_synced_document_update_ops_module()

    class _Txn:
        def __enter__(self):
            return "conn"

        def __exit__(self, exc_type, exc, tb):
            return False

    fetch_rows = [
        {"uuid": "media-uuid", "version": 1, "title": "Current Title", "content": "old body"},
    ]

    def _fetchone(_conn, _query, _params):
        return fetch_rows.pop(0)

    db = SimpleNamespace(
        client_id="sync-client",
        backend=object(),
        transaction=lambda: _Txn(),
        _get_current_utc_timestamp_str=lambda: "2026-03-22T20:00:00Z",
        _fetchone_with_connection=_fetchone,
        _execute_with_connection=lambda *_args, **_kwargs: SimpleNamespace(rowcount=0),
        create_document_version=lambda **_kwargs: None,
        _log_sync_event=lambda *_args, **_kwargs: None,
        _update_fts_media=lambda *_args, **_kwargs: None,
    )

    with pytest.raises(ConflictError):
        synced_document_update_ops_module.apply_synced_document_content_update(
            db,
            media_id=9,
            content="updated body",
        )


def test_apply_synced_document_content_update_updates_media_creates_version_logs_sync_and_refreshes_fts(
    monkeypatch,
) -> None:
    synced_document_update_ops_module = _load_synced_document_update_ops_module()

    events: list[tuple[str, object]] = []
    fetch_rows = [
        {"uuid": "media-uuid", "version": 1, "title": "Current Title", "content": "old body"},
        {"id": 9, "uuid": "media-uuid", "title": "Current Title", "version": 2},
    ]

    class _Txn:
        def __enter__(self):
            events.append(("transaction_enter", None))
            return "conn"

        def __exit__(self, exc_type, exc, tb):
            events.append(("transaction_exit", exc_type))
            return False

    def _fetchone(conn, query, params):
        normalized = " ".join(query.split())
        if normalized.startswith("SELECT uuid, version, title, content FROM Media"):
            events.append(("fetch_media", params))
        else:
            events.append(("fetch_updated_media", params))
        return fetch_rows.pop(0)

    execute_calls: list[tuple[str, tuple[object, ...]]] = []
    sync_payloads: list[dict[str, object]] = []
    fts_calls: list[tuple[object, int, str, str, str | None, str | None]] = []
    collection_calls: list[tuple[int, str]] = []
    vector_calls: list[str] = []

    def _execute(_conn, query, params):
        execute_calls.append((" ".join(query.split()), params))
        events.append(("update_media", params))
        return SimpleNamespace(rowcount=1)

    def _create_document_version(**kwargs):
        events.append(("create_document_version", kwargs))
        return {"uuid": "dv-uuid-4", "version_number": 4}

    def _log_sync_event(conn, entity, entity_uuid, operation, version, payload):
        events.append(("log_sync_event", version))
        sync_payloads.append(payload)

    def _update_fts_media(conn, media_id, title, content, *, old_title=None, old_content=None):
        events.append(("update_fts_media", media_id))
        fts_calls.append((conn, media_id, title, content, old_title, old_content))

    class _FakeCollectionsDatabase:
        @classmethod
        def from_backend(cls, *, user_id, backend):
            events.append(("collections_from_backend", (user_id, backend)))

            class _Instance:
                def mark_highlights_stale_if_content_changed(self, media_id, content_hash):
                    events.append(("collections_mark_stale", media_id))
                    collection_calls.append((media_id, content_hash))

            return _Instance()

    agentic_chunker_module = types.ModuleType(
        "tldw_Server_API.app.core.RAG.rag_service.agentic_chunker"
    )

    def _invalidate(media_key: str) -> None:
        events.append(("invalidate_vectors", media_key))
        vector_calls.append(media_key)

    agentic_chunker_module.invalidate_intra_doc_vectors = _invalidate  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules,
        "tldw_Server_API.app.core.RAG.rag_service.agentic_chunker",
        agentic_chunker_module,
    )
    monkeypatch.setattr(
        synced_document_update_ops_module,
        "_COLLECTIONS_DB",
        _FakeCollectionsDatabase,
        raising=False,
    )

    db = SimpleNamespace(
        client_id="sync-client",
        backend=object(),
        transaction=lambda: _Txn(),
        _get_current_utc_timestamp_str=lambda: "2026-03-22T20:00:00Z",
        _fetchone_with_connection=_fetchone,
        _execute_with_connection=_execute,
        create_document_version=_create_document_version,
        _log_sync_event=_log_sync_event,
        _update_fts_media=_update_fts_media,
    )

    result = synced_document_update_ops_module.apply_synced_document_content_update(
        db,
        media_id=9,
        content="updated body",
        prompt="sync prompt",
        analysis_content="sync analysis",
        safe_metadata='{"source":"sync"}',
    )

    expected_hash = hashlib.sha256("updated body".encode()).hexdigest()

    assert result == {
        "media_id": 9,
        "content_hash": expected_hash,
        "new_media_version": 2,
        "document_version_number": 4,
        "document_version_uuid": "dv-uuid-4",
    }
    assert execute_calls == [
        (
            "UPDATE Media SET content = ?, content_hash = ?, last_modified = ?, version = ?, client_id = ?, chunking_status = 'pending', vector_processing = 0 WHERE id = ? AND version = ?",
            (
                "updated body",
                expected_hash,
                "2026-03-22T20:00:00Z",
                2,
                "sync-client",
                9,
                1,
            ),
        )
    ]
    assert sync_payloads == [
        {
            "id": 9,
            "uuid": "media-uuid",
            "title": "Current Title",
            "version": 2,
            "created_doc_ver_uuid": "dv-uuid-4",
            "created_doc_ver_num": 4,
        }
    ]
    assert fts_calls == [
        ("conn", 9, "Current Title", "updated body", "Current Title", "old body")
    ]
    assert collection_calls == [(9, expected_hash)]
    assert vector_calls == ["9"]
    assert [name for name, _value in events] == [
        "transaction_enter",
        "fetch_media",
        "update_media",
        "create_document_version",
        "fetch_updated_media",
        "log_sync_event",
        "update_fts_media",
        "transaction_exit",
        "collections_from_backend",
        "collections_mark_stale",
        "invalidate_vectors",
    ]


def test_apply_synced_document_content_update_swallows_best_effort_post_commit_hook_failures(
    monkeypatch,
) -> None:
    synced_document_update_ops_module = _load_synced_document_update_ops_module()

    class _Txn:
        def __enter__(self):
            return "conn"

        def __exit__(self, exc_type, exc, tb):
            return False

    fetch_rows = [
        {"uuid": "media-uuid", "version": 1, "title": "Current Title", "content": "old body"},
        {"id": 9, "uuid": "media-uuid", "title": "Current Title", "version": 2},
    ]

    def _fetchone(_conn, _query, _params):
        return fetch_rows.pop(0)

    class _BrokenCollectionsDatabase:
        @classmethod
        def from_backend(cls, *, user_id, backend):
            class _Instance:
                def mark_highlights_stale_if_content_changed(self, media_id, content_hash):
                    raise RuntimeError("collections hook failed")

            return _Instance()

    agentic_chunker_module = types.ModuleType(
        "tldw_Server_API.app.core.RAG.rag_service.agentic_chunker"
    )

    def _invalidate(_media_key: str) -> None:
        raise RuntimeError("vector invalidation failed")

    agentic_chunker_module.invalidate_intra_doc_vectors = _invalidate  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules,
        "tldw_Server_API.app.core.RAG.rag_service.agentic_chunker",
        agentic_chunker_module,
    )
    monkeypatch.setattr(
        synced_document_update_ops_module,
        "_COLLECTIONS_DB",
        _BrokenCollectionsDatabase,
        raising=False,
    )

    db = SimpleNamespace(
        client_id="sync-client",
        backend=object(),
        transaction=lambda: _Txn(),
        _get_current_utc_timestamp_str=lambda: "2026-03-22T20:00:00Z",
        _fetchone_with_connection=_fetchone,
        _execute_with_connection=lambda *_args, **_kwargs: SimpleNamespace(rowcount=1),
        create_document_version=lambda **_kwargs: {
            "uuid": "dv-uuid-4",
            "version_number": 4,
        },
        _log_sync_event=lambda *_args, **_kwargs: None,
        _update_fts_media=lambda *_args, **_kwargs: None,
    )

    result = synced_document_update_ops_module.apply_synced_document_content_update(
        db,
        media_id=9,
        content="updated body",
    )

    assert result["document_version_number"] == 4


def test_apply_synced_document_content_update_wraps_sqlite_errors() -> None:
    synced_document_update_ops_module = _load_synced_document_update_ops_module()

    class _Txn:
        def __enter__(self):
            return "conn"

        def __exit__(self, exc_type, exc, tb):
            return False

    db = SimpleNamespace(
        client_id="sync-client",
        backend=object(),
        transaction=lambda: _Txn(),
        _get_current_utc_timestamp_str=lambda: "2026-03-22T20:00:00Z",
        _fetchone_with_connection=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            sqlite3.OperationalError("database is locked")
        ),
        _execute_with_connection=lambda *_args, **_kwargs: None,
        create_document_version=lambda **_kwargs: None,
        _log_sync_event=lambda *_args, **_kwargs: None,
        _update_fts_media=lambda *_args, **_kwargs: None,
    )

    with pytest.raises(DatabaseError, match="Synced content update failed"):
        synced_document_update_ops_module.apply_synced_document_content_update(
            db,
            media_id=9,
            content="updated body",
        )
