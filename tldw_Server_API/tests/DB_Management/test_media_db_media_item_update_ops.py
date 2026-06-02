from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sqlite3
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.media_db.errors import (
    ConflictError,
    DatabaseError,
    InputError,
)


pytestmark = pytest.mark.unit


def _load_media_item_update_ops_module():
    module_name = (
        "tldw_Server_API.app.core.DB_Management.media_db.runtime.media_item_update_ops"
    )
    assert importlib.util.find_spec(module_name) is not None
    return importlib.import_module(module_name)


class _Txn:
    def __enter__(self):
        return "conn"

    def __exit__(self, exc_type, exc, tb):
        return False


def _media_row(
    *,
    title: str = "Current Title",
    content: str = "existing body",
    content_hash: str | None = None,
    version: int = 1,
) -> dict[str, object]:
    return {
        "id": 9,
        "uuid": "media-uuid",
        "title": title,
        "content": content,
        "content_hash": content_hash or hashlib.sha256(content.encode()).hexdigest(),
        "version": version,
    }


def test_apply_media_item_update_rebinds_on_media_database() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.media_database import MediaDatabase

    media_item_update_ops_module = _load_media_item_update_ops_module()

    assert (
        MediaDatabase.apply_media_item_update
        is media_item_update_ops_module.apply_media_item_update
    )


def test_apply_media_item_update_rejects_empty_fields() -> None:
    media_item_update_ops_module = _load_media_item_update_ops_module()

    with pytest.raises(InputError, match="At least one media update field is required"):
        media_item_update_ops_module.apply_media_item_update(
            SimpleNamespace(),
            media_id=9,
            fields={},
        )


def test_apply_media_item_update_rejects_missing_media() -> None:
    media_item_update_ops_module = _load_media_item_update_ops_module()

    db = SimpleNamespace(
        client_id="api-client",
        backend=object(),
        transaction=lambda: _Txn(),
        _get_current_utc_timestamp_str=lambda: "2026-03-22T20:00:00Z",
        _fetchone_with_connection=lambda _conn, _query, _params: None,
        _execute_with_connection=lambda *_args, **_kwargs: None,
        create_document_version=lambda **_kwargs: None,
        _log_sync_event=lambda *_args, **_kwargs: None,
        _update_fts_media=lambda *_args, **_kwargs: None,
    )

    with pytest.raises(InputError, match="Media 9 not found or inactive/trashed"):
        media_item_update_ops_module.apply_media_item_update(
            db,
            media_id=9,
            fields={"title": "Updated"},
        )


def test_apply_media_item_update_rejects_optimistic_conflict() -> None:
    media_item_update_ops_module = _load_media_item_update_ops_module()
    fetch_rows = [_media_row()]

    db = SimpleNamespace(
        client_id="api-client",
        backend=object(),
        transaction=lambda: _Txn(),
        _get_current_utc_timestamp_str=lambda: "2026-03-22T20:00:00Z",
        _fetchone_with_connection=lambda _conn, _query, _params: fetch_rows.pop(0),
        _execute_with_connection=lambda *_args, **_kwargs: SimpleNamespace(rowcount=0),
        create_document_version=lambda **_kwargs: None,
        _log_sync_event=lambda *_args, **_kwargs: None,
        _update_fts_media=lambda *_args, **_kwargs: None,
    )

    with pytest.raises(ConflictError):
        media_item_update_ops_module.apply_media_item_update(
            db,
            media_id=9,
            fields={"title": "Updated"},
        )


def test_apply_media_item_update_updates_metadata_logs_sync_and_refreshes_title_fts() -> None:
    media_item_update_ops_module = _load_media_item_update_ops_module()

    fetch_rows = [
        _media_row(title="Current Title", content="existing body", version=3),
        {
            **_media_row(title="Updated Title", content="existing body", version=4),
            "client_id": "api-client",
        },
    ]
    execute_calls: list[tuple[str, tuple[object, ...]]] = []
    sync_payloads: list[dict[str, object]] = []
    fts_calls: list[tuple[object, int, str, str]] = []
    doc_versions: list[dict[str, object]] = []

    def _fetchone(_conn, _query, _params):
        return fetch_rows.pop(0)

    def _execute(_conn, query, params):
        execute_calls.append((" ".join(query.split()), params))
        return SimpleNamespace(rowcount=1)

    def _log_sync_event(_conn, entity, entity_uuid, operation, version, payload):
        assert (entity, entity_uuid, operation, version) == (
            "Media",
            "media-uuid",
            "update",
            4,
        )
        sync_payloads.append(payload)

    def _create_document_version(**kwargs):
        doc_versions.append(kwargs)
        return {"uuid": "dv-should-not-be-created", "version_number": 99}

    def _update_fts_media(conn, media_id, title, content):
        fts_calls.append((conn, media_id, title, content))

    db = SimpleNamespace(
        client_id="api-client",
        backend=object(),
        transaction=lambda: _Txn(),
        _get_current_utc_timestamp_str=lambda: "2026-03-22T20:00:00Z",
        _fetchone_with_connection=_fetchone,
        _execute_with_connection=_execute,
        create_document_version=_create_document_version,
        _log_sync_event=_log_sync_event,
        _update_fts_media=_update_fts_media,
    )

    result = media_item_update_ops_module.apply_media_item_update(
        db,
        media_id=9,
        fields={"title": "Updated Title"},
    )

    assert result == {
        "media_id": 9,
        "content_hash": hashlib.sha256("existing body".encode()).hexdigest(),
        "new_media_version": 4,
        "content_changed": False,
        "document_version_number": None,
        "document_version_uuid": None,
        "invalidate_rag": True,
    }
    assert execute_calls == [
        (
            "UPDATE Media SET last_modified = ?, version = ?, client_id = ?, title = ? WHERE id = ? AND version = ?",
            (
                "2026-03-22T20:00:00Z",
                4,
                "api-client",
                "Updated Title",
                9,
                3,
            ),
        )
    ]
    assert sync_payloads == [
        {
            **_media_row(title="Updated Title", content="existing body", version=4),
            "client_id": "api-client",
        }
    ]
    assert fts_calls == [("conn", 9, "Updated Title", "existing body")]
    assert doc_versions == []


def test_apply_media_item_update_changes_content_creates_version_logs_sync_and_marks_stale(
    monkeypatch,
) -> None:
    media_item_update_ops_module = _load_media_item_update_ops_module()

    old_hash = hashlib.sha256("existing body".encode()).hexdigest()
    expected_hash = hashlib.sha256("updated body".encode()).hexdigest()
    fetch_rows = [
        _media_row(title="Current Title", content="existing body", content_hash=old_hash, version=1),
        {
            **_media_row(
                title="Updated Title",
                content="updated body",
                content_hash=expected_hash,
                version=2,
            ),
            "client_id": "api-client",
        },
    ]
    execute_calls: list[tuple[str, tuple[object, ...]]] = []
    doc_versions: list[dict[str, object]] = []
    sync_payloads: list[dict[str, object]] = []
    fts_calls: list[tuple[object, int, str, str]] = []
    collection_calls: list[tuple[int, str]] = []

    class _FakeCollectionsDatabase:
        @classmethod
        def from_backend(cls, *, user_id, backend):
            assert user_id == "api-client"

            class _Instance:
                def mark_highlights_stale_if_content_changed(self, media_id, content_hash):
                    collection_calls.append((media_id, content_hash))

            return _Instance()

    monkeypatch.setattr(
        media_item_update_ops_module,
        "_COLLECTIONS_DB",
        _FakeCollectionsDatabase,
        raising=False,
    )

    def _fetchone(_conn, _query, _params):
        return fetch_rows.pop(0)

    def _execute(_conn, query, params):
        execute_calls.append((" ".join(query.split()), params))
        return SimpleNamespace(rowcount=1)

    def _create_document_version(**kwargs):
        doc_versions.append(kwargs)
        return {"uuid": "dv-uuid-4", "version_number": 4}

    def _log_sync_event(_conn, _entity, _entity_uuid, _operation, _version, payload):
        sync_payloads.append(payload)

    def _update_fts_media(conn, media_id, title, content):
        fts_calls.append((conn, media_id, title, content))

    db = SimpleNamespace(
        client_id="api-client",
        backend=object(),
        transaction=lambda: _Txn(),
        _get_current_utc_timestamp_str=lambda: "2026-03-22T20:00:00Z",
        _fetchone_with_connection=_fetchone,
        _execute_with_connection=_execute,
        create_document_version=_create_document_version,
        _log_sync_event=_log_sync_event,
        _update_fts_media=_update_fts_media,
    )

    result = media_item_update_ops_module.apply_media_item_update(
        db,
        media_id=9,
        fields={"title": "Updated Title", "content": "updated body"},
        prompt="api prompt",
        analysis_content="api analysis",
    )

    update_query, update_params = execute_calls[0]
    assert result == {
        "media_id": 9,
        "content_hash": expected_hash,
        "new_media_version": 2,
        "content_changed": True,
        "document_version_number": 4,
        "document_version_uuid": "dv-uuid-4",
        "invalidate_rag": True,
    }
    assert "content = ?" in update_query
    assert "content_hash = ?" in update_query
    assert "chunking_status = ?" in update_query
    assert "vector_processing = ?" in update_query
    assert update_params == (
        "2026-03-22T20:00:00Z",
        2,
        "api-client",
        "Updated Title",
        "updated body",
        expected_hash,
        "pending",
        0,
        9,
        1,
    )
    assert doc_versions == [
        {
            "media_id": 9,
            "content": "updated body",
            "prompt": "api prompt",
            "analysis_content": "api analysis",
        }
    ]
    assert sync_payloads == [
        {
            **_media_row(
                title="Updated Title",
                content="updated body",
                content_hash=expected_hash,
                version=2,
            ),
            "client_id": "api-client",
            "created_doc_ver_uuid": "dv-uuid-4",
            "created_doc_ver_num": 4,
        }
    ]
    assert fts_calls == [("conn", 9, "Updated Title", "updated body")]
    assert collection_calls == [(9, expected_hash)]


def test_apply_media_item_update_versions_identical_content_without_rechunking_or_fts() -> None:
    media_item_update_ops_module = _load_media_item_update_ops_module()

    existing_hash = hashlib.sha256("existing body".encode()).hexdigest()
    fetch_rows = [
        _media_row(content="existing body", content_hash=existing_hash, version=5),
        _media_row(content="existing body", content_hash=existing_hash, version=6),
    ]
    execute_calls: list[tuple[str, tuple[object, ...]]] = []
    doc_versions: list[dict[str, object]] = []
    fts_calls: list[tuple[object, int, str, str]] = []

    def _fetchone(_conn, _query, _params):
        return fetch_rows.pop(0)

    def _execute(_conn, query, params):
        execute_calls.append((" ".join(query.split()), params))
        return SimpleNamespace(rowcount=1)

    def _create_document_version(**kwargs):
        doc_versions.append(kwargs)
        return {"uuid": "dv-identical", "version_number": 6}

    def _update_fts_media(conn, media_id, title, content):
        fts_calls.append((conn, media_id, title, content))

    db = SimpleNamespace(
        client_id="api-client",
        backend=object(),
        transaction=lambda: _Txn(),
        _get_current_utc_timestamp_str=lambda: "2026-03-22T20:00:00Z",
        _fetchone_with_connection=_fetchone,
        _execute_with_connection=_execute,
        create_document_version=_create_document_version,
        _log_sync_event=lambda *_args, **_kwargs: None,
        _update_fts_media=_update_fts_media,
    )

    result = media_item_update_ops_module.apply_media_item_update(
        db,
        media_id=9,
        fields={"content": "existing body"},
    )

    update_query, update_params = execute_calls[0]
    assert result == {
        "media_id": 9,
        "content_hash": existing_hash,
        "new_media_version": 6,
        "content_changed": False,
        "document_version_number": 6,
        "document_version_uuid": "dv-identical",
        "invalidate_rag": True,
    }
    assert "content = ?" not in update_query
    assert "content_hash = ?" not in update_query
    assert "chunking_status" not in update_query
    assert "vector_processing" not in update_query
    assert update_params == ("2026-03-22T20:00:00Z", 6, "api-client", 9, 5)
    assert doc_versions == [
        {
            "media_id": 9,
            "content": "existing body",
            "prompt": None,
            "analysis_content": None,
        }
    ]
    assert fts_calls == []


def test_apply_media_item_update_wraps_sqlite_errors() -> None:
    media_item_update_ops_module = _load_media_item_update_ops_module()

    db = SimpleNamespace(
        client_id="api-client",
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

    with pytest.raises(DatabaseError, match="Media item update failed"):
        media_item_update_ops_module.apply_media_item_update(
            db,
            media_id=9,
            fields={"title": "Updated"},
        )
