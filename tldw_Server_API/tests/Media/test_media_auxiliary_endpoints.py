import sqlite3
from contextlib import contextmanager

import pytest
from fastapi import FastAPI, HTTPException, Response
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError, InputError

pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self):
        self.debug_calls = []
        self.error_calls = []
        self.info_calls = []
        self.warning_calls = []
        self.exception_calls = []

    def debug(self, *args, **kwargs):
        self.debug_calls.append((args, kwargs))

    def error(self, *args, **kwargs):
        self.error_calls.append((args, kwargs))

    def info(self, *args, **kwargs):
        self.info_calls.append((args, kwargs))

    def warning(self, *args, **kwargs):
        self.warning_calls.append((args, kwargs))

    def exception(self, *args, **kwargs):
        self.exception_calls.append((args, kwargs))


_SENSITIVE_MARKERS = (
    "driver failed",
    "invalid-private-id",
    "/private/tmp/media-listing-keywords.db",
    "progress backend leaked",
    "/private/tmp/reading-progress.db",
    "trash backend leaked",
    "/private/tmp/media-trash.db",
)


def _database_failure() -> DatabaseError:
    return DatabaseError("driver failed /private/tmp/media-listing-keywords.db")


def _reading_progress_failure() -> RuntimeError:
    return RuntimeError("progress backend leaked /private/tmp/reading-progress.db")


def _trash_failure() -> RuntimeError:
    return RuntimeError("trash backend leaked /private/tmp/media-trash.db")


def _patch_listing_logger(monkeypatch: pytest.MonkeyPatch) -> _LoggerStub:
    from tldw_Server_API.app.api.v1.endpoints.media import listing as listing_endpoints

    logger_stub = _LoggerStub()
    monkeypatch.setattr(listing_endpoints, "logger", logger_stub, raising=True)
    return logger_stub


def _patch_reading_progress_logger(monkeypatch: pytest.MonkeyPatch) -> _LoggerStub:
    from tldw_Server_API.app.api.v1.endpoints.media import reading_progress as reading_progress_endpoints

    logger_stub = _LoggerStub()
    monkeypatch.setattr(reading_progress_endpoints, "logger", logger_stub, raising=True)
    return logger_stub


def _assert_sanitized_error_log(
    logger_stub: _LoggerStub,
    expected_message: str,
) -> None:
    assert logger_stub.exception_calls == []
    assert logger_stub.error_calls

    matching_messages = [args[0] for args, _kwargs in logger_stub.error_calls if args]
    assert expected_message in matching_messages
    assert all(not kwargs.get("exc_info") for _args, kwargs in logger_stub.error_calls)

    rendered_calls = repr(logger_stub.error_calls)
    for marker in _SENSITIVE_MARKERS:
        assert marker not in rendered_calls


def _assert_sanitized_warning_log(
    logger_stub: _LoggerStub,
    expected_message: str,
) -> None:
    assert logger_stub.warning_calls

    matching_messages = [args[0] for args, _kwargs in logger_stub.warning_calls if args]
    assert expected_message in matching_messages
    assert all(not kwargs.get("exc_info") for _args, kwargs in logger_stub.warning_calls)

    rendered_calls = repr(logger_stub.warning_calls)
    for marker in _SENSITIVE_MARKERS:
        assert marker not in rendered_calls


class _FakeMediaAuxDb:
    def __init__(self, *, keywords: list[str] | None = None, media_exists: bool = True):
        from tldw_Server_API.app.core.DB_Management.media_db.schema.document_workspace_schema import (
            ensure_sqlite_document_workspace_schema,
        )

        self._keywords = keywords or []
        self._media_exists = media_exists
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        ensure_sqlite_document_workspace_schema(self._conn)

    def fetch_all_keywords(self) -> list[str]:
        return list(self._keywords)

    def get_media_by_id(self, media_id: int, include_deleted: bool = False, include_trash: bool = False):
        if not self._media_exists:
            return None
        return {
            "id": media_id,
            "title": f"Media {media_id}",
            "type": "document",
            "deleted": int(include_deleted),
            "is_trash": int(include_trash),
        }

    @contextmanager
    def transaction(self):
        try:
            yield self._conn
            self._conn.commit()
        finally:
            pass

    def _execute_with_connection(self, conn, query, params=None):
        cursor = conn.cursor()
        cursor.execute(query, params or ())
        return cursor

    def _fetchone_with_connection(self, conn, query, params=None):
        cursor = self._execute_with_connection(conn, query, params)
        row = cursor.fetchone()
        return dict(row) if row is not None else None

    def _fetchall_with_connection(self, conn, query, params=None):
        cursor = self._execute_with_connection(conn, query, params)
        return [dict(row) for row in cursor.fetchall()]

    def close(self) -> None:
        self._conn.close()


class _ErroringMediaAuxDb(_FakeMediaAuxDb):
    def __init__(self, fetch_keywords_exc: Exception):
        super().__init__()
        self._fetch_keywords_exc = fetch_keywords_exc

    def fetch_all_keywords(self) -> list[str]:
        raise self._fetch_keywords_exc


class _SearchByMetadataFailingDb(_FakeMediaAuxDb):
    def search_by_safe_metadata(self, **_kwargs):
        raise _database_failure()


class _SearchByMetadataDb(_FakeMediaAuxDb):
    def search_by_safe_metadata(self, **_kwargs):
        return (
            [
                {
                    "id": 7,
                    "title": "Metadata Search Result",
                    "safe_metadata": '{"doi":"10.1000/example"}',
                }
            ],
            25,
        )


class _FailingReadingProgressDb(_FakeMediaAuxDb):
    def __init__(self, operation_exc: Exception):
        super().__init__()
        self._operation_exc = operation_exc

    @contextmanager
    def transaction(self):
        raise self._operation_exc
        yield


class _FailingTrashDb:
    def execute_query(self, *_args, **_kwargs):
        raise _trash_failure()


@contextmanager
def _build_media_auxiliary_client(db: _FakeMediaAuxDb):
    from tldw_Server_API.app.api.v1.endpoints.media import listing as listing_endpoints
    from tldw_Server_API.app.api.v1.endpoints.media import reading_progress as reading_progress_endpoints

    app = FastAPI()
    app.include_router(listing_endpoints.router, prefix="/api/v1/media", tags=["media"])
    app.include_router(reading_progress_endpoints.router, prefix="/api/v1/media", tags=["media"])
    app.dependency_overrides[listing_endpoints.get_media_db_for_user] = lambda: db
    app.dependency_overrides[listing_endpoints.try_get_media_db_for_user] = lambda: db
    app.dependency_overrides[reading_progress_endpoints.get_media_db_for_user] = lambda: db
    app.dependency_overrides[listing_endpoints.get_request_user] = lambda: type("User", (), {"id": 1})()
    app.dependency_overrides[reading_progress_endpoints.get_request_user] = lambda: type("User", (), {"id": 1})()
    with TestClient(app) as client:
        yield client, db
    db.close()


@pytest.fixture
def media_auxiliary_client():
    with _build_media_auxiliary_client(_FakeMediaAuxDb(keywords=["alpha", "beta", "almanac"])) as client_and_db:
        yield client_and_db


def test_media_keywords_endpoint_returns_filtered_keywords(media_auxiliary_client):
    client, _db = media_auxiliary_client

    response = client.get("/api/v1/media/keywords", params={"query": "al"})

    assert response.status_code == 200, response.text  # nosec B101
    assert response.json() == {"keywords": ["alpha", "almanac"]}  # nosec B101


def test_media_keywords_maps_input_error_to_400():
    with _build_media_auxiliary_client(_ErroringMediaAuxDb(InputError("invalid keyword filter"))) as (client, _db):
        response = client.get("/api/v1/media/keywords")

    assert response.status_code == 400, response.text  # nosec B101
    assert response.json() == {"detail": "invalid keyword filter"}  # nosec B101


def test_media_keywords_preserves_database_error_detail(monkeypatch):
    logger_stub = _patch_listing_logger(monkeypatch)

    with _build_media_auxiliary_client(_ErroringMediaAuxDb(_database_failure())) as (client, _db):
        response = client.get("/api/v1/media/keywords")

    assert response.status_code == 500, response.text  # nosec B101
    assert response.json() == {"detail": "Failed to load media keywords"}  # nosec B101
    _assert_sanitized_error_log(logger_stub, "Failed to list media keywords")


def test_list_media_sanitizes_outer_failure_log(monkeypatch):
    logger_stub = _patch_listing_logger(monkeypatch)
    from tldw_Server_API.app.api.v1.endpoints.media import listing as listing_endpoints

    def _raise_paginated_files(*_args, **_kwargs):
        raise _database_failure()

    monkeypatch.setattr(listing_endpoints, "get_paginated_files", _raise_paginated_files)

    with _build_media_auxiliary_client(_FakeMediaAuxDb()) as (client, _db):
        response = client.get("/api/v1/media/")

    assert response.status_code == 500, response.text  # nosec B101
    assert response.json() == {"detail": "Failed to list media"}  # nosec B101
    _assert_sanitized_error_log(logger_stub, "Error listing media")


def test_list_media_invalid_row_id_log_is_sanitized(monkeypatch):
    logger_stub = _patch_listing_logger(monkeypatch)
    from tldw_Server_API.app.api.v1.endpoints.media import listing as listing_endpoints

    def _fake_paginated_files(*_args, **_kwargs):
        return (
            [
                {
                    "id": "invalid-private-id /private/tmp/media-listing-keywords.db",
                    "title": "Bad Row",
                    "type": "document",
                },
                {"id": 2, "title": "Good Row", "type": "document"},
            ],
            1,
            1,
            2,
        )

    monkeypatch.setattr(listing_endpoints, "get_paginated_files", _fake_paginated_files)

    with _build_media_auxiliary_client(_FakeMediaAuxDb()) as (client, _db):
        response = client.get("/api/v1/media/")

    assert response.status_code == 200, response.text  # nosec B101
    payload = response.json()
    assert payload["items"] == [
        {
            "id": 2,
            "title": "Good Row",
            "type": "document",
            "url": "/api/v1/media/2",
        }
    ]  # nosec B101
    assert payload["skipped_count"] == 1  # nosec B101
    _assert_sanitized_error_log(logger_stub, "Skipping media row with invalid id")


def test_list_media_exposes_source_picker_metadata(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints.media import listing as listing_endpoints

    def _fake_paginated_files(*_args, **_kwargs):
        return (
            [
                {
                    "id": 7,
                    "title": "Workspace QA Draft",
                    "type": "document",
                    "ingestion_date": "2026-05-14T10:00:00",
                    "last_modified": "2026-05-14T10:05:00",
                    "chunking_status": "processed",
                    "safe_metadata": (
                        '{"workspace_id":"workspace-qa",'
                        '"workspace_name":"QA Workspace",'
                        '"workspace_artifact":true,'
                        '"is_generated":true,'
                        '"test_artifact":true,'
                        '"artifact_kind":"workspace_artifact",'
                        '"private_notes":"do not leak"}'
                    ),
                },
            ],
            1,
            1,
            1,
        )

    monkeypatch.setattr(listing_endpoints, "get_paginated_files", _fake_paginated_files)

    with _build_media_auxiliary_client(_FakeMediaAuxDb()) as (client, _db):
        response = client.get("/api/v1/media/")

    assert response.status_code == 200, response.text  # nosec B101
    item = response.json()["items"][0]
    assert item == {
        "id": 7,
        "title": "Workspace QA Draft",
        "type": "document",
        "url": "/api/v1/media/7",
        "status": "processed",
        "created_at": "2026-05-14T10:00:00",
        "updated_at": "2026-05-14T10:05:00",
        "workspace_id": "workspace-qa",
        "workspace_name": "QA Workspace",
        "workspace_artifact": True,
        "is_generated": True,
        "test_artifact": True,
        "artifact_kind": "workspace_artifact",
    }  # nosec B101


def test_search_media_exposes_source_picker_metadata(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints.media import listing as listing_endpoints

    def _fake_search_media(*_args, **_kwargs):
        return (
            [
                {
                    "id": 11,
                    "title": "Generated Search Result",
                    "type": "document",
                    "ingestion_date": "2026-05-14T11:00:00",
                    "last_modified": "2026-05-14T11:05:00",
                    "chunking_status": "processed",
                    "safe_metadata": {
                        "workspace_id": "workspace-search",
                        "workspace_name": "Search Workspace",
                        "workspace_artifact": True,
                        "is_generated": True,
                        "test_artifact": True,
                        "artifact_kind": "workspace_artifact",
                        "private_notes": "do not leak",
                    },
                },
            ],
            1,
        )

    monkeypatch.setattr(listing_endpoints, "search_media", _fake_search_media)

    with _build_media_auxiliary_client(_FakeMediaAuxDb()) as (client, _db):
        response = client.post("/api/v1/media/search", json={"query": "generated", "fields": ["title"]})

    assert response.status_code == 200, response.text  # nosec B101
    item = response.json()["items"][0]
    assert item == {
        "id": 11,
        "title": "Generated Search Result",
        "type": "document",
        "url": "/api/v1/media/11",
        "status": "processed",
        "created_at": "2026-05-14T11:00:00",
        "updated_at": "2026-05-14T11:05:00",
        "workspace_id": "workspace-search",
        "workspace_name": "Search Workspace",
        "workspace_artifact": True,
        "is_generated": True,
        "test_artifact": True,
        "artifact_kind": "workspace_artifact",
    }  # nosec B101


def test_list_media_trash_sanitizes_outer_failure_log(monkeypatch):
    logger_stub = _patch_listing_logger(monkeypatch)
    from tldw_Server_API.app.api.v1.endpoints.media import listing as listing_endpoints

    def _raise_paginated_trash_files(*_args, **_kwargs):
        raise _database_failure()

    monkeypatch.setattr(listing_endpoints, "get_paginated_trash_files", _raise_paginated_trash_files)

    with _build_media_auxiliary_client(_FakeMediaAuxDb()) as (client, _db):
        response = client.get("/api/v1/media/trash")

    assert response.status_code == 500, response.text  # nosec B101
    assert response.json() == {"detail": "Failed to list trashed media"}  # nosec B101
    _assert_sanitized_error_log(logger_stub, "Error listing trashed media")


def test_list_media_trash_invalid_row_id_log_is_sanitized(monkeypatch):
    logger_stub = _patch_listing_logger(monkeypatch)
    from tldw_Server_API.app.api.v1.endpoints.media import listing as listing_endpoints

    def _fake_paginated_trash_files(*_args, **_kwargs):
        return (
            [
                {
                    "id": "invalid-private-id /private/tmp/media-listing-keywords.db",
                    "title": "Bad Trash Row",
                    "type": "document",
                },
                {"id": 3, "title": "Good Trash Row", "type": "document"},
            ],
            1,
            1,
            2,
        )

    monkeypatch.setattr(listing_endpoints, "get_paginated_trash_files", _fake_paginated_trash_files)

    with _build_media_auxiliary_client(_FakeMediaAuxDb()) as (client, _db):
        response = client.get("/api/v1/media/trash")

    assert response.status_code == 200, response.text  # nosec B101
    payload = response.json()
    assert payload["items"] == [
        {
            "id": 3,
            "title": "Good Trash Row",
            "type": "document",
            "url": "/api/v1/media/3",
        }
    ]  # nosec B101
    assert payload["skipped_count"] == 1  # nosec B101
    _assert_sanitized_error_log(logger_stub, "Skipping trashed media row with invalid id")


def test_metadata_search_sanitizes_outer_failure_log(monkeypatch):
    logger_stub = _patch_listing_logger(monkeypatch)

    with _build_media_auxiliary_client(_SearchByMetadataFailingDb()) as (client, _db):
        response = client.get(
            "/api/v1/media/metadata-search",
            params={"field": "title", "value": "paper"},
        )

    assert response.status_code == 500, response.text  # nosec B101
    assert response.json() == {"detail": "Error performing metadata search"}  # nosec B101
    _assert_sanitized_error_log(logger_stub, "Metadata search error")


def test_metadata_search_includes_canonical_page_pagination():
    with _build_media_auxiliary_client(_SearchByMetadataDb()) as (client, _db):
        response = client.get(
            "/api/v1/media/metadata-search",
            params={"field": "title", "value": "Metadata", "page": 2, "per_page": 10},
        )

    assert response.status_code == 200, response.text  # nosec B101
    payload = response.json()
    assert payload["results"][0]["safe_metadata"] == {"doi": "10.1000/example"}  # nosec B101
    assert payload["pagination"] == {  # nosec B101
        "mode": "page",
        "page": 2,
        "per_page": 10,
        "total": 25,
        "total_pages": 3,
        "has_more": True,
    }


def test_identifier_lookup_sanitizes_outer_failure_log(monkeypatch):
    logger_stub = _patch_listing_logger(monkeypatch)

    with _build_media_auxiliary_client(_SearchByMetadataFailingDb()) as (client, _db):
        response = client.get(
            "/api/v1/media/by-identifier",
            params={"s2_paper_id": "S2-abc123"},
        )

    assert response.status_code == 500, response.text  # nosec B101
    assert response.json() == {"detail": "Error in identifier lookup"}  # nosec B101
    _assert_sanitized_error_log(logger_stub, "Identifier lookup error")


def test_reading_progress_returns_no_progress_payload_instead_of_500(media_auxiliary_client):
    client, _db = media_auxiliary_client

    response = client.get("/api/v1/media/42/progress")

    assert response.status_code == 200, response.text  # nosec B101
    assert response.json() == {"media_id": 42, "has_progress": False}  # nosec B101


def test_reading_progress_endpoint_delegates_to_repository(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints.media import reading_progress as reading_progress_endpoints

    repo_calls = []

    class FakeRepository:
        def get_reading_progress(self, *, media_id, user_id):
            repo_calls.append(("get_reading_progress", media_id, user_id))
            return {
                "media_id": media_id,
                "user_id": user_id,
                "current_page": 3,
                "total_pages": 10,
                "zoom_level": 125,
                "view_mode": "continuous",
                "cfi": "epubcfi(/6/2)",
                "percentage": 30.0,
                "last_read_at": "2026-01-01T00:00:00+00:00",
            }

    class FakeRepositoryFactory:
        @staticmethod
        def from_media_db(db):
            repo_calls.append(("from_media_db", db))
            return FakeRepository()

    monkeypatch.setattr(
        reading_progress_endpoints,
        "DocumentWorkspaceRepository",
        FakeRepositoryFactory,
        raising=False,
    )

    with _build_media_auxiliary_client(_FakeMediaAuxDb()) as (client, db):
        response = client.get("/api/v1/media/42/progress")

    assert response.status_code == 200, response.text  # nosec B101
    payload = response.json()
    assert payload["media_id"] == 42  # nosec B101
    assert payload["current_page"] == 3  # nosec B101
    assert payload["percent_complete"] == 30.0  # nosec B101
    assert repo_calls == [("from_media_db", db), ("get_reading_progress", 42, "1")]  # nosec B101


def test_reading_progress_treats_corrupt_rows_as_missing_progress(media_auxiliary_client):
    client, db = media_auxiliary_client

    with db.transaction() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS document_reading_progress (
                media_id INTEGER NOT NULL,
                user_id TEXT NOT NULL,
                current_page INTEGER NOT NULL DEFAULT 1,
                total_pages INTEGER NOT NULL DEFAULT 1,
                zoom_level INTEGER NOT NULL DEFAULT 100,
                view_mode TEXT NOT NULL DEFAULT 'single',
                cfi TEXT,
                percentage REAL,
                last_read_at TEXT NOT NULL,
                PRIMARY KEY (media_id, user_id)
            )
            """
        )
        conn.execute(
            """
            INSERT INTO document_reading_progress
            (media_id, user_id, current_page, total_pages, zoom_level, view_mode, cfi, percentage, last_read_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (99, "1", 3, 10, 100, "broken-mode", None, None, "not-a-timestamp"),
        )

    response = client.get("/api/v1/media/99/progress")

    assert response.status_code == 200, response.text  # nosec B101
    assert response.json() == {"media_id": 99, "has_progress": False}  # nosec B101


def test_reading_progress_sanitizes_corrupt_row_warning(monkeypatch):
    logger_stub = _patch_reading_progress_logger(monkeypatch)

    with _build_media_auxiliary_client(_FakeMediaAuxDb()) as (client, db):
        with db.transaction() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS document_reading_progress (
                    media_id INTEGER NOT NULL,
                    user_id TEXT NOT NULL,
                    current_page INTEGER NOT NULL DEFAULT 1,
                    total_pages INTEGER NOT NULL DEFAULT 1,
                    zoom_level INTEGER NOT NULL DEFAULT 100,
                    view_mode TEXT NOT NULL DEFAULT 'single',
                    cfi TEXT,
                    percentage REAL,
                    last_read_at TEXT NOT NULL,
                    PRIMARY KEY (media_id, user_id)
                )
                """
            )
            conn.execute(
                """
                INSERT INTO document_reading_progress
                (media_id, user_id, current_page, total_pages, zoom_level, view_mode, cfi, percentage, last_read_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (99, "1", 3, 10, 100, "broken-mode", None, None, "not-a-timestamp"),
            )

        response = client.get("/api/v1/media/99/progress")

    assert response.status_code == 200, response.text  # nosec B101
    assert response.json() == {"media_id": 99, "has_progress": False}  # nosec B101
    _assert_sanitized_warning_log(
        logger_stub,
        "Ignoring corrupt reading progress row",
    )


def test_get_reading_progress_sanitizes_fetch_error_log(monkeypatch):
    logger_stub = _patch_reading_progress_logger(monkeypatch)

    with _build_media_auxiliary_client(_FailingReadingProgressDb(_reading_progress_failure())) as (client, _db):
        response = client.get("/api/v1/media/42/progress")

    assert response.status_code == 500, response.text  # nosec B101
    assert response.json() == {"detail": "Failed to fetch reading progress"}  # nosec B101
    _assert_sanitized_error_log(logger_stub, "Error fetching reading progress")


def test_update_reading_progress_sanitizes_update_error_log(monkeypatch):
    logger_stub = _patch_reading_progress_logger(monkeypatch)

    with _build_media_auxiliary_client(_FailingReadingProgressDb(_reading_progress_failure())) as (client, _db):
        response = client.put(
            "/api/v1/media/42/progress",
            json={
                "current_page": 2,
                "total_pages": 10,
                "zoom_level": 125,
                "view_mode": "continuous",
            },
        )

    assert response.status_code == 500, response.text  # nosec B101
    assert response.json() == {"detail": "Failed to update reading progress"}  # nosec B101
    _assert_sanitized_error_log(logger_stub, "Error updating reading progress")


def test_delete_reading_progress_sanitizes_delete_error_log(monkeypatch):
    logger_stub = _patch_reading_progress_logger(monkeypatch)

    with _build_media_auxiliary_client(_FailingReadingProgressDb(_reading_progress_failure())) as (client, _db):
        response = client.delete("/api/v1/media/42/progress")

    assert response.status_code == 500, response.text  # nosec B101
    assert response.json() == {"detail": "Failed to delete reading progress"}  # nosec B101
    _assert_sanitized_error_log(logger_stub, "Error deleting reading progress")


async def test_empty_media_trash_sanitizes_outer_failure_log(monkeypatch):
    logger_stub = _patch_listing_logger(monkeypatch)
    from tldw_Server_API.app.api.v1.endpoints.media import listing as listing_endpoints

    with pytest.raises(HTTPException) as exc_info:
        await listing_endpoints.empty_media_trash_endpoint(
            response=Response(),
            db=_FailingTrashDb(),
            current_user=type("User", (), {"id": 1})(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to empty media trash"
    _assert_sanitized_error_log(logger_stub, "Error emptying media trash")
