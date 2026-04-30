from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.API_Deps import Collections_DB_Deps as collections_deps
from tldw_Server_API.app.api.v1.API_Deps import DB_Deps as media_deps
from tldw_Server_API.app.api.v1.API_Deps import Watchlists_DB_Deps as watchlists_deps
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.DB_Management.backends.base import (
    BackendType,
    DatabaseError as BackendDatabaseError,
)
from tldw_Server_API.app.core.DB_Management.media_db.errors import (
    DatabaseError as MediaDatabaseError,
    SchemaError as MediaSchemaError,
)


def _user() -> User:
    return User(id=24, username="content-user")


@contextmanager
def _capture_error_messages(module_logger):
    messages: list[str] = []
    sink_id = module_logger.add(
        lambda message: messages.append(str(message.record.get("message") or "")),
        level="ERROR",
        format="{message}",
    )
    try:
        yield messages
    finally:
        module_logger.remove(sink_id)


@contextmanager
def _capture_debug_records(module_logger):
    rendered_records: list[str] = []

    def capture(message):
        rendered_records.append(
            "\n".join(
                (
                    str(message.record.get("message") or ""),
                    str(message.record.get("extra") or ""),
                    str(message.record.get("exception") or ""),
                    str(message),
                )
            )
        )

    sink_id = module_logger.add(capture, level="DEBUG", format="{message}")
    try:
        yield rendered_records
    finally:
        module_logger.remove(sink_id)


@pytest.mark.asyncio
async def test_get_collections_db_maps_backend_database_error(monkeypatch):
    class FailingCollectionsDatabase:
        @staticmethod
        def for_user(user_id):
            raise BackendDatabaseError("backend exploded")

    monkeypatch.setattr(collections_deps, "CollectionsDatabase", FailingCollectionsDatabase)

    with pytest.raises(HTTPException) as exc_info:
        await collections_deps.get_collections_db_for_user(current_user=_user())

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Collections DB unavailable"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exception_factory", "expected_type"),
    (
        (lambda message: BackendDatabaseError(message), "DatabaseError"),
        (lambda message: RuntimeError(message), "RuntimeError"),
    ),
)
async def test_get_collections_db_logs_safe_initialization_failure(
    monkeypatch,
    exception_factory,
    expected_type,
):
    sensitive_message = (
        "sqlite failed at /Users/alice/private/collections.db "
        "with password=super-secret"
    )

    class FailingCollectionsDatabase:
        @staticmethod
        def for_user(user_id):
            raise exception_factory(sensitive_message)

    monkeypatch.setattr(collections_deps, "CollectionsDatabase", FailingCollectionsDatabase)

    with _capture_error_messages(collections_deps.logger) as messages:
        with pytest.raises(HTTPException) as exc_info:
            await collections_deps.get_collections_db_for_user(current_user=_user())

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Collections DB unavailable"

    rendered_logs = "\n".join(messages)
    assert f"error_type={expected_type}" in rendered_logs
    assert "sqlite failed" not in rendered_logs
    assert "/Users/alice/private/collections.db" not in rendered_logs
    assert "super-secret" not in rendered_logs


@pytest.mark.asyncio
async def test_get_watchlists_db_maps_backend_database_error(monkeypatch):
    class FailingWatchlistsDatabase:
        @staticmethod
        def for_user(user_id):
            raise BackendDatabaseError("backend exploded")

    monkeypatch.setattr(watchlists_deps, "WatchlistsDatabase", FailingWatchlistsDatabase)

    with pytest.raises(HTTPException) as exc_info:
        await watchlists_deps.get_watchlists_db_for_user(current_user=_user())

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Watchlists DB unavailable"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exception_factory", "expected_type"),
    (
        (lambda message: BackendDatabaseError(message), "DatabaseError"),
        (lambda message: RuntimeError(message), "RuntimeError"),
    ),
)
async def test_get_watchlists_db_logs_safe_initialization_failure(
    monkeypatch,
    exception_factory,
    expected_type,
):
    sensitive_message = (
        "sqlite failed at /Users/alice/private/watchlists.db "
        "with password=super-secret"
    )

    class FailingWatchlistsDatabase:
        @staticmethod
        def for_user(user_id):
            raise exception_factory(sensitive_message)

    monkeypatch.setattr(watchlists_deps, "WatchlistsDatabase", FailingWatchlistsDatabase)

    with _capture_error_messages(watchlists_deps.logger) as messages:
        with pytest.raises(HTTPException) as exc_info:
            await watchlists_deps.get_watchlists_db_for_user(current_user=_user())

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Watchlists DB unavailable"

    rendered_logs = "\n".join(messages)
    assert f"error_type={expected_type}" in rendered_logs
    assert "sqlite failed" not in rendered_logs
    assert "/Users/alice/private/watchlists.db" not in rendered_logs
    assert "super-secret" not in rendered_logs


@pytest.mark.asyncio
async def test_get_watchlists_db_schema_ensure_fallback_log_is_sanitized(monkeypatch):
    sensitive_message = (
        "schema failed at /Users/alice/private/watchlists.db "
        "with token=super-secret"
    )

    class WatchlistsDbWithSchemaFailure:
        def ensure_schema(self):
            raise RuntimeError(sensitive_message)

    returned_db = WatchlistsDbWithSchemaFailure()

    class WatchlistsDatabaseFactory:
        @staticmethod
        def for_user(user_id):
            return returned_db

    monkeypatch.setattr(watchlists_deps, "WatchlistsDatabase", WatchlistsDatabaseFactory)

    with _capture_debug_records(watchlists_deps.logger) as records:
        db = await watchlists_deps.get_watchlists_db_for_user(current_user=_user())

    assert db is returned_db
    rendered_logs = "\n".join(records)
    assert "Watchlists DB schema ensure failed in dependency setup" in rendered_logs
    assert "schema failed" not in rendered_logs
    assert "/Users/alice/private/watchlists.db" not in rendered_logs
    assert "super-secret" not in rendered_logs
    assert "Traceback" not in rendered_logs
    assert "exc_info" not in rendered_logs


class _FakeSqliteBackend:
    backend_type = BackendType.SQLITE


def _patch_media_factory_failure(monkeypatch, exc: Exception) -> None:
    media_deps.reset_media_db_cache()
    monkeypatch.setenv("CONTENT_DB_MODE", "sqlite")
    monkeypatch.setattr(
        media_deps,
        "get_content_backend_instance",
        lambda: _FakeSqliteBackend(),
    )
    monkeypatch.setattr(
        media_deps,
        "_get_db_path_for_user",
        lambda user_id: Path(f"/tmp/media-{user_id}.db"),
    )

    def fail_for_sqlite_path(*args, **kwargs):
        raise exc

    monkeypatch.setattr(
        media_deps.MediaDbFactory,
        "for_sqlite_path",
        staticmethod(fail_for_sqlite_path),
    )


def test_get_media_db_factory_maps_schema_error(monkeypatch):
    _patch_media_factory_failure(monkeypatch, MediaSchemaError("schema exploded"))

    with pytest.raises(HTTPException) as exc_info:
        media_deps._get_or_create_media_db_factory(_user())

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Database schema error"


def test_get_media_db_factory_maps_database_error(monkeypatch):
    _patch_media_factory_failure(monkeypatch, MediaDatabaseError("backend exploded"))

    with pytest.raises(HTTPException) as exc_info:
        media_deps._get_or_create_media_db_factory(_user())

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Media DB unavailable"


def test_get_media_db_factory_sanitizes_oserror(monkeypatch):
    _patch_media_factory_failure(monkeypatch, OSError("media path exploded"))

    with pytest.raises(HTTPException) as exc_info:
        media_deps._get_or_create_media_db_factory(_user())

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Media DB unavailable"
