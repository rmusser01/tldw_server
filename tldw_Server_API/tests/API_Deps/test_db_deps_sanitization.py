from pathlib import Path

import pytest
from fastapi import HTTPException, status

from tldw_Server_API.app.api.v1.API_Deps import DB_Deps as deps
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError


class _User:
    id = "user-42"
    id_int = 42


class _SqliteBackend:
    backend_type = BackendType.SQLITE


def _capture_db_deps_logs() -> tuple[list[str], int]:
    messages: list[str] = []
    sink_id = deps.logger.add(
        lambda message: messages.append(str(message.record.get("message") or "")),
        format="{message}",
    )
    return messages, sink_id


def _assert_sensitive_text_not_logged(rendered_logs: str, *sensitive_values: str) -> None:
    for value in sensitive_values:
        assert value not in rendered_logs


def test_content_backend_initialization_failure_log_is_sanitized(monkeypatch):
    secret = "postgresql://user:super-secret-token@localhost/content"
    messages, sink_id = _capture_db_deps_logs()

    def _raise_backend_failure():
        raise RuntimeError(f"could not connect to {secret}")

    monkeypatch.setattr(deps, "get_content_backend_instance", _raise_backend_failure)

    try:
        with pytest.raises(HTTPException) as exc_info:
            deps._get_or_create_media_db_factory(_User())
    finally:
        deps.logger.remove(sink_id)

    assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert (
        exc_info.value.detail
        == "PostgreSQL content backend required but unavailable. Check server logs."
    )

    rendered_logs = "\n".join(messages)
    assert "Content backend initialization failed" in rendered_logs
    _assert_sensitive_text_not_logged(rendered_logs, "could not connect", secret, "super-secret-token")


def test_get_db_path_for_user_directory_resolution_failure_log_is_sanitized(monkeypatch):
    secret_path = "/private/tmp/user-4242/sensitive-media.db"
    secret_message = f"permission denied opening {secret_path}"
    messages, sink_id = _capture_db_deps_logs()

    def _raise_path_failure(_user_id: int):
        raise RuntimeError(secret_message)

    monkeypatch.setattr(deps.DatabasePaths, "get_media_db_path", _raise_path_failure)

    try:
        with pytest.raises(OSError) as exc_info:
            deps._get_db_path_for_user(4242)
    finally:
        deps.logger.remove(sink_id)

    assert str(exc_info.value) == "Could not initialize storage directory for user 4242."

    rendered_logs = "\n".join(messages)
    assert "Could not resolve database directory" in rendered_logs
    _assert_sensitive_text_not_logged(
        rendered_logs,
        "4242",
        secret_path,
        secret_message,
        "permission denied",
    )
    assert "RuntimeError" in rendered_logs


def test_media_db_factory_database_error_log_is_sanitized(monkeypatch):
    secret_path = "/private/tmp/user-42/secret-media.db"
    secret_message = f"sqlite failed for {secret_path}"
    messages, sink_id = _capture_db_deps_logs()
    deps._media_db_factories.clear()

    monkeypatch.setattr(deps, "is_test_mode", lambda: False)
    monkeypatch.setattr(deps, "get_content_backend_instance", lambda: _SqliteBackend())
    monkeypatch.setattr(deps, "_get_db_path_for_user", lambda _user_id: Path(secret_path))

    def _raise_database_error(*_args, **_kwargs):
        raise DatabaseError(secret_message)

    monkeypatch.setattr(deps.MediaDbFactory, "for_sqlite_path", _raise_database_error)

    try:
        with pytest.raises(HTTPException) as exc_info:
            deps._get_or_create_media_db_factory(_User())
    finally:
        deps.logger.remove(sink_id)
        deps._media_db_factories.clear()

    assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert exc_info.value.detail == "Media DB unavailable"

    rendered_logs = "\n".join(messages)
    assert "Failed to initialize database" in rendered_logs
    _assert_sensitive_text_not_logged(
        rendered_logs,
        "user 42",
        "user_id 42",
        secret_path,
        secret_message,
        "sqlite failed",
    )
    assert "DatabaseError" in rendered_logs


def test_media_db_factory_os_error_log_is_sanitized(monkeypatch):
    secret_path = "/private/tmp/user-42/os-media.db"
    secret_message = f"cannot create {secret_path}"
    messages, sink_id = _capture_db_deps_logs()
    deps._media_db_factories.clear()

    monkeypatch.setattr(deps, "is_test_mode", lambda: False)
    monkeypatch.setattr(deps, "get_content_backend_instance", lambda: _SqliteBackend())
    monkeypatch.setattr(deps, "_get_db_path_for_user", lambda _user_id: Path(secret_path))

    def _raise_os_error(*_args, **_kwargs):
        raise OSError(secret_message)

    monkeypatch.setattr(deps.MediaDbFactory, "for_sqlite_path", _raise_os_error)

    try:
        with pytest.raises(HTTPException) as exc_info:
            deps._get_or_create_media_db_factory(_User())
    finally:
        deps.logger.remove(sink_id)
        deps._media_db_factories.clear()

    assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert exc_info.value.detail == "Media DB unavailable"

    rendered_logs = "\n".join(messages)
    assert "Failed to get DB path" in rendered_logs
    _assert_sensitive_text_not_logged(
        rendered_logs,
        "user 42",
        "user_id 42",
        secret_path,
        secret_message,
        "cannot create",
    )
    assert "OSError" in rendered_logs


def test_media_db_factory_unexpected_error_log_is_sanitized(monkeypatch):
    secret_path = "/private/tmp/user-42/unexpected-media.db"
    secret_message = f"driver crashed at {secret_path}"
    messages, sink_id = _capture_db_deps_logs()
    deps._media_db_factories.clear()

    monkeypatch.setattr(deps, "is_test_mode", lambda: False)
    monkeypatch.setattr(deps, "get_content_backend_instance", lambda: _SqliteBackend())
    monkeypatch.setattr(deps, "_get_db_path_for_user", lambda _user_id: Path(secret_path))

    def _raise_unexpected_error(*_args, **_kwargs):
        raise ValueError(secret_message)

    monkeypatch.setattr(deps.MediaDbFactory, "for_sqlite_path", _raise_unexpected_error)

    try:
        with pytest.raises(HTTPException) as exc_info:
            deps._get_or_create_media_db_factory(_User())
    finally:
        deps.logger.remove(sink_id)
        deps._media_db_factories.clear()

    assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert exc_info.value.detail == "An unexpected error occurred during database setup for user."

    rendered_logs = "\n".join(messages)
    assert "Unexpected error initializing database" in rendered_logs
    _assert_sensitive_text_not_logged(
        rendered_logs,
        "user 42",
        "user_id 42",
        secret_path,
        secret_message,
        "driver crashed",
    )
    assert "ValueError" in rendered_logs
