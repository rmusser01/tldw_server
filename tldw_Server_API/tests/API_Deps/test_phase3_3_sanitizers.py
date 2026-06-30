from pathlib import Path

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.API_Deps import DB_Deps as db_deps
from tldw_Server_API.app.api.v1.API_Deps import kanban_deps
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError


pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self):
        self.messages = []

    def debug(self, message, *args, **kwargs):
        self.messages.append(("debug", str(message), args, kwargs))

    def info(self, message, *args, **kwargs):
        self.messages.append(("info", str(message), args, kwargs))

    def warning(self, message, *args, **kwargs):
        self.messages.append(("warning", str(message), args, kwargs))

    def error(self, message, *args, **kwargs):
        self.messages.append(("error", str(message), args, kwargs))


class _User:
    id = "secret-user-42"
    id_int = 42


class _SqliteBackend:
    backend_type = BackendType.SQLITE


def _render_logs(logger_stub: _LoggerStub) -> str:
    return "\n".join(
        " ".join(
            [
                level,
                message,
                repr(args),
                repr(kwargs),
            ]
        )
        for level, message, args, kwargs in logger_stub.messages
    )


def _assert_logs_are_sanitized(logger_stub: _LoggerStub) -> None:
    rendered = _render_logs(logger_stub)
    assert "backend exploded" not in rendered
    assert "/private/" not in rendered
    assert "SECRET_TOKEN" not in rendered
    assert "secret-user-42" not in rendered
    assert "exc_info" not in rendered


@pytest.mark.asyncio
async def test_optional_media_db_http_exception_log_is_sanitized(monkeypatch):
    logger_stub = _LoggerStub()

    def fail_resolve(_current_user):
        raise HTTPException(status_code=500, detail="backend exploded at /private/db/path SECRET_TOKEN")

    monkeypatch.setattr(db_deps, "logger", logger_stub)
    monkeypatch.setattr(db_deps, "_resolve_media_db_for_user", fail_resolve)

    generator = db_deps.try_get_media_db_for_user(request=object(), current_user=_User())
    try:
        assert await anext(generator) is None
    finally:
        await generator.aclose()

    assert any(
        level == "warning" and "Optional Media DB unavailable" in message
        for level, message, _args, _kwargs in logger_stub.messages
    )
    _assert_logs_are_sanitized(logger_stub)


@pytest.mark.asyncio
async def test_optional_media_db_unexpected_error_log_is_sanitized(monkeypatch):
    logger_stub = _LoggerStub()

    def fail_resolve(_current_user):
        raise RuntimeError("backend exploded at /private/db/path SECRET_TOKEN")

    monkeypatch.setattr(db_deps, "logger", logger_stub)
    monkeypatch.setattr(db_deps, "_resolve_media_db_for_user", fail_resolve)

    generator = db_deps.try_get_media_db_for_user(request=object(), current_user=_User())
    try:
        assert await anext(generator) is None
    finally:
        await generator.aclose()

    assert any(
        level == "warning" and "Optional Media DB unexpected error" in message
        for level, message, _args, _kwargs in logger_stub.messages
    )
    rendered = _render_logs(logger_stub)
    assert "RuntimeError" in rendered
    _assert_logs_are_sanitized(logger_stub)


def test_test_mode_db_path_fallback_log_is_sanitized(monkeypatch):
    logger_stub = _LoggerStub()

    class _BrokenSettings:
        def get(self, _key):
            raise RuntimeError("backend exploded at /private/db/path SECRET_TOKEN")

    monkeypatch.delenv("USER_DB_BASE_DIR", raising=False)
    monkeypatch.setenv("TESTING", "1")
    monkeypatch.setattr(db_deps, "logger", logger_stub)
    monkeypatch.setattr(db_deps, "settings", _BrokenSettings())
    monkeypatch.setattr(
        db_deps.DatabasePaths,
        "get_media_db_path",
        staticmethod(lambda _user_id: Path("/tmp/safe-media.db")),
    )

    assert db_deps._get_db_path_for_user(42) == Path("/tmp/safe-media.db")

    assert any(
        level == "warning" and "TESTING mode: failed to derive project-root user DB dir" in message
        for level, message, _args, _kwargs in logger_stub.messages
    )
    rendered = _render_logs(logger_stub)
    assert "RuntimeError" in rendered
    _assert_logs_are_sanitized(logger_stub)


def test_media_db_factory_database_error_log_does_not_attach_traceback_metadata(monkeypatch):
    logger_stub = _LoggerStub()
    db_deps._media_db_factories.clear()

    def fail_sqlite_factory(*_args, **_kwargs):
        raise DatabaseError("backend exploded at /private/db/path SECRET_TOKEN")

    monkeypatch.setattr(db_deps, "logger", logger_stub)
    monkeypatch.setattr(db_deps, "is_test_mode", lambda: False)
    monkeypatch.setattr(db_deps, "get_content_backend_instance", lambda: _SqliteBackend())
    monkeypatch.setattr(db_deps, "_get_db_path_for_user", lambda _user_id: Path("/tmp/safe-media.db"))
    monkeypatch.setattr(db_deps.MediaDbFactory, "for_sqlite_path", fail_sqlite_factory)

    try:
        with pytest.raises(HTTPException):
            db_deps._get_or_create_media_db_factory(_User())
    finally:
        db_deps._media_db_factories.clear()

    assert any(
        level == "error" and "Failed to initialize database" in message
        for level, message, _args, _kwargs in logger_stub.messages
    )
    rendered = _render_logs(logger_stub)
    assert "DatabaseError" in rendered
    _assert_logs_are_sanitized(logger_stub)


def test_kanban_health_last_error_uses_safe_error_type() -> None:
    with kanban_deps._KANBAN_HEALTH_LOCK:
        kanban_deps._KANBAN_HEALTH.update(
            {
                "init_attempts": 0,
                "init_failures": 0,
                "last_init_ms": None,
                "last_error": None,
                "last_success_ts": None,
                "last_failure_ts": None,
                "cached_instances": 0,
            }
        )
        kanban_deps._KANBAN_RECENT_INIT_FAILURES.clear()

    kanban_deps._record_init(
        12.5,
        False,
        RuntimeError("backend exploded at /private/db/path SECRET_TOKEN"),
    )

    snapshot = kanban_deps.get_kanban_health_snapshot()
    assert snapshot["last_error"] == "RuntimeError"
    assert "backend exploded" not in str(snapshot)
    assert "/private/" not in str(snapshot)
    assert "SECRET_TOKEN" not in str(snapshot)
