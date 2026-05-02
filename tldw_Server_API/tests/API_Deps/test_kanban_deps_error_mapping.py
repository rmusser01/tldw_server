import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.API_Deps import kanban_deps


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


def _assert_log_omits_raw_exception(logger_stub, expected_level, expected_message):
    assert any(
        level == expected_level and expected_message in message
        for level, message, _args, _kwargs in logger_stub.messages
    )
    rendered = "\n".join(message for _level, message, _args, _kwargs in logger_stub.messages)
    assert "/private/db/path" not in rendered
    assert "SECRET_TOKEN" not in rendered
    assert "backend exploded" not in rendered


@pytest.fixture(autouse=True)
def clear_kanban_dependency_state():
    with kanban_deps._kanban_db_lock:
        kanban_deps._kanban_db_instances.clear()
        kanban_deps._kanban_db_health_checks.clear()
    yield
    kanban_deps.shutdown_kanban_executor(wait=True)
    with kanban_deps._kanban_db_lock:
        kanban_deps._kanban_db_instances.clear()
        kanban_deps._kanban_db_health_checks.clear()


@pytest.mark.asyncio
async def test_kanban_init_sanitizes_runtime_errors(monkeypatch):
    logger_stub = _LoggerStub()

    def _raise_init_error(_user_id):
        raise RuntimeError("kanban backend exploded at /private/db/path SECRET_TOKEN")

    monkeypatch.setattr(kanban_deps, "logger", logger_stub)
    monkeypatch.setattr(kanban_deps, "_create_kanban_db", _raise_init_error)

    with pytest.raises(HTTPException) as exc_info:
        await kanban_deps._get_or_init_db_instance(123)

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Could not initialize Kanban database for user"
    _assert_log_omits_raw_exception(
        logger_stub,
        "error",
        "Kanban DB initialization failed for user 123",
    )


def test_kanban_health_probe_sanitizes_failure_log(monkeypatch):
    logger_stub = _LoggerStub()

    class _DBInstance:
        db_path = "/private/db/path/kanban.db"

    def _raise_connect_error(*_args, **_kwargs):
        raise RuntimeError("kanban backend exploded at /private/db/path SECRET_TOKEN")

    monkeypatch.setattr(kanban_deps, "logger", logger_stub)
    monkeypatch.setattr(kanban_deps.sqlite3, "connect", _raise_connect_error)

    assert kanban_deps._health_check_instance(_DBInstance()) is False
    _assert_log_omits_raw_exception(
        logger_stub,
        "warning",
        "Kanban health probe failed",
    )


def test_close_all_kanban_db_instances_sanitizes_close_failure_log(monkeypatch):
    logger_stub = _LoggerStub()

    class _DBInstance:
        def close(self):
            raise RuntimeError("kanban backend exploded at /private/db/path SECRET_TOKEN")

    monkeypatch.setattr(kanban_deps, "logger", logger_stub)
    with kanban_deps._kanban_db_lock:
        kanban_deps._kanban_db_instances["kanban::123"] = _DBInstance()

    kanban_deps.close_all_kanban_db_instances()

    _assert_log_omits_raw_exception(
        logger_stub,
        "debug",
        "Error closing KanbanDB instance kanban::123",
    )


def test_shutdown_kanban_executor_sanitizes_shutdown_failure_log(monkeypatch):
    logger_stub = _LoggerStub()

    class _Executor:
        def shutdown(self, **_kwargs):
            raise RuntimeError("kanban backend exploded at /private/db/path SECRET_TOKEN")

    monkeypatch.setattr(kanban_deps, "logger", logger_stub)
    monkeypatch.setattr(kanban_deps, "_KANBAN_EXECUTOR", _Executor())
    monkeypatch.setattr(kanban_deps, "_KANBAN_EXECUTOR_SHUTDOWN", False)

    kanban_deps.shutdown_kanban_executor(wait=False)

    _assert_log_omits_raw_exception(
        logger_stub,
        "debug",
        "Kanban executor shutdown error",
    )


def test_handle_kanban_db_error_sanitizes_unexpected_operation_log(monkeypatch):
    logger_stub = _LoggerStub()

    monkeypatch.setattr(kanban_deps, "logger", logger_stub)

    exc = kanban_deps.handle_kanban_db_error(
        RuntimeError("kanban backend exploded at /private/db/path SECRET_TOKEN")
    )

    assert exc.status_code == 500
    assert exc.detail == "An unexpected error occurred"
    _assert_log_omits_raw_exception(
        logger_stub,
        "error",
        "Unexpected error in Kanban operation",
    )
    assert all(not kwargs.get("exc_info") for _level, _message, _args, kwargs in logger_stub.messages)
