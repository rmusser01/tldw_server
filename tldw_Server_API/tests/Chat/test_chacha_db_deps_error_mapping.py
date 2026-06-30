from pathlib import Path
import threading

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.API_Deps import ChaCha_Notes_DB_Deps as deps
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDBError,
    ConflictError,
    InputError,
    SchemaError,
)
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType


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
    rendered = "\n".join(
        " ".join(
            [
                message,
                " ".join(str(arg) for arg in args),
                " ".join(f"{key}={value}" for key, value in kwargs.items()),
            ]
        )
        for _level, message, args, kwargs in logger_stub.messages
    )
    assert "/private/db/path" not in rendered
    assert "SECRET_TOKEN" not in rendered
    assert "backend exploded" not in rendered
    assert not any(kwargs.get("exc_info") is True for _level, _message, _args, kwargs in logger_stub.messages)


def _patch_chacha_init_failure(monkeypatch, tmp_path: Path, exc: Exception) -> None:
    deps.close_all_chacha_db_instances()
    deps.reset_chacha_shutdown_state()
    monkeypatch.setattr(
        deps.DatabasePaths,
        "get_user_base_directory",
        lambda user_id: tmp_path / str(user_id),
    )

    def fail_create(*args, **kwargs):
        raise exc

    monkeypatch.setattr(deps, "_create_and_prepare_db", fail_create)


def _seed_existing_chacha_init_failure(monkeypatch, tmp_path: Path, user_id: int, exc: Exception) -> None:
    deps.close_all_chacha_db_instances()
    deps.reset_chacha_shutdown_state()
    user_dir = tmp_path / str(user_id)
    monkeypatch.setattr(
        deps.DatabasePaths,
        "get_user_base_directory",
        lambda requested_user_id: tmp_path / str(requested_user_id),
    )
    init_event = threading.Event()
    init_event.set()
    with deps._chacha_db_lock:
        deps._chacha_db_init_events[str(user_dir)] = init_event
        deps._chacha_db_init_errors[str(user_dir)] = exc


@pytest.mark.asyncio
async def test_get_or_init_chacha_db_maps_schema_error(monkeypatch, tmp_path):
    _patch_chacha_init_failure(monkeypatch, tmp_path, SchemaError("schema exploded"))

    try:
        with pytest.raises(HTTPException) as exc_info:
            await deps._get_or_init_db_instance(61, "61")
    finally:
        deps.close_all_chacha_db_instances()

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Database schema error"


@pytest.mark.asyncio
async def test_get_or_init_chacha_db_maps_base_database_error(monkeypatch, tmp_path):
    _patch_chacha_init_failure(monkeypatch, tmp_path, CharactersRAGDBError("backend exploded"))

    try:
        with pytest.raises(HTTPException) as exc_info:
            await deps._get_or_init_db_instance(62, "62")
    finally:
        deps.close_all_chacha_db_instances()

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "ChaChaNotes DB unavailable"


@pytest.mark.asyncio
async def test_get_or_init_chacha_db_keeps_conflict_init_errors_as_500(monkeypatch, tmp_path):
    _patch_chacha_init_failure(monkeypatch, tmp_path, ConflictError("duplicate bootstrap state"))

    try:
        with pytest.raises(HTTPException) as exc_info:
            await deps._get_or_init_db_instance(63, "63")
    finally:
        deps.close_all_chacha_db_instances()

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "ChaChaNotes DB unavailable"


@pytest.mark.asyncio
async def test_get_or_init_chacha_db_keeps_input_init_errors_as_500(monkeypatch, tmp_path):
    _patch_chacha_init_failure(monkeypatch, tmp_path, InputError("invalid bootstrap state"))

    try:
        with pytest.raises(HTTPException) as exc_info:
            await deps._get_or_init_db_instance(67, "67")
    finally:
        deps.close_all_chacha_db_instances()

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "invalid bootstrap state"


@pytest.mark.asyncio
async def test_get_or_init_chacha_db_waiter_maps_schema_error(monkeypatch, tmp_path):
    _seed_existing_chacha_init_failure(monkeypatch, tmp_path, 64, SchemaError("schema exploded"))

    try:
        with pytest.raises(HTTPException) as exc_info:
            await deps._get_or_init_db_instance(64, "64")
    finally:
        deps.close_all_chacha_db_instances()

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Database schema error"


@pytest.mark.asyncio
async def test_get_or_init_chacha_db_waiter_maps_base_database_error(monkeypatch, tmp_path):
    _seed_existing_chacha_init_failure(monkeypatch, tmp_path, 65, CharactersRAGDBError("backend exploded"))

    try:
        with pytest.raises(HTTPException) as exc_info:
            await deps._get_or_init_db_instance(65, "65")
    finally:
        deps.close_all_chacha_db_instances()

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "ChaChaNotes DB unavailable"


@pytest.mark.asyncio
async def test_get_or_init_chacha_db_waiter_keeps_conflict_init_errors_as_500(monkeypatch, tmp_path):
    _seed_existing_chacha_init_failure(monkeypatch, tmp_path, 66, ConflictError("duplicate bootstrap state"))

    try:
        with pytest.raises(HTTPException) as exc_info:
            await deps._get_or_init_db_instance(66, "66")
    finally:
        deps.close_all_chacha_db_instances()

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "ChaChaNotes DB unavailable"


@pytest.mark.asyncio
async def test_get_or_init_chacha_db_waiter_keeps_input_init_errors_as_500(monkeypatch, tmp_path):
    _seed_existing_chacha_init_failure(monkeypatch, tmp_path, 68, InputError("invalid bootstrap state"))

    try:
        with pytest.raises(HTTPException) as exc_info:
            await deps._get_or_init_db_instance(68, "68")
    finally:
        deps.close_all_chacha_db_instances()

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "invalid bootstrap state"


def test_chacha_tuning_sanitizes_fail_open_log(monkeypatch):
    logger_stub = _LoggerStub()

    class _DBInstance:
        backend_type = BackendType.SQLITE

        def get_connection(self):
            raise RuntimeError("chacha backend exploded at /private/db/path SECRET_TOKEN")

    monkeypatch.setattr(deps, "logger", logger_stub)

    deps._apply_sqlite_tuning(_DBInstance())

    _assert_log_omits_raw_exception(
        logger_stub,
        "debug",
        "ChaChaNotes tuning skipped",
    )


def test_chacha_health_probe_sanitizes_failure_log(monkeypatch):
    logger_stub = _LoggerStub()

    class _DBInstance:
        def get_connection(self):
            raise RuntimeError("chacha backend exploded at /private/db/path SECRET_TOKEN")

    monkeypatch.setattr(deps, "logger", logger_stub)

    assert deps._health_check_instance(_DBInstance()) is False
    _assert_log_omits_raw_exception(
        logger_stub,
        "warning",
        "ChaChaNotes health probe failed",
    )


def test_close_all_chacha_db_instances_sanitizes_close_failure_log(monkeypatch):
    logger_stub = _LoggerStub()

    class _DBInstance:
        def close_all_connections(self):
            raise RuntimeError("chacha backend exploded at /private/db/path SECRET_TOKEN")

    monkeypatch.setattr(deps, "logger", logger_stub)
    with deps._chacha_db_lock:
        deps._chacha_db_instances["/private/db/path/chacha.db"] = _DBInstance()

    deps.close_all_chacha_db_instances()

    _assert_log_omits_raw_exception(
        logger_stub,
        "error",
        "Error closing ChaChaNotesDB instance",
    )


def test_shutdown_chacha_executor_sanitizes_shutdown_failure_log(monkeypatch):
    logger_stub = _LoggerStub()

    class _Executor:
        def shutdown(self, **_kwargs):
            raise RuntimeError("chacha backend exploded at /private/db/path SECRET_TOKEN")

    monkeypatch.setattr(deps, "logger", logger_stub)
    monkeypatch.setattr(deps, "_CHACHA_EXECUTOR", _Executor())
    monkeypatch.setattr(deps, "_CHACHA_EXECUTOR_SHUTDOWN", False)

    deps.shutdown_chacha_executor(wait=False)

    _assert_log_omits_raw_exception(
        logger_stub,
        "debug",
        "ChaChaNotes executor shutdown error",
    )


@pytest.mark.asyncio
async def test_warm_chacha_db_for_user_sanitizes_fail_open_log(monkeypatch):
    logger_stub = _LoggerStub()

    async def fail_init(*args, **kwargs):
        raise RuntimeError("chacha backend exploded at /private/db/path SECRET_TOKEN")

    monkeypatch.setattr(deps, "logger", logger_stub)
    monkeypatch.setattr(deps, "_get_or_init_db_instance", fail_init)

    await deps.warm_chacha_db_for_user(4242)

    _assert_log_omits_raw_exception(
        logger_stub,
        "warning",
        "Warm-up for ChaChaNotes failed",
    )
    rendered = "\n".join(message for _level, message, _args, _kwargs in logger_stub.messages)
    assert "4242" not in rendered


def test_maybe_dump_traceback_sanitizes_dump_failure_log(monkeypatch):
    logger_stub = _LoggerStub()

    def fail_dump_traceback(*_args, **_kwargs):
        raise RuntimeError("chacha backend exploded at /private/db/path SECRET_TOKEN")

    monkeypatch.setattr(deps, "logger", logger_stub)
    monkeypatch.setattr(deps.faulthandler, "dump_traceback", fail_dump_traceback)
    monkeypatch.setitem(deps._CHACHA_HEALTH, "last_warn_dump", None)

    deps._maybe_dump_traceback("watchdog test")

    _assert_log_omits_raw_exception(
        logger_stub,
        "debug",
        "Faulthandler dump failed",
    )


def test_create_and_prepare_db_sanitizes_secondary_mkdir_failure_log(monkeypatch, tmp_path):
    logger_stub = _LoggerStub()
    safe_db_path = tmp_path / "safe-chacha" / "ChaChaNotes.db"
    original_mkdir = deps.Path.mkdir

    class _DBInstance:
        pass

    def fail_parent_mkdir(self, *args, **kwargs):
        if self == safe_db_path.parent:
            raise OSError("chacha backend exploded at /private/db/path SECRET_TOKEN")
        return original_mkdir(self, *args, **kwargs)

    def make_db(*, db_path, client_id):
        assert db_path == str(safe_db_path)
        assert client_id == "safe-client"
        return _DBInstance()

    monkeypatch.setattr(deps, "logger", logger_stub)
    monkeypatch.setattr(deps, "_get_chacha_db_path_for_user", lambda _user_id: safe_db_path)
    monkeypatch.setattr(deps.Path, "mkdir", fail_parent_mkdir)
    monkeypatch.setattr(deps, "CharactersRAGDB", make_db)
    monkeypatch.setattr(deps, "_apply_sqlite_tuning", lambda _db_instance: None)

    assert isinstance(deps._create_and_prepare_db(4242, "safe-client"), _DBInstance)

    _assert_log_omits_raw_exception(
        logger_stub,
        "debug",
        "Secondary ensure for ChaChaNotes parent failed softly",
    )


@pytest.mark.asyncio
async def test_ensure_default_character_async_sanitizes_failure_log(monkeypatch):
    logger_stub = _LoggerStub()

    def fail_default_character(_db_instance):
        raise RuntimeError("chacha backend exploded at /private/db/path SECRET_TOKEN")

    monkeypatch.setattr(deps, "logger", logger_stub)
    monkeypatch.setattr(deps, "_ensure_default_character", fail_default_character)

    await deps._ensure_default_character_async(object(), 4242)

    _assert_log_omits_raw_exception(
        logger_stub,
        "warning",
        "Error ensuring default character",
    )
    rendered = "\n".join(
        " ".join([message, " ".join(str(arg) for arg in args)])
        for _level, message, args, _kwargs in logger_stub.messages
    )
    assert "4242" not in rendered


def test_ensure_default_character_sanitizes_conflict_fallback_logs(monkeypatch):
    logger_stub = _LoggerStub()

    class _DBInstance:
        client_id = "safe-client"

        def __init__(self):
            self.fetch_count = 0

        def ensure_character_tables_ready(self):
            return None

        def get_character_card_by_name(self, _name):
            self.fetch_count += 1
            if self.fetch_count == 1:
                raise ConflictError("chacha backend exploded at /private/db/path SECRET_TOKEN")
            return None

    monkeypatch.setattr(deps, "logger", logger_stub)

    assert deps._ensure_default_character(_DBInstance()) is None

    _assert_log_omits_raw_exception(
        logger_stub,
        "warning",
        "Conflict error while ensuring default character",
    )
    _assert_log_omits_raw_exception(
        logger_stub,
        "error",
        "Still could not get/create default character after conflict",
    )


def test_ensure_default_character_sanitizes_database_failure_log(monkeypatch):
    logger_stub = _LoggerStub()

    class _DBInstance:
        def ensure_character_tables_ready(self):
            raise CharactersRAGDBError("chacha backend exploded at /private/db/path SECRET_TOKEN")

    monkeypatch.setattr(deps, "logger", logger_stub)

    assert deps._ensure_default_character(_DBInstance()) is None

    _assert_log_omits_raw_exception(
        logger_stub,
        "error",
        "Database error while ensuring default character",
    )


def test_ensure_default_character_sanitizes_unexpected_failure_log(monkeypatch):
    logger_stub = _LoggerStub()

    class _DBInstance:
        def ensure_character_tables_ready(self):
            raise RuntimeError("chacha backend exploded at /private/db/path SECRET_TOKEN")

    monkeypatch.setattr(deps, "logger", logger_stub)

    assert deps._ensure_default_character(_DBInstance()) is None

    _assert_log_omits_raw_exception(
        logger_stub,
        "error",
        "Unexpected error while ensuring default character",
    )


def test_chacha_health_last_error_uses_safe_error_type():
    with deps._CHACHA_HEALTH_LOCK:
        deps._CHACHA_HEALTH.update(
            {
                "init_attempts": 0,
                "init_failures": 0,
                "last_init_ms": None,
                "last_error": None,
                "last_init_success": None,
                "last_warn_dump": None,
                "cached_instances": 0,
                "consecutive_failures": 0,
                "default_char_ensures": 0,
                "default_char_failures": 0,
                "warm_startups": 0,
                "last_failure": None,
            }
        )

    deps._record_init(
        9.5,
        False,
        RuntimeError("chacha backend exploded at /private/db/path SECRET_TOKEN"),
    )

    snapshot = deps.get_chacha_health_snapshot()
    assert snapshot["last_error"] == "RuntimeError"
    assert "chacha backend exploded" not in str(snapshot)
    assert "/private/" not in str(snapshot)
    assert "SECRET_TOKEN" not in str(snapshot)
