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
