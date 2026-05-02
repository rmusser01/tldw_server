import asyncio

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.API_Deps import Prompts_DB_Deps as deps
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.Prompt_Management.Prompts_Interop import (
    ConflictError,
    DatabaseError,
    InputError,
    SchemaError,
)


class _LoggerStub:
    def __init__(self):
        self.error_calls = []

    def debug(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        self.error_calls.append((args, kwargs))

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


class _RequestWithoutApp:
    pass


class _FailingClosePromptsDb:
    def __init__(self, exc: Exception):
        self.exc = exc
        self.close_calls = 0

    def close_connection(self):
        self.close_calls += 1
        raise self.exc


_SENSITIVE_MARKERS = (
    "prompts backend exploded",
    "/private/tmp/prompts-db.sqlite",
    "secret-token-123",
)


def _user() -> User:
    return User(id=52, username="prompts-user")


def _assert_error_log_sanitized(logger_stub: _LoggerStub, *, expected_message: str) -> None:
    assert logger_stub.error_calls
    matching_messages = [args[0] for args, _kwargs in logger_stub.error_calls if args]
    assert expected_message in matching_messages
    assert all(not kwargs.get("exc_info") for _args, kwargs in logger_stub.error_calls)

    rendered_calls = repr(logger_stub.error_calls)
    for marker in _SENSITIVE_MARKERS:
        assert marker not in rendered_calls


async def _assert_prompts_init_error_maps(
    monkeypatch,
    exc: Exception,
    *,
    expected_status: int,
    expected_detail: str,
) -> None:
    await deps.close_all_cached_prompts_db_instances()
    await deps.stop_prompts_pending_close_worker()

    def fail_create(*args, **kwargs):
        raise exc

    monkeypatch.setattr(deps, "_create_prompts_db_instance", fail_create)

    try:
        with pytest.raises(HTTPException) as exc_info:
            await deps.get_prompts_db_for_user(
                request=_RequestWithoutApp(),
                current_user=_user(),
            )

        assert exc_info.value.status_code == expected_status
        assert exc_info.value.detail == expected_detail
    finally:
        await deps.close_all_cached_prompts_db_instances()
        await deps.stop_prompts_pending_close_worker()


async def _assert_prompts_init_error_log_sanitized(
    monkeypatch,
    exc: Exception,
    *,
    expected_status: int,
    expected_detail: str,
    expected_message: str,
) -> None:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(deps, "logger", logger_stub, raising=True)

    await _assert_prompts_init_error_maps(
        monkeypatch,
        exc,
        expected_status=expected_status,
        expected_detail=expected_detail,
    )

    assert logger_stub.error_calls
    _assert_error_log_sanitized(logger_stub, expected_message=expected_message)


@pytest.mark.asyncio
async def test_get_prompts_db_maps_schema_error(monkeypatch):
    await _assert_prompts_init_error_maps(
        monkeypatch,
        SchemaError("schema exploded"),
        expected_status=500,
        expected_detail="Database schema error",
    )


@pytest.mark.asyncio
async def test_get_prompts_db_maps_database_error(monkeypatch):
    await _assert_prompts_init_error_maps(
        monkeypatch,
        DatabaseError("backend exploded"),
        expected_status=500,
        expected_detail="Prompts DB unavailable",
    )


@pytest.mark.asyncio
async def test_get_prompts_db_keeps_conflict_init_errors_as_500(monkeypatch):
    await _assert_prompts_init_error_maps(
        monkeypatch,
        ConflictError("duplicate bootstrap state"),
        expected_status=500,
        expected_detail="Prompts DB unavailable",
    )


@pytest.mark.asyncio
async def test_get_prompts_db_keeps_input_init_errors_as_500(monkeypatch):
    await _assert_prompts_init_error_maps(
        monkeypatch,
        InputError("invalid bootstrap state"),
        expected_status=500,
        expected_detail="invalid bootstrap state",
    )


@pytest.mark.asyncio
async def test_get_prompts_db_sanitizes_database_init_error_log(monkeypatch):
    await _assert_prompts_init_error_log_sanitized(
        monkeypatch,
        DatabaseError("prompts backend exploded /private/tmp/prompts-db.sqlite secret-token-123"),
        expected_status=500,
        expected_detail="Prompts DB unavailable",
        expected_message="Failed to initialize PromptsDatabase",
    )


@pytest.mark.asyncio
async def test_get_prompts_db_sanitizes_oserror_init_error_log(monkeypatch):
    await _assert_prompts_init_error_log_sanitized(
        monkeypatch,
        OSError("prompts backend exploded /private/tmp/prompts-db.sqlite secret-token-123"),
        expected_status=500,
        expected_detail="Prompts DB unavailable",
        expected_message="Failed to get PromptsDatabase path",
    )


@pytest.mark.asyncio
async def test_get_prompts_db_sanitizes_unexpected_init_error_log(monkeypatch):
    await _assert_prompts_init_error_log_sanitized(
        monkeypatch,
        RuntimeError("prompts backend exploded /private/tmp/prompts-db.sqlite secret-token-123"),
        expected_status=500,
        expected_detail="An unexpected error occurred during prompts database setup.",
        expected_message="Unexpected error initializing PromptsDatabase",
    )


@pytest.mark.asyncio
async def test_get_prompts_db_sanitizes_oserror_init_error(monkeypatch):
    await _assert_prompts_init_error_maps(
        monkeypatch,
        OSError("prompts path exploded"),
        expected_status=500,
        expected_detail="Prompts DB unavailable",
    )


def test_close_prompts_db_instance_sync_sanitizes_close_failure_log(monkeypatch):
    logger_stub = _LoggerStub()
    monkeypatch.setattr(deps, "logger", logger_stub, raising=True)
    db = _FailingClosePromptsDb(
        DatabaseError("prompts backend exploded /private/tmp/prompts-db.sqlite secret-token-123")
    )

    deps._close_prompts_db_instance_sync(
        (52, "secret-token-123"),
        db,
        reason="/private/tmp/prompts-db.sqlite",
    )

    assert db.close_calls == 1
    _assert_error_log_sanitized(
        logger_stub,
        expected_message="Failed to close PromptsDatabase instance",
    )


@pytest.mark.parametrize(
    "exc",
    [
        asyncio.QueueFull("prompts backend exploded /private/tmp/prompts-db.sqlite secret-token-123"),
        RuntimeError("prompts backend exploded /private/tmp/prompts-db.sqlite secret-token-123"),
    ],
)
def test_enqueue_prompts_db_close_sanitizes_queue_failure_log(monkeypatch, exc):
    class _FailingQueue:
        def put_nowait(self, item):
            raise exc

    logger_stub = _LoggerStub()
    monkeypatch.setattr(deps, "logger", logger_stub, raising=True)
    monkeypatch.setattr(deps, "_ensure_pending_close_worker", lambda: True, raising=True)
    monkeypatch.setattr(deps, "_pending_close_queue", _FailingQueue(), raising=True)
    db = _FailingClosePromptsDb(
        DatabaseError("prompts backend exploded /private/tmp/prompts-db.sqlite secret-token-123")
    )

    deps._enqueue_prompts_db_close(
        (52, "secret-token-123"),
        db,
        reason="/private/tmp/prompts-db.sqlite",
    )

    assert db.close_calls == 0
    _assert_error_log_sanitized(
        logger_stub,
        expected_message="Failed to enqueue PromptsDatabase close",
    )


@pytest.mark.asyncio
async def test_process_pending_closes_sanitizes_worker_close_failure_log(monkeypatch):
    def fail_close(*args, **kwargs):
        raise RuntimeError("prompts backend exploded /private/tmp/prompts-db.sqlite secret-token-123")

    logger_stub = _LoggerStub()
    queue = asyncio.Queue()
    await queue.put(
        (
            (52, "secret-token-123"),
            object(),
            "/private/tmp/prompts-db.sqlite",
        )
    )
    monkeypatch.setattr(deps, "logger", logger_stub, raising=True)
    monkeypatch.setattr(deps, "_pending_close_queue", queue, raising=True)
    monkeypatch.setattr(deps, "_close_prompts_db_instance_sync", fail_close, raising=True)

    task = asyncio.create_task(deps._process_pending_closes())
    try:
        await asyncio.wait_for(queue.join(), timeout=1)
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    _assert_error_log_sanitized(
        logger_stub,
        expected_message="Failed to process pending PromptsDatabase close",
    )


def test_evicting_lru_cache_popitem_sanitizes_eviction_callback_failure_log(monkeypatch):
    def fail_evict(*args, **kwargs):
        raise RuntimeError("prompts backend exploded /private/tmp/prompts-db.sqlite secret-token-123")

    logger_stub = _LoggerStub()
    monkeypatch.setattr(deps, "logger", logger_stub, raising=True)
    cache = deps._EvictingLRUCache(maxsize=1, on_evict=fail_evict)
    cache[(52, "secret-token-123")] = object()

    assert cache.popitem()[0] == (52, "secret-token-123")

    _assert_error_log_sanitized(
        logger_stub,
        expected_message="Prompts DB cache eviction callback failed",
    )


@pytest.mark.asyncio
async def test_close_all_cached_prompts_db_instances_sanitizes_close_failure_log(monkeypatch):
    logger_stub = _LoggerStub()
    db_instances = deps.LRUCache(maxsize=2)
    db_locks = deps.LRUCache(maxsize=2)
    db_instances[(52, "secret-token-123")] = _FailingClosePromptsDb(
        DatabaseError("prompts backend exploded /private/tmp/prompts-db.sqlite secret-token-123")
    )
    db_locks[(52, "secret-token-123")] = asyncio.Lock()
    monkeypatch.setattr(deps, "logger", logger_stub, raising=True)
    monkeypatch.setattr(deps, "_user_db_instances", db_instances, raising=True)
    monkeypatch.setattr(deps, "_user_db_locks", db_locks, raising=True)

    await deps.close_all_cached_prompts_db_instances()

    assert len(db_instances) == 0
    assert len(db_locks) == 0
    _assert_error_log_sanitized(
        logger_stub,
        expected_message="Failed to close cached PromptsDatabase instance",
    )
