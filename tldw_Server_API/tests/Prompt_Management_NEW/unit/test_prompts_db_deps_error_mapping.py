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


_SENSITIVE_MARKERS = (
    "prompts backend exploded",
    "/private/tmp/prompts-db.sqlite",
    "secret-token-123",
)


def _user() -> User:
    return User(id=52, username="prompts-user")


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
    matching_messages = [args[0] for args, _kwargs in logger_stub.error_calls if args]
    assert expected_message in matching_messages
    assert all(not kwargs.get("exc_info") for _args, kwargs in logger_stub.error_calls)

    rendered_calls = repr(logger_stub.error_calls)
    for marker in _SENSITIVE_MARKERS:
        assert marker not in rendered_calls


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
