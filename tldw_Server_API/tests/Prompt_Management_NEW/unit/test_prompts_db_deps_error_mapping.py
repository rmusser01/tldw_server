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


class _RequestWithoutApp:
    pass


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
async def test_get_prompts_db_sanitizes_oserror_init_error(monkeypatch):
    await _assert_prompts_init_error_maps(
        monkeypatch,
        OSError("prompts path exploded"),
        expected_status=500,
        expected_detail="Prompts DB unavailable",
    )
