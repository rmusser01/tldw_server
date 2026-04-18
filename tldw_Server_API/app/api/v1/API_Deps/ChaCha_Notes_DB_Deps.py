from __future__ import annotations

import inspect
from typing import Any

from fastapi import Depends, HTTPException, status
from loguru import logger

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.chacha.runtime import (
    DEFAULT_CHARACTER_DESCRIPTION,
    DEFAULT_CHARACTER_NAME,
    ChaChaRuntimeInitError,
    ChaChaRuntimeManager,
    ChaChaRuntimeUnavailableError,
    _apply_sqlite_tuning,
    _health_check_instance,
)

_CHACHA_RUNTIME = ChaChaRuntimeManager()


def reset_chacha_shutdown_state() -> None:
    _CHACHA_RUNTIME.reset_for_tests()


def get_chacha_health_snapshot() -> dict[str, Any]:
    return _CHACHA_RUNTIME.snapshot()


def resolve_chacha_user_base_dir():
    return DatabasePaths.get_user_db_base_dir()


async def warm_chacha_db_for_user(user_id: int, client_id: str | None = None) -> None:
    try:
        await _CHACHA_RUNTIME.warm_for_user(user_id, client_id)
    except (ChaChaRuntimeUnavailableError, ChaChaRuntimeInitError, OSError, RuntimeError, ValueError, TypeError) as e:
        logger.warning("Warm-up for ChaChaNotes user {} failed: {}", user_id, e)


async def _resolve_runtime_db(user_id: int, client_id: str | None) -> CharactersRAGDB:
    try:
        return await _CHACHA_RUNTIME.get_or_create(user_id, client_id or str(user_id))
    except ChaChaRuntimeUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except ChaChaRuntimeInitError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


async def get_chacha_db_for_user_id(user_id: int, client_id: str | None = None) -> CharactersRAGDB:
    if isinstance(user_id, bool) or not isinstance(user_id, int) or user_id <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid owner_user_id.",
        )
    db_instance = await _resolve_runtime_db(user_id, client_id)
    if not _CHACHA_RUNTIME.is_shutting_down():
        _CHACHA_RUNTIME.schedule_default_character_ensure(db_instance, user_id)
    return db_instance


async def get_chacha_db_for_user(current_user: User = Depends(get_request_user)) -> CharactersRAGDB:
    try:
        from tldw_Server_API.app.main import app as _app

        override_fn = _app.dependency_overrides.get(get_chacha_db_for_user)
        if override_fn is not None:
            result = override_fn()
            if inspect.isawaitable(result):
                result = await result  # type: ignore[func-returns-value]
            if isinstance(result, CharactersRAGDB):
                return result
    except (ImportError, AttributeError, RuntimeError, HTTPException, TypeError, ValueError):
        pass

    logger.info("<<<<< ACTUAL get_chacha_db_for_user CALLED >>>>>")
    if not current_user or not isinstance(current_user.id, int):
        logger.error("get_chacha_db_for_user called without a valid User object or user.id is not int.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User identification failed for ChaChaNotes DB.",
        )

    db_instance = await _resolve_runtime_db(current_user.id, str(current_user.id))
    if not _CHACHA_RUNTIME.is_shutting_down():
        _CHACHA_RUNTIME.schedule_default_character_ensure(db_instance, current_user.id)
    return db_instance


async def get_chacha_db_for_owner(owner_user_id: int) -> CharactersRAGDB:
    if isinstance(owner_user_id, bool) or not isinstance(owner_user_id, int) or owner_user_id <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid owner_user_id.",
        )
    return await _resolve_runtime_db(owner_user_id, str(owner_user_id))


async def shutdown_chacha_resources(wait_timeout: float = 5.0) -> None:
    await _CHACHA_RUNTIME.shutdown(wait_timeout=wait_timeout)


def shutdown_chacha_executor(wait: bool = False) -> None:
    _CHACHA_RUNTIME.shutdown_executor(wait=wait)


def close_all_chacha_db_instances() -> None:
    _CHACHA_RUNTIME.close_all_instances()
