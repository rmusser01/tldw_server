# tldw_Server_API/app/api/v1/API_Deps/ChaCha_Notes_DB_Deps.py
"""HTTP adapter for per-user ChaChaNotes databases.

``core/Character_Chat/chacha_db_provider.py`` opens the databases and raises core errors. This module maps
them to HTTP and re-exports the provider's public names for API callers and tests.
"""

import inspect
from collections.abc import Awaitable
from functools import partial

from fastapi import Depends, HTTPException, status
from loguru import logger

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.Character_Chat import chacha_db_provider
from tldw_Server_API.app.core.Character_Chat.chacha_db_provider import (
    DEFAULT_CHARACTER_NAME as DEFAULT_CHARACTER_NAME,
)
from tldw_Server_API.app.core.Character_Chat.chacha_db_provider import (
    SERVER_CLIENT_ID as SERVER_CLIENT_ID,
)
from tldw_Server_API.app.core.Character_Chat.chacha_db_provider import (
    close_all_chacha_db_instances as close_all_chacha_db_instances,
)
from tldw_Server_API.app.core.Character_Chat.chacha_db_provider import (
    get_chacha_health_snapshot as get_chacha_health_snapshot,
)
from tldw_Server_API.app.core.Character_Chat.chacha_db_provider import (
    reset_chacha_shutdown_state as reset_chacha_shutdown_state,
)
from tldw_Server_API.app.core.Character_Chat.chacha_db_provider import (
    resolve_chacha_user_base_dir as resolve_chacha_user_base_dir,
)
from tldw_Server_API.app.core.Character_Chat.chacha_db_provider import (
    shutdown_chacha_executor as shutdown_chacha_executor,
)
from tldw_Server_API.app.core.Character_Chat.chacha_db_provider import (
    shutdown_chacha_resources as shutdown_chacha_resources,
)
from tldw_Server_API.app.core.Character_Chat.chacha_db_provider import (
    snapshot_cached_chacha_db_instances as snapshot_cached_chacha_db_instances,
)
from tldw_Server_API.app.core.Character_Chat.chacha_db_provider import (
    warm_chacha_db_for_user as warm_chacha_db_for_user,
)
from tldw_Server_API.app.core.DB_Management.chacha.runtime import (
    _CHACHA_CORRUPTION_ERROR_DETAIL,
    _CHACHA_INIT_TIMEOUT_DETAIL,
    _CHACHA_SHUTDOWN_INIT_ERROR_DETAIL,
    ChaChaInitFailedError,
    ChaChaInitTimeoutError,
    ChaChaRuntimeUnavailableError,
    _is_sqlite_corruption_error,
    get_chacha_runtime,
)
from tldw_Server_API.app.core.DB_Management.chacha.runtime import (
    ChaChaDatabaseCorruptionError as ChaChaDatabaseCorruptionError,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    InputError,
)
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths


def _map_chacha_init_db_error(exc: Exception) -> HTTPException:
    from tldw_Server_API.app.api.v1.utils.http_errors import map_db_error_to_http

    if _is_sqlite_corruption_error(exc):
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=_CHACHA_CORRUPTION_ERROR_DETAIL,
        )

    return map_db_error_to_http(
        exc,
        default_detail="ChaChaNotes DB unavailable",
        input_status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        conflict_status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        conflict_detail="ChaChaNotes DB unavailable",
    )


def _to_http(exc: Exception) -> HTTPException:
    """Map a runtime error to its HTTP response; this function adds no ``str(exc)`` to a detail.

    A DB-error cause goes through ``_map_chacha_init_db_error``, whose details are unchanged.
    """
    # ChaChaInitAbortedError subclasses ChaChaRuntimeUnavailableError: both are the shutdown 503.
    if isinstance(exc, ChaChaRuntimeUnavailableError):
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=_CHACHA_SHUTDOWN_INIT_ERROR_DETAIL,
        )
    if isinstance(exc, ChaChaInitTimeoutError):
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=_CHACHA_INIT_TIMEOUT_DETAIL,
        )
    cause = exc.__cause__ if isinstance(exc.__cause__, Exception) else None
    if cause is None:
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not initialize character & notes database for user: unknown error",
        )
    if _is_sqlite_corruption_error(cause) or isinstance(cause, (CharactersRAGDBError, InputError)):
        return _map_chacha_init_db_error(cause)
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Could not initialize character & notes database for user",
    )


async def _as_http(opening: Awaitable[CharactersRAGDB]) -> CharactersRAGDB:
    """Await a provider accessor; its invalid-id ``ValueError`` is a 400, its runtime errors map via ``_to_http``."""
    try:
        return await opening
    except ValueError as exc:
        # The provider validates the id; its message may carry a path, so it is never echoed.
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid owner_user_id.") from exc
    except (ChaChaRuntimeUnavailableError, ChaChaInitTimeoutError, ChaChaInitFailedError) as exc:
        raise _to_http(exc) from exc


async def _get_or_init_db_instance(user_id: int, client_id: str) -> CharactersRAGDB:
    """Return the owner's cached database, initializing it through the runtime; errors become HTTP."""
    runtime = get_chacha_runtime()
    try:
        # Refuse before resolving the user directory: resolving it creates it.
        if runtime.shutting_down:
            raise ChaChaRuntimeUnavailableError(_CHACHA_SHUTDOWN_INIT_ERROR_DETAIL)
        return await runtime.get_db(
            str(DatabasePaths.get_user_base_directory(user_id)),
            partial(chacha_db_provider.create_and_prepare_chacha_db, user_id, client_id),
            user_id=user_id,
        )
    except (ChaChaRuntimeUnavailableError, ChaChaInitTimeoutError, ChaChaInitFailedError) as exc:
        raise _to_http(exc) from exc


async def get_chacha_db_for_user_id(user_id: int, client_id: str | None = None) -> CharactersRAGDB:
    """
    Fetch a CharactersRAGDB instance for a specific user ID.

    This helper mirrors get_chacha_db_for_user but is intended for non-request contexts
    (e.g., WebSocket handlers) where we already know the user id.
    """
    return await _as_http(chacha_db_provider.get_chacha_db_for_user_id(user_id, client_id))


# --- Main Dependency Function ---


async def get_chacha_db_for_user(current_user: User = Depends(get_request_user)) -> CharactersRAGDB:
    """
    FastAPI dependency to get the CharactersRAGDB instance for the identified user.
    Handles caching and health checks; heavy initialization runs in a dedicated executor.
    """
    # Respect FastAPI dependency overrides explicitly if they exist.
    # Some test environments reset overrides aggressively; checking here ensures
    # we still honor an override bound to this callable.
    try:
        from tldw_Server_API.app.main import app as _app  # Local import to avoid import cycles at module load

        override_fn = _app.dependency_overrides.get(get_chacha_db_for_user)
        if override_fn is not None:
            try:
                result = override_fn()
                if inspect.isawaitable(result):
                    result = await result  # type: ignore[func-returns-value]
                if isinstance(result, CharactersRAGDB):
                    return result
            except (HTTPException, TypeError, ValueError, RuntimeError, AttributeError):
                # Fall back to standard resolution on any override execution issue
                pass
    except (ImportError, AttributeError, RuntimeError):
        # If importing app or inspecting overrides fails, proceed normally
        pass

    logger.info("<<<<< ACTUAL get_chacha_db_for_user CALLED >>>>>")
    if not current_user or not isinstance(current_user.id, int):  # Ensure user_id is an int
        logger.error("get_chacha_db_for_user called without a valid User object or user.id is not int.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="User identification failed for ChaChaNotes DB."
        )

    return await _as_http(chacha_db_provider.get_chacha_db_for_user_id(current_user.id))


async def get_chacha_db_for_owner(owner_user_id: int) -> CharactersRAGDB:
    """
    Return a CharactersRAGDB for an arbitrary user (the workspace owner).

    Used by the sharing module to read sources from the owner's DB
    while the accessor chats with their own DB.
    """
    return await _as_http(chacha_db_provider.get_chacha_db_for_owner(owner_user_id))
