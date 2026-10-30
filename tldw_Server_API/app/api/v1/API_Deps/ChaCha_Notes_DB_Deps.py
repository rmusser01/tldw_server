# tldw_Server_API/app/api/v1/API_Deps/ChaCha_Notes_DB_Deps.py
import asyncio
import inspect
import json
import sqlite3
from functools import partial
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, HTTPException, status
from loguru import logger

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user

# Local Imports
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management import sqlite_policy
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.DB_Management.chacha.runtime import (
    _CHACHA_CORRUPTION_ERROR_DETAIL,
    _CHACHA_INIT_TIMEOUT_DETAIL,
    _CHACHA_SHUTDOWN_INIT_ERROR_DETAIL,
    ChaChaDatabaseCorruptionError,
    ChaChaInitFailedError,
    ChaChaInitTimeoutError,
    ChaChaRuntimeUnavailableError,
    _is_sqlite_corruption_error,
    _sanitize_chacha_db_identifier,
    get_chacha_runtime,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
    SchemaError,
)
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.exceptions import BuiltinCharacterSeedError

#
#######################################################################################################################


def reset_chacha_shutdown_state() -> None:
    get_chacha_runtime().resume()


def get_chacha_health_snapshot() -> dict[str, Any]:
    """Return current ChaChaNotes health plus lifetime counters (see ``ChaChaRuntimeManager.health_snapshot``)."""
    return get_chacha_runtime().health_snapshot()


def resolve_chacha_user_base_dir() -> Path:
    """Public helper to expose the resolved user database base directory."""
    return DatabasePaths.get_user_db_base_dir()


SERVER_CLIENT_ID = settings.get("SERVER_CLIENT_ID")
if not SERVER_CLIENT_ID:
    logger.error("CRITICAL: SERVER_CLIENT_ID is not configured in settings.")
    SERVER_CLIENT_ID = "default_server_client_id"
    logger.warning(f"SERVER_CLIENT_ID not set, using placeholder: {SERVER_CLIENT_ID}")

# Global directory creation for a *common* ChaChaNotes base is removed
# as each user gets their DB under their own USER_DB_BASE_DIR/user_id/

# +++ Default Character Configuration +++
from tldw_Server_API.app.core.Character_Chat.constants import (  # noqa: E402 - re-exported
    DEFAULT_CHARACTER_DESCRIPTION,
    DEFAULT_CHARACTER_NAME,
)


def snapshot_cached_chacha_db_instances() -> tuple[CharactersRAGDB, ...]:
    """Return a lock-protected snapshot of authenticated cached DB instances."""
    return get_chacha_runtime().cached_instances()


#######################################################################################################################

# --- Helper Functions ---


def _get_chacha_db_path_for_user(user_id: int) -> Path:
    """
    Resolve the per-user ChaChaNotes DB path under the configured base.

    Policy: store each user's notes/chats DB at
      USER_DB_BASE_DIR / <user_id> / "ChaChaNotes.db"

    Notes:
    - USER_DB_BASE_DIR is read from global settings and can be overridden by env.
    - This per-user layout is intentional to isolate data and simplify backups.
    - Falls back to the default `Databases/user_databases` under repo root when unset.
    """
    db_file = DatabasePaths.get_chacha_db_path(user_id)
    logger.info(f"Ensured ChaChaNotes DB directory for user {user_id}: {db_file.parent}")
    return db_file


def _verify_existing_chacha_db_integrity(db_path: Path, *, user_id: int | None = None) -> None:
    """Run a read-only quick check before opening an existing SQLite DB for writes."""
    if not db_path.exists():
        return

    try:
        check_values = sqlite_policy.run_sqlite_quick_check(
            db_path,
            timeout_s=1.0,
            busy_timeout_ms=1000,
        )
    except sqlite3.Error as exc:
        if _is_sqlite_corruption_error(exc):
            raise ChaChaDatabaseCorruptionError(
                _CHACHA_CORRUPTION_ERROR_DETAIL,
                user_id=user_id,
                db_path=db_path,
            ) from exc
        raise

    if check_values and all(value.lower() == "ok" for value in check_values):
        return

    raise ChaChaDatabaseCorruptionError(
        _CHACHA_CORRUPTION_ERROR_DETAIL,
        user_id=user_id,
        db_path=db_path,
    )


def _apply_sqlite_tuning(db_instance: CharactersRAGDB) -> None:
    if db_instance.backend_type != BackendType.SQLITE:
        return
    try:
        conn = db_instance.get_connection()
        sqlite_policy.configure_sqlite_connection(
            conn,
            use_wal=True,
            synchronous="NORMAL",
            foreign_keys=True,
            busy_timeout_ms=10000,
            temp_store=None,
        )
    except (CharactersRAGDBError, sqlite3.Error, OSError, RuntimeError, ValueError) as e:
        logger.debug("ChaChaNotes tuning skipped ({})", type(e).__name__)


@chacha_operation(independent=True)
def _create_and_prepare_db(user_id: int, client_id: str) -> CharactersRAGDB:
    """Prepare an owner database and seed its bundled content before publication."""
    db_path: Optional[Path] = None
    db_path = _get_chacha_db_path_for_user(user_id)
    try:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    except OSError as _mk2:
        logger.debug("Secondary ensure for ChaChaNotes parent failed softly ({})", type(_mk2).__name__)
    affected_db = _sanitize_chacha_db_identifier(user_id, db_path)
    logger.info("Initializing CharactersRAGDB instance for user {} ({})", user_id, affected_db)
    try:
        _verify_existing_chacha_db_integrity(db_path, user_id=user_id)
    except ChaChaDatabaseCorruptionError:
        logger.error("ChaChaNotes DB corruption preflight failed for user {} ({})", user_id, affected_db)
        raise
    db_instance = CharactersRAGDB(
        db_path=str(db_path), client_id=str(client_id), owner_user_id=str(user_id),
    )
    _apply_sqlite_tuning(db_instance)
    from tldw_Server_API.app.core.Visual_Identities.builtin_pixel_migu import ensure_pixel_migu_character

    try:
        ensure_pixel_migu_character(db_instance, owner_user_id=user_id)
    # Storage, decoding and legacy repository APIs can still raise built-ins.
    except (
        BuiltinCharacterSeedError,
        CharactersRAGDBError,
        sqlite3.Error,
        OSError,
        RuntimeError,
        ValueError,
        TypeError,
    ):
        db_instance.close_connection()
        raise
    return db_instance


async def _ensure_default_character_async(db_instance: CharactersRAGDB, user_id: int) -> None:
    runtime = get_chacha_runtime()
    loop = asyncio.get_running_loop()
    try:
        future = loop.run_in_executor(runtime.executor(), _ensure_default_character_owned, db_instance)
        runtime.track(future)
        await asyncio.wait_for(
            asyncio.shield(future),
            timeout=5,
        )
        runtime.count("default_char_ensures")
    except asyncio.TimeoutError:
        runtime.count("default_char_failures")
        logger.warning(f"Timed out ensuring default character for user {user_id}; will retry on next access.")
    except (
        CharactersRAGDBError,
        SchemaError,
        InputError,
        ConflictError,
        sqlite3.Error,
        OSError,
        RuntimeError,
        ValueError,
    ) as e:
        runtime.count("default_char_failures")
        logger.warning(
            "Error ensuring default character ({}); continuing; will retry on next access.",
            type(e).__name__,
        )


@chacha_operation(independent=True)
def _ensure_default_character_owned(db_instance: CharactersRAGDB) -> Optional[int]:
    """Keep executor maintenance independent of the request that scheduled it."""
    return _ensure_default_character(db_instance)


def _ensure_default_character(db_instance: CharactersRAGDB) -> Optional[int]:
    """
    Checks if the default character exists in the DB, creates it if not.
    Returns the character_id of the default character.
    """
    try:
        db_instance.ensure_character_tables_ready()
        default_char = db_instance.get_character_card_by_name(DEFAULT_CHARACTER_NAME)
        if default_char:
            logger.debug(f"Default character '{DEFAULT_CHARACTER_NAME}' already exists with ID: {default_char['id']}.")
            return default_char["id"]
        else:
            logger.info(f"Default character '{DEFAULT_CHARACTER_NAME}' not found. Creating now...")
            card_data = {
                "name": DEFAULT_CHARACTER_NAME,
                "description": DEFAULT_CHARACTER_DESCRIPTION,
                # All other fields will be None or default in the DB
                "personality": "Supportive, patient, and concise.",
                "scenario": "General assistance",
                "system_prompt": "You are a helpful AI assistant.",
                "image": None,
                "post_history_instructions": None,
                "first_message": "Hello! I'm your Helpful AI Assistant. How can I support you today?",
                "message_example": None,
                "creator_notes": "This character is automatically generated to provide a reliable default assistant persona.",
                "alternate_greetings": None,
                "tags": json.dumps(["default", "neutral", "assistant"]),  # Store as JSON string
                "creator": "System",
                "character_version": "1.0",
                "extensions": None,
                "client_id": db_instance.client_id,  # Ensure client_id is set
            }
            # The add_character_card in CharactersRAGDB handles versioning and timestamps.
            char_id = db_instance.add_character_card(card_data)
            if char_id:
                logger.info(f"Successfully created default character '{DEFAULT_CHARACTER_NAME}' with ID: {char_id}.")
                return char_id
            else:
                # This should ideally not happen if add_character_card raises on failure
                logger.error(
                    f"Failed to create default character '{DEFAULT_CHARACTER_NAME}'. add_character_card returned None."
                )
                return None
    except ConflictError as e:  # Should only happen if get_character_card_by_name had an issue or race condition
        logger.warning("Conflict error while ensuring default character ({}); re-fetching.", type(e).__name__)
        # Re-fetch, as it might have been created by another thread.
        refetched_char = db_instance.get_character_card_by_name(DEFAULT_CHARACTER_NAME)
        if refetched_char:
            return refetched_char["id"]
        logger.error("Still could not get/create default character after conflict ({})", type(e).__name__)
        return None
    except (CharactersRAGDBError, SchemaError, InputError) as e:
        logger.error(
            "Database error while ensuring default character '{}' ({})",
            DEFAULT_CHARACTER_NAME,
            type(e).__name__,
        )
        return None  # Indicate failure
    except (AttributeError, KeyError, TypeError, ValueError, OSError, RuntimeError) as e_gen:
        logger.error(
            "Unexpected error while ensuring default character '{}' ({})",
            DEFAULT_CHARACTER_NAME,
            type(e_gen).__name__,
        )
        return None


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
    """Map a runtime error to its HTTP response. No detail includes ``str(exc)``."""
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


async def _get_or_init_db_instance(user_id: int, client_id: str) -> CharactersRAGDB:
    """Return the owner's cached database, initializing it through the runtime; errors become HTTP."""
    try:
        return await get_chacha_runtime().get_db(
            str(DatabasePaths.get_user_base_directory(user_id)),
            partial(_create_and_prepare_db, user_id, client_id),
            user_id=user_id,
        )
    except (ChaChaRuntimeUnavailableError, ChaChaInitTimeoutError, ChaChaInitFailedError) as exc:
        raise _to_http(exc) from exc


async def warm_chacha_db_for_user(user_id: int, client_id: str | None = None) -> None:
    if get_chacha_runtime().shutting_down:
        logger.debug("ChaChaNotes shutdown in progress; skipping warmup for user {}", user_id)
        return
    try:
        db_instance = await _get_or_init_db_instance(
            user_id, _tenant_client_id(user_id, client_id)
        )
        runtime = get_chacha_runtime()
        runtime.count("warm_startups")
        runtime.track(asyncio.create_task(_ensure_default_character_async(db_instance, user_id)))
    except (HTTPException, OSError, RuntimeError, ValueError, TypeError) as e:
        logger.warning("Warm-up for ChaChaNotes failed ({})", type(e).__name__)


def _tenant_client_id(user_id: int, client_id: str | None) -> str:
    """Return the client_id to bind this instance to, which is always the user.

    client_id is not a free-form label. On PostgreSQL it becomes
    app.current_user_id (ChaChaNotes_DB._set_session_client_id), and every
    ChaCha row-level security policy compares client_id against that setting,
    so it is the tenant identity.

    Callers had been passing descriptive strings: "voice_assistant" is a
    constant shared by every user, which pooled them all into one tenant on
    PostgreSQL, and "chat-macro-worker-<id>" is per-user but does not match the
    user's normal tenant, so rows written under it were invisible to them
    afterwards. The instance cache compounds it: its key is the user directory
    alone, so whichever caller initialised a user first fixed that user's
    tenant for the life of the process.

    Attribution belongs in logs, not in the tenant key.
    """
    resolved = str(user_id)
    if client_id is not None and str(client_id) != resolved:
        logger.debug(
            "Ignoring non-tenant client_id {!r} for user {}; client_id is the "
            "PostgreSQL tenant key and must be the user id",
            client_id,
            user_id,
        )
    return resolved


async def get_chacha_db_for_user_id(user_id: int, client_id: str | None = None) -> CharactersRAGDB:
    """
    Fetch a CharactersRAGDB instance for a specific user ID.

    This helper mirrors get_chacha_db_for_user but is intended for non-request contexts
    (e.g., WebSocket handlers) where we already know the user id.
    """
    if isinstance(user_id, bool) or not isinstance(user_id, int) or user_id <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid owner_user_id.",
        )

    db_instance = await _get_or_init_db_instance(
        user_id, _tenant_client_id(user_id, client_id)
    )
    runtime = get_chacha_runtime()
    if not runtime.shutting_down:
        runtime.track(asyncio.create_task(_ensure_default_character_async(db_instance, user_id)))
    return db_instance


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

    user_id = current_user.id
    db_instance = await _get_or_init_db_instance(
        user_id, _tenant_client_id(user_id, None)
    )
    runtime = get_chacha_runtime()
    if not runtime.shutting_down:
        runtime.track(asyncio.create_task(_ensure_default_character_async(db_instance, user_id)))
    return db_instance


async def get_chacha_db_for_owner(owner_user_id: int) -> CharactersRAGDB:
    """
    Return a CharactersRAGDB for an arbitrary user (the workspace owner).

    Used by the sharing module to read sources from the owner's DB
    while the accessor chats with their own DB.
    """
    if not isinstance(owner_user_id, int):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid owner_user_id.",
        )
    return await _get_or_init_db_instance(
        owner_user_id, _tenant_client_id(owner_user_id, None)
    )


def close_all_chacha_db_instances():
    """Closes all cached ChaChaNotesDB connections. Useful for application shutdown."""
    get_chacha_runtime().close_all()


async def shutdown_chacha_resources(wait_timeout: float = 5.0) -> None:
    """Drain ChaChaNotes tasks and close resources without racing active threads."""
    await get_chacha_runtime().aclose(wait_timeout)


def shutdown_chacha_executor(wait: bool = False) -> None:
    """Shut down the ChaChaNotes executor; the next use recreates it."""
    get_chacha_runtime().shutdown_executor(wait=wait)


# Example of how to register for shutdown event in FastAPI:
# from fastapi import FastAPI
# app = FastAPI()
# @app.on_event("shutdown")
# async def shutdown_event():
#     close_all_chacha_db_instances()
#     # also close other DB instances if you have similar managers
