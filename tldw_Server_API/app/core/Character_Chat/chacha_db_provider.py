"""Per-user ChaChaNotes databases: the factory, default-character seeding, the tenant client id and the accessors.

Workers and core code open a user's database here. ``get_chacha_runtime()`` owns the cache, executor and
lifecycle. Every function raises core errors, never ``HTTPException``: the accessors raise
``ValueError("Invalid owner_user_id.")`` for an invalid id and the runtime's errors otherwise. The API layer's
ChaCha dependency module maps them to HTTP.
"""

import asyncio
import json
import sqlite3
from functools import partial
from pathlib import Path
from typing import Any, Optional

from loguru import logger

# Re-exported: callers import the default character's name from here.
from tldw_Server_API.app.core.Character_Chat.constants import (
    DEFAULT_CHARACTER_DESCRIPTION,
    DEFAULT_CHARACTER_NAME,
)
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management import sqlite_policy
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.DB_Management.chacha.runtime import (
    _CHACHA_CORRUPTION_ERROR_DETAIL,
    _CHACHA_SHUTDOWN_INIT_ERROR_DETAIL,
    ChaChaDatabaseCorruptionError,
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
from tldw_Server_API.app.core.Utils.app_services import app_services

_INVALID_OWNER_DETAIL = "Invalid owner_user_id."


def resolve_chacha_user_base_dir() -> Path:
    """Public helper to expose the resolved user database base directory."""
    return DatabasePaths.get_user_db_base_dir()


SERVER_CLIENT_ID = settings.get("SERVER_CLIENT_ID")
if not SERVER_CLIENT_ID:
    logger.error("CRITICAL: SERVER_CLIENT_ID is not configured in settings.")
    SERVER_CLIENT_ID = "default_server_client_id"
    logger.warning(f"SERVER_CLIENT_ID not set, using placeholder: {SERVER_CLIENT_ID}")


# --- Runtime lifecycle (delegations to get_chacha_runtime()) ---


def get_chacha_health_snapshot() -> dict[str, Any]:
    """Return current ChaChaNotes health plus lifetime counters (see ``ChaChaRuntimeManager.health_snapshot``)."""
    return get_chacha_runtime().health_snapshot()


def snapshot_cached_chacha_db_instances() -> tuple[CharactersRAGDB, ...]:
    """Return a lock-protected snapshot of authenticated cached DB instances."""
    return get_chacha_runtime().cached_instances()


def reset_chacha_shutdown_state() -> None:
    get_chacha_runtime().resume()


def close_all_chacha_db_instances() -> None:
    """Closes all cached ChaChaNotesDB connections. Useful for application shutdown."""
    get_chacha_runtime().close_all()


def shutdown_chacha_executor(wait: bool = False) -> None:
    """Shut down the ChaChaNotes executor; the next use recreates it."""
    get_chacha_runtime().shutdown_executor(wait=wait)


async def shutdown_chacha_resources(wait_timeout: float = 5.0) -> None:
    """Drain ChaChaNotes tasks and close resources; with no runtime built, there is nothing to close."""
    runtime = app_services().chacha_runtime
    if runtime is not None:
        await runtime.aclose(wait_timeout)


# --- The per-user database factory ---


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
def create_and_prepare_chacha_db(user_id: int, client_id: str) -> CharactersRAGDB:
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


# --- Default-character seeding ---


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
    return ensure_default_character(db_instance)


def ensure_default_character(db_instance: CharactersRAGDB) -> Optional[int]:
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


# --- The tenant and the accessors ---


def tenant_client_id(user_id: int, client_id: str | None) -> str:
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


async def _open(user_id: int, client_id: str | None) -> CharactersRAGDB:
    """Return the user's cached database, building it through the runtime under the user's tenant."""
    runtime = get_chacha_runtime()
    # Refuse before resolving the user directory: resolving it creates it.
    if runtime.shutting_down:
        raise ChaChaRuntimeUnavailableError(_CHACHA_SHUTDOWN_INIT_ERROR_DETAIL)
    return await runtime.get_db(
        str(DatabasePaths.get_user_base_directory(user_id)),
        partial(create_and_prepare_chacha_db, user_id, tenant_client_id(user_id, client_id)),
        user_id=user_id,
    )


async def get_chacha_db_for_user_id(user_id: int, client_id: str | None = None) -> CharactersRAGDB:
    """Return the database of a user whose id the caller already knows (workers, WebSocket handlers).

    Raises ``ValueError`` for a bool, non-int or non-positive id before it touches the runtime.
    Schedules the default character on the runtime.
    """
    if isinstance(user_id, bool) or not isinstance(user_id, int) or user_id <= 0:
        raise ValueError(_INVALID_OWNER_DETAIL)
    db_instance = await _open(user_id, client_id)
    runtime = get_chacha_runtime()
    if not runtime.shutting_down:
        runtime.track(asyncio.create_task(_ensure_default_character_async(db_instance, user_id)))
    return db_instance


async def get_chacha_db_for_owner(owner_user_id: int) -> CharactersRAGDB:
    """Return an arbitrary user's database (the workspace owner), as the sharing module reads sources from it.

    Raises ``ValueError`` for a non-int id.
    """
    if not isinstance(owner_user_id, int):
        raise ValueError(_INVALID_OWNER_DETAIL)
    return await _open(owner_user_id, None)


async def warm_chacha_db_for_user(user_id: int, client_id: str | None = None) -> None:
    if get_chacha_runtime().shutting_down:
        logger.debug("ChaChaNotes shutdown in progress; skipping warmup for user {}", user_id)
        return
    try:
        db_instance = await _open(user_id, client_id)
        runtime = get_chacha_runtime()
        runtime.count("warm_startups")
        runtime.track(asyncio.create_task(_ensure_default_character_async(db_instance, user_id)))
    # The runtime's errors are RuntimeErrors.
    except (OSError, RuntimeError, ValueError, TypeError) as e:
        logger.warning("Warm-up for ChaChaNotes failed ({})", type(e).__name__)
