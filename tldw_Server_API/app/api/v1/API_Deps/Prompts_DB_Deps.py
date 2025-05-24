# tldw_Server_API/app/api/v1/API_Deps/Prompts_DB_Deps.py
#
# Imports
import logging
import threading
from pathlib import Path
from typing import Dict, Optional
#
# Third-party imports
from fastapi import Depends, HTTPException, status
from cachetools import LRUCache # Assuming cachetools is available
from loguru import logger

#
# Local Imports
from tldw_Server_API.app.core.Prompt_Management.Prompts_Interop import (
    initialize_interop as initialize_prompts_interop,
    shutdown_interop as shutdown_prompts_interop,
    get_db_instance as get_prompts_db_instance_from_interop,
    is_initialized as is_prompts_interop_initialized,
    PromptsDatabase, DatabaseError, SchemaError, InputError, ConflictError
)
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User
#
########################################################################################################################
#
# Functions:

# --- Configuration ---
DEFAULT_PROMPTS_DB_SUBDIR = "prompts_user_dbs"
MAIN_USER_DATA_BASE_DIR = settings.get("USER_DB_BASE_DIR")

if not MAIN_USER_DATA_BASE_DIR:
    logger.critical("CRITICAL: USER_DB_BASE_DIR is not configured in settings. Cannot determine Prompts DB path structure.")
    MAIN_USER_DATA_BASE_DIR = Path("./app_data/user_databases_fallback").resolve() # Fallback
    logger.error(f"USER_DB_BASE_DIR missing from settings, using emergency fallback: {MAIN_USER_DATA_BASE_DIR}")

SERVER_CLIENT_ID = settings.get("SERVER_CLIENT_ID")
if not SERVER_CLIENT_ID:
    logger.error("CRITICAL: SERVER_CLIENT_ID is not configured in settings.")
    SERVER_CLIENT_ID = "default_server_client_id_prompts" # Unique default
    logger.warning(f"SERVER_CLIENT_ID not set for prompts, using placeholder: {SERVER_CLIENT_ID}")

# --- Global Cache for Prompts DB Instances (managed by prompts_interop, but we track paths) ---
MAX_CACHED_PROMPTS_DB_INSTANCES = settings.get("MAX_CACHED_PROMPTS_DB_INSTANCES", 20)
_prompts_db_paths_cache: LRUCache = LRUCache(maxsize=MAX_CACHED_PROMPTS_DB_INSTANCES)
_prompts_db_lock = threading.Lock() # Lock for path cache and initialization coordination

# --- Helper Functions ---

def _get_prompts_db_path_for_user(user_id: int) -> Path:
    """
    Determines the Prompts database file path for a given user ID.
    Ensures the user's specific directory exists.
    Path: USER_DB_BASE_DIR / <user_id> / prompts_user_dbs / user_prompts.sqlite
    """
    user_dir_name = str(user_id)
    user_specific_prompts_base_dir = MAIN_USER_DATA_BASE_DIR / user_dir_name / DEFAULT_PROMPTS_DB_SUBDIR
    db_file = user_specific_prompts_base_dir / "user_prompts_v2.sqlite" # Added v2 to filename

    try:
        user_specific_prompts_base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured Prompts DB directory for user {user_id}: {user_specific_prompts_base_dir}")
    except OSError as e:
        logger.error(
            f"Could not create Prompts DB directory for user_id {user_id} at {user_specific_prompts_base_dir}: {e}",
            exc_info=True)
        raise IOError(f"Could not initialize Prompts storage directory for user {user_id}.") from e
    return db_file

# --- Main Dependency Function ---

_user_db_instances: Dict[int, PromptsDatabase] = {} # Simple dict for instances per user_id for this request scope
_user_db_locks: Dict[int, threading.Lock] = {} # Per-user lock for initialization

async def get_prompts_db_for_user(
        current_user: User = Depends(get_request_user)
) -> PromptsDatabase:
    """
    FastAPI dependency to get the PromptsDatabase instance for the identified user,
    managed via the prompts_interop layer.
    """
    # More robust check for User object and its id
    if not isinstance(current_user, User) or not hasattr(current_user, 'id') or not isinstance(current_user.id, int):
        logger.error(
            f"get_prompts_db_for_user called with an invalid User object. "
            f"Expected User model with int id. Got type: {type(current_user)}, value: {current_user}"
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="User identification failed for Prompts DB (Invalid User object).")

    user_id = current_user.id

    # Get or create a lock for this specific user_id
    if user_id not in _user_db_locks:
        with _prompts_db_lock: # Protect access to _user_db_locks
            if user_id not in _user_db_locks: # Double check
                _user_db_locks[user_id] = threading.Lock()
    user_specific_lock = _user_db_locks[user_id]

    with user_specific_lock:
        # Check if an instance for this user_id already exists (cached for this request/app lifetime if persistent)
        # The prompts_interop itself doesn't manage multiple DBs, it manages ONE.
        # So, we need a way to tell prompts_interop WHICH db to use, or manage instances here.
        # Let's manage instances here and pass the db_path to prompts_interop.
        # This means prompts_interop's global _db_instance is less useful in a multi-user, multi-db context.
        #
        # REVISED APPROACH:
        # The `prompts_interop` as written manages a SINGLE global instance.
        # For a multi-user system where each user has their OWN DB, we cannot use
        # the interop's global instance directly.
        #
        # Option 1: Modify `prompts_interop` to NOT use a global singleton, but return instances. (More work)
        # Option 2: Instantiate `PromptsDatabase` directly here, and wrap its calls.
        #           This bypasses the benefit of the interop being a single point of call.
        # Option 3: (Chosen for simplicity given current interop)
        #           The interop layer is more of a "wrapper" for the DB methods.
        #           We will instantiate PromptsDatabase directly here, per user.
        #           The `prompts_interop.py` utility functions that take `db_instance` can still be used.
        #           The interop's own instance-based methods will be bypassed.

        if user_id in _user_db_instances:
            db_instance = _user_db_instances.get(user_id)
            if db_instance:
                try:
                    # Quick check if connection is alive
                    conn = db_instance.get_connection()
                    conn.execute("SELECT 1")
                    logger.debug(f"Using cached PromptsDatabase instance for user_id: {user_id}")
                    return db_instance
                except Exception as e:
                    logger.warning(f"Cached PromptsDatabase for user {user_id} inactive ({e}). Re-creating.")
                    _user_db_instances.pop(user_id, None) # Remove bad instance

        # If not cached or cache was bad, create a new one
        db_path: Optional[Path] = None
        try:
            db_path = _get_prompts_db_path_for_user(user_id)
            logger.info(f"Initializing PromptsDatabase instance for user {user_id} at path: {db_path}")

            # Instantiate PromptsDatabase directly
            # The client_id for the PromptsDatabase should be the SERVER_CLIENT_ID,
            # as it's the server application making changes on behalf of the user.
            # If you need to track the specific end-user initiating the change,
            # that would be a different field, or SERVER_CLIENT_ID could be user-specific.
            # For now, using a global server client ID.
            db_instance = PromptsDatabase(db_path=str(db_path), client_id=str(current_user.id))

            _user_db_instances[user_id] = db_instance # Cache it
            logger.info(f"PromptsDatabase instance created and cached for user {user_id}")
            return db_instance

        except (DatabaseError, SchemaError, InputError, ConflictError) as e:
            log_path_str = str(db_path) if db_path else f"directory for user_id {user_id}"
            logger.error(f"Failed to initialize PromptsDatabase for user {user_id} at {log_path_str}: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Could not initialize prompts database for user: {str(e)}"
            ) from e
        except IOError as e:
            logger.error(f"Failed to get PromptsDatabase path for user {user_id}: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
        except Exception as e:
            log_path_str = str(db_path) if db_path else f"directory for user_id {user_id}"
            logger.error(f"Unexpected error initializing PromptsDatabase for user {user_id} at {log_path_str}: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred during prompts database setup."
            ) from e


def close_all_cached_prompts_db_instances():
    """Closes all cached PromptsDatabase connections. Useful for application shutdown."""
    with _prompts_db_lock: # Protects _user_db_locks and _user_db_instances
        logger.info(f"Closing all cached PromptsDatabase instances ({len(_user_db_instances)})...")
        for user_id, db_instance in list(_user_db_instances.items()):
            try:
                db_instance.close_connection() # PromptsDatabase handles its own thread-local connections
                logger.info(f"Closed PromptsDatabase connection for current thread for user {user_id}.")
            except Exception as e:
                logger.error(f"Error closing PromptsDatabase instance for user {user_id}: {e}", exc_info=True)
        _user_db_instances.clear()
        _user_db_locks.clear() # Clear user-specific locks as well
        logger.info("All PromptsDatabase instances cleared from cache and locks removed.")

# Register for shutdown in your main FastAPI app:
# @app.on_event("shutdown")
# async def shutdown_event():
#     close_all_cached_prompts_db_instances()

#
# End of Prompts_DB_Deps.py
########################################################################################################################
