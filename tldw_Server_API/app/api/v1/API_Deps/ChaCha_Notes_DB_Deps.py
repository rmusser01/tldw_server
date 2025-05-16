# tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB_Deps.py
import json
import threading
from pathlib import Path
import logging
from typing import Dict, Optional

from fastapi import Depends, HTTPException, status
from cachetools import LRUCache
#
#    logging.warning("cachetools not found. ChaChaNotes DB instance cache will grow indefinitely. "
#                    "Install with: pip install cachetools")
#
# Local Imports
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError, SchemaError, \
    InputError, ConflictError
#
#######################################################################################################################


# --- Configuration ---
_HAS_CACHETOOLS = True
# Define a specific base directory for ChaChaNotes DBs, or fallback
# Ensure USER_CHACHA_DB_BASE_DIR is defined in your settings, or adjust the default.
DEFAULT_CHACHA_DB_SUBDIR = "chachanotes_user_dbs"
USER_CHACHA_DB_BASE_DIR_STR = settings.get("user_db_base_dir")
if USER_CHACHA_DB_BASE_DIR_STR:
    USER_CHACHA_DB_BASE_DIR = Path(USER_CHACHA_DB_BASE_DIR_STR)
else:
    # Fallback to a subdir under the general USER_DB_BASE_DIR if USER_CHACHA_DB_BASE_DIR is not set
    USER_DB_BASE_DIR_STR = settings.get("user_db_base_dir")
    if not USER_DB_BASE_DIR_STR:
        logging.error("CRITICAL: user_db_base_dir is not configured in settings. Cannot determine ChaChaNotes DB path.")
        # This should ideally prevent the app from starting or be handled gracefully.
        # For now, using a placeholder that will likely cause issues if not configured:
        USER_DB_BASE_DIR_STR = "./data/user_dbs"  # Placeholder, ensure this is configured
        logging.warning(f"USER_DB_BASE_DIR not set, using placeholder: {USER_DB_BASE_DIR_STR}")
    USER_CHACHA_DB_BASE_DIR = Path(USER_DB_BASE_DIR_STR) / DEFAULT_CHACHA_DB_SUBDIR
    logging.info(f"USER_CHACHA_DB_BASE_DIR not explicitly set, defaulting to: {USER_CHACHA_DB_BASE_DIR}")

SERVER_CLIENT_ID = settings.get("SERVER_CLIENT_ID")
if not SERVER_CLIENT_ID:
    logging.error("CRITICAL: SERVER_CLIENT_ID is not configured in settings.")
    SERVER_CLIENT_ID = "default_server_client_id"  # Placeholder
    logging.warning(f"SERVER_CLIENT_ID not set, using placeholder: {SERVER_CLIENT_ID}")

# Ensure base directory for ChaChaNotes exists
try:
    USER_CHACHA_DB_BASE_DIR.mkdir(parents=True, exist_ok=True)
    logging.info(f"ChaChaNotes base directory ensured: {USER_CHACHA_DB_BASE_DIR}")
except OSError as e:
    logging.error(f"Could not create ChaChaNotes base DB directory {USER_CHACHA_DB_BASE_DIR}: {e}", exc_info=True)
    # Depending on application startup, this might be a fatal error.

# +++ Default Character Configuration +++
DEFAULT_CHARACTER_NAME = "Default Character"
DEFAULT_CHARACTER_DESCRIPTION = "This is a default character created by the system."

# --- Global Cache for ChaChaNotes DB Instances ---
MAX_CACHED_CHACHA_DB_INSTANCES = settings.get("MAX_CACHED_CHACHA_DB_INSTANCES",
                                              20)  # Reduced default for potentially larger DBs

if _HAS_CACHETOOLS:
    _chacha_db_instances: LRUCache = LRUCache(maxsize=MAX_CACHED_CHACHA_DB_INSTANCES)
    logging.info(f"Using LRUCache for ChaChaNotes DB instances (maxsize={MAX_CACHED_CHACHA_DB_INSTANCES}).")
else:
    _chacha_db_instances: Dict[int, CharactersRAGDB] = {}

_chacha_db_lock = threading.Lock()


#######################################################################################################################

# --- Helper Functions ---

def _get_chacha_db_path_for_user(user_id: int) -> Path:
    """
    Determines the database file path for a given user ID for ChaChaNotes.
    Ensures the user's specific directory exists.
    """
    user_dir_name = str(user_id)
    user_dir = USER_CHACHA_DB_BASE_DIR / user_dir_name
    db_file = user_dir / "user_chacha_notes_rag.sqlite"  # Consistent naming

    try:
        user_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logging.error(f"Could not create ChaChaNotes DB directory for user_id {user_id} at {user_dir}: {e}",
                      exc_info=True)
        raise IOError(f"Could not initialize ChaChaNotes storage directory for user {user_id}.") from e
    return db_file

def _ensure_default_character(db_instance: CharactersRAGDB) -> Optional[int]:
    """
    Checks if the default character exists in the DB, creates it if not.
    Returns the character_id of the default character.
    """
    try:
        default_char = db_instance.get_character_card_by_name(DEFAULT_CHARACTER_NAME)
        if default_char:
            logging.debug(f"Default character '{DEFAULT_CHARACTER_NAME}' already exists with ID: {default_char['id']}.")
            return default_char['id']
        else:
            logging.info(f"Default character '{DEFAULT_CHARACTER_NAME}' not found. Creating now...")
            card_data = {
                'name': DEFAULT_CHARACTER_NAME,
                'description': DEFAULT_CHARACTER_DESCRIPTION,
                # All other fields will be None or default in the DB
                'personality': None,
                'scenario': None,
                'system_prompt': None, # Explicitly neutral system prompt
                'image': None,
                'post_history_instructions': None,
                'first_message': "Hello! How can I help you today?", # A generic greeting
                'message_example': None,
                'creator_notes': "This is an automatically generated default character.",
                'alternate_greetings': None,
                'tags': json.dumps(["default", "neutral"]), # Store as JSON string
                'creator': "System",
                'character_version': "1.0",
                'extensions': None,
                'client_id': db_instance.client_id # Ensure client_id is set
            }
            # The add_character_card in CharactersRAGDB handles versioning and timestamps.
            char_id = db_instance.add_character_card(card_data)
            if char_id:
                logging.info(f"Successfully created default character '{DEFAULT_CHARACTER_NAME}' with ID: {char_id}.")
                return char_id
            else:
                # This should ideally not happen if add_character_card raises on failure
                logging.error(f"Failed to create default character '{DEFAULT_CHARACTER_NAME}'. add_character_card returned None.")
                return None
    except ConflictError as e: # Should only happen if get_character_card_by_name had an issue or race condition
        logging.warning(f"Conflict error while ensuring default character (likely race condition, re-fetching): {e}")
        # Re-fetch, as it might have been created by another thread.
        refetched_char = db_instance.get_character_card_by_name(DEFAULT_CHARACTER_NAME)
        if refetched_char:
            return refetched_char['id']
        logging.error(f"Still could not get/create default character after conflict: {e}")
        return None
    except (CharactersRAGDBError, InputError) as e:
        logging.error(f"Database error while ensuring default character '{DEFAULT_CHARACTER_NAME}': {e}", exc_info=True)
        return None # Indicate failure
    except Exception as e_gen:
        logging.error(f"Unexpected error while ensuring default character '{DEFAULT_CHARACTER_NAME}': {e_gen}", exc_info=True)
        return None

# --- Main Dependency Function ---

async def get_chacha_db_for_user(
        current_user: User = Depends(get_request_user)
) -> CharactersRAGDB:
    """
    FastAPI dependency to get the CharactersRAGDB instance for the identified user.
    Handles caching, initialization, and schema checks.
    """
    if not current_user or not isinstance(current_user.id, int):  # Ensure user_id is an int
        logging.error("get_chacha_db_for_user called without a valid User object or user.id is not int.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="User identification failed for ChaChaNotes DB.")

    user_id = current_user.id
    db_instance: Optional[CharactersRAGDB] = None

    with _chacha_db_lock:  # Protects cache access
        db_instance = _chacha_db_instances.get(user_id)

    if db_instance:
        try:
            # Perform a quick check to see if the connection is alive
            # This is a basic check; the CharactersRAGDB class handles more robust connection checks internally
            conn = db_instance.get_connection()
            conn.execute("SELECT 1")
            logging.debug(f"Using cached and active ChaChaNotesDB instance for user_id: {user_id}")
            return db_instance
        except (CharactersRAGDBError, AttributeError, Exception) as e:  # Catch broader errors if connection is dead
            logging.warning(f"Cached ChaChaNotesDB instance for user {user_id} seems inactive ({e}). Re-initializing.")
            with _chacha_db_lock:  # Ensure exclusive access for removal
                if _chacha_db_instances.get(user_id) is db_instance:  # ensure it's the same instance
                    _chacha_db_instances.pop(user_id, None)
            db_instance = None  # Force re-initialization

    logging.info(f"No usable cached ChaChaNotesDB instance found for user_id: {user_id}. Initializing.")
    with _chacha_db_lock:  # Protects instance creation and cache update
        # Double-check cache in case another thread created it while waiting
        db_instance = _chacha_db_instances.get(user_id)
        if db_instance:  # pragma: no cover
            logging.debug(f"ChaChaNotesDB instance for user {user_id} created concurrently by another thread.")
            return db_instance

        db_path: Optional[Path] = None
        try:
            db_path = _get_chacha_db_path_for_user(user_id)
            logging.info(f"Initializing CharactersRAGDB instance for user {user_id} at path: {db_path}")

            db_instance = CharactersRAGDB(db_path=str(db_path), client_id=SERVER_CLIENT_ID)

            # +++ Ensure default character exists after DB instance is created +++
            default_char_id = _ensure_default_character(db_instance)
            if default_char_id is None:
                # This is a problem, the application might not function correctly without a default.
                logging.error(f"Failed to ensure default character for user {user_id}. This might impact functionality.")
                # Depending on strictness, you could raise an HTTPException here.
                # For now, we'll log and proceed, but chat saving might fail if it relies on this.

            _chacha_db_instances[user_id] = db_instance
            logging.info(f"CharactersRAGDB instance created and cached successfully for user {user_id}")

        except (CharactersRAGDBError, SchemaError, InputError, ConflictError) as e:
            log_path_str = str(db_path) if db_path else f"directory for user_id {user_id}"
            logging.error(f"Failed to initialize CharactersRAGDB for user {user_id} at {log_path_str}: {e}",
                          exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Could not initialize character & notes database for user: {e}"
            ) from e
        except IOError as e:  # Catch error from _get_chacha_db_path_for_user
            logging.error(f"Failed to get CharactersRAGDB path for user {user_id}: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            ) from e
        except Exception as e:
            log_path_str = str(db_path) if db_path else f"directory for user_id {user_id}"
            logging.error(f"Unexpected error initializing CharactersRAGDB for user {user_id} at {log_path_str}: {e}",
                          exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred during character & notes database setup for user."
            ) from e
    return db_instance


def close_all_chacha_db_instances():
    """Closes all cached ChaChaNotesDB connections. Useful for application shutdown."""
    with _chacha_db_lock:
        logging.info(f"Closing all cached ChaChaNotesDB instances ({len(_chacha_db_instances)})...")
        for user_id, db_instance in list(_chacha_db_instances.items()):
            try:
                db_instance.close_connection()
                logging.info(f"Closed ChaChaNotesDB instance for user {user_id}.")
            except Exception as e:
                logging.error(f"Error closing ChaChaNotesDB instance for user {user_id}: {e}", exc_info=True)
        _chacha_db_instances.clear()
        logging.info("All ChaChaNotesDB instances closed and cache cleared.")

# Example of how to register for shutdown event in FastAPI:
# from fastapi import FastAPI
# app = FastAPI()
# @app.on_event("shutdown")
# async def shutdown_event():
#     close_all_chacha_db_instances()
#     # also close other DB instances if you have similar managers