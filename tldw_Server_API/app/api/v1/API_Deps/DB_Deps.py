# DB_Deps.py
# Description: Manages user-specific database instances based on application mode.
#
# Imports
import threading
from pathlib import Path
import logging
from typing import Dict, Optional

# 3rd-party Libraries
from fastapi import Header, HTTPException, status, Depends
try:
    from cachetools import LRUCache
    _HAS_CACHETOOLS = True
except ImportError:
    _HAS_CACHETOOLS = False
    logging.warning("cachetools not found. User DB instance cache will grow indefinitely. "
                    "Install with: pip install cachetools")

# Local Imports
# Import the settings dictionary
from tldw_Server_API.app.core.config import settings
# Import the primary user identification dependency and User model
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User
# Import the specific Database class
from tldw_Server_API.app.core.DB_Management.Media_DB_v2 import MediaDatabase, DatabaseError, SchemaError # Adjust import path

#######################################################################################################################

# --- Configuration (using imported settings dictionary) ---
# Assign to module-level vars for convenience or use settings["KEY"] directly
USER_DB_BASE_DIR = settings["USER_DB_BASE_DIR"]
SERVER_CLIENT_ID = settings["SERVER_CLIENT_ID"]

# Base directory existence is now checked within config.py

# --- Global Cache for User DB Instances ---
MAX_CACHED_DB_INSTANCES = 100  # Adjust as needed

if _HAS_CACHETOOLS:
    # Keyed by user ID (int)
    _user_db_instances: LRUCache = LRUCache(maxsize=MAX_CACHED_DB_INSTANCES)
    logging.info(f"Using LRUCache for user DB instances (maxsize={MAX_CACHED_DB_INSTANCES}).")
else:
    # Keyed by user ID (int)
    _user_db_instances: Dict[int, MediaDatabase] = {} # Fallback to standard dict

_user_db_lock = threading.Lock() # Protects access to _user_db_instances

#######################################################################################################################

# --- Helper Functions ---

def _get_db_path_for_user(user_id: int) -> Path:
    """
    Determines the database file path for a given user ID.
    Ensures the user's specific directory exists.
    Uses USER_DB_BASE_DIR assigned from settings.
    """
    # user_id will be settings["SINGLE_USER_FIXED_ID"] in single-user mode
    user_dir_name = str(user_id)
    # Use the variable assigned from settings dict
    user_dir = USER_DB_BASE_DIR / user_dir_name
    db_file = user_dir / "user_media_library.sqlite" # Consistent naming

    try:
        user_dir.mkdir(parents=True, exist_ok=True)
        # Optional: logging.debug(f"Ensured directory exists for user {user_id}: {user_dir}")
    except OSError as e:
        logging.error(f"Could not create database directory for user_id {user_id} at {user_dir}: {e}", exc_info=True)
        # Raise a standard exception to be caught by the main dependency
        raise IOError(f"Could not initialize storage directory for user {user_id}.") from e
    return db_file

# --- Main Dependency Function ---

async def get_media_db_for_user(
    # Depends on the primary authentication/identification dependency
    current_user: User = Depends(get_request_user)
) -> MediaDatabase:
    """
    FastAPI dependency to get the Database instance for the identified user.

    Works in both single-user (using fixed ID from settings) and multi-user modes.
    Handles caching, initialization, and schema checks. Uses configuration
    values assigned from the 'settings' dictionary.

    Args:
        current_user: The User object (either fixed or fetched) provided by `get_request_user`.

    Returns:
        A Database instance connected to the appropriate user's database file.

    Raises:
        HTTPException: If authentication fails (handled by `get_request_user`),
                       or if the database cannot be initialized.
    """
    if not current_user or not isinstance(current_user.id, int):
        logging.error("get_media_db_for_user called without a valid User object/ID.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="User identification failed.")

    user_id = current_user.id # Will be SINGLE_USER_FIXED_ID in single-user mode
    db_instance: Optional[MediaDatabase] = None

    # --- Check Cache ---
    # Read lock implicitly handled by context manager
    with _user_db_lock:
        db_instance = _user_db_instances.get(user_id)

    if db_instance:
        # Optional: Add connection check if needed, though Database class might handle it
        logging.debug(f"Using cached Database instance for user_id: {user_id}")
        return db_instance

    # --- Instance Not Cached: Create New One ---
    logging.info(f"No cached DB instance found for user_id: {user_id}. Initializing.")
    # Acquire write lock
    with _user_db_lock:
        # Double-check cache in case another thread created it while waiting
        db_instance = _user_db_instances.get(user_id)
        if db_instance:
            logging.debug(f"DB instance for user {user_id} created concurrently.")
            return db_instance

        # --- Get Path and Initialize ---
        db_path: Optional[Path] = None # Define scope for logging in except block
        try:
            db_path = _get_db_path_for_user(user_id)
            logging.info(f"Initializing Database instance for user {user_id} at path: {db_path}")

            # Instantiate the Database class for the specific user ID's path
            # Use SERVER_CLIENT_ID assigned from settings dict
            db_instance = MediaDatabase(db_path=str(db_path), client_id=str(current_user.id))

            # --- Store in Cache ---
            _user_db_instances[user_id] = db_instance
            logging.info(f"Database instance created and cached successfully for user {user_id}")

        except (DatabaseError, SchemaError) as e:
            log_path = db_path or f"directory for user_id {user_id}"
            logging.error(f"Failed to initialize database for user {user_id} at {log_path}: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Could not initialize database for user: {e}"
            ) from e
        except IOError as e: # Catch error from _get_db_path_for_user
            logging.error(f"Failed to get DB path for user {user_id}: {e}", exc_info=True)
            raise HTTPException(
                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                 detail=str(e) # Use the message from IOError
             ) from e
        except Exception as e:
            log_path = db_path or f"directory for user_id {user_id}"
            logging.error(f"Unexpected error initializing database for user {user_id} at {log_path}: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"An unexpected error occurred during database setup for user."
            ) from e

    # Return the newly created and cached instance
    return db_instance

#
# End of DB_Deps.py
########################################################################################################################
