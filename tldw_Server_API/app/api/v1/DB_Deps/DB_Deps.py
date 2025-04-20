# DB_Deps.py
# Description: This file contains the database dependency management for the FastAPI application.
#
# Imports
import os
import threading
from pathlib import Path
import logging
# 3rd-party Libraries
from fastapi import Header, HTTPException, status, Depends
#
# Local Imports
from tldw_Server_API.app.core.DB_Management.Media_DB import Database # Adjust import path
#
#######################################################################################################################
#
# Functions:

# --- Configuration ---
# Load the fixed API key from an environment variable or use a default
# IMPORTANT: For real use, use environment variables or a secrets manager.
EXPECTED_API_KEY = os.environ.get("TEST_API_KEY", "default-secret-key-for-testing")

# Define the path for the single database file
# Use Path object for better path handling
SINGLE_DB_FILE_PATH = Path("./single-user-folder/main_media_library.sqlite") # Example: store in a 'data' subdir

# --- Ensure base directory exists ---
# This ensures the directory where the DB will live is created if it doesn't exist
try:
    SINGLE_DB_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Ensured database directory exists: {SINGLE_DB_FILE_PATH.parent}")
except OSError as e:
    logging.error(f"Could not create database directory {SINGLE_DB_FILE_PATH.parent}: {e}", exc_info=True)
    # Decide if you want the app to fail startup if the dir can't be created
    # raise RuntimeError(f"Failed to create database directory: {e}") from e


# --- Global cache for the single DB instance ---
# To avoid reconnecting constantly, we can cache the single instance.
_db_instance = None
_db_lock = threading.Lock() # To prevent race conditions during first creation




#########################################################
#
# Singleton pattern for the Database instance for a single user

async def get_database() -> Database:
    """
    Gets the singleton Database instance for the single, shared database file.
    Handles initialization and schema checks on first access.
    """
    global _db_instance
    if _db_instance is None:
        # Use a lock to ensure only one thread initializes the instance if multiple requests hit simultaneously
        with _db_lock:
            # Double-check if another thread initialized it while waiting for the lock
            if _db_instance is None:
                try:
                    logging.info(f"Initializing database instance for path: {SINGLE_DB_FILE_PATH}")
                    # Instantiate the Database class with the single path
                    # The __init__ method handles schema checks (_ensure_schema)
                    _db_instance = Database(db_path=str(SINGLE_DB_FILE_PATH))
                    logging.info(f"Database instance created successfully for {SINGLE_DB_FILE_PATH}")
                except Exception as e:
                    logging.error(f"Failed to initialize database instance at {SINGLE_DB_FILE_PATH}: {e}", exc_info=True)
                    # Reset instance on failure so next request might retry
                    _db_instance = None
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Could not initialize database: {e}"
                    )
    # Always return the cached instance (once created)
    if _db_instance is None:
         # Should not happen if initialization worked, but handle defensively
         raise HTTPException(status_code=500, detail="Database instance is not available.")
    return _db_instance


#
# End of DB_Deps.py
########################################################################################################################