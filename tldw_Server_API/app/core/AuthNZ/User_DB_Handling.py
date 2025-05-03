# User_DB_Handling.py
# Description: This file contains functions relating to user database handling
#
# Imports
import os
from pathlib import Path
from typing import Optional

#
# 3rd-party Libraries
from fastapi import Depends, HTTPException, status, Header
#
# Local Imports
# DB
from tldw_Server_API.app.core.DB_Management.Media_DB import Database
from tldw_Server_API.app.core.DB_Management.Users_DB import get_user_by_username
# Security
from tldw_Server_API.app.core.Security.Security import decode_access_token
# Utils
from tldw_Server_API.app.core.Utils.Utils import logging, load_and_log_configs
# API
from tldw_Server_API.app.api.v1.API_Deps.v1_endpoint_deps import oauth2_scheme
#
#######################################################################################################################
#
# Functions:


# FIXME
# Setup check from config for seeing if multiplayer is enabled
# Also add proper authentication etc. for multiplayer
MULTIPLAYER = False  # Placeholder for multiplayer mode

# --- Configuration ---
# Load the fixed API key from an environment variable or use a default
EXPECTED_API_KEY = os.environ.get("TEST_API_KEY", "default-secret-key-for-testing")
USER_DB_BASE_PATH = Path("./user_databases")


async def verify_api_key(api_key: str = Header(..., alias="X-API-KEY")): # Use a standard header like X-API-KEY
    """
    Simple dependency to verify a fixed API key.
    Raises 401 Unauthorized if the key is missing or invalid.
    """
    if api_key != EXPECTED_API_KEY:
        logging.warning(f"Invalid API Key received: '{api_key[:5]}...'") # Log carefully
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key"
        )
    # If the key is valid, the function completes without returning anything.
    logging.debug("API Key verified successfully.")


# Placeholder function to verify token and get user identifier
async def verify_token_and_get_user(token: Optional[str] = Header(None)) -> str:
    # Check if multiplayer is off AND token is missing
    if not MULTIPLAYER and token is None:
        # FIXME - integrate with config/install for allowing a custom user_id when set to singleuser mode
        #user_id = f"{SingleUser}"
        user_id = "SingleUser"  # Default user ID for single-user mode
        logging.info(f"Single-user mode, no token provided. Using default user_id: {user_id}")
        return user_id
    # Existing multiplayer or single-user-with-token logic
    elif MULTIPLAYER:
        if not token or not token.startswith("valid-token-"):
             raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token for multiplayer mode")
        user_id = token.split("-")[-1]
    elif not MULTIPLAYER and token is not None:
        # Single user mode but a token WAS provided (maybe for future use?)
        # You could choose to ignore it or validate it if needed later.
        # For now, let's assume we still use SingleUser for the DB path
        user_id = "SingleUser"
        logging.info(f"Single-user mode, token provided but ignored for DB path. Using default user_id: {user_id}")
    elif not MULTIPLAYER and token is "test_api_token_123":
        # Single user mode but a token WAS provided by test suite.
        user_id = "TestUser"
        logging.info(f"Single-user mode, test token provided. Using default test user_id: {user_id}")
    else:
        # Should not happen if logic above is correct
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unexpected state in token verification")

    if not user_id:
         raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not determine user ID") # Should be caught earlier

    logging.info(f"Token logic determined user: {user_id}")
    return user_id


def get_current_user(token: str = Depends(oauth2_scheme)):
    global MULTIPLAYER0
    if MULTIPLAYER == "True":
        payload = decode_access_token(token)
        if payload is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )
    else:
        payload = 'SingleUserMode'

    username: str = payload.get("sub")  # "sub" is subject
    if username is None:
        raise HTTPException(status_code=401, detail="Invalid token")

    # fetch user from DB
    user_dict = get_user_by_username(username)
    if not user_dict:
        raise HTTPException(status_code=401, detail="User not found")

    # optionally check if is_active, etc.
    return user_dict


# Dependency to get the correct DB instance for the user
async def get_db_for_user(user_id: str = Depends(verify_token_and_get_user)) -> Database:
    """
    Gets the Database instance for the specific user.
    Creates the user's directory and DB file if they don't exist.
    """
    try:
        user_db_dir = USER_DB_BASE_PATH / user_id
        user_db_dir.mkdir(parents=True, exist_ok=True) # Ensure user directory exists

        # Assuming the DB is named 'Media_DB.sqlite' within the user's folder
        db_path = user_db_dir / "Media_DB.sqlite"
        logging.info(f"Accessing database for user '{user_id}' at: {db_path}")

        # Initialize the Database object with the specific path
        # The Database class likely handles connection pooling or creation
        db = Database(db_path=str(db_path))
        # You might need to explicitly initialize the schema if the DB is new
        #db.initialize_schema_if_needed() # Add a method like this to your Database class
        return db
    except OSError as e:
        logging.error(f"Failed to create directory or access DB for user '{user_id}' at {USER_DB_BASE_PATH / user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not access user database storage.")
    except Exception as e:
        logging.error(f"Unexpected error getting DB for user '{user_id}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal error setting up user database.")

# --- Make sure your Database class can accept db_path ---
# Example modification to your Database class (in ./database.py)
# class Database:
#     def __init__(self, db_path="path/to/default.db"): # Accept db_path
#         self.db_path = db_path
#         self.conn = None # Or manage connection pool
#         self._ensure_schema() # Ensure tables exist

#     def _ensure_schema(self):
#          # Logic to create tables if they don't exist in self.db_path
#          # CREATE TABLE IF NOT EXISTS Media (...) etc.
#          pass

#     def get_connection(self):
#         # Return a new connection or one from pool for self.db_path
#         return sqlite3.connect(self.db_path)



# --- User Existence/Auth Checks ---




# --- Per-User Database Handling Functions ---






#
# End of User_DB_Handling.py
#######################################################################################################################
