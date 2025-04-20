# User_DB_Handling.py
# Description: This file contains functions relating to user database handling
#
# Imports
import os
from pathlib import Path
#
# 3rd-party Libraries
from fastapi import Depends, HTTPException, status, Header

from tldw_Server_API.app.api.v1.DB_Deps.DB_Deps import EXPECTED_API_KEY
#
# Local Imports
from tldw_Server_API.app.core.DB_Management.Media_DB import Database
from tldw_Server_API.app.core.Utils.Utils import logging

#
#######################################################################################################################
#
# Functions:

# FIXME - THIS IS PLACEHOLDER CODE, NOT CONFIRMED OR FULLY EVALUATED

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
async def verify_token_and_get_user(token: str = Header(...)) -> str:
    # In a real app, validate the token and return a unique user ID
    # For now, we'll use a dummy check
    if not token or not token.startswith("valid-token-"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token")
    user_id = token.split("-")[-1] # e.g., "user1" from "valid-token-user1"
    if not user_id:
         raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not extract user ID from token")
    logging.info(f"Token verified for user: {user_id}")
    return user_id

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
        # db.initialize_schema_if_needed() # Add a method like this to your Database class
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
