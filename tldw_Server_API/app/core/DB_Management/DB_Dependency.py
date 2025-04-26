# DB_Dependency.py
# Description: Simple hack for FastAPI to use a custom DB object as a dependency.
#
# Imports
#
# 3rd-Party Imports
#
# Local Imports
from fastapi import Header, HTTPException
from starlette import status
#
# Local Imports
from tldw_Server_API.app.core.DB_Management.Sessions import get_user_from_token
#
########################################################################################################################
#
# Functions:

# def get_db_manager():
#     """
#     Simple FastAPI dependency that returns the custom 'db' object.
#     """
#     return db


def get_db_manager_for_user(user_id: int):
    pass


async def get_current_db_manager(token: str = Header(...)):
    """
    Dependency function to get the database manager for the current user.
    Validates the token and returns the appropriate DB manager.
    """
    user = await get_user_from_token(token)  # Validate token, get user info
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Assuming get_db_manager_for_user returns the manager object you need
    db_manager = await get_db_manager_for_user(user.id)  # Or however you identify the user's DB
    if not db_manager:
        raise HTTPException(status_code=500, detail="Could not retrieve database manager for user.")

    # You might want to handle setup/teardown here if it's a session
    # For simple manager objects, just returning might be fine.
    # If it needs cleanup (like closing a session), use yield:
    # try:
    #     yield db_manager
    # finally:
    #     await db_manager.close() # Example cleanup

    return db_manager  # Return the manager object

#
# End of DB_Dependency.py
########################################################################################################################
