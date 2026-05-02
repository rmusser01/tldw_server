# Collections_DB_Deps.py
# FastAPI dependencies for Collections database access (per-user)

from typing import Optional

from fastapi import Depends, HTTPException, status
from loguru import logger

from tldw_Server_API.app.api.v1.utils.http_errors import map_db_error_to_http
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase


async def get_collections_db_for_user(
    current_user: User = Depends(get_request_user)
) -> CollectionsDatabase:
    """Return a CollectionsDatabase bound to the current user's Media DB.

    Mirrors get_media_db_for_user semantics: per-user isolation, lazy schema ensure.
    """
    if not current_user or current_user.id is None:
        logger.error("get_collections_db_for_user called without a valid User.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="User identification failed.")
    try:
        return CollectionsDatabase.for_user(user_id=current_user.id)
    except DatabaseError as e:
        logger.error(
            "Failed to initialize Collections DB for user {}: error_type={}",
            current_user.id,
            type(e).__name__,
        )
        raise map_db_error_to_http(e, default_detail="Collections DB unavailable") from e
    except Exception as e:
        logger.error(
            "Failed to initialize Collections DB for user {}: error_type={}",
            current_user.id,
            type(e).__name__,
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Collections DB unavailable") from e


async def try_get_collections_db_for_user(
    current_user: User = Depends(get_request_user)
) -> Optional[CollectionsDatabase]:
    try:
        return await get_collections_db_for_user(current_user=current_user)
    except HTTPException:
        return None
    except Exception:
        return None
