"""
FastAPI dependencies for Watchlists database access (per-user Media DB).
"""

from fastapi import Depends, HTTPException, status
from loguru import logger

from tldw_Server_API.app.api.v1.utils.http_errors import map_db_error_to_http
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError
from tldw_Server_API.app.core.DB_Management.Watchlists_DB import WatchlistsDatabase


async def get_watchlists_db_for_user(
    current_user: User = Depends(get_request_user)
) -> WatchlistsDatabase:
    if not current_user or current_user.id is None:
        logger.error("get_watchlists_db_for_user called without a valid User")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="User identification failed")
    try:
        db = WatchlistsDatabase.for_user(user_id=current_user.id)
        # Defensive: ensure schema exists for this user's DB in test/minimal app contexts
        try:
            db.ensure_schema()
        except Exception as schema_error:
            # Best-effort; creation may have already occurred or be gated by init
            logger.debug(
                "Watchlists DB schema ensure failed in dependency setup: error_type={}",
                type(schema_error).__name__,
            )
        return db
    except DatabaseError as e:
        logger.error(
            "Failed to init Watchlists DB for user {}: error_type={}",
            current_user.id,
            type(e).__name__,
        )
        raise map_db_error_to_http(e, default_detail="Watchlists DB unavailable") from e
    except Exception as e:
        logger.error(
            "Failed to init Watchlists DB for user {}: error_type={}",
            current_user.id,
            type(e).__name__,
        )
        raise HTTPException(status_code=500, detail="Watchlists DB unavailable") from e
