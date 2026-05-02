"""FastAPI dependency for SlidesDatabase per user."""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path

from fastapi import Depends, HTTPException, status
from loguru import logger

from tldw_Server_API.app.api.v1.utils.http_errors import map_db_error_to_http
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Slides.slides_db import SchemaError, SlidesDatabase, SlidesDatabaseError

_MAX_CACHED_SLIDES_DB = 20
_slides_db_lock = threading.Lock()
_slides_db_instances: OrderedDict[str, SlidesDatabase] = OrderedDict()


def _get_slides_db_path_for_user(user_id: int) -> Path:
    """Return the per-user Slides database path."""
    return DatabasePaths.get_slides_db_path(user_id)


def cleanup_slides_db_cache() -> None:
    """Close all cached SlidesDatabase connections on shutdown."""
    with _slides_db_lock:
        for user_id, db in list(_slides_db_instances.items()):
            try:
                db.close_connection()
            except Exception as exc:
                logger.warning(
                    "Failed to close Slides DB connection on shutdown; error_type={}",
                    type(exc).__name__,
                )
        _slides_db_instances.clear()


def get_slides_db_for_user(
    current_user: User = Depends(get_request_user),
) -> SlidesDatabase:
    """Resolve or initialize a per-user SlidesDatabase instance; raises HTTPException on failure."""
    if not current_user or current_user.id is None:
        logger.error("get_slides_db_for_user called without valid user")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="User identification failed")
    user_id = int(current_user.id)
    db_key = str(user_id)
    with _slides_db_lock:
        db_instance = _slides_db_instances.get(db_key)
        if db_instance:
            _slides_db_instances.move_to_end(db_key)
            return db_instance
        try:
            if len(_slides_db_instances) >= _MAX_CACHED_SLIDES_DB:
                oldest_key, oldest_db = _slides_db_instances.popitem(last=False)
                try:
                    oldest_db.close_connection()
                except Exception as exc:
                    logger.warning(
                        "Failed to close evicted Slides DB connection; error_type={}",
                        type(exc).__name__,
                    )
            db_path = _get_slides_db_path_for_user(user_id)
            db_instance = SlidesDatabase(db_path=str(db_path), client_id=str(current_user.id))
            _slides_db_instances[db_key] = db_instance
            return db_instance
        except (SlidesDatabaseError, SchemaError) as exc:
            logger.error(
                "Failed to initialize Slides DB; error_type={}",
                type(exc).__name__,
            )
            raise map_db_error_to_http(exc, default_detail="Slides DB unavailable") from exc
        except Exception as exc:
            logger.error(
                "Unexpected Slides DB init failure; error_type={}",
                type(exc).__name__,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Slides DB unavailable",
            ) from exc


def try_get_slides_db_for_user(
    current_user: User = Depends(get_request_user),
) -> SlidesDatabase | None:
    """Best-effort SlidesDatabase resolver that returns None on failure."""
    try:
        return get_slides_db_for_user(current_user=current_user)
    except HTTPException as exc:
        logger.debug(
            "Slides DB unavailable during best-effort lookup; status_code={}",
            exc.status_code,
        )
        return None
    except Exception as exc:
        logger.error(
            "Unexpected Slides DB init failure during best-effort lookup; error_type={}",
            type(exc).__name__,
        )
        return None
