"""FastAPI dependency for per-user ExplainerDatabase instances."""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path

from fastapi import Depends, HTTPException, status
from loguru import logger

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.Explainer_DB import (
    ExplainerDatabase,
    ExplainerDatabaseError,
)
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

_MAX_CACHED_EXPLAINER_DB = 20
_explainer_db_lock = threading.Lock()
_explainer_db_instances: OrderedDict[str, ExplainerDatabase] = OrderedDict()


def _get_explainer_db_path_for_user(user_id: int) -> Path:
    """Return the per-user Explainer database path."""
    return DatabasePaths.get_explainer_db_path(user_id)


def cleanup_explainer_db_cache() -> None:
    """Close all cached ExplainerDatabase connections on shutdown."""
    with _explainer_db_lock:
        for _user_id, db in list(_explainer_db_instances.items()):
            try:
                db.close_connection()
            except Exception as exc:
                logger.warning(
                    "Failed to close Explainer DB connection on shutdown; error_type={}",
                    type(exc).__name__,
                )
        _explainer_db_instances.clear()


def get_explainer_db(
    current_user: User = Depends(get_request_user),
) -> ExplainerDatabase:
    """Resolve or initialize a cached per-user ExplainerDatabase."""
    if not current_user or current_user.id is None:
        logger.error("get_explainer_db called without valid user")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="User identification failed")
    user_id = int(current_user.id)
    db_key = str(user_id)
    with _explainer_db_lock:
        db_instance = _explainer_db_instances.get(db_key)
        if db_instance is not None:
            _explainer_db_instances.move_to_end(db_key)
            return db_instance
        try:
            if len(_explainer_db_instances) >= _MAX_CACHED_EXPLAINER_DB:
                _oldest_key, oldest_db = _explainer_db_instances.popitem(last=False)
                try:
                    oldest_db.close_connection()
                except Exception as exc:
                    logger.warning(
                        "Failed to close evicted Explainer DB connection; error_type={}",
                        type(exc).__name__,
                    )
            db_path = _get_explainer_db_path_for_user(user_id)
            db_instance = ExplainerDatabase(db_path=db_path, client_id=str(user_id))
            _explainer_db_instances[db_key] = db_instance
            return db_instance
        except ExplainerDatabaseError as exc:
            logger.error("Failed to initialize Explainer DB; error_type={}", type(exc).__name__)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Explainer DB unavailable",
            ) from exc
        except Exception as exc:
            logger.error("Unexpected Explainer DB init failure; error_type={}", type(exc).__name__)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Explainer DB unavailable",
            ) from exc
