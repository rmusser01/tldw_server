"""FastAPI dependency for per-user ExplainerDatabase instances."""

from __future__ import annotations

import threading
from collections import OrderedDict

from fastapi import Depends, HTTPException, status
from loguru import logger

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.Explainer_DB import (
    ExplainerDatabase,
    ExplainerDatabaseError,
    explainer_db_for_user,
)

_MAX_CACHED_EXPLAINER_DB = 20
_explainer_db_lock = threading.Lock()
_explainer_db_instances: OrderedDict[str, ExplainerDatabase] = OrderedDict()


def _close_quietly(db: ExplainerDatabase, context: str) -> None:
    try:
        db.close_all_connections()
    except Exception as exc:
        logger.warning(
            "Failed to close Explainer DB connections ({}); error_type={}",
            context,
            type(exc).__name__,
        )


def cleanup_explainer_db_cache() -> None:
    """Close all cached ExplainerDatabase connections on shutdown."""
    with _explainer_db_lock:
        instances = list(_explainer_db_instances.values())
        _explainer_db_instances.clear()
    for db in instances:
        _close_quietly(db, "shutdown")


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

    # Construct outside the lock: schema init touches the filesystem and
    # must not stall every other user's requests.
    try:
        new_instance = explainer_db_for_user(user_id)
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

    to_close: list[ExplainerDatabase] = []
    with _explainer_db_lock:
        existing = _explainer_db_instances.get(db_key)
        if existing is not None:
            # Lost a construction race; keep the cached instance.
            _explainer_db_instances.move_to_end(db_key)
            db_instance = existing
            to_close.append(new_instance)
        else:
            while len(_explainer_db_instances) >= _MAX_CACHED_EXPLAINER_DB:
                _oldest_key, oldest_db = _explainer_db_instances.popitem(last=False)
                to_close.append(oldest_db)
            _explainer_db_instances[db_key] = new_instance
            db_instance = new_instance

    for db in to_close:
        _close_quietly(db, "eviction")
    return db_instance
