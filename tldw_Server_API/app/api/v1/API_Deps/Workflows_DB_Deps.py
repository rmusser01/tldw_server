"""FastAPI dependencies for optional Workflows database access."""
from __future__ import annotations

import sqlite3
from threading import RLock

from loguru import logger

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError as BackendDatabaseError
from tldw_Server_API.app.core.DB_Management.DB_Manager import (
    create_workflows_database,
    get_content_backend_instance,
)
from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase

_WORKFLOWS_DB_DEPENDENCY_EXCEPTIONS = (
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    sqlite3.Error,
    BackendDatabaseError,
)

_cached_workflows_db: WorkflowsDatabase | None = None
_cached_workflows_db_lock = RLock()


def _get_cached_workflows_db_for_user() -> WorkflowsDatabase:
    global _cached_workflows_db
    if _cached_workflows_db is not None:
        return _cached_workflows_db
    with _cached_workflows_db_lock:
        if _cached_workflows_db is None:
            _cached_workflows_db = create_workflows_database(backend=get_content_backend_instance())
        return _cached_workflows_db


def try_get_workflows_db_for_user() -> WorkflowsDatabase | None:
    """Optional Workflows DB dependency for routes that support mixed resource types."""
    try:
        return _get_cached_workflows_db_for_user()
    except _WORKFLOWS_DB_DEPENDENCY_EXCEPTIONS as exc:
        logger.warning("Optional Workflows DB unavailable ({})", type(exc).__name__)
        return None


def close_cached_workflows_db_for_user() -> None:
    """Close and clear the cached optional Workflows DB dependency instance."""
    global _cached_workflows_db
    with _cached_workflows_db_lock:
        db = _cached_workflows_db
        _cached_workflows_db = None
    if db is None:
        return
    close = getattr(db, "close", None) or getattr(db, "close_connection", None)
    if callable(close):
        close()
