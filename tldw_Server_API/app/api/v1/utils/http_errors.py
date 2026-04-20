"""
HTTP error mapping helpers for API v1.

This module centralizes translation of internal exceptions
(especially database-related ones) into FastAPI HTTPException
instances with consistent status codes and messages.

Handles both the unified hierarchy (db_errors.py) and legacy
module-specific hierarchies (media_db, Kanban_DB, ChaChaNotes_DB, etc.)
via isinstance checks that work across inheritance chains.

Usage (see media/item.py for a full exemplar)::

    from tldw_Server_API.app.api.v1.utils.http_errors import map_db_error_to_http

    try:
        result = db.some_operation(...)
    except (ConflictError, InputError, DatabaseError) as exc:
        raise map_db_error_to_http(exc, default_detail="...context...") from exc
"""

from __future__ import annotations

from fastapi import HTTPException, status
from loguru import logger

# Unified hierarchy (preferred — all DB modules should migrate to these)
from tldw_Server_API.app.core.DB_Management.db_errors import (
    ConflictError as UnifiedConflictError,
    DatabaseError as UnifiedDatabaseError,
    InputError as UnifiedInputError,
    NotFoundError as UnifiedNotFoundError,
    SchemaError as UnifiedSchemaError,
)

# Legacy media_db hierarchy (backward compat — checked after unified)
from tldw_Server_API.app.core.DB_Management.media_db.errors import (
    ConflictError,
    DatabaseError,
    InputError,
    SchemaError,
)

# All InputError-like types across DB modules (inherit from ValueError)
_INPUT_ERROR_TYPES: tuple[type, ...] = (UnifiedInputError, InputError)

# All ConflictError-like types
_CONFLICT_ERROR_TYPES: tuple[type, ...] = (UnifiedConflictError, ConflictError)

# All SchemaError-like types
_SCHEMA_ERROR_TYPES: tuple[type, ...] = (UnifiedSchemaError, SchemaError)

# All NotFoundError-like types
_NOT_FOUND_ERROR_TYPES: tuple[type, ...] = (UnifiedNotFoundError,)

# All DatabaseError-like base types (catch-all for DB layer)
_DATABASE_ERROR_TYPES: tuple[type, ...] = (UnifiedDatabaseError, DatabaseError)

# Extend with module-specific types that don't inherit from the unified base yet
try:
    from tldw_Server_API.app.core.DB_Management.Kanban_DB import (
        ConflictError as KanbanConflictError,
        InputError as KanbanInputError,
        KanbanDBError,
        NotFoundError as KanbanNotFoundError,
    )

    _INPUT_ERROR_TYPES = (*_INPUT_ERROR_TYPES, KanbanInputError)
    _CONFLICT_ERROR_TYPES = (*_CONFLICT_ERROR_TYPES, KanbanConflictError)
    _NOT_FOUND_ERROR_TYPES = (*_NOT_FOUND_ERROR_TYPES, KanbanNotFoundError)
    _DATABASE_ERROR_TYPES = (*_DATABASE_ERROR_TYPES, KanbanDBError)
except ImportError:
    pass

try:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
        CharactersRAGDBError,
        ConflictError as ChaChaConflictError,
        InputError as ChaChaInputError,
        SchemaError as ChaChaSchemaError,
    )

    _INPUT_ERROR_TYPES = (*_INPUT_ERROR_TYPES, ChaChaInputError)
    _CONFLICT_ERROR_TYPES = (*_CONFLICT_ERROR_TYPES, ChaChaConflictError)
    _SCHEMA_ERROR_TYPES = (*_SCHEMA_ERROR_TYPES, ChaChaSchemaError)
    _DATABASE_ERROR_TYPES = (*_DATABASE_ERROR_TYPES, CharactersRAGDBError)
except ImportError:
    pass

try:
    from tldw_Server_API.app.core.DB_Management.Prompts_DB import (
        ConflictError as PromptsConflictError,
        DatabaseError as PromptsDatabaseError,
        InputError as PromptsInputError,
        SchemaError as PromptsSchemaError,
    )

    _INPUT_ERROR_TYPES = (*_INPUT_ERROR_TYPES, PromptsInputError)
    _CONFLICT_ERROR_TYPES = (*_CONFLICT_ERROR_TYPES, PromptsConflictError)
    _SCHEMA_ERROR_TYPES = (*_SCHEMA_ERROR_TYPES, PromptsSchemaError)
    _DATABASE_ERROR_TYPES = (*_DATABASE_ERROR_TYPES, PromptsDatabaseError)
except ImportError:
    pass


def map_db_error_to_http(
    exc: Exception,
    *,
    default_detail: str = "Database error occurred",
) -> HTTPException:
    """Map a database-layer exception to a FastAPI HTTPException.

    Handles all DB module exception hierarchies (unified, media_db,
    Kanban_DB, ChaChaNotes_DB, Prompts_DB) via consolidated type tuples.

    Mapping rules:
    - InputError-like    -> 400 Bad Request
    - NotFoundError-like -> 404 Not Found
    - ConflictError-like -> 409 Conflict
    - SchemaError-like   -> 500 Internal Server Error (with logging)
    - DatabaseError-like -> 500 Internal Server Error
    - other Exception    -> 500 Internal Server Error
    """
    if isinstance(exc, _INPUT_ERROR_TYPES):
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc) or "Invalid input",
        )
    if isinstance(exc, _NOT_FOUND_ERROR_TYPES):
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc) or "Resource not found",
        )
    if isinstance(exc, _CONFLICT_ERROR_TYPES):
        return HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc) or "Conflict detected",
        )
    if isinstance(exc, _SCHEMA_ERROR_TYPES):
        logger.error(f"SchemaError from DB layer: {exc}", exc_info=True)
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database schema error",
        )
    if isinstance(exc, _DATABASE_ERROR_TYPES):
        logger.error(f"DatabaseError from DB layer: {exc}", exc_info=True)
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=default_detail,
        )

    # Fallback for unexpected errors.
    logger.error(f"Unexpected exception mapped to HTTP 500: {exc}", exc_info=True)
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Internal server error",
    )


__all__ = ["map_db_error_to_http"]
