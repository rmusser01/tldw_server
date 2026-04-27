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
    DataIntegrityError as UnifiedDataIntegrityError,
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

# All DataIntegrityError-like types
_DATA_INTEGRITY_ERROR_TYPES: tuple[type, ...] = (UnifiedDataIntegrityError,)

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
    input_status: int | None = None,
    not_found_status: int | None = None,
    default_detail: str = "Database error occurred",
    input_detail: str | None = None,
    conflict_detail: str | None = None,
    data_integrity_detail: str = "Data integrity violation",
    log_context: str | None = None,
) -> HTTPException:
    """Map a database-layer exception to a FastAPI HTTPException.

    Handles all DB module exception hierarchies (unified, media_db,
    Kanban_DB, ChaChaNotes_DB, Prompts_DB) via consolidated type tuples.

    Mapping rules:
    - InputError-like    -> `input_status` (defaults to 400 Bad Request)
    - NotFoundError-like -> 404 Not Found
    - DataIntegrityError -> 422 Unprocessable Entity
    - ConflictError-like -> 409 Conflict
    - SchemaError-like   -> 500 Internal Server Error (schema/migration issue)
    - DatabaseError-like -> 500 Internal Server Error
    - other Exception    -> 500 Internal Server Error

    `input_detail` and `conflict_detail` let call sites keep stable,
    endpoint-specific client messages instead of exposing raw DB exception
    strings. If omitted, the exception message is preserved for compatibility,
    with a safe generic fallback for empty messages. `log_context` lets callers
    preserve request identifiers such as `media_id` in server-side logs.
    `data_integrity_detail` provides the safe client message for 422 responses.
    """

    def _log_db_mapping_error(label: str) -> None:
        message = f"{log_context}: {label}" if log_context else label
        logger.error(message, exc_info=True)

    # Backward compatibility: some endpoints still model absence as InputError
    # and pass not_found_status=404. Real NotFoundError handling is below.
    legacy_input_fallback_status = (
        not_found_status if not_found_status is not None else status.HTTP_400_BAD_REQUEST
    )
    resolved_input_status = (
        input_status if input_status is not None else legacy_input_fallback_status
    )

    if isinstance(exc, _INPUT_ERROR_TYPES):
        if resolved_input_status >= status.HTTP_500_INTERNAL_SERVER_ERROR:
            _log_db_mapping_error("InputError from DB layer")
        return HTTPException(
            status_code=resolved_input_status,
            detail=input_detail if input_detail is not None else str(exc) or "Invalid input",
        )
    if isinstance(exc, _NOT_FOUND_ERROR_TYPES):
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Resource not found",
        )
    if isinstance(exc, _DATA_INTEGRITY_ERROR_TYPES):
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=data_integrity_detail,
        )
    if isinstance(exc, _CONFLICT_ERROR_TYPES):
        return HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(conflict_detail if conflict_detail is not None else str(exc) or "Conflict detected"),
        )
    if isinstance(exc, _SCHEMA_ERROR_TYPES):
        # Schema issues are serious; log with stack trace.
        _log_db_mapping_error("SchemaError from DB layer")
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database schema error",
        )
    if isinstance(exc, _DATABASE_ERROR_TYPES):
        _log_db_mapping_error("DatabaseError from DB layer")
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=default_detail,
        )

    # Fallback for unexpected errors.
    _log_db_mapping_error("Unexpected exception mapped to HTTP 500")
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Internal server error",
    )


__all__ = ["map_db_error_to_http"]
