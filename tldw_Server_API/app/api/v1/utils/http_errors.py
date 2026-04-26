"""
HTTP error mapping helpers for API v1.

This module centralizes translation of internal exceptions
(especially database-related ones) into FastAPI HTTPException
instances with consistent status codes and messages.

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

from tldw_Server_API.app.core.DB_Management.media_db.errors import (
    ConflictError,
    DatabaseError,
    InputError,
    SchemaError,
)


def map_db_error_to_http(
    exc: Exception,
    *,
    input_status: int | None = None,
    not_found_status: int | None = None,
    default_detail: str = "Database error occurred",
    input_detail: str | None = None,
    conflict_detail: str | None = None,
    log_context: str | None = None,
) -> HTTPException:
    """
    Map a database-layer exception to a FastAPI HTTPException.

    Mapping rules (aligned with Media refactor PRD):
    - InputError       -> `input_status` (defaults to 400 Bad Request)
    - ConflictError    -> 409 Conflict
    - SchemaError      -> 500 Internal Server Error (schema/migration issue)
    - DatabaseError    -> 500 Internal Server Error
    - other Exception  -> 500 Internal Server Error

    `input_detail` and `conflict_detail` let call sites keep stable,
    endpoint-specific client messages instead of exposing raw DB exception
    strings. If omitted, the exception message is preserved for compatibility,
    with a safe generic fallback for empty messages. `log_context` lets callers
    preserve request identifiers such as `media_id` in server-side logs.
    """

    def _log_db_mapping_error(label: str) -> None:
        prefix = f"{log_context}: " if log_context else ""
        logger.error(f"{prefix}{label}: {exc}", exc_info=True)

    resolved_input_status = (
        input_status
        if input_status is not None
        else (not_found_status if not_found_status is not None else status.HTTP_400_BAD_REQUEST)
    )

    if isinstance(exc, InputError):
        if resolved_input_status >= status.HTTP_500_INTERNAL_SERVER_ERROR:
            _log_db_mapping_error("InputError from DB layer")
        return HTTPException(
            status_code=resolved_input_status,
            detail=input_detail if input_detail is not None else str(exc) or "Invalid input",
        )
    if isinstance(exc, ConflictError):
        return HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(conflict_detail if conflict_detail is not None else str(exc) or "Conflict detected"),
        )
    if isinstance(exc, SchemaError):
        # Schema issues are serious; log with stack trace.
        _log_db_mapping_error("SchemaError from DB layer")
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database schema error",
        )
    if isinstance(exc, DatabaseError):
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
