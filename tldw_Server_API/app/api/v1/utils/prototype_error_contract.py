"""Shared OpenAPI and exception helpers for prototype endpoint errors."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, status

from ..schemas.prototype_workspace_schemas import (
    PrototypeErrorCategory,
    PrototypeErrorResponse,
    PrototypeFrontendState,
    prototype_error_detail,
)

_PROTOTYPE_VALIDATION_ANYOF_SCHEMA = {
    "anyOf": [
        {"$ref": "#/components/schemas/PrototypeErrorResponse"},
        {"$ref": "#/components/schemas/HTTPValidationError"},
    ],
}

_PROTOTYPE_VALIDATION_RESPONSE = {
    "description": (
        "Prototype request validation failed. FastAPI request parsing may return "
        "HTTPValidationError; domain validation returns PrototypeErrorResponse."
    ),
    "content": {
        "application/json": {
            "schema": _PROTOTYPE_VALIDATION_ANYOF_SCHEMA,
        },
    },
}

PROTOTYPE_ERROR_RESPONSE_MODELS: dict[int, dict[str, Any]] = {
    status.HTTP_403_FORBIDDEN: {
        "model": PrototypeErrorResponse,
        "description": "Prototype request is not authorized or the collaborator session is inactive.",
    },
    status.HTTP_404_NOT_FOUND: {
        "model": PrototypeErrorResponse,
        "description": "Prototype workspace, session, snapshot, promotion, or preview resource is unavailable.",
    },
    status.HTTP_409_CONFLICT: {
        "model": PrototypeErrorResponse,
        "description": "Prototype request conflicts with workspace, preview, bootstrap, or promotion state.",
    },
    status.HTTP_422_UNPROCESSABLE_ENTITY: _PROTOTYPE_VALIDATION_RESPONSE,
}

PROTOTYPE_LINK_ERROR_RESPONSES: dict[int, dict[str, Any]] = {
    status.HTTP_403_FORBIDDEN: {
        "model": PrototypeErrorResponse,
        "description": "Prototype link requires credentials or the workspace/session is unavailable.",
    },
    status.HTTP_404_NOT_FOUND: {
        "model": PrototypeErrorResponse,
        "description": "Prototype link is invalid, unavailable, exhausted, or cannot be resumed.",
    },
    status.HTTP_422_UNPROCESSABLE_ENTITY: _PROTOTYPE_VALIDATION_RESPONSE,
    status.HTTP_429_TOO_MANY_REQUESTS: {
        "description": "Prototype link public exchange rate limit exceeded.",
    },
}


def prototype_error_responses(*status_codes: int) -> dict[int, dict[str, Any]]:
    """Return OpenAPI response metadata for prototype contract errors."""
    return {status_code: PROTOTYPE_ERROR_RESPONSE_MODELS[status_code] for status_code in status_codes}


def prototype_http_error(
    *,
    status_code: int,
    category: PrototypeErrorCategory,
    message: str,
    frontend_state: PrototypeFrontendState,
    retryable: bool = False,
) -> HTTPException:
    """Build a prototype HTTPException with stable machine-readable detail."""
    return HTTPException(
        status_code=status_code,
        detail=prototype_error_detail(
            category=category,
            message=message,
            frontend_state=frontend_state,
            retryable=retryable,
        ),
    )
