"""Route-scoped typed error mapping for recipient shared workspaces."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from starlette.responses import JSONResponse

from tldw_Server_API.app.api.v1.schemas.shared_workspace_recipient_schemas import (
    SharedWorkspaceErrorDetail,
)

_ERRORS: dict[str, dict[str, Any]] = {
    "authentication_required": {
        "message": "Authentication is required.",
        "retryable": False,
    },
    "sharing_permission_required": {
        "message": "The sharing.read permission is required.",
        "retryable": False,
    },
    "invalid_shared_workspace_request": {
        "message": "The shared workspace request is invalid.",
        "retryable": False,
    },
    "invalid_shared_chat_request": {
        "message": "The shared chat request is invalid.",
        "retryable": False,
    },
    "shared_workspace_not_found": {
        "message": "Shared workspace not found.",
        "retryable": False,
    },
    "shared_workspace_unavailable": {
        "message": "Shared workspace is temporarily unavailable.",
        "retryable": True,
        "recovery_action": "retry",
    },
    "shared_workspace_rate_limited": {
        "message": "Shared workspace requests are temporarily rate limited.",
        "retryable": True,
        "recovery_action": "retry",
    },
    "shared_chat_rate_limited": {
        "message": "Shared chat requests are temporarily rate limited.",
        "retryable": True,
        "recovery_action": "retry",
    },
    "request_in_progress": {
        "message": "This question is still processing.",
        "retryable": True,
        "recovery_action": "retry",
    },
    "request_id_conflict": {
        "message": "This request ID was already used for another question.",
        "retryable": False,
    },
    "source_subset_required": {
        "message": "Select a smaller set of shared sources.",
        "retryable": False,
        "recovery_action": "reselect_sources",
    },
    "shared_source_changed": {
        "message": "The selected shared sources changed.",
        "retryable": False,
        "recovery_action": "refresh",
    },
    "no_relevant_evidence": {
        "message": "No relevant shared evidence was found.",
        "retryable": False,
        "recovery_action": "reselect_sources",
    },
    "shared_chat_context_too_large": {
        "message": "The shared chat question is too large for this model.",
        "retryable": False,
    },
    "retrieval_unavailable": {
        "message": "Shared workspace retrieval is temporarily unavailable.",
        "retryable": True,
        "recovery_action": "retry",
    },
    "no_provider_configured": {
        "message": "No usable generation provider is configured.",
        "retryable": False,
    },
    "generation_failed": {
        "message": "Shared workspace generation is temporarily unavailable.",
        "retryable": True,
        "recovery_action": "retry",
    },
}


def recipient_error_detail(code: str) -> dict[str, Any]:
    """Return a fresh validated detail mapping for a stable recipient error."""
    values = _ERRORS[code]
    return SharedWorkspaceErrorDetail(code=code, **values).model_dump(
        mode="json",
        exclude_none=True,
    )


class SharedWorkspaceRecipientRoute(APIRoute):
    """Map only recipient route dependency and validation errors."""

    def get_route_handler(self) -> Callable[[Request], Any]:
        original_handler = super().get_route_handler()
        is_chat_post = "POST" in self.methods and self.path.endswith("/chat")

        async def _handler(request: Request) -> Response:
            try:
                return await original_handler(request)
            except RequestValidationError:
                code = (
                    "invalid_shared_chat_request"
                    if is_chat_post
                    else "invalid_shared_workspace_request"
                )
                return JSONResponse(
                    status_code=422,
                    content={"detail": recipient_error_detail(code)},
                )
            except HTTPException as exc:
                if exc.status_code != 401:
                    raise
                return JSONResponse(
                    status_code=401,
                    content={"detail": recipient_error_detail("authentication_required")},
                    headers=exc.headers,
                )

        return _handler


__all__ = ["SharedWorkspaceRecipientRoute", "recipient_error_detail"]
