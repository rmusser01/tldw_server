"""Which sign-in session, if any, is this request? (TASK-15649)

Only a browser sign-in (multi-user access JWT with a ``session_id`` claim, or a single-user
cookie session) is an interactive session. API keys, virtual keys, service tokens and
impersonation tokens are not.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import HTTPException, Request, status

from tldw_Server_API.app.core.AuthNZ.auth_principal_resolver import bearer_credential
from tldw_Server_API.app.core.AuthNZ.exceptions import InvalidTokenError, SessionError, TokenExpiredError
from tldw_Server_API.app.core.AuthNZ.jwt_service import get_jwt_service
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.session_manager import SessionManager, get_session_manager
from tldw_Server_API.app.core.AuthNZ.single_user_session import SESSION_DEVICE_ID

INTERACTIVE_REQUIRED_CODE = "interactive_session_required"
SESSION_STORE_UNAVAILABLE_CODE = "session_store_unavailable"

# A malformed or expired token is "not interactive", never a crash.
_TOKEN_DECODE_NONCRITICAL_EXCEPTIONS = (InvalidTokenError, TokenExpiredError)


@dataclass(frozen=True)
class InteractiveSession:
    user_id: int
    session_id: int


def interactive_required_http_exception() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail={
            "code": INTERACTIVE_REQUIRED_CODE,
            "message": "This action needs a browser sign-in. API keys, virtual keys and service tokens cannot end sessions.",
        },
    )


def session_store_unavailable_http_exception() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "code": SESSION_STORE_UNAVAILABLE_CODE,
            "message": "Sessions could not be read. Try again.",
        },
    )


def install_id_or_none(device_id: str | None) -> str | None:
    """The install id behind ``device_id``; the bare single-user cookie tag is shared, so it is not one."""
    if not device_id or device_id == SESSION_DEVICE_ID:
        return None
    return device_id


def _claimed_session_id(request: Request, principal: AuthPrincipal) -> int | None:
    if principal.token_type == "single_user_session":
        claimed = getattr(request.state, "single_user_session_id", None)
    elif principal.token_type == "access":
        token = bearer_credential(request.headers.get("Authorization"))
        if not token:
            return None
        try:
            claimed = get_jwt_service().decode_access_token(token).get("session_id")
        except _TOKEN_DECODE_NONCRITICAL_EXCEPTIONS:
            return None
    else:
        return None
    return claimed if isinstance(claimed, int) and not isinstance(claimed, bool) else None


async def resolve_current_session_id(
    request: Request,
    principal: AuthPrincipal,
    session_manager: SessionManager | None = None,
) -> int | None:
    """The caller's own live session id, or None when the request is not an interactive sign-in."""
    user_id = principal.user_id
    if (
        principal.kind != "user"
        or not isinstance(user_id, int)
        or user_id <= 0
        or principal.impersonation
    ):
        return None
    claimed = _claimed_session_id(request, principal)
    if claimed is None:
        return None
    manager = session_manager or await get_session_manager()
    try:
        rows = await manager.get_user_sessions(user_id, strict=True)
    except SessionError:
        raise session_store_unavailable_http_exception() from None
    return claimed if any(row["id"] == claimed for row in rows) else None
