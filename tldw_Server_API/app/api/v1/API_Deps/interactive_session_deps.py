"""FastAPI dependencies over ``core.AuthNZ.interactive_session`` (TASK-15649)."""

from __future__ import annotations

from fastapi import Depends, Request

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal, get_session_manager_dep
from tldw_Server_API.app.core.AuthNZ.interactive_session import (
    InteractiveSession,
    interactive_required_http_exception,
    resolve_current_session_id,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.session_manager import SessionManager


async def get_current_session_id(
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
    session_manager: SessionManager = Depends(get_session_manager_dep),
) -> int | None:
    return await resolve_current_session_id(request, principal, session_manager)


async def require_interactive_session(
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
    session_manager: SessionManager = Depends(get_session_manager_dep),
) -> InteractiveSession:
    session_id = await resolve_current_session_id(request, principal, session_manager)
    if session_id is None:
        raise interactive_required_http_exception()
    return InteractiveSession(user_id=int(principal.user_id), session_id=session_id)
