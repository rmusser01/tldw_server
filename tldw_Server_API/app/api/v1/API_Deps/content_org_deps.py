"""Resolve one validated organization for content, quota, and billing operations."""

from fastapi import Header, Query, Request

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.API_Deps.billing_deps import _resolve_org_id
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.DB_Management.scope_context import get_scope, set_scope


async def _activate_selected_org(
    request: Request, principal: AuthPrincipal, org_id: int | None, x_tldw_org_id: int | None
) -> int | None:
    """Apply the billing resolver's validated organization to content scope."""
    selected = await _resolve_org_id(principal, org_id, x_tldw_org_id)
    if selected is None:
        return None

    request.state.org_id = selected
    request.state.active_org_id = selected
    scope = get_scope()
    previous_org_id = scope.effective_org_id if scope is not None else principal.active_org_id
    switching_org = previous_org_id != selected
    team_ids = [] if switching_org else scope.team_ids if scope is not None else principal.team_ids
    active_team_id = None if switching_org else scope.active_team_id if scope is not None else principal.active_team_id
    if switching_org:
        request.state.team_ids = []
        request.state.team_id = None
        request.state.active_team_id = None
    set_scope(
        user_id=scope.user_id if scope is not None else principal.user_id,
        org_ids=scope.org_ids if scope is not None else principal.org_ids,
        team_ids=team_ids,
        active_org_id=selected,
        active_team_id=active_team_id,
        is_admin=scope.is_admin if scope is not None else principal.is_admin,
        session_role=scope.session_role if scope is not None else None,
    )
    return selected


async def select_content_org(
    request: Request,
    x_tldw_org_id: int | None = Header(None, alias="X-TLDW-Org-Id"),
    org_id: int | None = Query(None, description="Organization ID"),
) -> int | None:
    """Select an explicitly requested org for email search and detail."""
    if org_id is None and x_tldw_org_id is None:
        return None
    principal = await get_auth_principal(request)
    return await _activate_selected_org(request, principal, org_id, x_tldw_org_id)
