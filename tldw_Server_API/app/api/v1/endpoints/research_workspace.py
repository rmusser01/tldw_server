from __future__ import annotations

from fastapi import APIRouter, Depends

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    RequirePermission,
    User,
    get_request_user,
    rbac_rate_limit,
)
from tldw_Server_API.app.api.v1.schemas.research_workspace_capabilities import (
    ResearchWorkspaceCapabilitiesResponse,
)
from tldw_Server_API.app.core.AuthNZ.permissions import MEDIA_READ
from tldw_Server_API.app.core.Research_Workspace.capabilities import (
    collect_research_workspace_capabilities,
)

router = APIRouter(prefix="/research-workspace", tags=["research-workspace"])


@router.get(
    "/capabilities",
    response_model=ResearchWorkspaceCapabilitiesResponse,
    dependencies=[
        Depends(RequirePermission(MEDIA_READ)),
        Depends(rbac_rate_limit("research_workspace.capabilities")),
    ],
)
async def research_workspace_capabilities(
    current_user: User = Depends(get_request_user),
) -> ResearchWorkspaceCapabilitiesResponse:
    """Return user-safe Research Workspace capability readiness."""
    return await collect_research_workspace_capabilities(user_id=current_user.id)
