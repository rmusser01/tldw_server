from __future__ import annotations

from fastapi import APIRouter, Depends

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    RequirePermission,
    User,
    get_request_user,
    rbac_rate_limit,
)
from tldw_Server_API.app.api.v1.schemas.research_studio_capabilities import (
    ResearchStudioCapabilitiesResponse,
)
from tldw_Server_API.app.core.AuthNZ.permissions import MEDIA_READ
from tldw_Server_API.app.core.Research_Studio.capabilities import (
    collect_research_studio_capabilities,
)

router = APIRouter(prefix="/research-studio", tags=["research-studio"])


@router.get(
    "/capabilities",
    response_model=ResearchStudioCapabilitiesResponse,
    dependencies=[
        Depends(RequirePermission(MEDIA_READ)),
        Depends(rbac_rate_limit("research_studio.capabilities")),
    ],
)
async def research_studio_capabilities(
    current_user: User = Depends(get_request_user),
) -> ResearchStudioCapabilitiesResponse:
    """Return user-safe Research Studio capability readiness."""
    return await collect_research_studio_capabilities(user_id=current_user.id)
