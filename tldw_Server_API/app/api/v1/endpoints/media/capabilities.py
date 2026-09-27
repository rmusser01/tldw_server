"""Authenticated Media affordances evaluated against canonical permissions."""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal, RequirePermission
from tldw_Server_API.app.core.AuthNZ.permissions import MEDIA_DELETE
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal

router = APIRouter()


class MediaCapabilities(BaseModel):
    """Actions this caller may attempt; item access remains enforced on writes."""
    can_delete: bool


@router.get("/capabilities", response_model=MediaCapabilities)
async def get_media_capabilities(
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> MediaCapabilities:
    """Report the caller's delete capability using the mutation's permission guard."""
    try:
        await RequirePermission(MEDIA_DELETE)(principal)
    except HTTPException as exc:
        if exc.status_code != 403:
            raise
        return MediaCapabilities(can_delete=False)
    return MediaCapabilities(can_delete=True)
