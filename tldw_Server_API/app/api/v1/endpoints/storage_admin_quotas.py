"""Storage admin quota routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.endpoints.storage_helpers import (
    _principal_is_storage_admin,
    _to_quota_status,
)
from tldw_Server_API.app.api.v1.schemas.storage_schemas import (
    OrgQuotaResponse,
    SetQuotaRequest,
    SetQuotaResponse,
    TeamQuotaResponse,
)
from tldw_Server_API.app.core.AuthNZ.exceptions import StorageError, UserNotFoundError
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.services.storage_quota_service import StorageQuotaService, get_storage_service

router = APIRouter()


async def _get_storage_service() -> StorageQuotaService:
    """Get initialized storage quota service."""
    return await get_storage_service()


async def require_storage_admin(principal: AuthPrincipal = Depends(get_auth_principal)) -> AuthPrincipal:
    """Authorize storage admin endpoints with legacy `is_admin` compatibility."""
    if _principal_is_storage_admin(principal):
        return principal
    raise HTTPException(
        status_code=403,
        detail="Access denied. Required role(s): admin",
    )


@router.put("/admin/quotas/user/{user_id}", response_model=SetQuotaResponse)
async def set_user_quota(
    user_id: int,
    request: SetQuotaRequest,
    _principal: AuthPrincipal = Depends(require_storage_admin),
) -> SetQuotaResponse:
    """Set storage quota for a user (admin only)."""
    service = await _get_storage_service()

    try:
        result = await service.set_user_quota(user_id, request.quota_mb)
        status_data = {
            "quota_mb": result.get("storage_quota_mb"),
            "used_mb": result.get("storage_used_mb", 0.0),
            "remaining_mb": result.get("available_mb", 0.0),
            "usage_pct": result.get("usage_percentage", 0.0),
            "at_soft_limit": result.get("usage_percentage", 0) >= request.soft_limit_pct,
            "at_hard_limit": result.get("usage_percentage", 0) >= request.hard_limit_pct,
            "has_quota": True,
        }
        return SetQuotaResponse(
            success=True,
            quota=_to_quota_status(status_data),
        )
    except UserNotFoundError:
        raise HTTPException(status_code=404, detail="User not found") from None
    except StorageError as e:
        raise HTTPException(status_code=500, detail="Failed to set user storage quota") from e


@router.put("/admin/quotas/team/{team_id}", response_model=SetQuotaResponse)
async def set_team_quota(
    team_id: int,
    request: SetQuotaRequest,
    _principal: AuthPrincipal = Depends(require_storage_admin),
) -> SetQuotaResponse:
    """Set storage quota for a team (admin only)."""
    service = await _get_storage_service()

    await service.set_team_quota(
        team_id,
        request.quota_mb,
        soft_limit_pct=request.soft_limit_pct,
        hard_limit_pct=request.hard_limit_pct,
    )

    quota_status = await service.get_team_quota(team_id)

    return SetQuotaResponse(
        success=True,
        quota=_to_quota_status(quota_status),
    )


@router.put("/admin/quotas/org/{org_id}", response_model=SetQuotaResponse)
async def set_org_quota(
    org_id: int,
    request: SetQuotaRequest,
    _principal: AuthPrincipal = Depends(require_storage_admin),
) -> SetQuotaResponse:
    """Set storage quota for an organization (admin only)."""
    service = await _get_storage_service()

    await service.set_org_quota(
        org_id,
        request.quota_mb,
        soft_limit_pct=request.soft_limit_pct,
        hard_limit_pct=request.hard_limit_pct,
    )

    quota_status = await service.get_org_quota(org_id)

    return SetQuotaResponse(
        success=True,
        quota=_to_quota_status(quota_status),
    )


@router.get("/admin/quotas/team/{team_id}", response_model=TeamQuotaResponse)
async def get_team_quota(
    team_id: int,
    _principal: AuthPrincipal = Depends(require_storage_admin),
) -> TeamQuotaResponse:
    """Get storage quota for a team (admin only)."""
    service = await _get_storage_service()
    quota_status = await service.get_team_quota(team_id)

    return TeamQuotaResponse(
        team_id=team_id,
        quota=_to_quota_status(quota_status),
    )


@router.get("/admin/quotas/org/{org_id}", response_model=OrgQuotaResponse)
async def get_org_quota(
    org_id: int,
    _principal: AuthPrincipal = Depends(require_storage_admin),
) -> OrgQuotaResponse:
    """Get storage quota for an organization (admin only)."""
    service = await _get_storage_service()
    quota_status = await service.get_org_quota(org_id)

    return OrgQuotaResponse(
        org_id=org_id,
        quota=_to_quota_status(quota_status),
    )
