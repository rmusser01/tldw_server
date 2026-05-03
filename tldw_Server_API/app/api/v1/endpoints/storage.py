"""Storage management API endpoints.

Provides endpoints for:
- Generated files listing, download, delete
- Virtual folder management
- Storage usage statistics
- Trash operations
- Admin quota management
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_auth_principal, get_request_user
from tldw_Server_API.app.api.v1.endpoints import (
    storage_trash,
    storage_usage,
    storage_user_files,
    storage_user_folders,
)
from tldw_Server_API.app.api.v1.endpoints.storage_helpers import (
    _principal_is_storage_admin,
    _resolve_storage_base_dir,
    _to_quota_status,
)
# Re-export moved handlers so direct imports/tests against storage.py keep working.
from tldw_Server_API.app.api.v1.endpoints.storage_user_files import (  # noqa: F401
    bulk_delete_files,
    bulk_move_files,
    delete_file,
    get_file,
    list_files,
    list_least_accessed_files,
    update_file,
)
from tldw_Server_API.app.api.v1.endpoints.storage_user_folders import (  # noqa: F401
    create_folder,
    list_folders,
)
from tldw_Server_API.app.api.v1.endpoints.storage_trash import (  # noqa: F401
    list_trashed_files,
    permanently_delete_file,
    restore_file,
)
from tldw_Server_API.app.api.v1.endpoints.storage_usage import (  # noqa: F401
    get_storage_usage,
    get_usage_breakdown,
)
from tldw_Server_API.app.api.v1.schemas.storage_schemas import (
    OrgQuotaResponse,
    QuotaStatus,
    SetQuotaRequest,
    SetQuotaResponse,
    TeamQuotaResponse,
)
from tldw_Server_API.app.core.AuthNZ.exceptions import (
    StorageError,
    UserNotFoundError,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths as _DatabasePaths
from tldw_Server_API.app.services.storage_quota_service import get_storage_service

router = APIRouter(prefix="/storage", tags=["storage"])

# Compatibility seam for existing download tests that monkeypatch
# storage.DatabasePaths; storage_helpers imports the same class object.
DatabasePaths = _DatabasePaths


# =========================================================================
# Helper Functions
# =========================================================================

async def _get_service():
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


# =========================================================================
# File Endpoints
# =========================================================================

router.include_router(storage_user_files.router)
router.include_router(storage_user_folders.router)
router.include_router(storage_usage.router)
router.include_router(storage_trash.router)


@router.get("/files/{file_id}/download")
async def download_file(
    file_id: int,
    user: User = Depends(get_request_user),
):
    """Download a generated file."""
    service = await _get_service()
    files_repo = await service.get_generated_files_repo()

    file_record = await files_repo.get_file_by_id(file_id)
    if not file_record:
        raise HTTPException(status_code=404, detail="File not found")

    # Verify ownership
    if file_record.get("user_id") != user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    if file_record.get("is_deleted"):
        raise HTTPException(status_code=410, detail="File has been deleted")

    # Resolve file path
    storage_path = file_record.get("storage_path", "")
    base_dir = _resolve_storage_base_dir(user.id, file_record)
    full_path = base_dir / storage_path

    # Path traversal protection: ensure resolved path is within user's directory
    try:
        resolved_path = full_path.resolve()
        if not resolved_path.is_relative_to(base_dir.resolve()):
            raise HTTPException(status_code=403, detail="Invalid file path")
        full_path = resolved_path
    except ValueError:
        raise HTTPException(status_code=403, detail="Invalid file path") from None

    if not full_path.exists():
        raise HTTPException(status_code=404, detail="File not found on disk")

    # Update accessed_at
    await files_repo.update_accessed_at(file_id)

    # Return file
    filename = file_record.get("original_filename") or file_record.get("filename", "download")
    mime_type = file_record.get("mime_type") or "application/octet-stream"

    return FileResponse(
        path=str(full_path),
        filename=filename,
        media_type=mime_type,
    )

# =========================================================================
# Admin Quota Endpoints
# =========================================================================

@router.put("/admin/quotas/user/{user_id}", response_model=SetQuotaResponse)
async def set_user_quota(
    user_id: int,
    request: SetQuotaRequest,
    _principal: AuthPrincipal = Depends(require_storage_admin),
):
    """Set storage quota for a user (admin only)."""
    service = await _get_service()

    try:
        result = await service.set_user_quota(user_id, request.quota_mb)
        return SetQuotaResponse(
            success=True,
            quota=QuotaStatus(
                quota_mb=result.get("storage_quota_mb"),
                used_mb=result.get("storage_used_mb", 0.0),
                remaining_mb=result.get("available_mb", 0.0),
                usage_pct=result.get("usage_percentage", 0.0),
                at_soft_limit=result.get("usage_percentage", 0) >= request.soft_limit_pct,
                at_hard_limit=result.get("usage_percentage", 0) >= request.hard_limit_pct,
                has_quota=True,
            ),
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
):
    """Set storage quota for a team (admin only)."""
    service = await _get_service()

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
):
    """Set storage quota for an organization (admin only)."""
    service = await _get_service()

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
):
    """Get storage quota for a team (admin only)."""
    service = await _get_service()
    quota_status = await service.get_team_quota(team_id)

    return TeamQuotaResponse(
        team_id=team_id,
        quota=_to_quota_status(quota_status),
    )


@router.get("/admin/quotas/org/{org_id}", response_model=OrgQuotaResponse)
async def get_org_quota(
    org_id: int,
    _principal: AuthPrincipal = Depends(require_storage_admin),
):
    """Get storage quota for an organization (admin only)."""
    service = await _get_service()
    quota_status = await service.get_org_quota(org_id)

    return OrgQuotaResponse(
        org_id=org_id,
        quota=_to_quota_status(quota_status),
    )
