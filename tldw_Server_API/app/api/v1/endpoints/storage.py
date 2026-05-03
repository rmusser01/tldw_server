"""Storage management API endpoints.

Provides endpoints for:
- Generated files listing, download, delete
- Virtual folder management
- Storage usage statistics
- Trash operations
- Admin quota management
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_auth_principal, get_request_user
from tldw_Server_API.app.api.v1.endpoints._pagination_utils import build_offset_pagination_meta
from tldw_Server_API.app.api.v1.endpoints.storage_helpers import (
    _principal_is_storage_admin,
    _resolve_storage_base_dir,
    _to_generated_file,
    _to_quota_status,
)
from tldw_Server_API.app.api.v1.schemas.storage_schemas import (
    BulkDeleteRequest,
    BulkDeleteResponse,
    BulkMoveRequest,
    BulkMoveResponse,
    CategoryUsage,
    FileCategory,
    FolderCreateRequest,
    FolderInfo,
    FolderListResponse,
    GeneratedFileResponse,
    GeneratedFilesListResponse,
    GeneratedFileUpdate,
    OrgQuotaResponse,
    PermanentDeleteResponse,
    QuotaStatus,
    RestoreResponse,
    SetQuotaRequest,
    SetQuotaResponse,
    SourceFeature,
    StorageUsage,
    StorageUsageResponse,
    TeamQuotaResponse,
    TrashListResponse,
    UsageBreakdownResponse,
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

@router.get("/files", response_model=GeneratedFilesListResponse)
async def list_files(
    user: User = Depends(get_request_user),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    file_category: FileCategory | None = Query(default=None),
    source_feature: SourceFeature | None = Query(default=None),
    folder_tag: str | None = Query(default=None),
    search: str | None = Query(default=None, max_length=100),
    include_deleted: bool = Query(default=False),
):
    """
    List generated files for the current user.

    Supports filtering by category, source feature, folder, and search term.
    """
    service = await _get_service()
    files_repo = await service.get_generated_files_repo()

    files, total = await files_repo.list_files(
        user_id=user.id,
        offset=offset,
        limit=limit,
        file_category=file_category,
        source_feature=source_feature,
        folder_tag=folder_tag,
        search=search,
        include_deleted=include_deleted,
    )

    return GeneratedFilesListResponse(
        files=[_to_generated_file(f) for f in files],
        total=total,
        offset=offset,
        limit=limit,
        pagination=build_offset_pagination_meta(
            total=total,
            offset=offset,
            limit=limit,
            count=len(files),
        ),
    )


@router.get("/files/{file_id}", response_model=GeneratedFileResponse)
async def get_file(
    file_id: int,
    user: User = Depends(get_request_user),
):
    """Get metadata for a specific file."""
    service = await _get_service()
    files_repo = await service.get_generated_files_repo()

    file_record = await files_repo.get_file_by_id(file_id)
    if not file_record:
        raise HTTPException(status_code=404, detail="File not found")

    # Verify ownership
    if file_record.get("user_id") != user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Update accessed_at
    await files_repo.update_accessed_at(file_id)

    return GeneratedFileResponse(file=_to_generated_file(file_record))


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


@router.delete("/files/{file_id}")
async def delete_file(
    file_id: int,
    user: User = Depends(get_request_user),
    hard_delete: bool = Query(default=False),
):
    """
    Delete a generated file.

    By default, performs a soft delete (moves to trash).
    Use hard_delete=true for permanent deletion.
    """
    service = await _get_service()
    files_repo = await service.get_generated_files_repo()

    file_record = await files_repo.get_file_by_id(file_id)
    if not file_record:
        raise HTTPException(status_code=404, detail="File not found")

    # Verify ownership
    if file_record.get("user_id") != user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    success = await service.unregister_generated_file(file_id, hard_delete=hard_delete)

    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete file")

    return {"success": True, "file_id": file_id, "hard_delete": hard_delete}


@router.patch("/files/{file_id}", response_model=GeneratedFileResponse)
async def update_file(
    file_id: int,
    update: GeneratedFileUpdate,
    user: User = Depends(get_request_user),
):
    """Update file metadata (folder, tags, retention)."""
    service = await _get_service()
    files_repo = await service.get_generated_files_repo()

    file_record = await files_repo.get_file_by_id(file_id)
    if not file_record:
        raise HTTPException(status_code=404, detail="File not found")

    # Verify ownership
    if file_record.get("user_id") != user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    updated = await files_repo.update_file(
        file_id,
        folder_tag=update.folder_tag,
        tags=update.tags,
        retention_policy=update.retention_policy,
        expires_at=update.expires_at,
    )

    return GeneratedFileResponse(file=_to_generated_file(updated or file_record))


# =========================================================================
# Bulk Operations
# =========================================================================

@router.post("/files/bulk-delete", response_model=BulkDeleteResponse)
async def bulk_delete_files(
    request: BulkDeleteRequest,
    user: User = Depends(get_request_user),
):
    """Bulk delete multiple files."""
    service = await _get_service()
    files_repo = await service.get_generated_files_repo()

    # Verify ownership for all files
    for file_id in request.file_ids:
        file_record = await files_repo.get_file_by_id(file_id)
        if file_record and file_record.get("user_id") != user.id:
            raise HTTPException(status_code=403, detail=f"Access denied for file {file_id}")

    if request.hard_delete:
        # Hard delete each file with usage tracking
        deleted_count = 0
        for file_id in request.file_ids:
            if await service.unregister_generated_file(file_id, hard_delete=True):
                deleted_count += 1
    else:
        # Soft delete each file with usage tracking (not bulk, to update quotas properly)
        deleted_count = 0
        for file_id in request.file_ids:
            if await service.unregister_generated_file(file_id, hard_delete=False):
                deleted_count += 1

    return BulkDeleteResponse(
        deleted_count=deleted_count,
        file_ids=request.file_ids,
    )


@router.post("/files/bulk-move", response_model=BulkMoveResponse)
async def bulk_move_files(
    request: BulkMoveRequest,
    user: User = Depends(get_request_user),
):
    """Move multiple files to a folder."""
    service = await _get_service()
    files_repo = await service.get_generated_files_repo()

    # Verify ownership for all files
    for file_id in request.file_ids:
        file_record = await files_repo.get_file_by_id(file_id)
        if file_record and file_record.get("user_id") != user.id:
            raise HTTPException(status_code=403, detail=f"Access denied for file {file_id}")

    moved_count = await files_repo.bulk_move_to_folder(request.file_ids, request.folder_tag)

    return BulkMoveResponse(
        moved_count=moved_count,
        file_ids=request.file_ids,
        folder_tag=request.folder_tag,
    )


# =========================================================================
# Folder Endpoints
# =========================================================================

@router.get("/folders", response_model=FolderListResponse)
async def list_folders(
    user: User = Depends(get_request_user),
):
    """List virtual folders for the current user."""
    service = await _get_service()
    folders = await service.get_user_folders(user.id)

    return FolderListResponse(
        folders=[
            FolderInfo(
                folder_tag=f["folder_tag"],
                file_count=f["file_count"],
                total_bytes=f["total_bytes"],
                total_mb=round(f["total_bytes"] / (1024 * 1024), 2),
            )
            for f in folders
        ]
    )


@router.post("/folders")
async def create_folder(
    request: FolderCreateRequest,
    user: User = Depends(get_request_user),
):
    """
    Create a virtual folder.

    Note: Folders are virtual (tag-based). This endpoint validates the name
    but the folder only exists when files are assigned to it.
    """
    # Validate folder name
    name = request.name.strip()
    if not name or "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail="Invalid folder name")

    return {"success": True, "folder_tag": name, "message": "Folder created (virtual)"}


@router.get("/files/least-accessed", response_model=GeneratedFilesListResponse)
async def list_least_accessed_files(
    user: User = Depends(get_request_user),
    limit: int = Query(default=20, ge=1, le=100),
):
    """
    List least recently accessed files (candidates for cleanup).

    Useful for users approaching quota limits who need to free up space.
    Returns files sorted by access time (oldest first), with never-accessed
    files sorted by creation time.
    """
    service = await _get_service()
    files_repo = await service.get_generated_files_repo()

    files = await files_repo.list_least_accessed(user.id, limit=limit)

    return GeneratedFilesListResponse(
        files=[_to_generated_file(f) for f in files],
        total=len(files),
        offset=0,
        limit=limit,
    )


# =========================================================================
# Usage Endpoints
# =========================================================================

@router.get("/usage", response_model=StorageUsageResponse)
async def get_storage_usage(
    user: User = Depends(get_request_user),
):
    """Get storage usage summary for the current user."""
    service = await _get_service()
    usage_data = await service.get_user_generated_files_usage(user.id)

    # Build category usage
    by_category = {}
    for cat, data in usage_data.get("by_category", {}).items():
        by_category[cat] = CategoryUsage(
            file_count=data.get("file_count", 0),
            total_bytes=data.get("total_bytes", 0),
            total_mb=round(data.get("total_bytes", 0) / (1024 * 1024), 2),
        )

    usage = StorageUsage(
        total_bytes=usage_data.get("total_bytes", 0),
        total_mb=usage_data.get("total_mb", 0.0),
        by_category=by_category,
        trash_bytes=usage_data.get("trash_bytes", 0),
        trash_mb=usage_data.get("trash_mb", 0.0),
    )

    quota_mb = usage_data.get("quota_mb", 0)
    quota_used_mb = usage_data.get("quota_used_mb", 0.0)
    available_mb = max(0, quota_mb - quota_used_mb) if quota_mb else None

    # Calculate limit status
    usage_pct = (quota_used_mb / quota_mb * 100) if quota_mb else 0
    at_soft_limit = usage_pct >= 80
    at_hard_limit = usage_pct >= 100
    warning_message = None
    if at_hard_limit:
        warning_message = "Storage quota exceeded - delete files to continue"
    elif at_soft_limit:
        warning_message = "Approaching storage limit (80%+)"

    return StorageUsageResponse(
        usage=usage,
        quota_mb=quota_mb if quota_mb else None,
        quota_used_mb=quota_used_mb if quota_used_mb else None,
        available_mb=available_mb,
        usage_percentage=round(usage_pct, 1),
        at_soft_limit=at_soft_limit,
        at_hard_limit=at_hard_limit,
        warning=warning_message,
    )


@router.get("/usage/breakdown", response_model=UsageBreakdownResponse)
async def get_usage_breakdown(
    user: User = Depends(get_request_user),
):
    """Get detailed storage usage breakdown."""
    service = await _get_service()

    usage_data = await service.get_user_generated_files_usage(user.id)
    folders = await service.get_user_folders(user.id)

    # Build category usage
    by_category = {}
    for cat, data in usage_data.get("by_category", {}).items():
        by_category[cat] = CategoryUsage(
            file_count=data.get("file_count", 0),
            total_bytes=data.get("total_bytes", 0),
            total_mb=round(data.get("total_bytes", 0) / (1024 * 1024), 2),
        )

    quota_mb = usage_data.get("quota_mb", 0) or 0
    total_mb = usage_data.get("total_mb", 0.0)

    return UsageBreakdownResponse(
        user_id=user.id,
        by_category=by_category,
        by_folder=[
            FolderInfo(
                folder_tag=f["folder_tag"],
                file_count=f["file_count"],
                total_bytes=f["total_bytes"],
                total_mb=round(f["total_bytes"] / (1024 * 1024), 2),
            )
            for f in folders
        ],
        total_bytes=usage_data.get("total_bytes", 0),
        total_mb=total_mb,
        quota_mb=quota_mb,
        available_mb=max(0, quota_mb - total_mb),
        usage_percentage=round((total_mb / quota_mb * 100) if quota_mb else 0, 1),
    )


# =========================================================================
# Trash Endpoints
# =========================================================================

@router.get("/trash", response_model=TrashListResponse)
async def list_trashed_files(
    user: User = Depends(get_request_user),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
):
    """List files in trash."""
    service = await _get_service()
    files_repo = await service.get_generated_files_repo()

    files, total = await files_repo.list_trashed_files(
        user.id,
        offset=offset,
        limit=limit,
    )

    return TrashListResponse(
        files=[_to_generated_file(f) for f in files],
        total=total,
        offset=offset,
        limit=limit,
        pagination=build_offset_pagination_meta(
            total=total,
            offset=offset,
            limit=limit,
            count=len(files),
        ),
    )


@router.post("/trash/restore/{file_id}", response_model=RestoreResponse)
async def restore_file(
    file_id: int,
    user: User = Depends(get_request_user),
):
    """Restore a file from trash."""
    service = await _get_service()
    files_repo = await service.get_generated_files_repo()

    file_record = await files_repo.get_file_by_id(file_id)
    if not file_record:
        raise HTTPException(status_code=404, detail="File not found")

    # Verify ownership
    if file_record.get("user_id") != user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    if not file_record.get("is_deleted"):
        raise HTTPException(status_code=400, detail="File is not in trash")

    success = await files_repo.restore_file(file_id)

    if success:
        # Re-add to usage counters
        file_size = file_record.get("file_size_bytes", 0)
        await service.update_usage(user.id, file_size, operation="add")

        if file_record.get("org_id"):
            await service.update_org_usage(file_record["org_id"], file_size)
        if file_record.get("team_id"):
            await service.update_team_usage(file_record["team_id"], file_size)

        updated = await files_repo.get_file_by_id(file_id)
        return RestoreResponse(success=True, file=_to_generated_file(updated or file_record))

    return RestoreResponse(success=False, file=None)


@router.delete("/trash/{file_id}", response_model=PermanentDeleteResponse)
async def permanently_delete_file(
    file_id: int,
    user: User = Depends(get_request_user),
):
    """Permanently delete a file from trash."""
    service = await _get_service()
    files_repo = await service.get_generated_files_repo()

    file_record = await files_repo.get_file_by_id(file_id)
    if not file_record:
        raise HTTPException(status_code=404, detail="File not found")

    # Verify ownership
    if file_record.get("user_id") != user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    if not file_record.get("is_deleted"):
        raise HTTPException(status_code=400, detail="File is not in trash")

    success = await files_repo.hard_delete_file(file_id)

    return PermanentDeleteResponse(success=success, file_id=file_id)


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
