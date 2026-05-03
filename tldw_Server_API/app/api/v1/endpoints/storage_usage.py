"""User-owned storage usage routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_request_user
from tldw_Server_API.app.api.v1.schemas.storage_schemas import (
    CategoryUsage,
    FolderInfo,
    StorageUsage,
    StorageUsageResponse,
    UsageBreakdownResponse,
)

router = APIRouter()


async def _get_storage_service():
    """Resolve the storage service through storage.py for legacy monkeypatch seams."""
    from tldw_Server_API.app.api.v1.endpoints import storage as storage_endpoint

    return await storage_endpoint._get_service()


@router.get("/usage", response_model=StorageUsageResponse)
async def get_storage_usage(
    user: User = Depends(get_request_user),
):
    """Get storage usage summary for the current user."""
    service = await _get_storage_service()
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
    service = await _get_storage_service()

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
