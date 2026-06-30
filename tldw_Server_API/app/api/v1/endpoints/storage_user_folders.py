"""User-owned storage folder routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_request_user
from tldw_Server_API.app.api.v1.endpoints.storage_helpers import _normalize_folder_tag
from tldw_Server_API.app.api.v1.schemas.storage_schemas import (
    FolderCreateRequest,
    FolderInfo,
    FolderListResponse,
)
from tldw_Server_API.app.services.storage_quota_service import StorageQuotaService

router = APIRouter()


async def _get_storage_service() -> StorageQuotaService:
    """Resolve the storage service through storage.py for legacy monkeypatch seams."""
    from tldw_Server_API.app.api.v1.endpoints import storage as storage_endpoint

    return await storage_endpoint._get_service()


@router.get("/folders", response_model=FolderListResponse)
async def list_folders(
    user: User = Depends(get_request_user),
    service: StorageQuotaService = Depends(_get_storage_service),
) -> FolderListResponse:
    """List virtual folders for the current user."""
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
) -> dict[str, bool | str]:
    """
    Create a virtual folder.

    Note: Folders are virtual (tag-based). This endpoint validates the name
    but the folder only exists when files are assigned to it.
    """
    try:
        name = _normalize_folder_tag(request.name)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid folder name")

    return {"success": True, "folder_tag": name, "message": "Folder created (virtual)"}
