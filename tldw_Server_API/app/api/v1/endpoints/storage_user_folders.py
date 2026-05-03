"""User-owned storage folder routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_request_user
from tldw_Server_API.app.api.v1.schemas.storage_schemas import (
    FolderCreateRequest,
    FolderInfo,
    FolderListResponse,
)

router = APIRouter()


async def _get_storage_service():
    """Resolve the storage service through storage.py for legacy monkeypatch seams."""
    from tldw_Server_API.app.api.v1.endpoints import storage as storage_endpoint

    return await storage_endpoint._get_service()


@router.get("/folders", response_model=FolderListResponse)
async def list_folders(
    user: User = Depends(get_request_user),
):
    """List virtual folders for the current user."""
    service = await _get_storage_service()
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
