"""User-owned storage trash routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_request_user
from tldw_Server_API.app.api.v1.endpoints._pagination_utils import build_offset_pagination_meta
from tldw_Server_API.app.api.v1.endpoints.storage_helpers import _to_generated_file
from tldw_Server_API.app.api.v1.schemas.storage_schemas import (
    PermanentDeleteResponse,
    RestoreResponse,
    TrashListResponse,
)
from tldw_Server_API.app.services.storage_quota_service import StorageQuotaService

router = APIRouter()


async def _get_storage_service() -> StorageQuotaService:
    """Resolve the storage service through storage.py for legacy monkeypatch seams."""
    from tldw_Server_API.app.api.v1.endpoints import storage as storage_endpoint

    return await storage_endpoint._get_service()


@router.get("/trash", response_model=TrashListResponse)
async def list_trashed_files(
    user: User = Depends(get_request_user),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    service: StorageQuotaService = Depends(_get_storage_service),
) -> TrashListResponse:
    """List files in trash."""
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
    service: StorageQuotaService = Depends(_get_storage_service),
) -> RestoreResponse:
    """Restore a file from trash."""
    files_repo = await service.get_generated_files_repo()

    file_record = await files_repo.get_file_by_id(file_id)
    if not file_record:
        raise HTTPException(status_code=404, detail="File not found")

    # Verify ownership
    if file_record.get("user_id") != user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    if not file_record.get("is_deleted"):
        raise HTTPException(status_code=400, detail="File is not in trash")

    restored = await service.restore_generated_file(file_id, file_record=file_record)
    if restored:
        return RestoreResponse(success=True, file=_to_generated_file(restored))

    return RestoreResponse(success=False, file=None)


@router.delete("/trash/{file_id}", response_model=PermanentDeleteResponse)
async def permanently_delete_file(
    file_id: int,
    user: User = Depends(get_request_user),
    service: StorageQuotaService = Depends(_get_storage_service),
) -> PermanentDeleteResponse:
    """Permanently delete a file from trash."""
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
