"""User-owned storage file routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_request_user
from tldw_Server_API.app.api.v1.endpoints._pagination_utils import build_offset_pagination_meta
from tldw_Server_API.app.api.v1.endpoints.storage_helpers import _to_generated_file
from tldw_Server_API.app.api.v1.schemas.storage_schemas import (
    BulkDeleteRequest,
    BulkDeleteResponse,
    BulkMoveRequest,
    BulkMoveResponse,
    FileCategory,
    GeneratedFileResponse,
    GeneratedFilesListResponse,
    GeneratedFileUpdate,
    SourceFeature,
)

router = APIRouter()


async def _get_storage_service():
    """Resolve the storage service through storage.py for legacy monkeypatch seams."""
    from tldw_Server_API.app.api.v1.endpoints import storage as storage_endpoint

    return await storage_endpoint._get_service()


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
    service = await _get_storage_service()
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
    service = await _get_storage_service()
    files_repo = await service.get_generated_files_repo()

    files = await files_repo.list_least_accessed(user.id, limit=limit)

    return GeneratedFilesListResponse(
        files=[_to_generated_file(f) for f in files],
        total=len(files),
        offset=0,
        limit=limit,
        pagination=build_offset_pagination_meta(
            total=len(files),
            offset=0,
            limit=limit,
            count=len(files),
        ),
    )


@router.post("/files/bulk-delete", response_model=BulkDeleteResponse)
async def bulk_delete_files(
    request: BulkDeleteRequest,
    user: User = Depends(get_request_user),
):
    """Bulk delete multiple files."""
    service = await _get_storage_service()
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
    service = await _get_storage_service()
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


@router.get("/files/{file_id}", response_model=GeneratedFileResponse)
async def get_file(
    file_id: int,
    user: User = Depends(get_request_user),
):
    """Get metadata for a specific file."""
    service = await _get_storage_service()
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
    service = await _get_storage_service()
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
    service = await _get_storage_service()
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
