"""User-owned storage file routes."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_request_user
from tldw_Server_API.app.api.v1.endpoints._pagination_utils import build_offset_pagination_meta
from tldw_Server_API.app.api.v1.endpoints.storage_helpers import _normalize_folder_tag, _to_generated_file
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
    StorageDeleteResponse,
)
from tldw_Server_API.app.services.storage_quota_service import StorageQuotaService

router = APIRouter()


async def _get_storage_service() -> StorageQuotaService:
    """Resolve the storage service through storage.py for legacy monkeypatch seams."""
    from tldw_Server_API.app.api.v1.endpoints import storage as storage_endpoint

    return await storage_endpoint._get_service()


async def _get_files_by_ids(files_repo: Any, file_ids: list[int]) -> list[dict[str, Any]]:
    """Load file records with a single repo call when the backend supports it."""
    get_files_by_ids = getattr(files_repo, "get_files_by_ids", None)
    if callable(get_files_by_ids):
        return await get_files_by_ids(file_ids)

    records: list[dict[str, Any]] = []
    for file_id in file_ids:
        file_record = await files_repo.get_file_by_id(file_id)
        if file_record:
            records.append(file_record)
    return records


async def _load_owned_file_records(
    files_repo: Any,
    file_ids: list[int],
    user_id: int,
) -> list[dict[str, Any]]:
    """Load bulk file records and reject records owned by another user."""
    file_records = await _get_files_by_ids(files_repo, file_ids)
    for file_record in file_records:
        if file_record.get("user_id") != user_id:
            raise HTTPException(status_code=403, detail=f"Access denied for file {file_record.get('id')}")
    return file_records


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
    service: StorageQuotaService = Depends(_get_storage_service),
) -> GeneratedFilesListResponse:
    """
    List generated files for the current user.

    Supports filtering by category, source feature, folder, and search term.
    """
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
    service: StorageQuotaService = Depends(_get_storage_service),
) -> GeneratedFilesListResponse:
    """
    List least recently accessed files (candidates for cleanup).

    Useful for users approaching quota limits who need to free up space.
    Returns files sorted by access time (oldest first), with never-accessed
    files sorted by creation time.
    """
    files_repo = await service.get_generated_files_repo()

    files = await files_repo.list_least_accessed(user.id, limit=limit)
    count_least_accessed = getattr(files_repo, "count_least_accessed", None)
    total = await count_least_accessed(user.id) if callable(count_least_accessed) else len(files)

    return GeneratedFilesListResponse(
        files=[_to_generated_file(f) for f in files],
        total=total,
        offset=0,
        limit=limit,
        pagination=build_offset_pagination_meta(
            total=total,
            offset=0,
            limit=limit,
            count=len(files),
        ),
    )


@router.post("/files/bulk-delete", response_model=BulkDeleteResponse)
async def bulk_delete_files(
    request: BulkDeleteRequest,
    user: User = Depends(get_request_user),
    service: StorageQuotaService = Depends(_get_storage_service),
) -> BulkDeleteResponse:
    """Bulk delete multiple files."""
    files_repo = await service.get_generated_files_repo()
    file_records = await _load_owned_file_records(files_repo, request.file_ids, user.id)

    deleted_count = await service.unregister_generated_files(
        file_records,
        hard_delete=request.hard_delete,
    )

    return BulkDeleteResponse(
        deleted_count=deleted_count,
        file_ids=request.file_ids,
    )


@router.post("/files/bulk-move", response_model=BulkMoveResponse)
async def bulk_move_files(
    request: BulkMoveRequest,
    user: User = Depends(get_request_user),
    service: StorageQuotaService = Depends(_get_storage_service),
) -> BulkMoveResponse:
    """Move multiple files to a folder."""
    files_repo = await service.get_generated_files_repo()
    try:
        folder_tag = _normalize_folder_tag(request.folder_tag)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid folder name") from None

    await _load_owned_file_records(files_repo, request.file_ids, user.id)

    moved_count = await files_repo.bulk_move_to_folder(request.file_ids, folder_tag)

    return BulkMoveResponse(
        moved_count=moved_count,
        file_ids=request.file_ids,
        folder_tag=folder_tag,
    )


@router.get("/files/{file_id}", response_model=GeneratedFileResponse)
async def get_file(
    file_id: int,
    user: User = Depends(get_request_user),
    service: StorageQuotaService = Depends(_get_storage_service),
) -> GeneratedFileResponse:
    """Get metadata for a specific file."""
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


@router.delete("/files/{file_id}", response_model=StorageDeleteResponse)
async def delete_file(
    file_id: int,
    user: User = Depends(get_request_user),
    hard_delete: bool = Query(default=False),
    service: StorageQuotaService = Depends(_get_storage_service),
) -> StorageDeleteResponse:
    """
    Delete a generated file.

    By default, performs a soft delete (moves to trash).
    Use hard_delete=true for permanent deletion.
    """
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

    return StorageDeleteResponse(success=True, file_id=file_id, hard_delete=hard_delete)


@router.patch("/files/{file_id}", response_model=GeneratedFileResponse)
async def update_file(
    file_id: int,
    update: GeneratedFileUpdate,
    user: User = Depends(get_request_user),
    service: StorageQuotaService = Depends(_get_storage_service),
) -> GeneratedFileResponse:
    """Update file metadata (folder, tags, retention)."""
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
