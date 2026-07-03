"""Storage file download route."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_request_user
from tldw_Server_API.app.api.v1.endpoints.storage_helpers import (
    _resolve_storage_base_dir,
)
from tldw_Server_API.app.core.Utils.path_utils import safe_join
from tldw_Server_API.app.services.storage_quota_service import StorageQuotaService, get_storage_service

router = APIRouter()


async def _get_storage_service() -> StorageQuotaService:
    """Get initialized storage quota service."""
    return await get_storage_service()


@router.get(
    "/files/{file_id}/download",
    response_class=FileResponse,
    responses={
        200: {
            "description": "Generated file download.",
            "content": {
                "application/octet-stream": {},
            },
        },
    },
)
async def download_file(
    file_id: int,
    user: User = Depends(get_request_user),
) -> FileResponse:
    """Download a generated file."""
    service = await _get_storage_service()
    files_repo = await service.get_generated_files_repo()

    file_record = await files_repo.get_file_by_id(file_id)
    if not file_record:
        raise HTTPException(status_code=404, detail="File not found")

    if file_record.get("user_id") != user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    if file_record.get("is_deleted"):
        raise HTTPException(status_code=410, detail="File has been deleted")

    storage_path = file_record.get("storage_path")
    if not isinstance(storage_path, str) or not storage_path.strip():
        raise HTTPException(status_code=404, detail="File not found")

    base_dir = _resolve_storage_base_dir(user.id, file_record)
    try:
        # lgtm[py/path-injection]: base_dir comes from the generated-file record owner scope.
        base_dir_resolved = base_dir.resolve()
        resolved_path_str = safe_join(str(base_dir_resolved), storage_path)
        if resolved_path_str is None:
            raise HTTPException(status_code=403, detail="Invalid file path")
        full_path = Path(resolved_path_str)
    except ValueError:
        raise HTTPException(status_code=403, detail="Invalid file path") from None

    # lgtm[py/path-injection]: full_path is produced by safe_join under base_dir_resolved.
    if not full_path.is_file():
        raise HTTPException(status_code=404, detail="File not found on disk")

    await files_repo.update_accessed_at(file_id)

    filename = file_record.get("original_filename") or file_record.get("filename", "download")
    mime_type = file_record.get("mime_type") or "application/octet-stream"

    # lgtm[py/path-injection] full_path is resolved by safe_join under the record's storage base dir.
    return FileResponse(
        path=str(full_path),
        filename=filename,
        media_type=mime_type,
    )
