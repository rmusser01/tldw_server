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

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_request_user
from tldw_Server_API.app.api.v1.endpoints import (
    storage_admin_quotas,
    storage_trash,
    storage_usage,
    storage_user_files,
    storage_user_folders,
)
from tldw_Server_API.app.api.v1.endpoints.storage_helpers import (
    _resolve_storage_base_dir,
)
from tldw_Server_API.app.api.v1.endpoints.storage_admin_quotas import (  # noqa: F401
    get_org_quota,
    get_team_quota,
    require_storage_admin,
    set_org_quota,
    set_team_quota,
    set_user_quota,
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


router.include_router(storage_user_files.router)
router.include_router(storage_user_folders.router)
router.include_router(storage_usage.router)
router.include_router(storage_trash.router)
router.include_router(storage_admin_quotas.router)


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
