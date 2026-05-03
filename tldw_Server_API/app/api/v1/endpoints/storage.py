"""Storage management API endpoints.

Provides endpoints for:
- Generated files listing, download, delete
- Virtual folder management
- Storage usage statistics
- Trash operations
- Admin quota management
"""
from __future__ import annotations

from fastapi import APIRouter

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user  # noqa: F401
from tldw_Server_API.app.api.v1.endpoints import (
    storage_admin_quotas,
    storage_download,
    storage_trash,
    storage_usage,
    storage_user_files,
    storage_user_folders,
)
from tldw_Server_API.app.api.v1.endpoints.storage_admin_quotas import (  # noqa: F401
    get_org_quota,
    get_team_quota,
    require_storage_admin,
    set_org_quota,
    set_team_quota,
    set_user_quota,
)
from tldw_Server_API.app.api.v1.endpoints.storage_download import (  # noqa: F401
    download_file,
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
router.include_router(storage_download.router)
