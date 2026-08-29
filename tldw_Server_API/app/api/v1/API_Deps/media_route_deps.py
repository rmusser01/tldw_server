from __future__ import annotations

from typing import Any

from fastapi import Depends

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    RequirePermission,
    rbac_rate_limit,
)
from tldw_Server_API.app.core.AuthNZ.permissions import MEDIA_CREATE


def media_create_dependencies() -> list[Any]:
    """Return the shared authorization dependencies for media processing writes."""
    return [
        Depends(RequirePermission(MEDIA_CREATE)),
        Depends(rbac_rate_limit(MEDIA_CREATE)),
    ]
