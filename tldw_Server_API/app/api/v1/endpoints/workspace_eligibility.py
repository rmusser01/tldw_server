"""Workspace active-context eligibility endpoint."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_request_user
from tldw_Server_API.app.api.v1.endpoints.workspaces_rate_limit_policy import WORKSPACES_READ_RATE_LIMIT
from tldw_Server_API.app.api.v1.schemas.workspace_schemas import (
    WorkspaceEligibilityCheckRequest,
    WorkspaceEligibilityCheckResponse,
)
from tldw_Server_API.app.api.v1.utils.http_errors import map_db_error_to_http
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)
from tldw_Server_API.app.core.Workspaces.eligibility import (
    WorkspaceEligibilityRequest,
    WorkspaceEligibilityService,
)


router = APIRouter()


@router.post(
    "/check",
    response_model=WorkspaceEligibilityCheckResponse,
    dependencies=[Depends(WORKSPACES_READ_RATE_LIMIT)],
    summary="Check whether a resource can be used in the active workspace context",
)
def check_workspace_eligibility(
    request: WorkspaceEligibilityCheckRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
) -> WorkspaceEligibilityCheckResponse:
    """Return an eligibility decision without mutating resource or workspace state."""
    _ = current_user
    try:
        result = WorkspaceEligibilityService(db).check(
            WorkspaceEligibilityRequest(
                operation=request.operation,
                active_workspace_id=request.active_workspace_id,
                resource_type=request.resource_type,
                resource_id=request.resource_id,
                runtime_state=request.runtime_state,
                permission_state=request.permission_state,
            )
        )
    except (ConflictError, InputError, CharactersRAGDBError) as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to check workspace eligibility") from exc
    return WorkspaceEligibilityCheckResponse(**result.to_dict())
