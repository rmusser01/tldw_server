"""Workspace membership reverse lookup endpoints."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_request_user
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import try_get_media_db_for_user
from tldw_Server_API.app.api.v1.endpoints.workspaces_rate_limit_policy import WORKSPACES_READ_RATE_LIMIT
from tldw_Server_API.app.api.v1.schemas.workspace_schemas import WorkspaceResourceMembershipListResponse
from tldw_Server_API.app.api.v1.utils.http_errors import map_db_error_to_http
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)
from tldw_Server_API.app.core.Workspaces.membership_service import (
    WorkspaceMembershipService,
    WorkspaceMembershipServiceError,
)

router = APIRouter()


def _request_user_id(user: User) -> str:
    return str(getattr(user, "id", ""))


def _membership_service(db: CharactersRAGDB) -> WorkspaceMembershipService:
    return WorkspaceMembershipService(db)


def _membership_service_error_to_http(exc: WorkspaceMembershipServiceError) -> HTTPException:
    return HTTPException(
        status_code=exc.status_code,
        detail={
            "code": exc.code,
            "message": exc.message,
            "details": exc.details,
        },
    )


@router.get(
    "/resources/{resource_type}/{resource_id}",
    response_model=WorkspaceResourceMembershipListResponse,
    dependencies=[Depends(WORKSPACES_READ_RATE_LIMIT)],
    summary="List workspace memberships for a resource",
)
async def list_resource_workspace_memberships(
    resource_type: str,
    resource_id: str,
    include_deleted: bool = Query(default=False),
    resolve: bool = Query(default=True),
    limit: int = Query(default=100, ge=1, le=1000),
    cursor: str | None = Query(default=None),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    media_db: Any | None = Depends(try_get_media_db_for_user),
    current_user: User = Depends(get_request_user),
) -> WorkspaceResourceMembershipListResponse:
    """List memberships that connect the current user's workspaces to one resource."""
    try:
        payload = _membership_service(db).list_resource_memberships(
            resource_type,
            resource_id,
            include_deleted=include_deleted,
            resolve=resolve,
            limit=limit,
            cursor=cursor,
            media_db=media_db,
            user_id=_request_user_id(current_user),
        )
    except WorkspaceMembershipServiceError as exc:
        raise _membership_service_error_to_http(exc) from exc
    except (ConflictError, InputError, CharactersRAGDBError) as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to fetch resource workspace memberships") from exc
    return WorkspaceResourceMembershipListResponse(**payload)
