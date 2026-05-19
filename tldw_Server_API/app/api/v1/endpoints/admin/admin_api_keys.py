"""Admin endpoints for managing user API keys and virtual keys."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Query

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    get_auth_principal,
    get_db_transaction,
)
from tldw_Server_API.app.api.v1.schemas.admin_schemas import (
    ApiKeyUsageSummary,
    ApiKeyUsageTopResponse,
    ApiKeyUsageTopItem,
)
from tldw_Server_API.app.api.v1.schemas.api_key_schemas import (
    APIKeyAuditListResponse,
    APIKeyCreateRequest,
    APIKeyCreateResponse,
    APIKeyMetadata,
    APIKeyRevokeResponse,
    APIKeyRotateRequest,
    APIKeyUpdateRequest,
)
from tldw_Server_API.app.api.v1.schemas.auth_schemas import MessageResponse
from tldw_Server_API.app.api.v1.schemas.org_team_schemas import VirtualKeyCreateRequest
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.services import admin_api_keys_service
from tldw_Server_API.app.services.admin_system_ops_service import (
    get_api_key_usage as svc_get_api_key_usage,
    list_api_key_usage as svc_list_api_key_usage,
)

router = APIRouter()


@router.get("/users/{user_id}/api-keys", response_model=list[APIKeyMetadata])
async def admin_list_user_api_keys(
    user_id: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
    include_revoked: bool = False,
) -> list[APIKeyMetadata]:
    """List API keys for a specific user (admin)."""
    return await admin_api_keys_service.list_user_api_keys(
        principal,
        user_id,
        include_revoked=include_revoked,
    )


@router.post("/users/{user_id}/api-keys", response_model=APIKeyCreateResponse)
async def admin_create_user_api_key(
    user_id: int,
    request: APIKeyCreateRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> APIKeyCreateResponse:
    """Create a new API key for the given user (admin)."""
    return await admin_api_keys_service.create_user_api_key(
        principal,
        user_id,
        request,
    )


@router.post("/users/{user_id}/api-keys/{key_id}/rotate", response_model=APIKeyCreateResponse)
async def admin_rotate_user_api_key(
    user_id: int,
    key_id: int,
    request: APIKeyRotateRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> APIKeyCreateResponse:
    """Rotate an API key for the given user and return the new key (admin)."""
    return await admin_api_keys_service.rotate_user_api_key(
        principal,
        user_id,
        key_id,
        request,
    )


@router.delete("/users/{user_id}/api-keys/{key_id}", response_model=APIKeyRevokeResponse)
async def admin_revoke_user_api_key(
    user_id: int,
    key_id: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> APIKeyRevokeResponse:
    """Revoke an API key for the given user (admin)."""
    return await admin_api_keys_service.revoke_user_api_key(
        principal,
        user_id,
        key_id,
    )


@router.patch("/users/{user_id}/api-keys/{key_id}", response_model=APIKeyMetadata)
async def admin_update_user_api_key(
    user_id: int,
    key_id: int,
    request: APIKeyUpdateRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Any = Depends(get_db_transaction),
) -> APIKeyMetadata:
    """Update per-key limits like rate_limit and allowed_ips (admin)."""
    return await admin_api_keys_service.update_user_api_key(
        principal,
        user_id,
        key_id,
        request,
        db,
        is_pg_fn=_get_is_pg_fn(),
    )


@router.post("/users/{user_id}/virtual-keys", response_model=APIKeyCreateResponse)
async def admin_create_virtual_key(
    user_id: int,
    payload: VirtualKeyCreateRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> APIKeyCreateResponse:
    """Create a virtual API key for the given user (admin)."""
    return await admin_api_keys_service.create_virtual_key(
        principal,
        user_id,
        payload,
    )


def _get_is_pg_fn():
    """Return the backend check function used for Postgres-specific logic."""
    from tldw_Server_API.app.api.v1.endpoints import admin as admin_mod

    return admin_mod._is_postgres_backend


@router.get("/users/{user_id}/virtual-keys", response_model=list[APIKeyMetadata])
async def admin_list_virtual_keys(
    user_id: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Any = Depends(get_db_transaction),
    name: str | None = Query(None, description="Filter by key name (case-insensitive substring)"),
    status_filter: str | None = Query(None, alias="status", description="Filter by key status"),
    org_id: int | None = Query(None, description="Filter by org_id"),
    team_id: int | None = Query(None, description="Filter by team_id"),
    created_after: datetime | None = Query(None, description="ISO-8601 created_at lower bound (UTC)"),
    created_before: datetime | None = Query(None, description="ISO-8601 created_at upper bound (UTC)"),
) -> list[APIKeyMetadata]:
    """List virtual API keys for a user with optional filters (admin)."""
    return await admin_api_keys_service.list_virtual_keys(
        principal,
        user_id,
        db=db,
        name=name,
        status_filter=status_filter,
        org_id=org_id,
        team_id=team_id,
        created_after=created_after,
        created_before=created_before,
        is_pg_fn=_get_is_pg_fn(),
    )


@router.get("/api-keys/usage/top", response_model=ApiKeyUsageTopResponse)
async def admin_get_top_api_key_usage(
    limit: int = Query(10, ge=1, le=100, description="Max keys to return"),
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> ApiKeyUsageTopResponse:
    """Return top API keys ranked by total token consumption (admin)."""
    items_raw = svc_list_api_key_usage(limit=limit)
    items = [
        ApiKeyUsageTopItem(
            key_id=str(item.get("key_id", "")),
            request_count=int(item.get("request_count", 0)),
            total_tokens=int(item.get("total_tokens", 0)),
            estimated_cost_usd=float(item.get("estimated_cost_usd", 0.0)),
            last_used_at=item.get("last_used_at"),
        )
        for item in items_raw
    ]
    return ApiKeyUsageTopResponse(items=items)


@router.get("/api-keys/{key_id}/usage", response_model=ApiKeyUsageSummary)
async def admin_get_api_key_usage(
    key_id: str,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> ApiKeyUsageSummary:
    """Return usage summary for a specific API key (admin)."""
    data = svc_get_api_key_usage(key_id)
    return ApiKeyUsageSummary(**data)


@router.delete("/users/{user_id}/virtual-keys/{key_id}", response_model=MessageResponse)
async def admin_delete_virtual_key(
    user_id: int,
    key_id: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Any = Depends(get_db_transaction),
) -> MessageResponse:
    """Delete (revoke) a virtual API key for the given user (admin)."""
    await admin_api_keys_service.revoke_api_key(
        principal, user_id, key_id, db=db, is_pg_fn=_get_is_pg_fn()
    )
    return MessageResponse(message=f"Virtual key {key_id} revoked")


@router.get("/api-keys/{key_id}/audit-log", response_model=APIKeyAuditListResponse)
async def admin_get_api_key_audit_log(
    key_id: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Any = Depends(get_db_transaction),
) -> APIKeyAuditListResponse:
    """Return audit log entries for a specific API key (admin)."""
    return await admin_api_keys_service.get_api_key_audit_log(
        principal,
        key_id,
        limit=limit,
        offset=offset,
        db=db,
        is_pg_fn=_get_is_pg_fn(),
    )
