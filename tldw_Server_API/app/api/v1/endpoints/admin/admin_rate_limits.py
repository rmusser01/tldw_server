from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from pydantic import BaseModel

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    get_auth_principal,
    get_db_transaction,
)
from tldw_Server_API.app.api.v1.schemas.admin_rbac_schemas import (
    RateLimitResponse,
    RateLimitUpsertRequest,
)
from tldw_Server_API.app.api.v1.schemas.auth_schemas import MessageResponse
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.repos import rbac_rate_limits_repo
from tldw_Server_API.app.services import admin_rate_limits_service

router = APIRouter()


_RATE_LIMITS_NONCRITICAL_EXCEPTIONS = (
    asyncio.TimeoutError,
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    UnicodeDecodeError,
    HTTPException,
)


async def _enforce_admin_user_scope(
    principal: AuthPrincipal,
    target_user_id: int,
    *,
    require_hierarchy: bool,
) -> None:
    from tldw_Server_API.app.api.v1.endpoints import admin as admin_mod

    await admin_mod._enforce_admin_user_scope(
        principal,
        target_user_id,
        require_hierarchy=require_hierarchy,
    )


def _get_is_postgres_backend_fn() -> Callable[[], Awaitable[bool]]:
    from tldw_Server_API.app.api.v1.endpoints import admin as admin_mod

    return admin_mod._is_postgres_backend


def _require_platform_admin(principal: AuthPrincipal) -> None:
    from tldw_Server_API.app.api.v1.endpoints import admin as admin_mod

    admin_mod._require_platform_admin(principal)


@router.get("/rate-limits", response_model=list[RateLimitResponse])
async def list_admin_rate_limits(db=Depends(get_db_transaction)) -> list[RateLimitResponse]:
    try:
        rows = await rbac_rate_limits_repo.list_all(db, is_postgres=await _get_is_postgres_backend_fn()())

        return [RateLimitResponse(**row) for row in rows]
    except _RATE_LIMITS_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to list admin rate limits")
        raise HTTPException(status_code=500, detail="Failed to list admin rate limits") from e


@router.post("/roles/{role_id}/rate-limits", response_model=RateLimitResponse)
async def upsert_role_rate_limit(role_id: int, payload: RateLimitUpsertRequest, db=Depends(get_db_transaction)) -> RateLimitResponse:
    try:
        await rbac_rate_limits_repo.upsert_role_limit(
            db,
            is_postgres=await _get_is_postgres_backend_fn()(),
            role_id=role_id,
            resource=payload.resource,
            limit_per_min=payload.limit_per_min,
            burst=payload.burst,
        )
        return RateLimitResponse(scope="role", id=role_id, resource=payload.resource, limit_per_min=payload.limit_per_min, burst=payload.burst)
    except _RATE_LIMITS_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to upsert role rate limit")
        raise HTTPException(status_code=500, detail="Failed to upsert role rate limit") from e


@router.delete("/roles/{role_id}/rate-limits", response_model=MessageResponse)
async def clear_role_rate_limits(role_id: int, db=Depends(get_db_transaction)) -> MessageResponse:
    try:
        await rbac_rate_limits_repo.clear_role_limits(
            db, is_postgres=await _get_is_postgres_backend_fn()(), role_id=role_id
        )
        return MessageResponse(message="Role rate limits cleared", details={"role_id": role_id})
    except _RATE_LIMITS_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to clear role rate limits")
        raise HTTPException(status_code=500, detail="Failed to clear role rate limits") from e


@router.post("/users/{user_id}/rate-limits", response_model=RateLimitResponse)
async def upsert_user_rate_limit(
    user_id: int,
    payload: RateLimitUpsertRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db=Depends(get_db_transaction),
) -> RateLimitResponse:
    try:
        await _enforce_admin_user_scope(principal, user_id, require_hierarchy=True)
        await rbac_rate_limits_repo.upsert_user_limit(
            db,
            is_postgres=await _get_is_postgres_backend_fn()(),
            user_id=user_id,
            resource=payload.resource,
            limit_per_min=payload.limit_per_min,
            burst=payload.burst,
        )
        return RateLimitResponse(scope="user", id=user_id, resource=payload.resource, limit_per_min=payload.limit_per_min, burst=payload.burst)
    except _RATE_LIMITS_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to upsert user rate limit")
        raise HTTPException(status_code=500, detail="Failed to upsert user rate limit") from e


class RateLimitSimRequest(BaseModel):
    user_id: int
    endpoint: str = ""


class RateLimitSimResponse(BaseModel):
    user_id: int
    endpoint: str
    effective_limit_per_min: int | None = None
    effective_burst: int | None = None
    limit_source: str = "none"
    would_allow: bool = True
    user_limits: list[dict[str, Any]] = []
    role_limits: list[dict[str, Any]] = []


@router.post("/debug/simulate-rate-limit", response_model=RateLimitSimResponse)
async def simulate_rate_limit(
    payload: RateLimitSimRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db=Depends(get_db_transaction),
) -> RateLimitSimResponse:
    """Simulate a rate-limit check for a given user/key and endpoint."""
    _require_platform_admin(principal)
    result = await admin_rate_limits_service.simulate_rate_limit(
        db=db,
        user_id=int(payload.user_id),
        endpoint=payload.endpoint,
        is_postgres=await _get_is_postgres_backend_fn()(),
    )
    return RateLimitSimResponse(**result)
