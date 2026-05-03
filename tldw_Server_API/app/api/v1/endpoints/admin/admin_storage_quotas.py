"""
admin_storage_quotas.py

Admin endpoints for storage quota management.
All endpoints require admin role (enforced by the parent admin router).
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from loguru import logger
from pydantic import BaseModel, Field, model_validator

from tldw_Server_API.app.api.v1.endpoints._pagination_utils import build_offset_pagination_meta
from tldw_Server_API.app.api.v1.schemas.pagination import (
    OffsetPaginationMeta,
    default_offset_pagination_aliases,
)
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.repos.storage_quotas_repo import (
    AuthnzStorageQuotasRepo,
    DEFAULT_HARD_LIMIT_PCT,
    DEFAULT_ORG_QUOTA_MB,
    DEFAULT_SOFT_LIMIT_PCT,
    DEFAULT_TEAM_QUOTA_MB,
)

router = APIRouter(prefix="/storage-quotas", tags=["admin-storage-quotas"])

_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    ConnectionError,
    KeyError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class StorageQuotaResponse(BaseModel):
    """Storage quota information."""

    quota_mb: int | None = None
    used_mb: float = 0.0
    remaining_mb: float | None = None
    usage_pct: float = 0.0
    at_soft_limit: bool = False
    at_hard_limit: bool = False
    has_quota: bool = False


class UpdateQuotaRequest(BaseModel):
    """Request body for updating a storage quota."""

    quota_mb: int = Field(..., gt=0, description="Storage quota in megabytes")
    soft_limit_pct: int = Field(
        DEFAULT_SOFT_LIMIT_PCT,
        ge=0,
        le=100,
        description="Soft limit percentage (warning threshold)",
    )
    hard_limit_pct: int = Field(
        DEFAULT_HARD_LIMIT_PCT,
        ge=0,
        le=100,
        description="Hard limit percentage (enforcement threshold)",
    )


class StorageQuotaSummaryResponse(BaseModel):
    """Overall storage quota summary."""

    total_quotas: int
    items: list[dict[str, Any]]
    limit: int | None = Field(default=None, ge=1, description="Alias for pagination.limit")
    offset: int | None = Field(default=None, ge=0, description="Alias for pagination.offset")
    pagination: OffsetPaginationMeta
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return default_offset_pagination_aliases(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _get_repo() -> AuthnzStorageQuotasRepo:
    """Build and return a storage quotas repo."""
    pool = await get_db_pool()
    return AuthnzStorageQuotasRepo(db_pool=pool)


# ---------------------------------------------------------------------------
# User-level endpoints (org/team scope)
# ---------------------------------------------------------------------------


@router.get(
    "/users/{user_id}",
    response_model=StorageQuotaResponse,
    summary="Get user storage quota and usage",
)
async def get_user_storage_quota(user_id: int) -> StorageQuotaResponse:
    """Get a user's storage quota status.

    Currently delegates to the org-level quota that the user belongs to.
    User-level quotas are stored on the users table; this endpoint provides
    a unified view.
    """
    try:
        repo = await _get_repo()
        # User-level quotas are on the users table; for org/team quotas
        # we would need the user's org_id. For now, return org-based status.
        quota_status = await repo.check_quota_status(org_id=user_id)
        return StorageQuotaResponse(**quota_status)
    except _NONCRITICAL_EXCEPTIONS as exc:
        logger.warning("Failed to get user storage quota")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve storage quota",
        ) from exc


@router.put(
    "/users/{user_id}",
    response_model=dict[str, Any],
    summary="Update user storage quota",
)
async def update_user_storage_quota(
    user_id: int,
    body: UpdateQuotaRequest,
) -> dict[str, Any]:
    """Update a user's storage quota.

    This sets the org-level quota for the user's primary organization.
    """
    try:
        repo = await _get_repo()
        result = await repo.upsert_org_quota(
            user_id,
            quota_mb=body.quota_mb,
            soft_limit_pct=body.soft_limit_pct,
            hard_limit_pct=body.hard_limit_pct,
        )
        return result
    except _NONCRITICAL_EXCEPTIONS as exc:
        logger.warning("Failed to update user storage quota")
        raise HTTPException(
            status_code=500,
            detail="Failed to update storage quota",
        ) from exc


# ---------------------------------------------------------------------------
# Org-level endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/orgs/{org_id}",
    response_model=StorageQuotaResponse,
    summary="Get org storage quota",
)
async def get_org_storage_quota(org_id: int) -> StorageQuotaResponse:
    """Get an organization's storage quota and current usage."""
    try:
        repo = await _get_repo()
        result = await repo.check_quota_status(org_id=org_id)
        return StorageQuotaResponse(**result)
    except _NONCRITICAL_EXCEPTIONS as exc:
        logger.warning("Failed to get org storage quota")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve org storage quota",
        ) from exc


@router.put(
    "/orgs/{org_id}",
    response_model=dict[str, Any],
    summary="Update org storage quota",
)
async def update_org_storage_quota(
    org_id: int,
    body: UpdateQuotaRequest,
) -> dict[str, Any]:
    """Update an organization's storage quota."""
    try:
        repo = await _get_repo()
        result = await repo.upsert_org_quota(
            org_id,
            quota_mb=body.quota_mb,
            soft_limit_pct=body.soft_limit_pct,
            hard_limit_pct=body.hard_limit_pct,
        )
        return result
    except _NONCRITICAL_EXCEPTIONS as exc:
        logger.warning("Failed to update org storage quota")
        raise HTTPException(
            status_code=500,
            detail="Failed to update org storage quota",
        ) from exc


# ---------------------------------------------------------------------------
# Summary endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/summary",
    response_model=StorageQuotaSummaryResponse,
    summary="Overall storage usage summary",
)
async def get_storage_quota_summary(
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
) -> StorageQuotaSummaryResponse:
    """Get an overall summary of all storage quotas."""
    try:
        repo = await _get_repo()
        items = await repo.list_all_quotas(offset=offset, limit=limit)
        return StorageQuotaSummaryResponse(
            total_quotas=len(items),
            items=items,
            pagination=build_offset_pagination_meta(limit=limit, offset=offset, count=len(items)),
        )
    except _NONCRITICAL_EXCEPTIONS as exc:
        logger.warning("Failed to get storage quota summary")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve storage quota summary",
        ) from exc
