from __future__ import annotations

import csv
import io
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.schemas.privileges import (
    PrivilegeDetailResponse,
    PrivilegeOrgResponse,
    PrivilegeSelfResponse,
    PrivilegeSnapshotAcceptedResponse,
    PrivilegeSnapshotCreateRequest,
    PrivilegeSnapshotDetailResponse,
    PrivilegeSnapshotListResponse,
    PrivilegeSnapshotRecord,
    PrivilegeSnapshotSummary,
)
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Logging.log_context import ensure_request_id, get_ps_logger
from tldw_Server_API.app.core.PrivilegeMaps import (
    PaginationLimitExceeded,
    PrivilegeMapService,
    PrivilegeSnapshotStore,
    get_privilege_map_service,
    get_privilege_snapshot_store,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal

router = APIRouter(prefix="/privileges", tags=["privileges"])
_ADMIN_CLAIM_PERMISSIONS = frozenset({"*", "system.configure"})


def _has_privilege_admin_claim(principal: AuthPrincipal) -> bool:
    roles = {
        str(role).strip().lower()
        for role in (principal.roles or [])
        if str(role).strip()
    }
    if roles & {"admin", "owner", "platform_admin"}:
        return True
    permissions = {
        str(permission).strip().lower()
        for permission in (principal.permissions or [])
        if str(permission).strip()
    }
    return bool(permissions & _ADMIN_CLAIM_PERMISSIONS)


async def require_privilege_admin(
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> AuthPrincipal:
    if _has_privilege_admin_claim(principal):
        return principal
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Administrator privileges required to access privilege maps.",
    )


async def require_privilege_admin_or_self(
    user_id: str,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> AuthPrincipal:
    if _has_privilege_admin_claim(principal):
        return principal
    if principal.user_id is not None and str(principal.user_id) == str(user_id):
        return principal
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Insufficient privileges to inspect requested user.",
    )


@router.get("/self", response_model=PrivilegeSelfResponse)
async def get_self_privilege_map(
    *,
    resource: str | None = Query(None),
    principal: AuthPrincipal = Depends(get_auth_principal),
    service: PrivilegeMapService = Depends(get_privilege_map_service),
) -> PrivilegeSelfResponse:
    if principal.user_id is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User principal required to inspect self privilege map.",
        )
    user_id = str(principal.user_id)
    try:
        return await service.get_self_map(user_id=user_id, resource=resource)
    except ValueError as exc:
        logger.warning("Invalid self privilege request: {}", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/org", response_model=PrivilegeOrgResponse)
async def get_org_privilege_map(
    *,
    group_by: str = Query("role", pattern="^(role|team|resource)$"),
    include_trends: bool = Query(False),
    since: datetime | None = Query(None),
    view: str = Query("summary", pattern="^(summary|detail)$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=500),
    resource: str | None = Query(None),
    role: str | None = Query(None),
    dependency: str | None = Query(None),
    org_id: str | None = Query(None, description="Filter to a specific organization"),
    principal: AuthPrincipal = Depends(require_privilege_admin),
    service: PrivilegeMapService = Depends(get_privilege_map_service),
) -> PrivilegeOrgResponse:
    del principal  # dependency enforcement only
    try:
        if view == "detail":
            return await service.get_org_detail(
                page=page,
                page_size=page_size,
                resource=resource,
                dependency=dependency,
                role_filter=role,
                org_id=org_id,
            )
        return await service.get_org_summary(
            group_by=group_by,
            include_trends=include_trends,
            since=since,
            org_id=org_id,
        )
    except PaginationLimitExceeded as exc:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)) from exc
    except ValueError as exc:
        logger.warning("Invalid privilege detail request: {}", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/teams/{team_id}", response_model=PrivilegeOrgResponse)
async def get_team_privilege_map(
    *,
    team_id: str,
    group_by: str = Query("member", pattern="^(member|resource)$"),
    view: str = Query("summary", pattern="^(summary|detail)$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=500),
    resource: str | None = Query(None),
    role: str | None = Query(None),
    dependency: str | None = Query(None),
    include_trends: bool = Query(False),
    since: datetime | None = Query(None),
    principal: AuthPrincipal = Depends(require_privilege_admin),
    service: PrivilegeMapService = Depends(get_privilege_map_service),
) -> PrivilegeOrgResponse:
    del principal
    try:
        if view == "detail":
            return await service.get_team_detail(
                team_id=team_id,
                page=page,
                page_size=page_size,
                resource=resource,
                dependency=dependency,
                role_filter=role,
            )
        return await service.get_team_summary(
            team_id=team_id,
            group_by=group_by,
            include_trends=include_trends,
            since=since,
        )
    except PaginationLimitExceeded as exc:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)) from exc
    except ValueError as exc:
        logger.warning("Invalid team privilege request: {}", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/users/{user_id}", response_model=PrivilegeDetailResponse)
async def get_user_privilege_map(
    *,
    user_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=500),
    resource: str | None = Query(None),
    principal: AuthPrincipal = Depends(require_privilege_admin_or_self),
    service: PrivilegeMapService = Depends(get_privilege_map_service),
) -> PrivilegeDetailResponse:
    del principal
    try:
        return await service.get_user_detail(
            user_id=user_id,
            page=page,
            page_size=page_size,
            resource=resource,
        )
    except PaginationLimitExceeded as exc:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)) from exc
    except ValueError as exc:
        logger.warning("Invalid user privilege request: {}", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/snapshots", response_model=PrivilegeSnapshotListResponse)
async def list_privilege_snapshots(
    *,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    date_from: datetime | None = Query(None),
    date_to: datetime | None = Query(None),
    generated_by: str | None = Query(None),
    org_id: str | None = Query(None),
    team_id: str | None = Query(None),
    catalog_version: str | None = Query(None),
    scope: str | None = Query(None),
    include_counts: bool = Query(False),
    principal: AuthPrincipal = Depends(require_privilege_admin),
    store: PrivilegeSnapshotStore = Depends(get_privilege_snapshot_store),
) -> PrivilegeSnapshotListResponse:
    del principal
    if org_id and team_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide either org_id or team_id filter, not both.",
        )
    if scope and not include_counts:
        include_counts = True

    payload = await store.list_snapshots(
        page=page,
        page_size=page_size,
        date_from=date_from,
        date_to=date_to,
        generated_by=generated_by,
        org_id=org_id,
        team_id=team_id,
        catalog_version=catalog_version,
        scope=scope,
        include_counts=include_counts,
    )
    return payload


@router.get("/snapshots/{snapshot_id}", response_model=PrivilegeSnapshotDetailResponse)
async def get_privilege_snapshot(
    *,
    snapshot_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(500, ge=1, le=500),
    principal: AuthPrincipal = Depends(require_privilege_admin),
    store: PrivilegeSnapshotStore = Depends(get_privilege_snapshot_store),
) -> PrivilegeSnapshotDetailResponse:
    del principal
    snapshot = await store.get_snapshot(snapshot_id=snapshot_id, page=page, page_size=page_size)
    if not snapshot:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Snapshot not found.")
    if snapshot.get("_downsampled"):
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Snapshot detail has been downsampled to a weekly summary and is no longer available.",
        )
    return snapshot


@router.get("/snapshots/{snapshot_id}/export.json")
async def export_privilege_snapshot_json(
    *,
    snapshot_id: str,
    principal: AuthPrincipal = Depends(require_privilege_admin),
    store: PrivilegeSnapshotStore = Depends(get_privilege_snapshot_store),
) -> JSONResponse:
    del principal
    export_data = await store.export_snapshot(snapshot_id=snapshot_id)
    if not export_data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Snapshot not found.")

    generated_at = export_data.get("generated_at")
    if isinstance(generated_at, datetime):
        generated_at_iso = generated_at.astimezone(timezone.utc).isoformat()
    else:
        generated_at_iso = str(generated_at) if generated_at else None

    payload = {
        "snapshot_id": export_data.get("snapshot_id"),
        "catalog_version": export_data.get("catalog_version"),
        "generated_at": generated_at_iso,
        "generated_by": export_data.get("generated_by"),
        "target_scope": export_data.get("target_scope"),
        "org_id": export_data.get("org_id"),
        "team_id": export_data.get("team_id"),
        "summary": export_data.get("summary"),
        "total_items": export_data.get("total_items"),
        "detail_items": export_data.get("detail_items", []),
        "metadata": {
            "exported_at": datetime.now(timezone.utc).isoformat(),
        },
    }

    response = JSONResponse(content=jsonable_encoder(payload))
    filename = f"privilege-snapshot-{snapshot_id}.json"
    response.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    etag = export_data.get("etag")
    if etag:
        response.headers["ETag"] = etag
    return response


@router.get(
    "/snapshots/{snapshot_id}/export.csv",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Privilege snapshot export as CSV.",
            "content": {
                "text/csv": {},
            },
        },
    },
)
async def export_privilege_snapshot_csv(
    *,
    snapshot_id: str,
    principal: AuthPrincipal = Depends(require_privilege_admin),
    store: PrivilegeSnapshotStore = Depends(get_privilege_snapshot_store),
) -> StreamingResponse:
    del principal
    export_data = await store.export_snapshot(snapshot_id=snapshot_id)
    if not export_data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Snapshot not found.")

    generated_at = export_data.get("generated_at")
    if isinstance(generated_at, datetime):
        generated_at_iso = generated_at.astimezone(timezone.utc).isoformat()
    else:
        generated_at_iso = str(generated_at) if generated_at else None

    detail_items = export_data.get("detail_items", [])
    fieldnames = [
        "snapshot_id",
        "catalog_version",
        "generated_at",
        "generated_by",
        "target_scope",
        "org_id",
        "team_id",
        "user_id",
        "endpoint",
        "method",
        "privilege_scope_id",
        "status",
        "blocked_reason",
        "feature_flag_id",
        "sensitivity_tier",
        "ownership_predicates",
        "dependencies",
        "dependency_modules",
        "rate_limit_class",
        "rate_limit_resources",
        "source_module",
        "summary",
        "tags",
    ]

    def _flatten_detail(detail: dict[str, Any]) -> dict[str, Any]:
        dependencies = detail.get("dependencies") or []
        dependency_ids = ";".join(dep.get("id", "") for dep in dependencies if dep)
        dependency_modules = ";".join(
            dep.get("module", "") for dep in dependencies if dep and dep.get("module")
        )
        ownership_predicates = ";".join(detail.get("ownership_predicates") or [])
        rate_limit_resources = ";".join(detail.get("rate_limit_resources") or [])
        tags = ";".join(detail.get("tags") or [])

        return {
            "snapshot_id": export_data.get("snapshot_id"),
            "catalog_version": export_data.get("catalog_version"),
            "generated_at": generated_at_iso,
            "generated_by": export_data.get("generated_by"),
            "target_scope": export_data.get("target_scope"),
            "org_id": export_data.get("org_id"),
            "team_id": export_data.get("team_id"),
            "user_id": detail.get("user_id"),
            "endpoint": detail.get("endpoint"),
            "method": detail.get("method"),
            "privilege_scope_id": detail.get("privilege_scope_id"),
            "status": detail.get("status"),
            "blocked_reason": detail.get("blocked_reason"),
            "feature_flag_id": detail.get("feature_flag_id"),
            "sensitivity_tier": detail.get("sensitivity_tier"),
            "ownership_predicates": ownership_predicates,
            "dependencies": dependency_ids,
            "dependency_modules": dependency_modules,
            "rate_limit_class": detail.get("rate_limit_class"),
            "rate_limit_resources": rate_limit_resources,
            "source_module": detail.get("source_module"),
            "summary": detail.get("summary"),
            "tags": tags,
        }

    def iter_csv():
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        yield buffer.getvalue()
        buffer.seek(0)
        buffer.truncate(0)

        for detail in detail_items:
            writer.writerow(_flatten_detail(detail))
            yield buffer.getvalue()
            buffer.seek(0)
            buffer.truncate(0)

    response = StreamingResponse(iter_csv(), media_type="text/csv")
    filename = f"privilege-snapshot-{snapshot_id}.csv"
    response.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    etag = export_data.get("etag")
    if etag:
        response.headers["ETag"] = etag
    return response


@router.post(
    "/snapshots",
    response_model=PrivilegeSnapshotRecord,
    status_code=status.HTTP_201_CREATED,
    responses={
        202: {
            "model": PrivilegeSnapshotAcceptedResponse,
            "description": "Snapshot generation queued for async processing.",
        },
    },
)
async def create_privilege_snapshot(
    *,
    payload: PrivilegeSnapshotCreateRequest,
    principal: AuthPrincipal = Depends(require_privilege_admin),
    service: PrivilegeMapService = Depends(get_privilege_map_service),
    store: PrivilegeSnapshotStore = Depends(get_privilege_snapshot_store),
    request: Request,
):
    generated_by = str(principal.user_id if principal.user_id is not None else (principal.subject or "system"))
    user_ids = list(payload.user_ids or [])
    if payload.target_scope == "user" and not user_ids:
        user_ids = [generated_by]
    if payload.async_job:
        snapshot_id = f"snap-{uuid4()}"
        job_payload = {
            "action": "generate_snapshot",
            "snapshot_id": snapshot_id,
            "target_scope": payload.target_scope,
            "org_id": payload.org_id,
            "team_id": payload.team_id,
            "user_ids": user_ids,
            "catalog_version": payload.catalog_version or service.catalog.version,
            "requested_by": generated_by,
            "notes": payload.notes,
        }
        try:
            job_manager = JobManager()
            job_manager.create_job(
                domain="privilege_maps",
                queue="default",
                job_type="snapshot",
                payload=job_payload,
                owner_user_id=generated_by,
                request_id=snapshot_id,
            )
        except Exception as exc:
            rid = ensure_request_id(request) if request is not None else None
            get_ps_logger(request_id=rid, ps_component="endpoint", ps_job_kind="privilege_maps").error(
                "Failed to enqueue privilege snapshot job: %s", exc
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Unable to queue privilege snapshot job.",
            ) from exc
        accepted = PrivilegeSnapshotAcceptedResponse(
            request_id=snapshot_id,
            status="queued",
            estimated_ready_at=datetime.now(timezone.utc),
        )
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content=jsonable_encoder(accepted),
        )
    try:
        summary_data, snapshot_users = await service.build_snapshot_summary(
            target_scope=payload.target_scope,
            org_id=payload.org_id,
            team_id=payload.team_id,
            user_ids=user_ids,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    generated_at = datetime.now(timezone.utc)
    snapshot_id = f"snap-{generated_at.strftime('%Y%m%d-%H%M%S')}"
    summary_model = PrivilegeSnapshotSummary(**summary_data)
    detail_items = service.build_snapshot_detail(
        snapshot_users,
        restrict_to_team=payload.team_id if payload.target_scope == "team" else None,
    )
    record = PrivilegeSnapshotRecord(
        snapshot_id=snapshot_id,
        generated_at=generated_at,
        generated_by=generated_by,
        target_scope=payload.target_scope,
        org_id=payload.org_id,
        team_id=payload.team_id,
        catalog_version=payload.catalog_version or service.catalog.version,
        summary=summary_model,
    )
    await store.add_snapshot(
        {
            "snapshot_id": record.snapshot_id,
            "generated_at": record.generated_at,
            "generated_by": record.generated_by,
            "target_scope": record.target_scope,
            "org_id": record.org_id,
            "team_id": record.team_id,
            "catalog_version": record.catalog_version,
            "summary": summary_model.dict(),
        },
        detail_items=detail_items,
    )
    return record
