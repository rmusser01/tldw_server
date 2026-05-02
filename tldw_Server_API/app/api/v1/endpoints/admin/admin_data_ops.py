from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.endpoints._pagination_utils import (
    build_offset_pagination_meta,
)
from tldw_Server_API.app.api.v1.schemas.admin_schemas import (
    BackupCreateRequest,
    BackupCreateResponse,
    BackupItem,
    BackupListResponse,
    BackupRestoreRequest,
    BackupRestoreResponse,
    BackupScheduleCreateRequest,
    BackupScheduleItem,
    BackupScheduleListResponse,
    BackupScheduleMutationResponse,
    BackupScheduleUpdateRequest,
    DataSubjectRequestCreateRequest,
    DataSubjectRequestCreateResponse,
    DataSubjectRequestItem,
    DataSubjectRequestListResponse,
    DataSubjectRequestPreviewRequest,
    DataSubjectRequestPreviewResponse,
    RetentionPoliciesResponse,
    RetentionPolicy,
    RetentionPolicyPreviewRequest,
    RetentionPolicyPreviewResponse,
    RetentionPolicyUpdateRequest,
)
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.repos.data_subject_requests_repo import (
    AuthnzDataSubjectRequestsRepo,
)
from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
from tldw_Server_API.app.services.admin_backup_schedules_service import (
    AdminBackupSchedulesService,
)
from tldw_Server_API.app.services.admin_data_ops_service import (
    build_retention_preview_signature as svc_build_retention_preview_signature,
)
from tldw_Server_API.app.services.admin_data_ops_service import (
    create_backup_snapshot as svc_create_backup_snapshot,
)
from tldw_Server_API.app.services.admin_data_ops_service import (
    list_backup_items as svc_list_backup_items,
)
from tldw_Server_API.app.services.admin_data_ops_service import (
    list_retention_policies as svc_list_retention_policies,
)
from tldw_Server_API.app.services.admin_data_ops_service import (
    preview_retention_policy as svc_preview_retention_policy,
)
from tldw_Server_API.app.services.admin_data_ops_service import (
    restore_backup_snapshot as svc_restore_backup_snapshot,
)
from tldw_Server_API.app.services.admin_data_ops_service import (
    update_retention_policy as svc_update_retention_policy,
)
from tldw_Server_API.app.services.admin_data_ops_service import (
    verify_retention_preview_signature as svc_verify_retention_preview_signature,
)
from tldw_Server_API.app.services.admin_data_subject_requests_service import (
    execute_dsr_erasure as svc_execute_dsr_erasure,
)
from tldw_Server_API.app.services.admin_data_subject_requests_service import (
    list_data_subject_requests as svc_list_data_subject_requests,
)
from tldw_Server_API.app.services.admin_data_subject_requests_service import (
    preview_data_subject_request as svc_preview_data_subject_request,
)
from tldw_Server_API.app.services.admin_data_subject_requests_service import (
    record_data_subject_request as svc_record_data_subject_request,
)

if TYPE_CHECKING:
    from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
    from tldw_Server_API.app.core.AuthNZ.repos.backup_schedules_repo import (
        AuthnzBackupSchedulesRepo,
    )

router = APIRouter()


_DATA_OPS_NONCRITICAL_EXCEPTIONS = (
    asyncio.CancelledError,
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


async def _build_dsr_repos() -> tuple[AuthnzUsersRepo, AuthnzDataSubjectRequestsRepo]:
    db_pool = await get_db_pool()
    return (
        AuthnzUsersRepo(db_pool=db_pool),
        AuthnzDataSubjectRequestsRepo(db_pool=db_pool),
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


async def _emit_admin_audit_event(
    request: Request,
    principal: AuthPrincipal,
    *,
    event_type: str,
    category: str,
    resource_type: str,
    resource_id: str | None,
    action: str,
    metadata: dict[str, Any],
) -> None:
    from tldw_Server_API.app.api.v1.endpoints import admin as admin_mod

    await admin_mod._emit_admin_audit_event(
        request,
        principal,
        event_type=event_type,
        category=category,
        resource_type=resource_type,
        resource_id=resource_id,
        action=action,
        metadata=metadata,
    )


_BACKUP_DATASETS = {"media", "chacha", "prompts", "evaluations", "audit", "authnz"}
_PER_USER_BACKUP_DATASETS = _BACKUP_DATASETS - {"authnz"}


def _require_user_id_for_dataset(dataset: str, user_id: int | None) -> None:
    """Require a target user for per-user datasets."""
    if dataset in _PER_USER_BACKUP_DATASETS and user_id is None:
        raise HTTPException(status_code=400, detail="user_id_required")


def _build_schedule_description(item: dict[str, Any]) -> str:
    """Return human-readable schedule copy for an API response row."""
    service = AdminBackupSchedulesService(repo=None)
    return service.describe_schedule(item)


def _serialize_backup_schedule_item(item: dict[str, Any]) -> BackupScheduleItem:
    """Convert a repository schedule row into the public API schema."""
    return BackupScheduleItem(
        id=str(item["id"]),
        dataset=str(item["dataset"]),
        target_user_id=item.get("target_user_id"),
        frequency=str(item["frequency"]),
        time_of_day=str(item["time_of_day"]),
        timezone=str(item.get("timezone") or "UTC"),
        anchor_day_of_week=item.get("anchor_day_of_week"),
        anchor_day_of_month=item.get("anchor_day_of_month"),
        retention_count=int(item["retention_count"]),
        is_paused=bool(item.get("is_paused", False)),
        schedule_description=_build_schedule_description(item),
        next_run_at=item.get("next_run_at"),
        last_run_at=item.get("last_run_at"),
        last_status=item.get("last_status"),
        last_job_id=item.get("last_job_id"),
        last_error=item.get("last_error"),
        created_at=item["created_at"],
        updated_at=item["updated_at"],
        deleted_at=item.get("deleted_at"),
    )


async def _get_backup_schedules_repo(
    pool: DatabasePool = Depends(get_db_pool),
) -> AuthnzBackupSchedulesRepo:
    """Build the backup schedules repository from the shared AuthNZ pool."""
    from tldw_Server_API.app.core.AuthNZ.repos.backup_schedules_repo import AuthnzBackupSchedulesRepo

    repo = AuthnzBackupSchedulesRepo(pool)
    await repo.ensure_schema()
    return repo


async def _get_backup_schedule_service(
    repo: AuthnzBackupSchedulesRepo = Depends(_get_backup_schedules_repo),
) -> AdminBackupSchedulesService:
    """Build the backup schedules service from the injected repository."""
    return AdminBackupSchedulesService(repo=repo)


def _dsr_item_from_record(record: dict[str, Any]) -> DataSubjectRequestItem:
    return DataSubjectRequestItem(**record)


@router.get("/backups", response_model=BackupListResponse)
async def list_backups(
    dataset: str | None = Query(None, description="Dataset key to filter"),
    user_id: int | None = Query(None, description="User ID for per-user datasets"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> BackupListResponse:
    try:
        normalized_dataset = None
        if dataset:
            normalized_dataset = dataset.strip().lower()
            if normalized_dataset not in _BACKUP_DATASETS:
                raise HTTPException(status_code=400, detail="unknown_dataset")
            _require_user_id_for_dataset(normalized_dataset, user_id)
        if user_id is not None:
            await _enforce_admin_user_scope(principal, user_id, require_hierarchy=False)
        items, total = svc_list_backup_items(
            dataset=normalized_dataset,
            user_id=user_id,
            limit=limit,
            offset=offset,
        )
        payload = [
            BackupItem(
                id=item.filename,
                dataset=item.dataset,
                user_id=item.user_id,
                status="ready",
                size_bytes=item.size_bytes,
                created_at=item.created_at,
            )
            for item in items
        ]
        return BackupListResponse(
            items=payload,
            total=total,
            limit=limit,
            offset=offset,
            pagination=build_offset_pagination_meta(
                total=total,
                offset=offset,
                limit=limit,
                count=len(payload),
            ),
        )
    except HTTPException:
        raise
    except _DATA_OPS_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Failed to list backups")
        raise HTTPException(status_code=500, detail="Failed to list backups") from exc


@router.post("/backups", response_model=BackupCreateResponse)
async def create_backup(
    payload: BackupCreateRequest,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> BackupCreateResponse:
    try:
        dataset = payload.dataset.strip().lower()
        if dataset not in _BACKUP_DATASETS:
            raise HTTPException(status_code=400, detail="unknown_dataset")
        _require_user_id_for_dataset(dataset, payload.user_id)
        if payload.user_id is not None:
            await _enforce_admin_user_scope(principal, payload.user_id, require_hierarchy=False)
        item = svc_create_backup_snapshot(
            dataset=dataset,
            user_id=payload.user_id,
            backup_type=payload.backup_type or "full",
            max_backups=payload.max_backups,
        )
        await _emit_admin_audit_event(
            request,
            principal,
            event_type="data.write",
            category="system",
            resource_type="backup",
            resource_id=item.filename,
            action="backup.create",
            metadata={
                "dataset": dataset,
                "user_id": item.user_id,
                "size_bytes": item.size_bytes,
            },
        )
        return BackupCreateResponse(
            item=BackupItem(
                id=item.filename,
                dataset=item.dataset,
                user_id=item.user_id,
                status="ready",
                size_bytes=item.size_bytes,
                created_at=item.created_at,
            )
        )
    except ValueError as exc:
        if str(exc) == "unknown_dataset":
            raise HTTPException(status_code=400, detail="unknown_dataset") from exc
        raise HTTPException(status_code=400, detail="invalid_backup_request") from exc
    except HTTPException:
        raise
    except _DATA_OPS_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Failed to create backup")
        raise HTTPException(status_code=500, detail="Failed to create backup") from exc


@router.post("/backups/{backup_id}/restore", response_model=BackupRestoreResponse)
async def restore_backup(
    backup_id: str,
    payload: BackupRestoreRequest,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> BackupRestoreResponse:
    if not payload.confirm:
        raise HTTPException(status_code=400, detail="confirm_required")
    try:
        dataset = payload.dataset.strip().lower()
        if dataset not in _BACKUP_DATASETS:
            raise HTTPException(status_code=400, detail="unknown_dataset")
        _require_user_id_for_dataset(dataset, payload.user_id)
        if payload.user_id is not None:
            await _enforce_admin_user_scope(principal, payload.user_id, require_hierarchy=False)
        result = svc_restore_backup_snapshot(
            dataset=dataset,
            user_id=payload.user_id,
            backup_id=backup_id,
        )
        await _emit_admin_audit_event(
            request,
            principal,
            event_type="data.import",
            category="system",
            resource_type="backup",
            resource_id=backup_id,
            action="backup.restore",
            metadata={
                "dataset": dataset,
                "user_id": payload.user_id,
            },
        )
        return BackupRestoreResponse(status="restored", message=result)
    except ValueError as exc:
        if str(exc) == "unknown_dataset":
            raise HTTPException(status_code=400, detail="unknown_dataset") from exc
        raise HTTPException(status_code=400, detail="invalid_restore_request") from exc
    except HTTPException:
        raise
    except _DATA_OPS_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Failed to restore backup")
        raise HTTPException(status_code=500, detail="Failed to restore backup") from exc


@router.get("/backup-schedules", response_model=BackupScheduleListResponse)
async def list_backup_schedules(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    principal: AuthPrincipal = Depends(get_auth_principal),
    service: AdminBackupSchedulesService = Depends(_get_backup_schedule_service),
) -> BackupScheduleListResponse:
    try:
        repo = service.repo
        items, total = await repo.list_schedules(
            limit=limit,
            offset=offset,
            exclude_authnz=not service.is_platform_admin(principal),
        )
        items = service.filter_visible_items(items, principal=principal)
        payload = [_serialize_backup_schedule_item(item) for item in items]
        return BackupScheduleListResponse(
            items=payload,
            total=total,
            limit=limit,
            offset=offset,
            pagination=build_offset_pagination_meta(
                total=total,
                limit=limit,
                offset=offset,
                count=len(payload),
            ),
        )
    except HTTPException:
        raise
    except _DATA_OPS_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Failed to list backup schedules")
        raise HTTPException(status_code=500, detail="Failed to list backup schedules") from exc


@router.post("/backup-schedules", response_model=BackupScheduleMutationResponse)
async def create_backup_schedule(
    payload: BackupScheduleCreateRequest,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
    service: AdminBackupSchedulesService = Depends(_get_backup_schedule_service),
) -> BackupScheduleMutationResponse:
    try:
        dataset = service.normalize_dataset(payload.dataset)
        service.validate_target_rules(dataset, payload.target_user_id)
        if service.requires_platform_admin(dataset):
            service.require_platform_admin(principal)
        if payload.target_user_id is not None:
            await _enforce_admin_user_scope(principal, payload.target_user_id, require_hierarchy=False)

        created = await service.create_schedule(
            dataset=dataset,
            target_user_id=payload.target_user_id,
            frequency=payload.frequency,
            time_of_day=payload.time_of_day,
            timezone_name=payload.timezone,
            retention_count=payload.retention_count,
            principal_user_id=principal.user_id,
        )
        item = _serialize_backup_schedule_item(created)
        await _emit_admin_audit_event(
            request,
            principal,
            event_type="data.write",
            category="system",
            resource_type="backup_schedule",
            resource_id=str(item.id),
            action="backup_schedule.create",
            metadata={
                "dataset": item.dataset,
                "target_user_id": item.target_user_id,
                "frequency": item.frequency,
            },
        )
        return BackupScheduleMutationResponse(status="created", item=item)
    except HTTPException:
        raise
    except _DATA_OPS_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Failed to create backup schedule")
        raise HTTPException(status_code=500, detail="Failed to create backup schedule") from exc


@router.patch("/backup-schedules/{schedule_id}", response_model=BackupScheduleMutationResponse)
async def update_backup_schedule(
    schedule_id: str,
    payload: BackupScheduleUpdateRequest,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
    service: AdminBackupSchedulesService = Depends(_get_backup_schedule_service),
) -> BackupScheduleMutationResponse:
    try:
        repo = service.repo
        current = await repo.get_schedule(schedule_id)
        if not current:
            raise HTTPException(status_code=404, detail="schedule_not_found")
        if service.requires_platform_admin(str(current.get("dataset"))):
            service.require_platform_admin(principal)
        elif current.get("target_user_id") is not None:
            await _enforce_admin_user_scope(principal, int(current["target_user_id"]), require_hierarchy=False)

        updated = await service.update_schedule(
            schedule_id=schedule_id,
            current=current,
            frequency=payload.frequency,
            time_of_day=payload.time_of_day,
            timezone_name=payload.timezone,
            retention_count=payload.retention_count,
            principal_user_id=principal.user_id,
        )
        if not updated:
            raise HTTPException(status_code=404, detail="schedule_not_found")
        item = _serialize_backup_schedule_item(updated)
        await _emit_admin_audit_event(
            request,
            principal,
            event_type="data.write",
            category="system",
            resource_type="backup_schedule",
            resource_id=str(item.id),
            action="backup_schedule.update",
            metadata={
                "dataset": item.dataset,
                "target_user_id": item.target_user_id,
                "frequency": item.frequency,
            },
        )
        return BackupScheduleMutationResponse(status="updated", item=item)
    except HTTPException:
        raise
    except _DATA_OPS_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Failed to update backup schedule")
        raise HTTPException(status_code=500, detail="Failed to update backup schedule") from exc


@router.post("/backup-schedules/{schedule_id}/pause", response_model=BackupScheduleMutationResponse)
async def pause_backup_schedule(
    schedule_id: str,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
    service: AdminBackupSchedulesService = Depends(_get_backup_schedule_service),
) -> BackupScheduleMutationResponse:
    try:
        repo = service.repo
        current = await repo.get_schedule(schedule_id)
        if not current:
            raise HTTPException(status_code=404, detail="schedule_not_found")
        if service.requires_platform_admin(str(current.get("dataset"))):
            service.require_platform_admin(principal)
        elif current.get("target_user_id") is not None:
            await _enforce_admin_user_scope(principal, int(current["target_user_id"]), require_hierarchy=False)

        updated = await repo.pause_schedule(schedule_id, updated_by_user_id=principal.user_id)
        if not updated:
            raise HTTPException(status_code=404, detail="schedule_not_found")
        item = _serialize_backup_schedule_item(updated)
        await _emit_admin_audit_event(
            request,
            principal,
            event_type="data.write",
            category="system",
            resource_type="backup_schedule",
            resource_id=str(item.id),
            action="backup_schedule.pause",
            metadata={"dataset": item.dataset, "target_user_id": item.target_user_id},
        )
        return BackupScheduleMutationResponse(status="paused", item=item)
    except HTTPException:
        raise
    except _DATA_OPS_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Failed to pause backup schedule")
        raise HTTPException(status_code=500, detail="Failed to pause backup schedule") from exc


@router.post("/backup-schedules/{schedule_id}/resume", response_model=BackupScheduleMutationResponse)
async def resume_backup_schedule(
    schedule_id: str,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
    service: AdminBackupSchedulesService = Depends(_get_backup_schedule_service),
) -> BackupScheduleMutationResponse:
    try:
        repo = service.repo
        current = await repo.get_schedule(schedule_id)
        if not current:
            raise HTTPException(status_code=404, detail="schedule_not_found")
        if service.requires_platform_admin(str(current.get("dataset"))):
            service.require_platform_admin(principal)
        elif current.get("target_user_id") is not None:
            await _enforce_admin_user_scope(principal, int(current["target_user_id"]), require_hierarchy=False)

        updated = await repo.resume_schedule(schedule_id, updated_by_user_id=principal.user_id)
        if not updated:
            raise HTTPException(status_code=404, detail="schedule_not_found")
        item = _serialize_backup_schedule_item(updated)
        await _emit_admin_audit_event(
            request,
            principal,
            event_type="data.write",
            category="system",
            resource_type="backup_schedule",
            resource_id=str(item.id),
            action="backup_schedule.resume",
            metadata={"dataset": item.dataset, "target_user_id": item.target_user_id},
        )
        return BackupScheduleMutationResponse(status="resumed", item=item)
    except HTTPException:
        raise
    except _DATA_OPS_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Failed to resume backup schedule")
        raise HTTPException(status_code=500, detail="Failed to resume backup schedule") from exc


@router.delete("/backup-schedules/{schedule_id}", response_model=BackupScheduleMutationResponse)
async def delete_backup_schedule(
    schedule_id: str,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
    service: AdminBackupSchedulesService = Depends(_get_backup_schedule_service),
) -> BackupScheduleMutationResponse:
    try:
        repo = service.repo
        current = await repo.get_schedule(schedule_id)
        if not current:
            raise HTTPException(status_code=404, detail="schedule_not_found")
        if service.requires_platform_admin(str(current.get("dataset"))):
            service.require_platform_admin(principal)
        elif current.get("target_user_id") is not None:
            await _enforce_admin_user_scope(principal, int(current["target_user_id"]), require_hierarchy=False)

        from datetime import datetime
        from datetime import timezone as dt_timezone

        deleted_at = datetime.now(dt_timezone.utc).isoformat()
        deleted = await repo.delete_schedule(schedule_id, deleted_at=deleted_at)
        if not deleted:
            raise HTTPException(status_code=404, detail="schedule_not_found")
        deleted_row = await repo.get_schedule(schedule_id, include_deleted=True)
        if not deleted_row:
            raise HTTPException(status_code=404, detail="schedule_not_found")
        item = _serialize_backup_schedule_item(deleted_row)
        await _emit_admin_audit_event(
            request,
            principal,
            event_type="data.write",
            category="system",
            resource_type="backup_schedule",
            resource_id=str(item.id),
            action="backup_schedule.delete",
            metadata={"dataset": item.dataset, "target_user_id": item.target_user_id},
        )
        return BackupScheduleMutationResponse(status="deleted", item=item)
    except HTTPException:
        raise
    except _DATA_OPS_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Failed to delete backup schedule")
        raise HTTPException(status_code=500, detail="Failed to delete backup schedule") from exc


@router.post("/data-subject-requests/preview", response_model=DataSubjectRequestPreviewResponse)
async def preview_data_subject_request(
    payload: DataSubjectRequestPreviewRequest,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> DataSubjectRequestPreviewResponse:
    try:
        users_repo, _ = await _build_dsr_repos()
        preview = await svc_preview_data_subject_request(
            requester_identifier=payload.requester_identifier,
            request_type=payload.request_type,
            categories=payload.categories,
            principal=principal,
            users_repo=users_repo,
        )
        await _emit_admin_audit_event(
            request,
            principal,
            event_type="data.read",
            category="compliance",
            resource_type="data_subject_request",
            resource_id=str(preview["resolved_user_id"]),
            action="data_subject_request.preview",
            metadata={
                "requester_identifier": preview["requester_identifier"],
                "resolved_user_id": preview["resolved_user_id"],
                "selected_categories": preview["selected_categories"],
            },
        )
        return DataSubjectRequestPreviewResponse(**preview)
    except HTTPException:
        raise
    except _DATA_OPS_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Failed to preview data subject request")
        raise HTTPException(status_code=500, detail="Failed to preview data subject request") from exc


@router.post("/data-subject-requests", response_model=DataSubjectRequestCreateResponse)
async def create_data_subject_request(
    payload: DataSubjectRequestCreateRequest,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> DataSubjectRequestCreateResponse:
    try:
        users_repo, requests_repo = await _build_dsr_repos()
        preview = await svc_preview_data_subject_request(
            requester_identifier=payload.requester_identifier,
            request_type=payload.request_type,
            categories=payload.categories,
            principal=principal,
            users_repo=users_repo,
        )
        record = await svc_record_data_subject_request(
            principal=principal,
            client_request_id=payload.client_request_id,
            requester_identifier=payload.requester_identifier,
            request_type=payload.request_type,
            categories=payload.categories,
            users_repo=users_repo,
            requests_repo=requests_repo,
            preview=preview,
            notes=payload.notes,
        )
        await _emit_admin_audit_event(
            request,
            principal,
            event_type="data.write",
            category="compliance",
            resource_type="data_subject_request",
            resource_id=str(record.get("id") or payload.client_request_id),
            action="data_subject_request.record",
            metadata={
                "requester_identifier": record.get("requester_identifier"),
                "resolved_user_id": record.get("resolved_user_id"),
                "request_type": record.get("request_type"),
                "selected_categories": record.get("selected_categories"),
            },
        )
        return DataSubjectRequestCreateResponse(item=_dsr_item_from_record(record))
    except HTTPException:
        raise
    except _DATA_OPS_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Failed to record data subject request")
        raise HTTPException(status_code=500, detail="Failed to record data subject request") from exc


@router.get("/data-subject-requests", response_model=DataSubjectRequestListResponse)
async def list_data_subject_requests(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> DataSubjectRequestListResponse:
    try:
        _, requests_repo = await _build_dsr_repos()
        items, total = await svc_list_data_subject_requests(
            principal,
            limit=limit,
            offset=offset,
            requests_repo=requests_repo,
        )
        return DataSubjectRequestListResponse(
            items=[_dsr_item_from_record(item) for item in items],
            total=total,
            limit=limit,
            offset=offset,
            pagination=build_offset_pagination_meta(
                total=total,
                limit=limit,
                offset=offset,
                count=len(items),
            ),
        )
    except HTTPException:
        raise
    except _DATA_OPS_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Failed to list data subject requests")
        raise HTTPException(status_code=500, detail="Failed to list data subject requests") from exc


@router.post("/data-subject-requests/{request_id}/execute")
async def execute_data_subject_request(
    request_id: int,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> dict[str, Any]:
    """Execute a recorded DSR erasure request."""
    try:
        _, requests_repo = await _build_dsr_repos()
        await requests_repo.ensure_schema()

        record = await requests_repo.get_request_by_id(request_id)
        if record is None:
            raise HTTPException(status_code=404, detail="request_not_found")

        if record.get("request_type") != "erasure":
            raise HTTPException(
                status_code=400,
                detail="only_erasure_requests_can_be_executed",
            )

        current_status = record.get("status", "")
        if current_status in {"completed", "executing"}:
            raise HTTPException(
                status_code=409,
                detail=f"request_already_{current_status}",
            )

        resolved_user_id = record.get("resolved_user_id")
        if resolved_user_id is None:
            raise HTTPException(status_code=400, detail="resolved_user_id_missing")

        await _enforce_admin_user_scope(
            principal,
            int(resolved_user_id),
            require_hierarchy=True,
        )

        selected_categories = record.get("selected_categories", [])
        if not selected_categories:
            raise HTTPException(status_code=400, detail="no_categories_selected")

        result = await svc_execute_dsr_erasure(
            request_id=request_id,
            user_id=int(resolved_user_id),
            selected_categories=selected_categories,
            dsr_repo=requests_repo,
            principal=principal,
        )

        await _emit_admin_audit_event(
            request,
            principal,
            event_type="data.write",
            category="compliance",
            resource_type="data_subject_request",
            resource_id=str(request_id),
            action="data_subject_request.execute",
            metadata={
                "resolved_user_id": resolved_user_id,
                "selected_categories": selected_categories,
                "status": result.get("status"),
            },
        )
        return result
    except HTTPException:
        raise
    except _DATA_OPS_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Failed to execute data subject request")
        raise HTTPException(status_code=500, detail="Failed to execute data subject request") from exc


@router.get("/retention-policies", response_model=RetentionPoliciesResponse)
async def list_retention_policies(
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> RetentionPoliciesResponse:
    try:
        del principal
        policies = [RetentionPolicy(**item) for item in await svc_list_retention_policies()]
        return RetentionPoliciesResponse(policies=policies)
    except _DATA_OPS_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Failed to list retention policies")
        raise HTTPException(status_code=500, detail="Failed to list retention policies") from exc


@router.post("/retention-policies/{policy_key}/preview", response_model=RetentionPolicyPreviewResponse)
async def preview_retention_policy(
    policy_key: str,
    payload: RetentionPolicyPreviewRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> RetentionPolicyPreviewResponse:
    try:
        preview = await svc_preview_retention_policy(
            policy_key=policy_key,
            current_days=payload.current_days,
            days=payload.days,
        )
        signature = svc_build_retention_preview_signature(
            principal=principal,
            policy_key=policy_key,
            current_days=int(preview["current_days"]),
            new_days=int(preview["new_days"]),
        )
        return RetentionPolicyPreviewResponse(**preview, preview_signature=signature)
    except ValueError as exc:
        detail = str(exc)
        if detail == "unknown_policy":
            raise HTTPException(status_code=404, detail="unknown_policy") from exc
        if detail == "invalid_range":
            raise HTTPException(status_code=400, detail="invalid_range") from exc
        if detail == "stale_current_days":
            raise HTTPException(status_code=409, detail="stale_current_days") from exc
        raise HTTPException(status_code=400, detail="invalid_retention_preview") from exc
    except _DATA_OPS_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Failed to preview retention policy")
        raise HTTPException(status_code=500, detail="Failed to preview retention policy") from exc


@router.put("/retention-policies/{policy_key}", response_model=RetentionPolicy)
async def update_retention_policy(
    policy_key: str,
    payload: RetentionPolicyUpdateRequest,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> RetentionPolicy:
    try:
        await svc_verify_retention_preview_signature(
            principal=principal,
            policy_key=policy_key,
            days=payload.days,
            preview_signature=payload.preview_signature,
        )
        updated = await svc_update_retention_policy(policy_key, payload.days)
        await _emit_admin_audit_event(
            request,
            principal,
            event_type="config.changed",
            category="system",
            resource_type="retention_policy",
            resource_id=policy_key,
            action="retention.update",
            metadata={"days": payload.days},
        )
        return RetentionPolicy(**updated)
    except ValueError as exc:
        detail = str(exc)
        if detail == "unknown_policy":
            raise HTTPException(status_code=404, detail="unknown_policy") from exc
        if detail == "invalid_range":
            raise HTTPException(status_code=400, detail="invalid_range") from exc
        if detail in {"preview_signature_required", "invalid_preview_signature", "expired_preview_signature"}:
            raise HTTPException(status_code=400, detail=detail) from exc
        raise HTTPException(status_code=400, detail="invalid_retention_update") from exc
    except _DATA_OPS_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Failed to update retention policy")
        raise HTTPException(status_code=500, detail="Failed to update retention policy") from exc
