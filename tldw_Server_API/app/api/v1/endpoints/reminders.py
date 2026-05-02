from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Path, status
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.Collections_DB_Deps import get_collections_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import RequirePermission, rbac_rate_limit
from tldw_Server_API.app.api.v1.schemas.reminders_schemas import (
    ReminderTaskCreateRequest,
    ReminderTaskDeleteResponse,
    ReminderTaskListResponse,
    ReminderTaskResponse,
    ReminderTaskUpdateRequest,
)
from tldw_Server_API.app.core.AuthNZ.permissions import TASKS_CONTROL, TASKS_READ
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase, ReminderTaskRow
from tldw_Server_API.app.core.Personalization.companion_activity import (
    record_reminder_task_created,
    record_reminder_task_deleted,
    record_reminder_task_updated,
)
from tldw_Server_API.app.services.reminders_scheduler import get_reminders_scheduler

router = APIRouter(prefix="/tasks", tags=["tasks"])

_REMINDERS_ENDPOINT_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


async def _reconcile_task_best_effort(*, task_id: str, user_id: int) -> None:
    try:
        await get_reminders_scheduler().reconcile_task(task_id=task_id, user_id=user_id)
    except _REMINDERS_ENDPOINT_NONCRITICAL_EXCEPTIONS:
        logger.warning("reminders endpoint reconcile_task failed")


async def _unschedule_task_best_effort(*, task_id: str) -> None:
    try:
        await get_reminders_scheduler().unschedule_task(task_id=task_id)
    except _REMINDERS_ENDPOINT_NONCRITICAL_EXCEPTIONS:
        logger.warning("reminders endpoint unschedule_task failed")


def _row_to_response(row: ReminderTaskRow) -> ReminderTaskResponse:
    return ReminderTaskResponse(
        id=row.id,
        user_id=row.user_id,
        tenant_id=row.tenant_id,
        title=row.title,
        body=row.body,
        link_type=row.link_type,
        link_id=row.link_id,
        link_url=row.link_url,
        schedule_kind=row.schedule_kind,  # type: ignore[arg-type]
        run_at=row.run_at,
        cron=row.cron,
        timezone=row.timezone,
        enabled=row.enabled,
        last_run_at=row.last_run_at,
        next_run_at=row.next_run_at,
        last_status=row.last_status,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=ReminderTaskResponse,
    dependencies=[Depends(rbac_rate_limit("tasks.control"))],
)
async def create_task(
    payload: ReminderTaskCreateRequest,
    db: CollectionsDatabase = Depends(get_collections_db_for_user),
    _principal=Depends(RequirePermission(TASKS_CONTROL)),  # noqa: B008
) -> ReminderTaskResponse:
    task_id = db.create_reminder_task(
        title=payload.title,
        body=payload.body,
        schedule_kind=payload.schedule_kind,
        run_at=payload.run_at,
        cron=payload.cron,
        timezone=payload.timezone,
        enabled=payload.enabled,
        link_type=payload.link_type,
        link_id=payload.link_id,
        link_url=payload.link_url,
    )
    await _reconcile_task_best_effort(task_id=task_id, user_id=int(db.user_id))
    task = _row_to_response(db.get_reminder_task(task_id))
    record_reminder_task_created(user_id=db.user_id, task=task)
    return task


@router.get(
    "",
    response_model=ReminderTaskListResponse,
    dependencies=[Depends(rbac_rate_limit("tasks.read"))],
)
async def list_tasks(
    db: CollectionsDatabase = Depends(get_collections_db_for_user),
    _principal=Depends(RequirePermission(TASKS_READ)),  # noqa: B008
) -> ReminderTaskListResponse:
    rows = db.list_reminder_tasks()
    return ReminderTaskListResponse(items=[_row_to_response(row) for row in rows], total=len(rows))


@router.get(
    "/{task_id}",
    response_model=ReminderTaskResponse,
    dependencies=[Depends(rbac_rate_limit("tasks.read"))],
)
async def get_task(
    task_id: str = Path(..., min_length=1),
    db: CollectionsDatabase = Depends(get_collections_db_for_user),
    _principal=Depends(RequirePermission(TASKS_READ)),  # noqa: B008
) -> ReminderTaskResponse:
    try:
        return _row_to_response(db.get_reminder_task(task_id))
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="task_not_found") from exc


@router.patch(
    "/{task_id}",
    response_model=ReminderTaskResponse,
    dependencies=[Depends(rbac_rate_limit("tasks.control"))],
)
async def update_task(
    payload: ReminderTaskUpdateRequest,
    task_id: str = Path(..., min_length=1),
    db: CollectionsDatabase = Depends(get_collections_db_for_user),
    _principal=Depends(RequirePermission(TASKS_CONTROL)),  # noqa: B008
) -> ReminderTaskResponse:
    patch = payload.model_dump(exclude_unset=True)
    try:
        updated = db.update_reminder_task(task_id, patch)
        await _reconcile_task_best_effort(task_id=task_id, user_id=int(db.user_id))
        response = _row_to_response(updated)
        if patch:
            record_reminder_task_updated(user_id=db.user_id, task=response, patch=patch)
        return response
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="task_not_found") from exc


@router.delete(
    "/{task_id}",
    response_model=ReminderTaskDeleteResponse,
    dependencies=[Depends(rbac_rate_limit("tasks.control"))],
)
async def delete_task(
    task_id: str = Path(..., min_length=1),
    db: CollectionsDatabase = Depends(get_collections_db_for_user),
    _principal=Depends(RequirePermission(TASKS_CONTROL)),  # noqa: B008
) -> ReminderTaskDeleteResponse:
    existing_task = None
    try:
        existing_task = _row_to_response(db.get_reminder_task(task_id))
    except KeyError:
        existing_task = None
    deleted = db.delete_reminder_task(task_id)
    if deleted:
        await _unschedule_task_best_effort(task_id=task_id)
        if existing_task is not None:
            record_reminder_task_deleted(user_id=db.user_id, task=existing_task)
    return ReminderTaskDeleteResponse(deleted=deleted)
