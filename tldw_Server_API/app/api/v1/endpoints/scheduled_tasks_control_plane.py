from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Path, status
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user, rbac_rate_limit, RequirePermission, User

from tldw_Server_API.app.api.v1.schemas.reminders_schemas import (
    ReminderTaskCreateRequest,
    ReminderTaskUpdateRequest,
)
from tldw_Server_API.app.api.v1.schemas.scheduled_tasks_control_plane_schemas import (
    ScheduledTask,
    ScheduledTaskDeleteResponse,
    ScheduledTaskListResponse,
)
from tldw_Server_API.app.core.AuthNZ.permissions import TASKS_CONTROL, TASKS_READ
from tldw_Server_API.app.services.scheduled_tasks_control_plane_service import ScheduledTasksControlPlaneService

router = APIRouter(prefix="/scheduled-tasks", tags=["scheduled-tasks"])


def get_scheduled_tasks_control_plane_service() -> ScheduledTasksControlPlaneService:
    return ScheduledTasksControlPlaneService()


@router.get(
    "",
    response_model=ScheduledTaskListResponse,
    dependencies=[Depends(rbac_rate_limit("tasks.read"))],
)
async def list_scheduled_tasks(
    current_user: User = Depends(get_request_user),
    _principal=Depends(RequirePermission(TASKS_READ)),  # noqa: B008
    service: ScheduledTasksControlPlaneService = Depends(get_scheduled_tasks_control_plane_service),
) -> ScheduledTaskListResponse:
    return await service.list_tasks(user_id=int(current_user.id))


@router.get(
    "/{task_id}",
    response_model=ScheduledTask,
    dependencies=[Depends(rbac_rate_limit("tasks.read"))],
)
async def get_scheduled_task(
    task_id: str = Path(..., min_length=1),
    current_user: User = Depends(get_request_user),
    _principal=Depends(RequirePermission(TASKS_READ)),  # noqa: B008
    service: ScheduledTasksControlPlaneService = Depends(get_scheduled_tasks_control_plane_service),
) -> ScheduledTask:
    try:
        return await service.get_task(user_id=int(current_user.id), task_id=task_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="scheduled_task_not_found") from exc


@router.post(
    "/reminders",
    response_model=ScheduledTask,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(rbac_rate_limit("tasks.control"))],
)
async def create_scheduled_task_reminder(
    payload: ReminderTaskCreateRequest,
    current_user: User = Depends(get_request_user),
    _principal=Depends(RequirePermission(TASKS_CONTROL)),  # noqa: B008
    service: ScheduledTasksControlPlaneService = Depends(get_scheduled_tasks_control_plane_service),
) -> ScheduledTask:
    return await service.create_reminder(user_id=int(current_user.id), payload=payload)


@router.patch(
    "/reminders/{task_id}",
    response_model=ScheduledTask,
    dependencies=[Depends(rbac_rate_limit("tasks.control"))],
)
async def update_scheduled_task_reminder(
    payload: ReminderTaskUpdateRequest,
    task_id: str = Path(..., min_length=1),
    current_user: User = Depends(get_request_user),
    _principal=Depends(RequirePermission(TASKS_CONTROL)),  # noqa: B008
    service: ScheduledTasksControlPlaneService = Depends(get_scheduled_tasks_control_plane_service),
) -> ScheduledTask:
    try:
        return await service.update_reminder(user_id=int(current_user.id), task_id=task_id, payload=payload)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="task_not_found") from exc


@router.delete(
    "/reminders/{task_id}",
    response_model=ScheduledTaskDeleteResponse,
    dependencies=[Depends(rbac_rate_limit("tasks.control"))],
)
async def delete_scheduled_task_reminder(
    task_id: str = Path(..., min_length=1),
    current_user: User = Depends(get_request_user),
    _principal=Depends(RequirePermission(TASKS_CONTROL)),  # noqa: B008
    service: ScheduledTasksControlPlaneService = Depends(get_scheduled_tasks_control_plane_service),
) -> ScheduledTaskDeleteResponse:
    return await service.delete_reminder(user_id=int(current_user.id), task_id=task_id)
