from __future__ import annotations

from typing import NoReturn

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user, rbac_rate_limit, RequirePermission, User

from tldw_Server_API.app.api.v1.schemas.reminders_schemas import (
    ReminderTaskCreateRequest,
    ReminderTaskUpdateRequest,
)
from tldw_Server_API.app.api.v1.schemas.scheduled_tasks_automation_schemas import (
    ScheduledTaskAuditListResponse,
    ScheduledTaskAutomationCapabilitiesResponse,
    ScheduledTaskDefinitionCreateRequest,
    ScheduledTaskDefinitionListResponse,
    ScheduledTaskDefinitionResponse,
    ScheduledTaskDefinitionUpdateRequest,
    ScheduledTaskDuplicateRequest,
    ScheduledTaskPreviewCreateRequest,
    ScheduledTaskPreviewListResponse,
    ScheduledTaskPreviewResponse,
)
from tldw_Server_API.app.api.v1.schemas.scheduled_tasks_control_plane_schemas import (
    ScheduledTask,
    ScheduledTaskDeleteResponse,
    ScheduledTaskListResponse,
)
from tldw_Server_API.app.core.AuthNZ.permissions import TASKS_CONTROL, TASKS_READ
from tldw_Server_API.app.services.scheduled_task_automation_service import ScheduledTaskAutomationService
from tldw_Server_API.app.services.scheduled_tasks_control_plane_service import ScheduledTasksControlPlaneService

router = APIRouter(prefix="/scheduled-tasks", tags=["scheduled-tasks"])


def get_scheduled_tasks_control_plane_service() -> ScheduledTasksControlPlaneService:
    return ScheduledTasksControlPlaneService()


def get_scheduled_task_automation_service() -> ScheduledTaskAutomationService:
    return ScheduledTaskAutomationService()


def _raise_automation_not_implemented(detail: str) -> NoReturn:
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=detail)


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
    "/capabilities",
    response_model=ScheduledTaskAutomationCapabilitiesResponse,
    dependencies=[Depends(rbac_rate_limit("tasks.read"))],
)
async def get_scheduled_task_automation_capabilities(
    _principal=Depends(RequirePermission(TASKS_READ)),  # noqa: B008
    service: ScheduledTaskAutomationService = Depends(get_scheduled_task_automation_service),
) -> ScheduledTaskAutomationCapabilitiesResponse:
    return service.get_capabilities()


@router.get(
    "/previews",
    response_model=ScheduledTaskPreviewListResponse,
    dependencies=[Depends(rbac_rate_limit("tasks.read"))],
)
async def list_scheduled_task_automation_previews(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    _principal=Depends(RequirePermission(TASKS_READ)),  # noqa: B008
    service: ScheduledTaskAutomationService = Depends(get_scheduled_task_automation_service),
) -> ScheduledTaskPreviewListResponse:
    return service.list_previews(limit=limit, offset=offset)


@router.post(
    "/previews",
    response_model=ScheduledTaskPreviewResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(rbac_rate_limit("tasks.control"))],
)
async def create_scheduled_task_automation_preview(
    _payload: ScheduledTaskPreviewCreateRequest,
    _principal=Depends(RequirePermission(TASKS_CONTROL)),  # noqa: B008
) -> ScheduledTaskPreviewResponse:
    _raise_automation_not_implemented("scheduled_task_preview_storage_not_implemented")


@router.get(
    "/previews/{preview_id}",
    response_model=ScheduledTaskPreviewResponse,
    dependencies=[Depends(rbac_rate_limit("tasks.read"))],
)
async def get_scheduled_task_automation_preview(
    preview_id: str = Path(..., min_length=1),
    _principal=Depends(RequirePermission(TASKS_READ)),  # noqa: B008
) -> ScheduledTaskPreviewResponse:
    _raise_automation_not_implemented("scheduled_task_preview_storage_not_implemented")


@router.get(
    "/definitions",
    response_model=ScheduledTaskDefinitionListResponse,
    dependencies=[Depends(rbac_rate_limit("tasks.read"))],
)
async def list_scheduled_task_automation_definitions(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    _principal=Depends(RequirePermission(TASKS_READ)),  # noqa: B008
    service: ScheduledTaskAutomationService = Depends(get_scheduled_task_automation_service),
) -> ScheduledTaskDefinitionListResponse:
    return service.list_definitions(limit=limit, offset=offset)


@router.post(
    "/definitions",
    response_model=ScheduledTaskDefinitionResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(rbac_rate_limit("tasks.control"))],
)
async def create_scheduled_task_automation_definition(
    _payload: ScheduledTaskDefinitionCreateRequest,
    _principal=Depends(RequirePermission(TASKS_CONTROL)),  # noqa: B008
) -> ScheduledTaskDefinitionResponse:
    _raise_automation_not_implemented("scheduled_task_definition_storage_not_implemented")


@router.get(
    "/definitions/{definition_id}/audit",
    response_model=ScheduledTaskAuditListResponse,
    dependencies=[Depends(rbac_rate_limit("tasks.read"))],
)
async def list_scheduled_task_automation_definition_audit(
    definition_id: str = Path(..., min_length=1),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    _principal=Depends(RequirePermission(TASKS_READ)),  # noqa: B008
    service: ScheduledTaskAutomationService = Depends(get_scheduled_task_automation_service),
) -> ScheduledTaskAuditListResponse:
    return service.list_audit_events(limit=limit, offset=offset)


@router.get(
    "/definitions/{definition_id}",
    response_model=ScheduledTaskDefinitionResponse,
    dependencies=[Depends(rbac_rate_limit("tasks.read"))],
)
async def get_scheduled_task_automation_definition(
    definition_id: str = Path(..., min_length=1),
    _principal=Depends(RequirePermission(TASKS_READ)),  # noqa: B008
) -> ScheduledTaskDefinitionResponse:
    _raise_automation_not_implemented("scheduled_task_definition_storage_not_implemented")


@router.patch(
    "/definitions/{definition_id}",
    response_model=ScheduledTaskDefinitionResponse,
    dependencies=[Depends(rbac_rate_limit("tasks.control"))],
)
async def update_scheduled_task_automation_definition(
    _payload: ScheduledTaskDefinitionUpdateRequest,
    definition_id: str = Path(..., min_length=1),
    _principal=Depends(RequirePermission(TASKS_CONTROL)),  # noqa: B008
) -> ScheduledTaskDefinitionResponse:
    _raise_automation_not_implemented("scheduled_task_definition_storage_not_implemented")


@router.post(
    "/definitions/{definition_id}/pause",
    response_model=ScheduledTaskDefinitionResponse,
    dependencies=[Depends(rbac_rate_limit("tasks.control"))],
)
async def pause_scheduled_task_automation_definition(
    definition_id: str = Path(..., min_length=1),
    _principal=Depends(RequirePermission(TASKS_CONTROL)),  # noqa: B008
) -> ScheduledTaskDefinitionResponse:
    _raise_automation_not_implemented("scheduled_task_definition_lifecycle_not_implemented")


@router.post(
    "/definitions/{definition_id}/resume",
    response_model=ScheduledTaskDefinitionResponse,
    dependencies=[Depends(rbac_rate_limit("tasks.control"))],
)
async def resume_scheduled_task_automation_definition(
    definition_id: str = Path(..., min_length=1),
    _principal=Depends(RequirePermission(TASKS_CONTROL)),  # noqa: B008
) -> ScheduledTaskDefinitionResponse:
    _raise_automation_not_implemented("scheduled_task_definition_lifecycle_not_implemented")


@router.post(
    "/definitions/{definition_id}/archive",
    response_model=ScheduledTaskDefinitionResponse,
    dependencies=[Depends(rbac_rate_limit("tasks.control"))],
)
async def archive_scheduled_task_automation_definition(
    definition_id: str = Path(..., min_length=1),
    _principal=Depends(RequirePermission(TASKS_CONTROL)),  # noqa: B008
) -> ScheduledTaskDefinitionResponse:
    _raise_automation_not_implemented("scheduled_task_definition_lifecycle_not_implemented")


@router.post(
    "/definitions/{definition_id}/duplicate",
    response_model=ScheduledTaskDefinitionResponse,
    dependencies=[Depends(rbac_rate_limit("tasks.control"))],
)
async def duplicate_scheduled_task_automation_definition(
    _payload: ScheduledTaskDuplicateRequest,
    definition_id: str = Path(..., min_length=1),
    _principal=Depends(RequirePermission(TASKS_CONTROL)),  # noqa: B008
) -> ScheduledTaskDefinitionResponse:
    _raise_automation_not_implemented("scheduled_task_definition_lifecycle_not_implemented")


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
