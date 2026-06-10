from __future__ import annotations

from typing import Any, NoReturn

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, status
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


def _scheduled_task_error(
    *,
    request: Request,
    status_code: int,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
    retryable: bool = False,
) -> HTTPException:
    correlation_id = getattr(getattr(request, "state", None), "request_id", None)
    return HTTPException(
        status_code=status_code,
        detail={
            "code": code,
            "message": message,
            "details": details or {},
            "field_errors": [],
            "retryable": retryable,
            "correlation_id": correlation_id,
        },
    )


_VALUE_ERROR_MAP: dict[str, tuple[int, str, str]] = {
    "scheduled_task_family_unavailable": (
        status.HTTP_409_CONFLICT,
        "scheduled_task_family_unavailable",
        "Scheduled task automation family is unavailable.",
    ),
    "scheduled_task_preview_required": (
        status.HTTP_400_BAD_REQUEST,
        "scheduled_task_preview_required",
        "A valid scheduled task preview is required.",
    ),
    "scheduled_task_preview_mismatch": (
        status.HTTP_409_CONFLICT,
        "scheduled_task_preview_mismatch",
        "Scheduled task preview does not match the requested operation.",
    ),
    "scheduled_task_preview_expired": (
        status.HTTP_409_CONFLICT,
        "scheduled_task_preview_expired",
        "Scheduled task preview has expired.",
    ),
    "scheduled_task_schedule_invalid": (
        status.HTTP_422_UNPROCESSABLE_ENTITY,
        "scheduled_task_schedule_invalid",
        "Scheduled task schedule is invalid.",
    ),
    "scheduled_task_scope_invalid": (
        status.HTTP_422_UNPROCESSABLE_ENTITY,
        "scheduled_task_scope_invalid",
        "Scheduled task scope is invalid.",
    ),
    "scheduled_task_agent_ref_invalid": (
        status.HTTP_422_UNPROCESSABLE_ENTITY,
        "scheduled_task_agent_ref_invalid",
        "Scheduled task agent reference is invalid.",
    ),
    "scheduled_task_permission_policy_invalid": (
        status.HTTP_422_UNPROCESSABLE_ENTITY,
        "scheduled_task_permission_policy_invalid",
        "Scheduled task permission policy is invalid.",
    ),
    "scheduled_task_execution_unavailable": (
        status.HTTP_409_CONFLICT,
        "scheduled_task_execution_unavailable",
        "Scheduled task execution is unavailable.",
    ),
    "scheduled_task_definition_version_conflict": (
        status.HTTP_409_CONFLICT,
        "scheduled_task_definition_version_conflict",
        "Scheduled task definition version conflict.",
    ),
    "scheduled_task_definition_archived": (
        status.HTTP_409_CONFLICT,
        "scheduled_task_definition_archived",
        "Scheduled task definition is archived.",
    ),
    "scheduled_task_lifecycle_transition_invalid": (
        status.HTTP_409_CONFLICT,
        "scheduled_task_lifecycle_transition_invalid",
        "Scheduled task lifecycle transition is invalid.",
    ),
    "scheduled_task_idempotency_conflict": (
        status.HTTP_409_CONFLICT,
        "scheduled_task_idempotency_conflict",
        "Idempotency key was already used with a different payload.",
    ),
    "preview_not_found": (
        status.HTTP_400_BAD_REQUEST,
        "scheduled_task_preview_required",
        "A valid scheduled task preview is required.",
    ),
    "preview_invalid": (
        status.HTTP_422_UNPROCESSABLE_ENTITY,
        "scheduled_task_schedule_invalid",
        "Scheduled task preview is invalid.",
    ),
    "preview_expired": (
        status.HTTP_409_CONFLICT,
        "scheduled_task_preview_expired",
        "Scheduled task preview has expired.",
    ),
    "preview_consumed": (
        status.HTTP_409_CONFLICT,
        "scheduled_task_preview_mismatch",
        "Scheduled task preview has already been consumed.",
    ),
    "preview_mode_mismatch": (
        status.HTTP_409_CONFLICT,
        "scheduled_task_preview_mismatch",
        "Scheduled task preview mode does not match the requested operation.",
    ),
    "preview_definition_mismatch": (
        status.HTTP_409_CONFLICT,
        "scheduled_task_preview_mismatch",
        "Scheduled task preview definition does not match the requested definition.",
    ),
    "definition_not_found": (
        status.HTTP_404_NOT_FOUND,
        "scheduled_task_definition_not_found",
        "Scheduled task definition was not found.",
    ),
    "definition_archived": (
        status.HTTP_409_CONFLICT,
        "scheduled_task_definition_archived",
        "Scheduled task definition is archived.",
    ),
    "definition_version_mismatch": (
        status.HTTP_409_CONFLICT,
        "scheduled_task_definition_version_conflict",
        "Scheduled task definition version conflict.",
    ),
    "definition_disabled": (
        status.HTTP_409_CONFLICT,
        "scheduled_task_lifecycle_transition_invalid",
        "Scheduled task definition is disabled.",
    ),
    "definition_disabled_locked": (
        status.HTTP_409_CONFLICT,
        "scheduled_task_lifecycle_transition_invalid",
        "Scheduled task definition is locked by policy.",
    ),
}


def _raise_automation_error(request: Request, exc: Exception) -> NoReturn:
    raw_code = str(exc)
    if isinstance(exc, KeyError):
        raw_code = raw_code.strip("'")
        if "definition" in raw_code:
            status_code, code, message = _VALUE_ERROR_MAP["definition_not_found"]
        else:
            status_code, code, message = _VALUE_ERROR_MAP["preview_not_found"]
    else:
        status_code, code, message = _VALUE_ERROR_MAP.get(
            raw_code,
            (
                status.HTTP_409_CONFLICT,
                "scheduled_task_lifecycle_transition_invalid",
                "Scheduled task automation request could not be completed.",
            ),
        )
    raise _scheduled_task_error(
        request=request,
        status_code=status_code,
        code=code,
        message=message,
        details={"reason": raw_code},
    ) from exc


def _actor_from_principal(principal: Any, current_user: User) -> str:
    subject = getattr(principal, "subject", None)
    return str(subject or current_user.id)


def _idempotency_key(request: Request) -> str | None:
    value = request.headers.get("Idempotency-Key")
    return value.strip() if isinstance(value, str) and value.strip() else None


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
    family: str | None = Query(default=None),
    mode: str | None = Query(default=None),
    status: str | None = Query(default=None),
    definition_id: str | None = Query(default=None),
    expired: bool | None = Query(default=None),
    current_user: User = Depends(get_request_user),
    _principal=Depends(RequirePermission(TASKS_READ)),  # noqa: B008
    service: ScheduledTaskAutomationService = Depends(get_scheduled_task_automation_service),
) -> ScheduledTaskPreviewListResponse:
    return service.list_previews(
        owner_id=int(current_user.id),
        limit=limit,
        offset=offset,
        family=family,
        mode=mode,
        status=status,
        definition_id=definition_id,
        expired=expired,
    )


@router.post(
    "/previews",
    response_model=ScheduledTaskPreviewResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(rbac_rate_limit("tasks.control"))],
)
async def create_scheduled_task_automation_preview(
    payload: ScheduledTaskPreviewCreateRequest,
    request: Request,
    current_user: User = Depends(get_request_user),
    principal=Depends(RequirePermission(TASKS_CONTROL)),  # noqa: B008
    service: ScheduledTaskAutomationService = Depends(get_scheduled_task_automation_service),
) -> ScheduledTaskPreviewResponse:
    try:
        return service.create_preview(
            owner_id=int(current_user.id),
            actor=_actor_from_principal(principal, current_user),
            payload=payload,
            idempotency_key=_idempotency_key(request),
        )
    except (KeyError, ValueError) as exc:
        _raise_automation_error(request, exc)


@router.get(
    "/previews/{preview_id}",
    response_model=ScheduledTaskPreviewResponse,
    dependencies=[Depends(rbac_rate_limit("tasks.read"))],
)
async def get_scheduled_task_automation_preview(
    request: Request,
    preview_id: str = Path(..., min_length=1),
    current_user: User = Depends(get_request_user),
    _principal=Depends(RequirePermission(TASKS_READ)),  # noqa: B008
    service: ScheduledTaskAutomationService = Depends(get_scheduled_task_automation_service),
) -> ScheduledTaskPreviewResponse:
    try:
        return service.get_preview(owner_id=int(current_user.id), preview_id=preview_id)
    except (KeyError, ValueError) as exc:
        _raise_automation_error(request, exc)


@router.get(
    "/definitions",
    response_model=ScheduledTaskDefinitionListResponse,
    dependencies=[Depends(rbac_rate_limit("tasks.read"))],
)
async def list_scheduled_task_automation_definitions(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    family: str | None = Query(default=None),
    lifecycle: str | None = Query(default=None),
    health: str | None = Query(default=None),
    visibility_policy: str | None = Query(default=None),
    q: str | None = Query(default=None),
    created_from: str | None = Query(default=None),
    created_to: str | None = Query(default=None),
    current_user: User = Depends(get_request_user),
    _principal=Depends(RequirePermission(TASKS_READ)),  # noqa: B008
    service: ScheduledTaskAutomationService = Depends(get_scheduled_task_automation_service),
) -> ScheduledTaskDefinitionListResponse:
    return service.list_definitions(
        owner_id=int(current_user.id),
        limit=limit,
        offset=offset,
        family=family,
        lifecycle=lifecycle,
        health=health,
        visibility_policy=visibility_policy,
        query=q,
        created_from=created_from,
        created_to=created_to,
    )


@router.post(
    "/definitions",
    response_model=ScheduledTaskDefinitionResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(rbac_rate_limit("tasks.control"))],
)
async def create_scheduled_task_automation_definition(
    payload: ScheduledTaskDefinitionCreateRequest,
    request: Request,
    current_user: User = Depends(get_request_user),
    principal=Depends(RequirePermission(TASKS_CONTROL)),  # noqa: B008
    service: ScheduledTaskAutomationService = Depends(get_scheduled_task_automation_service),
) -> ScheduledTaskDefinitionResponse:
    try:
        return service.create_definition(
            owner_id=int(current_user.id),
            actor=_actor_from_principal(principal, current_user),
            payload=payload,
            idempotency_key=_idempotency_key(request),
        )
    except (KeyError, ValueError) as exc:
        _raise_automation_error(request, exc)


@router.get(
    "/definitions/{definition_id}/audit",
    response_model=ScheduledTaskAuditListResponse,
    dependencies=[Depends(rbac_rate_limit("tasks.read"))],
)
async def list_scheduled_task_automation_definition_audit(
    request: Request,
    definition_id: str = Path(..., min_length=1),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    event_type: str | None = Query(default=None),
    actor: str | None = Query(default=None),
    created_from: str | None = Query(default=None),
    created_to: str | None = Query(default=None),
    idempotency_key: str | None = Query(default=None),
    request_id: str | None = Query(default=None),
    current_user: User = Depends(get_request_user),
    _principal=Depends(RequirePermission(TASKS_READ)),  # noqa: B008
    service: ScheduledTaskAutomationService = Depends(get_scheduled_task_automation_service),
) -> ScheduledTaskAuditListResponse:
    try:
        return service.list_audit_events(
            owner_id=int(current_user.id),
            definition_id=definition_id,
            limit=limit,
            offset=offset,
            event_type=event_type,
            actor=actor,
            created_from=created_from,
            created_to=created_to,
            idempotency_key=idempotency_key,
            request_id=request_id,
        )
    except (KeyError, ValueError) as exc:
        _raise_automation_error(request, exc)


@router.get(
    "/definitions/{definition_id}",
    response_model=ScheduledTaskDefinitionResponse,
    dependencies=[Depends(rbac_rate_limit("tasks.read"))],
)
async def get_scheduled_task_automation_definition(
    request: Request,
    definition_id: str = Path(..., min_length=1),
    current_user: User = Depends(get_request_user),
    _principal=Depends(RequirePermission(TASKS_READ)),  # noqa: B008
    service: ScheduledTaskAutomationService = Depends(get_scheduled_task_automation_service),
) -> ScheduledTaskDefinitionResponse:
    try:
        return service.get_definition(owner_id=int(current_user.id), definition_id=definition_id)
    except (KeyError, ValueError) as exc:
        _raise_automation_error(request, exc)


@router.patch(
    "/definitions/{definition_id}",
    response_model=ScheduledTaskDefinitionResponse,
    dependencies=[Depends(rbac_rate_limit("tasks.control"))],
)
async def update_scheduled_task_automation_definition(
    payload: ScheduledTaskDefinitionUpdateRequest,
    request: Request,
    definition_id: str = Path(..., min_length=1),
    current_user: User = Depends(get_request_user),
    principal=Depends(RequirePermission(TASKS_CONTROL)),  # noqa: B008
    service: ScheduledTaskAutomationService = Depends(get_scheduled_task_automation_service),
) -> ScheduledTaskDefinitionResponse:
    try:
        return service.update_definition(
            owner_id=int(current_user.id),
            actor=_actor_from_principal(principal, current_user),
            definition_id=definition_id,
            payload=payload,
            idempotency_key=_idempotency_key(request),
        )
    except (KeyError, ValueError) as exc:
        _raise_automation_error(request, exc)


@router.post(
    "/definitions/{definition_id}/pause",
    response_model=ScheduledTaskDefinitionResponse,
    dependencies=[Depends(rbac_rate_limit("tasks.control"))],
)
async def pause_scheduled_task_automation_definition(
    request: Request,
    definition_id: str = Path(..., min_length=1),
    current_user: User = Depends(get_request_user),
    principal=Depends(RequirePermission(TASKS_CONTROL)),  # noqa: B008
    service: ScheduledTaskAutomationService = Depends(get_scheduled_task_automation_service),
) -> ScheduledTaskDefinitionResponse:
    try:
        return service.pause_definition(
            owner_id=int(current_user.id),
            actor=_actor_from_principal(principal, current_user),
            definition_id=definition_id,
            idempotency_key=_idempotency_key(request),
        )
    except (KeyError, ValueError) as exc:
        _raise_automation_error(request, exc)


@router.post(
    "/definitions/{definition_id}/resume",
    response_model=ScheduledTaskDefinitionResponse,
    dependencies=[Depends(rbac_rate_limit("tasks.control"))],
)
async def resume_scheduled_task_automation_definition(
    request: Request,
    definition_id: str = Path(..., min_length=1),
    current_user: User = Depends(get_request_user),
    principal=Depends(RequirePermission(TASKS_CONTROL)),  # noqa: B008
    service: ScheduledTaskAutomationService = Depends(get_scheduled_task_automation_service),
) -> ScheduledTaskDefinitionResponse:
    try:
        return service.resume_definition(
            owner_id=int(current_user.id),
            actor=_actor_from_principal(principal, current_user),
            definition_id=definition_id,
            idempotency_key=_idempotency_key(request),
        )
    except (KeyError, ValueError) as exc:
        _raise_automation_error(request, exc)


@router.post(
    "/definitions/{definition_id}/archive",
    response_model=ScheduledTaskDefinitionResponse,
    dependencies=[Depends(rbac_rate_limit("tasks.control"))],
)
async def archive_scheduled_task_automation_definition(
    request: Request,
    definition_id: str = Path(..., min_length=1),
    current_user: User = Depends(get_request_user),
    principal=Depends(RequirePermission(TASKS_CONTROL)),  # noqa: B008
    service: ScheduledTaskAutomationService = Depends(get_scheduled_task_automation_service),
) -> ScheduledTaskDefinitionResponse:
    try:
        return service.archive_definition(
            owner_id=int(current_user.id),
            actor=_actor_from_principal(principal, current_user),
            definition_id=definition_id,
            idempotency_key=_idempotency_key(request),
        )
    except (KeyError, ValueError) as exc:
        _raise_automation_error(request, exc)


@router.post(
    "/definitions/{definition_id}/duplicate",
    response_model=ScheduledTaskDefinitionResponse,
    dependencies=[Depends(rbac_rate_limit("tasks.control"))],
)
async def duplicate_scheduled_task_automation_definition(
    payload: ScheduledTaskDuplicateRequest,
    request: Request,
    definition_id: str = Path(..., min_length=1),
    current_user: User = Depends(get_request_user),
    principal=Depends(RequirePermission(TASKS_CONTROL)),  # noqa: B008
    service: ScheduledTaskAutomationService = Depends(get_scheduled_task_automation_service),
) -> ScheduledTaskDefinitionResponse:
    try:
        return service.duplicate_definition(
            owner_id=int(current_user.id),
            actor=_actor_from_principal(principal, current_user),
            definition_id=definition_id,
            payload=payload,
            idempotency_key=_idempotency_key(request),
        )
    except (KeyError, ValueError) as exc:
        _raise_automation_error(request, exc)


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
