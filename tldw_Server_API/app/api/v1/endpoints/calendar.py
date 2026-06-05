from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    RequirePermission,
    User,
    get_request_user,
    rbac_rate_limit,
)
from tldw_Server_API.app.api.v1.schemas.calendar_schemas import (
    CalendarAnnotationCreateRequest,
    CalendarAnnotationResponse,
    CalendarCreateRequest,
    CalendarItemCopyRequest,
    CalendarItemCreateRequest,
    CalendarItemResponse,
    CalendarItemUpdateRequest,
    CalendarLinkCreateRequest,
    CalendarLinkResponse,
    CalendarListResponse,
    CalendarMembershipCreateRequest,
    CalendarMembershipDeleteResponse,
    CalendarMembershipListResponse,
    CalendarMembershipResponse,
    CalendarReminderCreateRequest,
    CalendarReminderProjectionResponse,
    CalendarReminderResponse,
    CalendarResponse,
    CalendarSyncTriggerResponse,
    CalendarViewItemResponse,
    CalendarViewLinkResponse,
    CalendarViewResponse,
    ExternalCalendarAccountCreateRequest,
    ExternalCalendarAccountListResponse,
    ExternalCalendarAccountResponse,
    ExternalCalendarBindingCreateRequest,
    ExternalCalendarBindingListResponse,
    ExternalCalendarBindingResponse,
)
from tldw_Server_API.app.api.v1.schemas.reminders_schemas import ReminderTaskCreateRequest
from tldw_Server_API.app.core.AuthNZ.permissions import (
    CALENDAR_READ,
    CALENDAR_SYNC,
    CALENDAR_WRITE,
)
from tldw_Server_API.app.core.Calendar.calendar_service import CalendarService
from tldw_Server_API.app.core.Calendar.errors import (
    CalendarItemNotFound,
    CalendarNotFound,
    CalendarPermissionDenied,
    CalendarReadOnlyError,
    CalendarValidationError,
)
from tldw_Server_API.app.core.Calendar.view_service import (
    CalendarViewFilters,
    CalendarViewResult,
    CalendarViewService,
)
from tldw_Server_API.app.core.DB_Management.Calendar_DB import (
    CalendarDatabase,
    CalendarItemRow,
    CalendarRecurrenceRow,
)
from tldw_Server_API.app.services.scheduled_tasks_control_plane_service import (
    ScheduledTasksControlPlaneService,
)

router = APIRouter(prefix="/calendar", tags=["calendar"])


def get_calendar_database() -> CalendarDatabase:
    return CalendarDatabase()


def get_scheduled_tasks_service() -> ScheduledTasksControlPlaneService:
    return ScheduledTasksControlPlaneService()


def get_calendar_service(
    current_user: User = Depends(get_request_user),
    db: CalendarDatabase = Depends(get_calendar_database),
) -> CalendarService:
    return CalendarService(db=db, tenant_id=_tenant_id(current_user))


def get_calendar_view_service(
    calendar_service: CalendarService = Depends(get_calendar_service),
    scheduled_tasks_service: ScheduledTasksControlPlaneService = Depends(get_scheduled_tasks_service),
) -> CalendarViewService:
    return CalendarViewService(
        calendar_service=calendar_service,
        scheduled_tasks_service=scheduled_tasks_service,
    )


def _user_id(current_user: User) -> int:
    try:
        return int(current_user.id)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "invalid_user_id", "message": "Calendar API requires a numeric user id"},
        ) from exc


def _tenant_id(current_user: User) -> str:
    return current_user.tenant_id or "default"


def _http_error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message},
    )


def _map_calendar_error(exc: Exception) -> HTTPException:
    if isinstance(exc, CalendarReadOnlyError):
        return _http_error(status.HTTP_409_CONFLICT, "item_read_only", str(exc))
    if isinstance(exc, (CalendarNotFound, CalendarItemNotFound)):
        return _http_error(status.HTTP_404_NOT_FOUND, "calendar_not_found", str(exc))
    if isinstance(exc, CalendarPermissionDenied):
        return _http_error(status.HTTP_403_FORBIDDEN, "calendar_permission_denied", str(exc))
    if isinstance(exc, CalendarValidationError):
        return _http_error(status.HTTP_400_BAD_REQUEST, "calendar_validation_error", str(exc))
    return _http_error(status.HTTP_500_INTERNAL_SERVER_ERROR, "calendar_error", "Calendar request failed")


def _recurrences_for_item(db: CalendarDatabase, item: CalendarItemRow) -> CalendarRecurrenceRow | None:
    return db.list_recurrences_for_items([item.id]).get(item.id)


def _item_response(db: CalendarDatabase, item: CalendarItemRow) -> CalendarItemResponse:
    return CalendarItemResponse.from_row(item, recurrence=_recurrences_for_item(db, item))


def _assert_external_account_owner(
    db: CalendarDatabase,
    *,
    account_id: int,
    current_user: User,
) -> None:
    account = db.get_external_account(account_id)
    if account.user_id != _user_id(current_user) or account.tenant_id != _tenant_id(current_user):
        raise CalendarPermissionDenied("External calendar account is outside the current user scope")


def _assert_external_binding_owner(
    db: CalendarDatabase,
    *,
    binding_id: int,
    current_user: User,
) -> None:
    binding = db.get_external_binding(binding_id)
    _assert_external_account_owner(db, account_id=binding.account_id, current_user=current_user)


def _upsert_recurrence_if_present(
    db: CalendarDatabase,
    *,
    item_id: int,
    payload: CalendarItemCreateRequest | CalendarItemUpdateRequest,
) -> None:
    if "recurrence" not in payload.model_fields_set or payload.recurrence is None:
        return
    recurrence = payload.recurrence
    db.upsert_recurrence(
        calendar_item_id=item_id,
        rrule=recurrence.rrule,
        rdate_json=recurrence.rdate,
        exdate_json=recurrence.exdate,
        timezone=recurrence.timezone,
    )


def _view_response(result: CalendarViewResult) -> CalendarViewResponse:
    items: list[CalendarViewItemResponse] = []
    for item in result.items:
        link = None
        if item.link is not None:
            link = CalendarViewLinkResponse(
                target_type=item.link.target_type,
                target_id=item.link.target_id,
                label=item.link.label,
                url=item.link.url,
                metadata=item.link.metadata,
            )
        items.append(
            CalendarViewItemResponse(
                id=item.id,
                title=item.title,
                source_owner=item.source_owner,
                start_at=item.start_at,
                end_at=item.end_at,
                due_at=item.due_at,
                calendar_id=item.calendar_id,
                calendar_item_id=item.calendar_item_id,
                description=item.description,
                location=item.location,
                all_day=item.all_day,
                status=item.status,
                read_only_reason=item.read_only_reason,
                recurrence_id=item.recurrence_id,
                occurrence_index=item.occurrence_index,
                link=link,
                metadata=item.metadata,
            )
        )
    return CalendarViewResponse(start_at=result.start_at, end_at=result.end_at, items=items)


@router.post(
    "/calendars",
    response_model=CalendarResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(rbac_rate_limit("calendar.write"))],
)
async def create_calendar(
    payload: CalendarCreateRequest,
    current_user: User = Depends(get_request_user),
    _principal=Depends(RequirePermission(CALENDAR_WRITE)),  # noqa: B008
    service: CalendarService = Depends(get_calendar_service),
) -> CalendarResponse:
    try:
        row = service.create_calendar(
            actor_user_id=_user_id(current_user),
            name=payload.name,
            timezone=payload.timezone,
            org_id=payload.org_id,
            color=payload.color,
            description=payload.description,
            visibility=payload.visibility,
            default_reminder_policy_json=payload.default_reminder_policy,
            rbac_policy_ref=payload.rbac_policy_ref,
        )
    except (CalendarReadOnlyError, CalendarNotFound, CalendarItemNotFound, CalendarPermissionDenied, CalendarValidationError) as exc:
        raise _map_calendar_error(exc) from exc
    return CalendarResponse.from_row(row)


@router.get(
    "/calendars",
    response_model=CalendarListResponse,
    dependencies=[Depends(rbac_rate_limit("calendar.read"))],
)
async def list_calendars(
    include_archived: bool = Query(default=False),
    current_user: User = Depends(get_request_user),
    _principal=Depends(RequirePermission(CALENDAR_READ)),  # noqa: B008
    service: CalendarService = Depends(get_calendar_service),
) -> CalendarListResponse:
    try:
        rows = service.list_calendars(
            actor_user_id=_user_id(current_user),
            include_archived=include_archived,
        )
    except (CalendarReadOnlyError, CalendarNotFound, CalendarItemNotFound, CalendarPermissionDenied, CalendarValidationError) as exc:
        raise _map_calendar_error(exc) from exc
    items = [CalendarResponse.from_row(row) for row in rows]
    return CalendarListResponse(items=items, total=len(items))


@router.post(
    "/calendars/{calendar_id}/memberships",
    response_model=CalendarMembershipResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(rbac_rate_limit("calendar.write"))],
)
async def add_calendar_membership(
    payload: CalendarMembershipCreateRequest,
    calendar_id: int = Path(..., ge=1),
    current_user: User = Depends(get_request_user),
    _principal=Depends(RequirePermission(CALENDAR_WRITE)),  # noqa: B008
    service: CalendarService = Depends(get_calendar_service),
) -> CalendarMembershipResponse:
    try:
        row = service.add_membership(
            actor_user_id=_user_id(current_user),
            calendar_id=calendar_id,
            principal_type=payload.principal_type,
            principal_id=payload.principal_id,
            role=payload.role,
        )
    except (CalendarReadOnlyError, CalendarNotFound, CalendarItemNotFound, CalendarPermissionDenied, CalendarValidationError) as exc:
        raise _map_calendar_error(exc) from exc
    return CalendarMembershipResponse.from_row(row)


@router.get(
    "/calendars/{calendar_id}/memberships",
    response_model=CalendarMembershipListResponse,
    dependencies=[Depends(rbac_rate_limit("calendar.read"))],
)
async def list_calendar_memberships(
    calendar_id: int = Path(..., ge=1),
    current_user: User = Depends(get_request_user),
    _principal=Depends(RequirePermission(CALENDAR_READ)),  # noqa: B008
    service: CalendarService = Depends(get_calendar_service),
) -> CalendarMembershipListResponse:
    try:
        rows = service.list_memberships(actor_user_id=_user_id(current_user), calendar_id=calendar_id)
    except (CalendarReadOnlyError, CalendarNotFound, CalendarItemNotFound, CalendarPermissionDenied, CalendarValidationError) as exc:
        raise _map_calendar_error(exc) from exc
    items = [CalendarMembershipResponse.from_row(row) for row in rows]
    return CalendarMembershipListResponse(items=items, total=len(items))


@router.delete(
    "/calendars/{calendar_id}/memberships/{principal_type}/{principal_id}",
    response_model=CalendarMembershipDeleteResponse,
    dependencies=[Depends(rbac_rate_limit("calendar.write"))],
)
async def remove_calendar_membership(
    calendar_id: int = Path(..., ge=1),
    principal_type: str = Path(..., min_length=1),
    principal_id: str = Path(..., min_length=1),
    current_user: User = Depends(get_request_user),
    _principal=Depends(RequirePermission(CALENDAR_WRITE)),  # noqa: B008
    service: CalendarService = Depends(get_calendar_service),
) -> CalendarMembershipDeleteResponse:
    try:
        removed = service.remove_membership(
            actor_user_id=_user_id(current_user),
            calendar_id=calendar_id,
            principal_type=principal_type,
            principal_id=principal_id,
        )
    except (CalendarReadOnlyError, CalendarNotFound, CalendarItemNotFound, CalendarPermissionDenied, CalendarValidationError) as exc:
        raise _map_calendar_error(exc) from exc
    return CalendarMembershipDeleteResponse(removed=removed)


@router.post(
    "/items",
    response_model=CalendarItemResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(rbac_rate_limit("calendar.write"))],
)
async def create_calendar_item(
    payload: CalendarItemCreateRequest,
    current_user: User = Depends(get_request_user),
    _principal=Depends(RequirePermission(CALENDAR_WRITE)),  # noqa: B008
    service: CalendarService = Depends(get_calendar_service),
    db: CalendarDatabase = Depends(get_calendar_database),
) -> CalendarItemResponse:
    try:
        item = service.create_item(
            actor_user_id=_user_id(current_user),
            calendar_id=payload.calendar_id,
            kind=payload.kind,
            title=payload.title,
            description=payload.description,
            location=payload.location,
            start_at=payload.start_at,
            end_at=payload.end_at,
            due_at=payload.due_at,
            timezone=payload.timezone,
            all_day=payload.all_day,
            status=payload.status,
            local_tags_json=payload.local_tags,
            metadata_json=payload.metadata,
        )
        _upsert_recurrence_if_present(db, item_id=item.id, payload=payload)
        item = db.get_item(item.id)
    except (CalendarReadOnlyError, CalendarNotFound, CalendarItemNotFound, CalendarPermissionDenied, CalendarValidationError) as exc:
        raise _map_calendar_error(exc) from exc
    return _item_response(db, item)


@router.patch(
    "/items/{item_id}",
    response_model=CalendarItemResponse,
    dependencies=[Depends(rbac_rate_limit("calendar.write"))],
)
async def update_calendar_item(
    payload: CalendarItemUpdateRequest,
    item_id: int = Path(..., ge=1),
    current_user: User = Depends(get_request_user),
    _principal=Depends(RequirePermission(CALENDAR_WRITE)),  # noqa: B008
    service: CalendarService = Depends(get_calendar_service),
    db: CalendarDatabase = Depends(get_calendar_database),
) -> CalendarItemResponse:
    try:
        item = service.update_item(
            actor_user_id=_user_id(current_user),
            item_id=item_id,
            **payload.service_updates(),
        )
        _upsert_recurrence_if_present(db, item_id=item.id, payload=payload)
        item = db.get_item(item.id)
    except (CalendarReadOnlyError, CalendarNotFound, CalendarItemNotFound, CalendarPermissionDenied, CalendarValidationError) as exc:
        raise _map_calendar_error(exc) from exc
    return _item_response(db, item)


@router.get(
    "/views/agenda",
    response_model=CalendarViewResponse,
    dependencies=[Depends(rbac_rate_limit("calendar.read"))],
)
async def get_calendar_agenda(
    start_at: str = Query(..., min_length=1),
    end_at: str = Query(..., min_length=1),
    calendar_ids: list[int] | None = Query(default=None),
    include_scheduled_tasks: bool = Query(default=True),
    current_user: User = Depends(get_request_user),
    _principal=Depends(RequirePermission(CALENDAR_READ)),  # noqa: B008
    view_service: CalendarViewService = Depends(get_calendar_view_service),
) -> CalendarViewResponse:
    try:
        result = await view_service.agenda(
            actor_user_id=_user_id(current_user),
            start_at=start_at,
            end_at=end_at,
            filters=CalendarViewFilters(
                calendar_ids=calendar_ids,
                include_scheduled_tasks=include_scheduled_tasks,
                include_provider_tombstones=False,
            ),
        )
    except (CalendarReadOnlyError, CalendarNotFound, CalendarItemNotFound, CalendarPermissionDenied, CalendarValidationError) as exc:
        raise _map_calendar_error(exc) from exc
    except (TypeError, ValueError) as exc:
        raise _map_calendar_error(CalendarValidationError(str(exc))) from exc
    return _view_response(result)


@router.post(
    "/items/{item_id}/annotations",
    response_model=CalendarAnnotationResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(rbac_rate_limit("calendar.write"))],
)
async def create_calendar_annotation(
    payload: CalendarAnnotationCreateRequest,
    item_id: int = Path(..., ge=1),
    current_user: User = Depends(get_request_user),
    _principal=Depends(RequirePermission(CALENDAR_WRITE)),  # noqa: B008
    service: CalendarService = Depends(get_calendar_service),
) -> CalendarAnnotationResponse:
    try:
        row = service.create_annotation(
            actor_user_id=_user_id(current_user),
            item_id=item_id,
            body=payload.body,
            tags=payload.tags,
        )
    except (CalendarReadOnlyError, CalendarNotFound, CalendarItemNotFound, CalendarPermissionDenied, CalendarValidationError) as exc:
        raise _map_calendar_error(exc) from exc
    return CalendarAnnotationResponse.from_row(row)


@router.post(
    "/items/{item_id}/links",
    response_model=CalendarLinkResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(rbac_rate_limit("calendar.write"))],
)
async def create_calendar_link(
    payload: CalendarLinkCreateRequest,
    item_id: int = Path(..., ge=1),
    current_user: User = Depends(get_request_user),
    _principal=Depends(RequirePermission(CALENDAR_WRITE)),  # noqa: B008
    service: CalendarService = Depends(get_calendar_service),
) -> CalendarLinkResponse:
    try:
        row = service.create_link(
            actor_user_id=_user_id(current_user),
            item_id=item_id,
            target_type=payload.target_type,
            target_id=payload.target_id,
            label=payload.label,
            url=payload.url,
            metadata_json=payload.metadata,
        )
    except (CalendarReadOnlyError, CalendarNotFound, CalendarItemNotFound, CalendarPermissionDenied, CalendarValidationError) as exc:
        raise _map_calendar_error(exc) from exc
    return CalendarLinkResponse.from_row(row)


@router.post(
    "/items/{item_id}/copy",
    response_model=CalendarItemResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(rbac_rate_limit("calendar.write"))],
)
async def copy_calendar_item(
    payload: CalendarItemCopyRequest,
    item_id: int = Path(..., ge=1),
    current_user: User = Depends(get_request_user),
    _principal=Depends(RequirePermission(CALENDAR_WRITE)),  # noqa: B008
    service: CalendarService = Depends(get_calendar_service),
    db: CalendarDatabase = Depends(get_calendar_database),
) -> CalendarItemResponse:
    try:
        item = service.copy_provider_item(
            actor_user_id=_user_id(current_user),
            item_id=item_id,
            target_calendar_id=payload.target_calendar_id,
            title=payload.title,
        )
    except (CalendarReadOnlyError, CalendarNotFound, CalendarItemNotFound, CalendarPermissionDenied, CalendarValidationError) as exc:
        raise _map_calendar_error(exc) from exc
    return _item_response(db, item)


@router.post(
    "/reminders",
    response_model=CalendarReminderResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(rbac_rate_limit("calendar.write"))],
)
async def create_calendar_reminder(
    payload: CalendarReminderCreateRequest,
    current_user: User = Depends(get_request_user),
    _principal=Depends(RequirePermission(CALENDAR_WRITE)),  # noqa: B008
    service: CalendarService = Depends(get_calendar_service),
    scheduled_tasks_service: ScheduledTasksControlPlaneService = Depends(get_scheduled_tasks_service),
) -> CalendarReminderResponse:
    user_id = _user_id(current_user)
    try:
        service.get_item(actor_user_id=user_id, item_id=payload.calendar_item_id)
        reminder_payload = ReminderTaskCreateRequest(
            **{
                **payload.model_dump(exclude={"calendar_item_id"}),
                "link_type": "calendar_item",
                "link_id": str(payload.calendar_item_id),
            }
        )
        task = await scheduled_tasks_service.create_reminder(user_id=user_id, payload=reminder_payload)
    except (CalendarReadOnlyError, CalendarNotFound, CalendarItemNotFound, CalendarPermissionDenied, CalendarValidationError) as exc:
        raise _map_calendar_error(exc) from exc
    return CalendarReminderResponse(
        calendar_item_id=payload.calendar_item_id,
        scheduled_task=task,
        projection=CalendarReminderProjectionResponse(
            link_id=str(payload.calendar_item_id),
            next_run_at=task.next_run_at,
        ),
    )


@router.get(
    "/external/accounts",
    response_model=ExternalCalendarAccountListResponse,
    dependencies=[Depends(rbac_rate_limit("calendar.sync"))],
)
async def list_external_calendar_accounts(
    current_user: User = Depends(get_request_user),
    _principal=Depends(RequirePermission(CALENDAR_SYNC)),  # noqa: B008
    db: CalendarDatabase = Depends(get_calendar_database),
) -> ExternalCalendarAccountListResponse:
    rows = db.list_external_accounts_for_user(user_id=_user_id(current_user), tenant_id=_tenant_id(current_user))
    items = [ExternalCalendarAccountResponse.from_row(row) for row in rows]
    return ExternalCalendarAccountListResponse(items=items, total=len(items))


@router.post(
    "/external/accounts",
    response_model=ExternalCalendarAccountResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(rbac_rate_limit("calendar.sync"))],
)
async def create_external_calendar_account(
    payload: ExternalCalendarAccountCreateRequest,
    current_user: User = Depends(get_request_user),
    _principal=Depends(RequirePermission(CALENDAR_SYNC)),  # noqa: B008
    db: CalendarDatabase = Depends(get_calendar_database),
) -> ExternalCalendarAccountResponse:
    try:
        row = db.create_external_account(
            tenant_id=_tenant_id(current_user),
            user_id=_user_id(current_user),
            provider=payload.provider,
            display_name=payload.display_name,
            secret_ref=payload.secret_ref,
            account_metadata_json=payload.account_metadata,
        )
    except CalendarValidationError as exc:
        raise _map_calendar_error(exc) from exc
    return ExternalCalendarAccountResponse.from_row(row)


@router.post(
    "/external/bindings",
    response_model=ExternalCalendarBindingResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(rbac_rate_limit("calendar.sync"))],
)
async def create_external_calendar_binding(
    payload: ExternalCalendarBindingCreateRequest,
    current_user: User = Depends(get_request_user),
    _principal=Depends(RequirePermission(CALENDAR_SYNC)),  # noqa: B008
    db: CalendarDatabase = Depends(get_calendar_database),
) -> ExternalCalendarBindingResponse:
    try:
        _assert_external_account_owner(db, account_id=payload.account_id, current_user=current_user)
        row = db.create_external_binding(
            account_id=payload.account_id,
            calendar_id=payload.calendar_id,
            remote_calendar_id=payload.remote_calendar_id,
            remote_display_name=payload.remote_display_name,
            sync_enabled=payload.sync_enabled,
            sync_interval_minutes=payload.sync_interval_minutes,
            lookback_days=payload.lookback_days,
            lookahead_days=payload.lookahead_days,
            provider_capabilities_json=payload.provider_capabilities,
        )
    except (CalendarNotFound, CalendarPermissionDenied, CalendarValidationError) as exc:
        raise _map_calendar_error(exc) from exc
    return ExternalCalendarBindingResponse.from_row(row)


@router.get(
    "/external/accounts/{account_id}/bindings",
    response_model=ExternalCalendarBindingListResponse,
    dependencies=[Depends(rbac_rate_limit("calendar.sync"))],
)
async def list_external_calendar_bindings(
    account_id: int = Path(..., ge=1),
    current_user: User = Depends(get_request_user),
    _principal=Depends(RequirePermission(CALENDAR_SYNC)),  # noqa: B008
    db: CalendarDatabase = Depends(get_calendar_database),
) -> ExternalCalendarBindingListResponse:
    try:
        _assert_external_account_owner(db, account_id=account_id, current_user=current_user)
        rows = db.list_external_bindings_for_account(account_id)
    except (CalendarNotFound, CalendarPermissionDenied, CalendarValidationError) as exc:
        raise _map_calendar_error(exc) from exc
    items = [ExternalCalendarBindingResponse.from_row(row) for row in rows]
    return ExternalCalendarBindingListResponse(items=items, total=len(items))


@router.post(
    "/external/bindings/{binding_id}/sync",
    response_model=CalendarSyncTriggerResponse,
    dependencies=[Depends(rbac_rate_limit("calendar.sync"))],
)
async def trigger_external_calendar_sync(
    binding_id: int = Path(..., ge=1),
    current_user: User = Depends(get_request_user),
    _principal=Depends(RequirePermission(CALENDAR_SYNC)),  # noqa: B008
    db: CalendarDatabase = Depends(get_calendar_database),
) -> CalendarSyncTriggerResponse:
    try:
        _assert_external_binding_owner(db, binding_id=binding_id, current_user=current_user)
    except (CalendarNotFound, CalendarPermissionDenied, CalendarValidationError) as exc:
        raise _map_calendar_error(exc) from exc
    return CalendarSyncTriggerResponse(binding_id=binding_id)
