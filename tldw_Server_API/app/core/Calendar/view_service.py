"""Calendar agenda/week view expansion and linked projections."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Protocol
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dateutil import parser as date_parser

from tldw_Server_API.app.api.v1.schemas.scheduled_tasks_control_plane_schemas import (
    ScheduledTaskListResponse,
)
from tldw_Server_API.app.core.Calendar.calendar_service import CalendarService
from tldw_Server_API.app.core.Calendar.constants import (
    CALENDAR_SOURCE_OWNER_LINKED_PROJECTION,
    CALENDAR_SOURCE_OWNER_PROVIDER,
    CALENDAR_SOURCE_OWNER_TLDW,
)
from tldw_Server_API.app.core.Calendar.errors import (
    CalendarItemNotFound,
    CalendarPermissionDenied,
    CalendarValidationError,
)
from tldw_Server_API.app.core.Calendar.recurrence import (
    LocalRecurrenceRule,
    RecurrenceOccurrence,
    expand_rrule,
    validate_query_window,
)
from tldw_Server_API.app.core.DB_Management.Calendar_DB import (
    CalendarItemRow,
    CalendarRecurrenceRow,
)
from tldw_Server_API.app.services.scheduled_tasks_control_plane_service import (
    ScheduledTasksControlPlaneService,
)

TemporalValue = date | datetime | str


class ScheduledTasksListService(Protocol):
    """Subset of the scheduled-task control plane used by Calendar views."""

    async def list_tasks(self, *, user_id: int) -> ScheduledTaskListResponse:
        """Return normalized scheduled tasks for a user."""


@dataclass(frozen=True)
class CalendarViewFilters:
    """Filters accepted by backend Calendar views."""

    calendar_ids: list[int] | None = None
    include_scheduled_tasks: bool = True
    include_provider_tombstones: bool = False


@dataclass(frozen=True)
class CalendarViewLink:
    """Typed link attached to a Calendar view item."""

    target_type: str
    target_id: str
    label: str | None = None
    url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CalendarViewItem:
    """Expanded Calendar item or read-only linked projection."""

    id: str
    title: str
    source_owner: str
    start_at: str | None
    end_at: str | None = None
    due_at: str | None = None
    calendar_id: int | None = None
    calendar_item_id: int | None = None
    description: str | None = None
    location: str | None = None
    all_day: bool = False
    status: str | None = None
    read_only_reason: str | None = None
    recurrence_id: int | None = None
    occurrence_index: int | None = None
    link: CalendarViewLink | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CalendarViewResult:
    """Expanded Calendar view response."""

    start_at: str
    end_at: str
    items: list[CalendarViewItem]


class CalendarViewService:
    """Build permission-aware Calendar agenda and week views."""

    def __init__(
        self,
        *,
        calendar_service: CalendarService,
        scheduled_tasks_service: ScheduledTasksListService | None = None,
    ) -> None:
        self.calendar_service = calendar_service
        self.scheduled_tasks_service = scheduled_tasks_service or ScheduledTasksControlPlaneService()

    async def agenda(
        self,
        *,
        actor_user_id: int,
        start_at: TemporalValue,
        end_at: TemporalValue,
        filters: CalendarViewFilters | None = None,
    ) -> CalendarViewResult:
        """Return a bounded agenda window with local items and linked projections."""

        filters = filters or CalendarViewFilters()
        validate_query_window(start_at, end_at)
        window_start = _coerce_datetime(start_at)
        window_end = _coerce_datetime(end_at)
        items = self.expand_items_window(
            actor_user_id=actor_user_id,
            start_at=window_start,
            end_at=window_end,
            filters=filters,
        )
        if filters.include_scheduled_tasks:
            items.extend(
                await self.load_scheduled_task_projections(
                    actor_user_id=actor_user_id,
                    start_at=window_start,
                    end_at=window_end,
                )
            )
        items.sort(key=_view_sort_key)
        return CalendarViewResult(
            start_at=window_start.isoformat(),
            end_at=window_end.isoformat(),
            items=items,
        )

    async def week(
        self,
        *,
        actor_user_id: int,
        week_start: TemporalValue,
        timezone: str,
        filters: CalendarViewFilters | None = None,
    ) -> CalendarViewResult:
        """Return a seven-day Calendar view starting at local midnight."""

        tz = _zoneinfo(timezone)
        if isinstance(week_start, datetime):
            local_start = week_start.astimezone(tz) if week_start.tzinfo is not None else week_start.replace(tzinfo=tz)
            local_start = local_start.replace(hour=0, minute=0, second=0, microsecond=0)
        elif isinstance(week_start, date):
            local_start = datetime.combine(week_start, time.min, tzinfo=tz)
        else:
            parsed = date_parser.isoparse(week_start)
            local_start = parsed.astimezone(tz) if parsed.tzinfo is not None else parsed.replace(tzinfo=tz)
            local_start = local_start.replace(hour=0, minute=0, second=0, microsecond=0)
        return await self.agenda(
            actor_user_id=actor_user_id,
            start_at=local_start,
            end_at=local_start + timedelta(days=7),
            filters=filters,
        )

    def expand_items_window(
        self,
        *,
        actor_user_id: int,
        start_at: TemporalValue,
        end_at: TemporalValue,
        filters: CalendarViewFilters | None = None,
    ) -> list[CalendarViewItem]:
        """Expand local Calendar rows and local recurrence rows inside a window."""

        filters = filters or CalendarViewFilters()
        validate_query_window(start_at, end_at)
        window_start = _coerce_datetime(start_at)
        window_end = _coerce_datetime(end_at)
        calendar_ids = self._readable_calendar_ids(actor_user_id=actor_user_id, filters=filters)
        rows = self.calendar_service.db.list_items_for_expansion(
            calendar_ids=calendar_ids,
            window_start=window_start.isoformat(),
            window_end=window_end.isoformat(),
            include_remote_deleted=filters.include_provider_tombstones,
        )
        readable_rows = self._filter_readable_items(actor_user_id=actor_user_id, rows=rows)
        recurrences = self.calendar_service.db.list_recurrences_for_items(row.id for row in readable_rows)
        view_items: list[CalendarViewItem] = []
        for item in readable_rows:
            recurrence = recurrences.get(item.id)
            if recurrence and recurrence.rrule and item.source_owner == CALENDAR_SOURCE_OWNER_TLDW:
                view_items.extend(
                    self._expand_recurring_item(
                        item=item,
                        recurrence=recurrence,
                        window_start=window_start,
                        window_end=window_end,
                    )
                )
                continue
            if _item_overlaps_window(item, window_start, window_end):
                view_items.append(_view_item_from_row(item))
        return view_items

    async def load_scheduled_task_projections(
        self,
        *,
        actor_user_id: int,
        start_at: TemporalValue,
        end_at: TemporalValue,
    ) -> list[CalendarViewItem]:
        """Load read-only linked projections for scheduled tasks in the window."""

        validate_query_window(start_at, end_at)
        window_start = _coerce_datetime(start_at)
        window_end = _coerce_datetime(end_at)
        response = await self.scheduled_tasks_service.list_tasks(user_id=actor_user_id)
        projections: list[CalendarViewItem] = []
        for task in response.items:
            if not task.next_run_at:
                continue
            try:
                next_run_at = _coerce_datetime(task.next_run_at)
            except CalendarValidationError:
                continue
            if not (window_start <= next_run_at <= window_end):
                continue
            projections.append(
                CalendarViewItem(
                    id=f"projection:scheduled_task:{task.id}",
                    title=task.title,
                    description=task.description,
                    source_owner=CALENDAR_SOURCE_OWNER_LINKED_PROJECTION,
                    start_at=next_run_at.isoformat(),
                    status=task.status,
                    read_only_reason=CALENDAR_SOURCE_OWNER_LINKED_PROJECTION,
                    link=CalendarViewLink(
                        target_type="scheduled_task",
                        target_id=task.id,
                        label=task.schedule_summary,
                        url=task.manage_url,
                        metadata={
                            "primitive": task.primitive,
                            "edit_mode": task.edit_mode,
                            "source_ref": task.source_ref,
                        },
                    ),
                    metadata={
                        "enabled": task.enabled,
                        "timezone": task.timezone,
                        "last_run_at": task.last_run_at,
                    },
                )
            )
        return projections

    def _readable_calendar_ids(
        self,
        *,
        actor_user_id: int,
        filters: CalendarViewFilters,
    ) -> list[int]:
        visible_ids = {
            calendar.id
            for calendar in self.calendar_service.list_calendars(actor_user_id=actor_user_id)
        }
        if filters.calendar_ids is None:
            return sorted(visible_ids)
        return [calendar_id for calendar_id in filters.calendar_ids if calendar_id in visible_ids]

    def _filter_readable_items(
        self,
        *,
        actor_user_id: int,
        rows: list[CalendarItemRow],
    ) -> list[CalendarItemRow]:
        readable: list[CalendarItemRow] = []
        for row in rows:
            try:
                readable.append(
                    self.calendar_service.get_item(
                        actor_user_id=actor_user_id,
                        item_id=row.id,
                    )
                )
            except (CalendarItemNotFound, CalendarPermissionDenied):
                continue
        return readable

    def _expand_recurring_item(
        self,
        *,
        item: CalendarItemRow,
        recurrence: CalendarRecurrenceRow,
        window_start: datetime,
        window_end: datetime,
    ) -> list[CalendarViewItem]:
        master_start = item.start_at or item.due_at
        if master_start is None:
            return []
        occurrences = expand_rrule(
            master_start=master_start,
            master_end=item.end_at,
            rrule_text=recurrence.rrule or "",
            window_start=window_start,
            window_end=window_end,
            timezone_name=recurrence.timezone or item.timezone,
            all_day=item.all_day,
        )
        return [
            _view_item_from_occurrence(
                item=item,
                recurrence=recurrence,
                occurrence=occurrence,
            )
            for occurrence in occurrences
        ]


def _view_item_from_row(item: CalendarItemRow) -> CalendarViewItem:
    start_at = item.start_at or item.due_at
    return CalendarViewItem(
        id=f"calendar_item:{item.id}",
        calendar_id=item.calendar_id,
        calendar_item_id=item.id,
        title=item.title,
        description=item.description,
        location=item.location,
        source_owner=item.source_owner,
        start_at=start_at,
        end_at=item.end_at,
        due_at=item.due_at,
        all_day=item.all_day,
        status=item.status,
        read_only_reason="provider" if item.provider_owned or item.source_owner == CALENDAR_SOURCE_OWNER_PROVIDER else None,
    )


def _view_item_from_occurrence(
    *,
    item: CalendarItemRow,
    recurrence: CalendarRecurrenceRow,
    occurrence: RecurrenceOccurrence,
) -> CalendarViewItem:
    start_at = _serialize_temporal(occurrence.start_at)
    end_at = _serialize_temporal(occurrence.end_at)
    return CalendarViewItem(
        id=f"calendar_item:{item.id}:occurrence:{occurrence.occurrence_index}:{start_at}",
        calendar_id=item.calendar_id,
        calendar_item_id=item.id,
        title=item.title,
        description=item.description,
        location=item.location,
        source_owner=item.source_owner,
        start_at=start_at,
        end_at=end_at,
        due_at=start_at if item.kind == "todo" and item.due_at else None,
        all_day=item.all_day,
        status=item.status,
        recurrence_id=recurrence.id,
        occurrence_index=occurrence.occurrence_index,
    )


def _item_overlaps_window(item: CalendarItemRow, window_start: datetime, window_end: datetime) -> bool:
    start_value = item.start_at or item.due_at
    if start_value is None:
        return False
    start_at = _coerce_datetime(start_value)
    end_at = _coerce_datetime(item.end_at) if item.end_at else start_at
    return end_at >= window_start and start_at <= window_end


def _coerce_datetime(value: TemporalValue | None) -> datetime:
    if value is None:
        raise CalendarValidationError("Calendar query window boundaries are required")
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime.combine(value, time.min)
    elif isinstance(value, str):
        parsed = date_parser.isoparse(value)
    else:
        raise CalendarValidationError(f"Unsupported temporal value: {value!r}")
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def _serialize_temporal(value: date | datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def _view_sort_key(item: CalendarViewItem) -> tuple[datetime, str]:
    value = item.start_at or item.due_at
    if value is None:
        return datetime.max.replace(tzinfo=timezone.utc), item.id
    return _coerce_datetime(value), item.id


def _zoneinfo(timezone_name: str) -> ZoneInfo:
    try:
        return ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise CalendarValidationError(f"Unsupported calendar timezone: {timezone_name}") from exc


__all__ = [
    "CalendarViewFilters",
    "CalendarViewItem",
    "CalendarViewLink",
    "CalendarViewResult",
    "CalendarViewService",
    "LocalRecurrenceRule",
]
