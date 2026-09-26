from __future__ import annotations

import importlib
import json
import sqlite3
from collections.abc import Generator
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType

import pytest

from tldw_Server_API.app.api.v1.schemas.scheduled_tasks_control_plane_schemas import (
    ScheduledTask,
    ScheduledTaskListResponse,
)
from tldw_Server_API.app.core.Calendar.calendar_service import CalendarService
from tldw_Server_API.app.core.Calendar.errors import (
    CalendarPermissionDenied,
    CalendarReadOnlyError,
    CalendarValidationError,
)
from tldw_Server_API.app.core.DB_Management.Calendar_DB import CalendarDatabase, CalendarItemRow, CalendarRow

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("zone", ["Unknown/Zone", "../etc/passwd", "/etc/passwd", "", "UTC\x00"])
def test_calendar_creation_rejects_invalid_zone_without_writes(
    calendar_db: CalendarDatabase, zone: str,
) -> None:
    """Non-HTTP creation must reject unusable zone keys before persistence."""
    with pytest.raises(CalendarValidationError):
        CalendarService(db=calendar_db).create_calendar(actor_user_id=1, name="Invalid", timezone=zone)
    assert calendar_db.list_calendars(tenant_id="default") == []


@pytest.mark.parametrize("zone", ["UTC", "Europe/Paris", "America/Los_Angeles"])
def test_calendar_creation_preserves_valid_iana_zone(calendar_db: CalendarDatabase, zone: str) -> None:
    """The creation boundary retains valid zone names verbatim."""
    created = CalendarService(db=calendar_db).create_calendar(actor_user_id=1, name="Valid", timezone=zone)
    assert created.timezone == zone


def test_org_calendar_creation_denies_unverified_membership(calendar_db: CalendarDatabase) -> None:
    """Non-HTTP callers cannot attach calendars to an unverified organization."""
    service = CalendarService(db=calendar_db)
    with pytest.raises(CalendarPermissionDenied):
        service.create_calendar(actor_user_id=1, name="Forged org", org_id=42)
    assert calendar_db.list_calendars(tenant_id="default") == []


def test_org_calendar_creation_uses_actor_scoped_membership(calendar_db: CalendarDatabase) -> None:
    """Verified members may create only in the organization and actor they resolved."""
    service = CalendarService(
        db=calendar_db, org_membership_resolver=lambda actor, org: actor == 1 and org == 42,
    )
    created = service.create_calendar(actor_user_id=1, name="Verified org", org_id=42)
    for actor, org in [(2, 42), (1, 43)]:
        with pytest.raises(CalendarPermissionDenied):
            service.create_calendar(actor_user_id=actor, name="Unrelated", org_id=org)
    assert [row.id for row in calendar_db.list_calendars(tenant_id="default")] == [created.id]


def test_batched_visibility_preserves_private_provider_rows_and_tenant_scope(
    calendar_db: CalendarDatabase,
) -> None:
    """Sharing a calendar never shares its private imports or foreign-tenant rows."""
    from dataclasses import replace

    calendar, provider_item = _create_provider_item(calendar_db)
    service = CalendarService(db=calendar_db)
    service.add_membership(actor_user_id=1, calendar_id=calendar.id, principal_type="user", principal_id="2", role="viewer")
    native = service.create_item(
        actor_user_id=1, calendar_id=calendar.id, kind="event", title="Shared native", start_at="2026-06-05T09:00:00Z",
    )
    assert service.filter_readable_items(actor_user_id=2, items=[provider_item, native]) == [native]
    assert service.filter_readable_items(actor_user_id=1, items=[provider_item, native]) == [provider_item, native]
    assert CalendarService(db=calendar_db, tenant_id="other").filter_readable_items(
        actor_user_id=1, items=[provider_item, native],
    ) == []
    assert service.filter_readable_items(actor_user_id=2, items=[replace(provider_item, external_binding_id=99999)]) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("next_run", ["2026-06-06T00:00:00Z", "2026-06-06T02:00:00+02:00"])
async def test_scheduled_projection_uses_exclusive_window_end(
    calendar_db: CalendarDatabase, next_run: str,
) -> None:
    """A task on the shared boundary belongs only to the following window."""
    from tldw_Server_API.app.core.Calendar.view_service import CalendarViewService

    task = ScheduledTask(
        id="reminder:boundary", primitive="reminder_task", title="Boundary", status="scheduled",
        enabled=True, next_run_at=next_run, edit_mode="native",
    )
    view = CalendarViewService(
        calendar_service=CalendarService(db=calendar_db), scheduled_tasks_service=_ScheduledTasksStub([task]),
    )
    before = await view.load_scheduled_task_projections(
        actor_user_id=1, start_at="2026-06-05T00:00:00Z", end_at="2026-06-06T00:00:00Z",
    )
    after = await view.load_scheduled_task_projections(
        actor_user_id=1, start_at="2026-06-06T00:00:00Z", end_at="2026-06-07T00:00:00Z",
    )
    assert [item.id for item in before] == []
    assert [item.id for item in after] == ["projection:scheduled_task:reminder:boundary"]


@pytest.mark.parametrize("candidate_count", [5, 50])
def test_agenda_authorization_query_count_is_bounded(
    calendar_db: CalendarDatabase, monkeypatch: pytest.MonkeyPatch, candidate_count: int,
) -> None:
    """Loaded agenda rows are not fetched again for each permission check."""
    import contextlib

    from tldw_Server_API.app.core.Calendar.view_service import CalendarViewService

    service = CalendarService(db=calendar_db)
    calendar = service.create_calendar(actor_user_id=1, name="Large agenda")
    for index in range(candidate_count):
        service.create_item(
            actor_user_id=1, calendar_id=calendar.id, kind="event", title=f"Item {index}",
            start_at="2026-06-05T09:00:00Z",
        )
    selects: list[str] = []
    original_connection = calendar_db.connection

    @contextlib.contextmanager
    def counted_connection() -> Generator[sqlite3.Connection, None, None]:
        """Count SELECT statements across real isolated SQLite connections."""
        with original_connection() as connection:
            connection.set_trace_callback(lambda sql: selects.append(sql) if sql.lstrip().upper().startswith("SELECT") else None)
            try:
                yield connection
            finally:
                connection.set_trace_callback(None)

    monkeypatch.setattr(calendar_db, "connection", counted_connection)
    items = CalendarViewService(calendar_service=service).expand_items_window(
        actor_user_id=1, start_at="2026-06-05T00:00:00Z", end_at="2026-06-06T00:00:00Z",
    )
    assert len(items) == candidate_count
    assert len(selects) <= 12, selects


@pytest.mark.parametrize(
    "values",
    [
        {"start_at": "garbage"},
        {"end_at": "garbage"},
        {"due_at": "garbage"},
        {"end_at": "2026-06-05T08:00:00Z"},
        {"timezone": "No/SuchZone"},
    ],
)
def test_service_rejects_invalid_temporal_values_on_create_and_update(
    calendar_db: CalendarDatabase, values: dict[str, str],
) -> None:
    """Invalid timestamps, intervals and zones fail before changing persisted item times."""
    service = CalendarService(db=calendar_db)
    calendar = service.create_calendar(actor_user_id=1, name="Times", timezone="UTC")
    base = {"kind": "event", "title": "Event", "start_at": "2026-06-05T09:00:00Z"}
    with pytest.raises(CalendarValidationError):
        service.create_item(actor_user_id=1, calendar_id=calendar.id, **(base | values))
    item = service.create_item(actor_user_id=1, calendar_id=calendar.id, **base)
    with pytest.raises(CalendarValidationError):
        service.update_item(actor_user_id=1, item_id=item.id, **values)
    assert calendar_db.get_item(item.id).start_at == base["start_at"]


@pytest.mark.asyncio
@pytest.mark.parametrize("rrule", [None, "FREQ=DAILY;COUNT=3"])
async def test_agenda_applies_recurrence_additions_and_exclusions(
    calendar_db: CalendarDatabase, rrule: str | None,
) -> None:
    from tldw_Server_API.app.core.Calendar.view_service import CalendarViewService

    service = CalendarService(db=calendar_db)
    calendar = service.create_calendar(actor_user_id=1, name="Dates", timezone="UTC")
    item = service.create_item(
        actor_user_id=1, calendar_id=calendar.id, kind="event", title="Dates", start_at="2026-06-05T09:00:00Z"
    )
    calendar_db.upsert_recurrence(
        calendar_item_id=item.id, rrule=rrule, rdate_json=["2026-06-10T09:00:00Z"], exdate_json=["2026-06-06T09:00:00Z"]
    )
    result = await CalendarViewService(calendar_service=service).agenda(
        actor_user_id=1, start_at="2026-06-01T00:00:00Z", end_at="2026-06-12T00:00:00Z"
    )
    assert [entry.start_at[:10] for entry in result.items] == (
        ["2026-06-05", "2026-06-07", "2026-06-10"] if rrule else ["2026-06-05", "2026-06-10"]
    )


@pytest.mark.asyncio
async def test_all_day_without_end_overlaps_afternoon_but_not_next_day(calendar_db: CalendarDatabase) -> None:
    from tldw_Server_API.app.core.Calendar.view_service import CalendarViewService

    service = CalendarService(db=calendar_db)
    calendar = service.create_calendar(actor_user_id=1, name="Dates", timezone="UTC")
    service.create_item(
        actor_user_id=1, calendar_id=calendar.id, kind="event", title="Holiday", start_at="2026-06-05", all_day=True
    )
    view = CalendarViewService(calendar_service=service)
    afternoon = await view.agenda(actor_user_id=1, start_at="2026-06-05T12:00:00Z", end_at="2026-06-05T18:00:00Z")
    tomorrow = await view.agenda(actor_user_id=1, start_at="2026-06-06T00:00:00Z", end_at="2026-06-07T00:00:00Z")
    assert [item.title for item in afternoon.items] == ["Holiday"]
    assert tomorrow.items == []


@pytest.mark.asyncio
async def test_rdate_before_master_is_not_lost_by_database_prefilter(calendar_db: CalendarDatabase) -> None:
    from tldw_Server_API.app.core.Calendar.view_service import CalendarViewService

    service = CalendarService(db=calendar_db)
    calendar = service.create_calendar(actor_user_id=1, name="Dates", timezone="UTC")
    item = service.create_item(
        actor_user_id=1,
        calendar_id=calendar.id,
        kind="event",
        title="Earlier addition",
        start_at="2026-07-05T09:00:00Z",
    )
    calendar_db.upsert_recurrence(calendar_item_id=item.id, rrule=None, rdate_json=["2026-06-05T09:00:00Z"])
    result = await CalendarViewService(calendar_service=service).agenda(
        actor_user_id=1, start_at="2026-06-01T00:00:00Z", end_at="2026-06-08T00:00:00Z"
    )
    assert [entry.title for entry in result.items] == ["Earlier addition"]


@pytest.fixture
def calendar_db(tmp_path: Path) -> CalendarDatabase:
    db = CalendarDatabase(db_path=tmp_path / "calendar.db")
    db.ensure_schema()
    return db


def _create_provider_item(
    calendar_db: CalendarDatabase, *, owner_user_id: int = 1,
) -> tuple[CalendarRow, CalendarItemRow]:
    calendar = calendar_db.create_calendar(
        tenant_id="default",
        owner_user_id=owner_user_id,
        org_id=None,
        name="Imported",
        timezone="UTC",
        color="#2563eb",
    )
    account = calendar_db.create_external_account(
        tenant_id="default",
        user_id=owner_user_id,
        provider="caldav",
        display_name="Fastmail",
        secret_ref=None,
    )
    binding = calendar_db.create_external_binding(
        account_id=account.id,
        calendar_id=calendar.id,
        remote_calendar_id="remote-calendar",
    )
    item = calendar_db.upsert_provider_item(
        calendar_id=calendar.id,
        external_binding_id=binding.id,
        source_uid="remote-event-1",
        title="Imported meeting",
        start_at="2026-06-05T10:00:00Z",
        end_at="2026-06-05T11:00:00Z",
        provider_payload_json={"uid": "remote-event-1"},
        source_etag="etag-1",
        source_ctag="ctag-1",
    )
    return calendar, item


def _view_service_module() -> ModuleType:
    try:
        return importlib.import_module("tldw_Server_API.app.core.Calendar.view_service")
    except ModuleNotFoundError as exc:
        pytest.fail(f"calendar view service module is missing: {exc}")


class _ScheduledTasksStub:
    def __init__(self, tasks: list[ScheduledTask]) -> None:
        self.tasks = tasks
        self.calls: list[int] = []

    async def list_tasks(self, *, user_id: int) -> ScheduledTaskListResponse:
        self.calls.append(user_id)
        return ScheduledTaskListResponse(items=self.tasks, total=len(self.tasks))


def test_viewer_cannot_edit_local_item(calendar_db: CalendarDatabase) -> None:
    service = CalendarService(db=calendar_db)
    calendar = service.create_calendar(actor_user_id=1, name="Shared", timezone="UTC")
    local_item = service.create_item(
        actor_user_id=1,
        calendar_id=calendar.id,
        kind="event",
        title="Owner local item",
        start_at="2026-06-05T09:00:00Z",
    )
    service.add_membership(
        actor_user_id=1,
        calendar_id=calendar.id,
        principal_type="user",
        principal_id="2",
        role="viewer",
    )

    with pytest.raises(CalendarPermissionDenied):
        service.create_item(
            actor_user_id=2,
            calendar_id=calendar.id,
            kind="event",
            title="Nope",
            start_at="2026-06-05T10:00:00Z",
        )

    with pytest.raises(CalendarPermissionDenied):
        service.update_item(actor_user_id=2, item_id=local_item.id, title="Nope")


def test_owner_can_manage_membership(calendar_db: CalendarDatabase) -> None:
    service = CalendarService(db=calendar_db)
    calendar = service.create_calendar(actor_user_id=1, name="Shared", timezone="UTC")

    membership = service.add_membership(
        actor_user_id=1,
        calendar_id=calendar.id,
        principal_type="user",
        principal_id="2",
        role="editor",
    )
    memberships = service.list_memberships(actor_user_id=1, calendar_id=calendar.id)

    assert membership.role == "editor"
    assert any(row.principal_type == "user" and row.principal_id == "2" for row in memberships)


def test_non_owner_cannot_manage_membership(calendar_db: CalendarDatabase) -> None:
    service = CalendarService(db=calendar_db)
    calendar = service.create_calendar(actor_user_id=1, name="Shared", timezone="UTC")
    service.add_membership(
        actor_user_id=1,
        calendar_id=calendar.id,
        principal_type="user",
        principal_id="2",
        role="editor",
    )

    with pytest.raises(CalendarPermissionDenied):
        service.add_membership(
            actor_user_id=2,
            calendar_id=calendar.id,
            principal_type="user",
            principal_id="3",
            role="viewer",
        )


def test_org_role_membership_grants_access_only_through_resolver(calendar_db: CalendarDatabase) -> None:
    resolver_calls: list[tuple[int, int | None, str]] = []

    def resolver(user_id: int, org_id: int | None, role: str) -> bool:
        resolver_calls.append((user_id, org_id, role))
        return user_id == 2 and org_id == 42 and role == "researcher"

    denied_service = CalendarService(db=calendar_db)
    allowed_service = CalendarService(db=calendar_db, org_role_resolver=resolver)
    creator = CalendarService(db=calendar_db, org_membership_resolver=lambda actor, org: actor == 1 and org == 42)
    calendar = creator.create_calendar(
        actor_user_id=1,
        name="Org research",
        timezone="UTC",
        org_id=42,
    )
    denied_service.add_membership(
        actor_user_id=1,
        calendar_id=calendar.id,
        principal_type="org_role",
        principal_id="researcher",
        role="editor",
    )

    with pytest.raises(CalendarPermissionDenied):
        denied_service.create_item(
            actor_user_id=2,
            calendar_id=calendar.id,
            kind="event",
            title="Denied",
            start_at="2026-06-05T10:00:00Z",
        )

    item = allowed_service.create_item(
        actor_user_id=2,
        calendar_id=calendar.id,
        kind="event",
        title="Allowed",
        start_at="2026-06-05T10:00:00Z",
    )

    assert item.title == "Allowed"
    assert resolver_calls


def test_editor_can_create_and_edit_local_items(calendar_db: CalendarDatabase) -> None:
    service = CalendarService(db=calendar_db)
    calendar = service.create_calendar(actor_user_id=1, name="Shared", timezone="UTC")
    service.add_membership(
        actor_user_id=1,
        calendar_id=calendar.id,
        principal_type="user",
        principal_id="2",
        role="editor",
    )

    item = service.create_item(
        actor_user_id=2,
        calendar_id=calendar.id,
        kind="event",
        title="Draft title",
        start_at="2026-06-05T10:00:00Z",
    )
    updated = service.update_item(actor_user_id=2, item_id=item.id, title="Final title")

    assert updated.title == "Final title"


def test_update_item_validates_effective_event_and_todo_times(calendar_db: CalendarDatabase) -> None:
    service = CalendarService(db=calendar_db)
    calendar = service.create_calendar(actor_user_id=1, name="Personal", timezone="UTC")
    event = service.create_item(
        actor_user_id=1,
        calendar_id=calendar.id,
        kind="event",
        title="Planning",
        start_at="2026-06-05T10:00:00Z",
    )
    due_only_todo = service.create_item(
        actor_user_id=1,
        calendar_id=calendar.id,
        kind="todo",
        title="Submit notes",
        due_at="2026-06-05T17:00:00Z",
    )
    scheduled_todo = service.create_item(
        actor_user_id=1,
        calendar_id=calendar.id,
        kind="todo",
        title="Read paper",
        start_at="2026-06-05T14:00:00Z",
        due_at="2026-06-05T17:00:00Z",
    )

    with pytest.raises(CalendarValidationError):
        service.update_item(actor_user_id=1, item_id=event.id, start_at=None)

    with pytest.raises(CalendarValidationError):
        service.update_item(actor_user_id=1, item_id=due_only_todo.id, due_at=None)

    updated_todo = service.update_item(actor_user_id=1, item_id=scheduled_todo.id, due_at=None)

    assert calendar_db.get_item(event.id).start_at == "2026-06-05T10:00:00Z"
    assert calendar_db.get_item(due_only_todo.id).due_at == "2026-06-05T17:00:00Z"
    assert updated_todo.due_at is None
    assert updated_todo.start_at == "2026-06-05T14:00:00Z"


def test_commenter_can_annotate_local_item_but_cannot_edit_it(calendar_db: CalendarDatabase) -> None:
    service = CalendarService(db=calendar_db)
    calendar = service.create_calendar(actor_user_id=1, name="Shared", timezone="UTC")
    local_item = service.create_item(
        actor_user_id=1,
        calendar_id=calendar.id,
        kind="event",
        title="Shared local meeting",
        start_at="2026-06-05T10:00:00Z",
    )
    service.add_membership(
        actor_user_id=1,
        calendar_id=calendar.id,
        principal_type="user",
        principal_id="2",
        role="commenter",
    )

    annotation = service.create_annotation(
        actor_user_id=2,
        item_id=local_item.id,
        body="Ask about this meeting",
        tags=["follow-up"],
    )
    tag_overlay = service.update_local_tags(
        actor_user_id=2,
        item_id=local_item.id,
        tags=["needs-review"],
    )

    with pytest.raises(CalendarPermissionDenied):
        service.update_item(actor_user_id=2, item_id=local_item.id, title="Nope")

    owner_link = service.create_link(
        actor_user_id=1,
        item_id=local_item.id,
        target_type="note",
        target_id="note-123",
    )

    with pytest.raises(CalendarPermissionDenied):
        service.create_link(
            actor_user_id=2,
            item_id=local_item.id,
            target_type="note",
            target_id="note-456",
        )

    with pytest.raises(CalendarPermissionDenied):
        service.delete_link(actor_user_id=2, link_id=owner_link.id)

    assert annotation.author_user_id == 2
    assert json.loads(annotation.tags_json or "[]") == ["follow-up"]
    assert json.loads(tag_overlay.tags_json or "[]") == ["needs-review"]
    assert calendar_db.get_item(local_item.id).local_tags_json is None


def test_provider_owned_item_edits_raise_read_only_error(calendar_db: CalendarDatabase) -> None:
    _, provider_item = _create_provider_item(calendar_db)
    service = CalendarService(db=calendar_db)

    with pytest.raises(CalendarReadOnlyError):
        service.update_item(actor_user_id=1, item_id=provider_item.id, title="Nope")

    with pytest.raises(CalendarReadOnlyError):
        service.delete_item(actor_user_id=1, item_id=provider_item.id)


def test_copied_provider_item_becomes_local_and_independent(calendar_db: CalendarDatabase) -> None:
    calendar, provider_item = _create_provider_item(calendar_db)
    service = CalendarService(db=calendar_db)

    copied = service.copy_provider_item(
        actor_user_id=1,
        item_id=provider_item.id,
        target_calendar_id=calendar.id,
        title="Local copy",
    )
    edited = service.update_item(actor_user_id=1, item_id=copied.id, title="Edited local copy")
    provider_after = calendar_db.get_item(provider_item.id)

    assert copied.source_owner == "tldw"
    assert copied.provider_owned is False
    assert copied.external_binding_id is None
    assert copied.source_uid is None
    assert copied.copied_from_item_id == provider_item.id
    assert edited.title == "Edited local copy"
    assert provider_after.title == "Imported meeting"


def test_shared_viewer_can_read_local_item_but_not_personal_provider_import(calendar_db: CalendarDatabase) -> None:
    calendar, provider_item = _create_provider_item(calendar_db)
    service = CalendarService(db=calendar_db)
    local_item = service.create_item(
        actor_user_id=1,
        calendar_id=calendar.id,
        kind="event",
        title="Shared local meeting",
        start_at="2026-06-05T12:00:00Z",
        end_at="2026-06-05T13:00:00Z",
    )
    service.add_membership(
        actor_user_id=1,
        calendar_id=calendar.id,
        principal_type="user",
        principal_id="2",
        role="viewer",
    )

    assert service.get_item(actor_user_id=2, item_id=local_item.id).title == "Shared local meeting"
    with pytest.raises(CalendarPermissionDenied):
        service.get_item(actor_user_id=2, item_id=provider_item.id)

    visible_items = service.list_items_window(
        actor_user_id=2,
        calendar_ids=[calendar.id],
        window_start="2026-06-05T00:00:00Z",
        window_end="2026-06-06T00:00:00Z",
    )

    assert {item.id for item in visible_items} == {local_item.id}


def test_calendar_owner_can_read_and_list_personal_provider_import(calendar_db: CalendarDatabase) -> None:
    calendar, provider_item = _create_provider_item(calendar_db)
    service = CalendarService(db=calendar_db)

    fetched = service.get_item(actor_user_id=1, item_id=provider_item.id)
    visible_items = service.list_items_window(
        actor_user_id=1,
        calendar_ids=[calendar.id],
        window_start="2026-06-05T00:00:00Z",
        window_end="2026-06-06T00:00:00Z",
    )

    assert fetched.id == provider_item.id
    assert provider_item.id in {item.id for item in visible_items}


def test_copied_provider_item_is_shared_by_normal_membership(calendar_db: CalendarDatabase) -> None:
    calendar, provider_item = _create_provider_item(calendar_db)
    service = CalendarService(db=calendar_db)
    copied = service.copy_provider_item(
        actor_user_id=1,
        item_id=provider_item.id,
        target_calendar_id=calendar.id,
        title="Shared provider copy",
    )
    service.add_membership(
        actor_user_id=1,
        calendar_id=calendar.id,
        principal_type="user",
        principal_id="2",
        role="viewer",
    )

    fetched = service.get_item(actor_user_id=2, item_id=copied.id)
    visible_items = service.list_items_window(
        actor_user_id=2,
        calendar_ids=[calendar.id],
        window_start="2026-06-05T00:00:00Z",
        window_end="2026-06-06T00:00:00Z",
    )

    assert fetched.id == copied.id
    assert fetched.provider_owned is False
    assert copied.id in {item.id for item in visible_items}
    assert provider_item.id not in {item.id for item in visible_items}


def test_service_rejects_cross_tenant_calendar_item_list_and_annotation_paths(calendar_db: CalendarDatabase) -> None:
    tenant_a_service = CalendarService(db=calendar_db, tenant_id="tenant-a")
    tenant_b_service = CalendarService(db=calendar_db, tenant_id="tenant-b")
    tenant_b_calendar = tenant_b_service.create_calendar(
        actor_user_id=1,
        name="Tenant B",
        timezone="UTC",
    )
    tenant_b_item = tenant_b_service.create_item(
        actor_user_id=1,
        calendar_id=tenant_b_calendar.id,
        kind="event",
        title="Tenant B item",
        start_at="2026-06-05T10:00:00Z",
    )

    with pytest.raises(CalendarPermissionDenied):
        tenant_a_service.update_calendar(
            actor_user_id=1,
            calendar_id=tenant_b_calendar.id,
            name="Leaked calendar",
        )

    with pytest.raises(CalendarPermissionDenied):
        tenant_a_service.get_item(actor_user_id=1, item_id=tenant_b_item.id)

    with pytest.raises(CalendarPermissionDenied):
        tenant_a_service.update_item(
            actor_user_id=1,
            item_id=tenant_b_item.id,
            title="Leaked B item",
        )

    visible_items = tenant_a_service.list_items_window(
        actor_user_id=1,
        calendar_ids=[tenant_b_calendar.id],
        window_start="2026-06-05T00:00:00Z",
        window_end="2026-06-06T00:00:00Z",
    )

    with pytest.raises(CalendarPermissionDenied):
        tenant_a_service.create_annotation(
            actor_user_id=1,
            item_id=tenant_b_item.id,
            body="Cross-tenant note",
        )

    assert visible_items == []
    assert calendar_db.get_calendar(tenant_b_calendar.id).name == "Tenant B"
    assert tenant_b_service.get_item(actor_user_id=1, item_id=tenant_b_item.id).title == "Tenant B item"


def test_calendar_links_follow_calendar_permissions(calendar_db: CalendarDatabase) -> None:
    service = CalendarService(db=calendar_db)
    calendar = service.create_calendar(actor_user_id=1, name="Shared", timezone="UTC")
    service.add_membership(
        actor_user_id=1,
        calendar_id=calendar.id,
        principal_type="user",
        principal_id="2",
        role="editor",
    )
    item = service.create_item(
        actor_user_id=2,
        calendar_id=calendar.id,
        kind="todo",
        title="Read paper",
        due_at="2026-06-05T10:00:00Z",
    )

    link = service.create_link(
        actor_user_id=2,
        item_id=item.id,
        target_type="note",
        target_id="note-123",
        label="Notes",
    )
    links = service.list_links(actor_user_id=2, item_id=item.id)
    deleted_count = service.delete_link(actor_user_id=2, link_id=link.id)

    assert links == [link]
    assert deleted_count == 1


@pytest.mark.asyncio
async def test_agenda_expands_recurring_local_items_from_persisted_recurrence(calendar_db: CalendarDatabase) -> None:
    view_service = _view_service_module()
    service = CalendarService(db=calendar_db)
    calendar = service.create_calendar(actor_user_id=1, name="Research", timezone="UTC")
    start = datetime(2026, 6, 5, 9, 0, tzinfo=timezone.utc)
    item = service.create_item(
        actor_user_id=1,
        calendar_id=calendar.id,
        kind="event",
        title="Standup",
        start_at=start.isoformat(),
        end_at=(start + timedelta(minutes=30)).isoformat(),
    )
    calendar_db.upsert_recurrence(
        calendar_item_id=item.id,
        rrule=view_service.LocalRecurrenceRule(frequency="daily", count=3).to_rrule(),
        timezone="UTC",
    )

    result = await view_service.CalendarViewService(
        calendar_service=service,
        scheduled_tasks_service=_ScheduledTasksStub([]),
    ).agenda(
        actor_user_id=1,
        start_at=start - timedelta(days=1),
        end_at=start + timedelta(days=5),
        filters=view_service.CalendarViewFilters(calendar_ids=[calendar.id], include_scheduled_tasks=False),
    )

    standup_items = [item for item in result.items if item.title == "Standup"]
    assert [item.start_at for item in standup_items] == [
        "2026-06-05T09:00:00+00:00",
        "2026-06-06T09:00:00+00:00",
        "2026-06-07T09:00:00+00:00",
    ]


@pytest.mark.asyncio
async def test_agenda_includes_offset_aware_items_after_authoritative_datetime_overlap(calendar_db: CalendarDatabase) -> None:
    view_service = _view_service_module()
    service = CalendarService(db=calendar_db)
    calendar = service.create_calendar(actor_user_id=1, name="Research", timezone="America/Los_Angeles")
    service.create_item(
        actor_user_id=1,
        calendar_id=calendar.id,
        kind="event",
        title="Offset meeting",
        start_at="2026-06-05T09:00:00-07:00",
        end_at="2026-06-05T10:00:00-07:00",
    )

    result = await view_service.CalendarViewService(
        calendar_service=service,
        scheduled_tasks_service=_ScheduledTasksStub([]),
    ).agenda(
        actor_user_id=1,
        start_at=datetime(2026, 6, 5, 15, 0, tzinfo=timezone.utc),
        end_at=datetime(2026, 6, 5, 18, 0, tzinfo=timezone.utc),
        filters=view_service.CalendarViewFilters(calendar_ids=[calendar.id], include_scheduled_tasks=False),
    )

    assert [item.title for item in result.items] == ["Offset meeting"]
    assert result.items[0].start_at == "2026-06-05T09:00:00-07:00"


@pytest.mark.asyncio
async def test_week_includes_offset_aware_items_crossing_raw_iso_boundary(calendar_db: CalendarDatabase) -> None:
    view_service = _view_service_module()
    service = CalendarService(db=calendar_db)
    calendar = service.create_calendar(actor_user_id=1, name="Research", timezone="UTC")
    service.create_item(
        actor_user_id=1,
        calendar_id=calendar.id,
        kind="event",
        title="Offset week boundary",
        start_at="2026-06-08T00:30:00+02:00",
        end_at="2026-06-08T01:00:00+02:00",
    )

    result = await view_service.CalendarViewService(
        calendar_service=service,
        scheduled_tasks_service=_ScheduledTasksStub([]),
    ).week(
        actor_user_id=1,
        week_start=date(2026, 6, 1),
        timezone="UTC",
        filters=view_service.CalendarViewFilters(calendar_ids=[calendar.id], include_scheduled_tasks=False),
    )

    assert [item.title for item in result.items] == ["Offset week boundary"]


@pytest.mark.asyncio
async def test_agenda_includes_all_day_date_only_items_for_overlapping_day_windows(calendar_db: CalendarDatabase) -> None:
    view_service = _view_service_module()
    service = CalendarService(db=calendar_db)
    calendar = service.create_calendar(actor_user_id=1, name="Research", timezone="UTC")
    service.create_item(
        actor_user_id=1,
        calendar_id=calendar.id,
        kind="event",
        title="All-day writing retreat",
        start_at="2026-06-05",
        end_at="2026-06-06",
        all_day=True,
    )

    result = await view_service.CalendarViewService(
        calendar_service=service,
        scheduled_tasks_service=_ScheduledTasksStub([]),
    ).agenda(
        actor_user_id=1,
        start_at=datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc),
        end_at=datetime(2026, 6, 5, 13, 0, tzinfo=timezone.utc),
        filters=view_service.CalendarViewFilters(calendar_ids=[calendar.id], include_scheduled_tasks=False),
    )

    assert [item.title for item in result.items] == ["All-day writing retreat"]
    assert result.items[0].start_at == "2026-06-05"
    assert result.items[0].all_day is True


@pytest.mark.asyncio
async def test_agenda_returns_read_only_scheduled_task_linked_projections(calendar_db: CalendarDatabase) -> None:
    view_service = _view_service_module()
    service = CalendarService(db=calendar_db)
    calendar = service.create_calendar(actor_user_id=1, name="Research", timezone="UTC")
    in_window = ScheduledTask(
        id="reminder:task-1",
        primitive="reminder_task",
        title="Review notes",
        status="scheduled",
        enabled=True,
        next_run_at="2026-06-05T12:30:00+00:00",
        edit_mode="native",
    )
    outside_window = ScheduledTask(
        id="watchlist_job:99",
        primitive="watchlist_job",
        title="Outside scan",
        status="scheduled",
        enabled=True,
        next_run_at="2026-07-05T12:30:00+00:00",
        edit_mode="external",
    )
    scheduled_tasks = _ScheduledTasksStub([in_window, outside_window])

    result = await view_service.CalendarViewService(
        calendar_service=service,
        scheduled_tasks_service=scheduled_tasks,
    ).agenda(
        actor_user_id=1,
        start_at=datetime(2026, 6, 5, tzinfo=timezone.utc),
        end_at=datetime(2026, 6, 6, tzinfo=timezone.utc),
        filters=view_service.CalendarViewFilters(calendar_ids=[calendar.id], include_scheduled_tasks=True),
    )

    projection = next(item for item in result.items if item.title == "Review notes")
    assert scheduled_tasks.calls == [1]
    assert projection.source_owner == "linked_projection"
    assert projection.read_only_reason == "linked_projection"
    assert projection.link.target_type == "scheduled_task"
    assert projection.link.target_id == "reminder:task-1"
    assert "Outside scan" not in {item.title for item in result.items}
