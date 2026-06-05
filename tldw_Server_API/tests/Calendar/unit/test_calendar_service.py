from __future__ import annotations

import json

import pytest

from tldw_Server_API.app.core.Calendar.calendar_service import CalendarService
from tldw_Server_API.app.core.Calendar.errors import (
    CalendarPermissionDenied,
    CalendarReadOnlyError,
    CalendarValidationError,
)
from tldw_Server_API.app.core.DB_Management.Calendar_DB import CalendarDatabase


@pytest.fixture
def calendar_db(tmp_path):
    db = CalendarDatabase(db_path=tmp_path / "calendar.db")
    db.ensure_schema()
    return db


def _create_provider_item(calendar_db: CalendarDatabase, *, owner_user_id: int = 1):
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


def test_viewer_cannot_edit_local_item(calendar_db):
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


def test_owner_can_manage_membership(calendar_db):
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


def test_non_owner_cannot_manage_membership(calendar_db):
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


def test_org_role_membership_grants_access_only_through_resolver(calendar_db):
    resolver_calls: list[tuple[int, int | None, str]] = []

    def resolver(user_id: int, org_id: int | None, role: str) -> bool:
        resolver_calls.append((user_id, org_id, role))
        return user_id == 2 and org_id == 42 and role == "researcher"

    denied_service = CalendarService(db=calendar_db)
    allowed_service = CalendarService(db=calendar_db, org_role_resolver=resolver)
    calendar = denied_service.create_calendar(
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


def test_editor_can_create_and_edit_local_items(calendar_db):
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


def test_update_item_validates_effective_event_and_todo_times(calendar_db):
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


def test_commenter_can_annotate_local_item_but_cannot_edit_it(calendar_db):
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


def test_provider_owned_item_edits_raise_read_only_error(calendar_db):
    _, provider_item = _create_provider_item(calendar_db)
    service = CalendarService(db=calendar_db)

    with pytest.raises(CalendarReadOnlyError):
        service.update_item(actor_user_id=1, item_id=provider_item.id, title="Nope")

    with pytest.raises(CalendarReadOnlyError):
        service.delete_item(actor_user_id=1, item_id=provider_item.id)


def test_copied_provider_item_becomes_local_and_independent(calendar_db):
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


def test_shared_viewer_can_read_local_item_but_not_personal_provider_import(calendar_db):
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


def test_calendar_owner_can_read_and_list_personal_provider_import(calendar_db):
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


def test_copied_provider_item_is_shared_by_normal_membership(calendar_db):
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


def test_service_rejects_cross_tenant_calendar_item_list_and_annotation_paths(calendar_db):
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


def test_calendar_links_follow_calendar_permissions(calendar_db):
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
