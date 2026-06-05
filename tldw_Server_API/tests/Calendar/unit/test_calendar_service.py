from __future__ import annotations

import json

import pytest

from tldw_Server_API.app.core.Calendar.calendar_service import CalendarService
from tldw_Server_API.app.core.Calendar.errors import (
    CalendarPermissionDenied,
    CalendarReadOnlyError,
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


def test_commenter_can_annotate_provider_item_but_cannot_edit_provider_fields(calendar_db):
    calendar, provider_item = _create_provider_item(calendar_db)
    service = CalendarService(db=calendar_db)
    service.add_membership(
        actor_user_id=1,
        calendar_id=calendar.id,
        principal_type="user",
        principal_id="2",
        role="commenter",
    )

    annotation = service.create_annotation(
        actor_user_id=2,
        item_id=provider_item.id,
        body="Ask about this meeting",
        tags=["follow-up"],
    )
    tag_overlay = service.update_local_tags(
        actor_user_id=2,
        item_id=provider_item.id,
        tags=["needs-review"],
    )

    with pytest.raises(CalendarReadOnlyError):
        service.update_item(actor_user_id=2, item_id=provider_item.id, title="Nope")

    provider_after = calendar_db.get_item(provider_item.id)
    assert annotation.author_user_id == 2
    assert json.loads(annotation.tags_json or "[]") == ["follow-up"]
    assert json.loads(tag_overlay.tags_json or "[]") == ["needs-review"]
    assert provider_after.local_tags_json is None


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
