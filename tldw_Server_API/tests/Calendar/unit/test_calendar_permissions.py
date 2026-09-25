from __future__ import annotations

from dataclasses import replace

import pytest

from tldw_Server_API.app.core.Calendar.errors import CalendarPermissionDenied
from tldw_Server_API.app.core.Calendar.permissions import (
    CalendarAccessContext,
    assert_calendar_access,
    can_comment,
    can_manage_calendar,
    can_read_calendar,
    can_write_items,
)
from tldw_Server_API.app.core.DB_Management.Calendar_DB import (
    CalendarDatabase,
    CalendarMembershipRow,
)


@pytest.fixture
def calendar_db(tmp_path):
    db = CalendarDatabase(db_path=tmp_path / "calendar.db")
    db.ensure_schema()
    return db


def _create_calendar(db: CalendarDatabase, *, owner_user_id: int = 1, org_id: int | None = None):
    return db.create_calendar(
        tenant_id="default",
        owner_user_id=owner_user_id,
        org_id=org_id,
        name="Research",
        timezone="UTC",
        color="#2563eb",
    )


def _membership_with_role(membership: CalendarMembershipRow, role: str) -> CalendarMembershipRow:
    return replace(membership, role=role)


@pytest.mark.parametrize(
    ("role", "expected"),
    [
        ("owner", (True, True, True, True)),
        ("editor", (True, True, True, False)),
        ("commenter", (True, False, True, False)),
        ("viewer", (True, False, False, False)),
    ],
)
def test_role_helpers_map_calendar_membership_to_capabilities(calendar_db, role, expected):
    calendar = _create_calendar(calendar_db, owner_user_id=99)
    membership = calendar_db.create_membership(
        calendar_id=calendar.id,
        principal_type="user",
        principal_id="1",
        role=role,
    )
    context = CalendarAccessContext(
        actor_user_id=1,
        calendar=calendar,
        memberships=[_membership_with_role(membership, role)],
    )

    assert (
        can_read_calendar(context),
        can_write_items(context),
        can_comment(context),
        can_manage_calendar(context),
    ) == expected


def test_assert_calendar_access_raises_for_missing_capability(calendar_db):
    calendar = _create_calendar(calendar_db)
    calendar_db.create_membership(
        calendar_id=calendar.id,
        principal_type="user",
        principal_id="2",
        role="viewer",
    )
    context = CalendarAccessContext(
        actor_user_id=2,
        calendar=calendar,
        memberships=calendar_db.list_memberships(calendar.id),
    )

    with pytest.raises(CalendarPermissionDenied):
        assert_calendar_access(context, "write")


def test_org_role_membership_requires_injected_resolver(calendar_db):
    calendar = _create_calendar(calendar_db, org_id=42)
    calendar_db.create_membership(
        calendar_id=calendar.id,
        principal_type="org_role",
        principal_id="researcher",
        role="editor",
    )
    memberships = calendar_db.list_memberships(calendar.id)

    unresolved_context = CalendarAccessContext(
        actor_user_id=2,
        calendar=calendar,
        memberships=memberships,
    )
    denied_context = CalendarAccessContext(
        actor_user_id=2,
        calendar=calendar,
        memberships=memberships,
        org_role_resolver=lambda user_id, org_id, role: False,
    )
    allowed_context = CalendarAccessContext(
        actor_user_id=2,
        calendar=calendar,
        memberships=memberships,
        org_role_resolver=lambda user_id, org_id, role: (
            user_id == 2 and org_id == 42 and role == "researcher"
        ),
    )

    assert can_read_calendar(unresolved_context) is False
    assert can_write_items(denied_context) is False
    assert can_write_items(allowed_context) is True
