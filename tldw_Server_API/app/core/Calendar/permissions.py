"""Calendar role evaluation helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from tldw_Server_API.app.core.Calendar.constants import (
    CALENDAR_ROLE_COMMENTER,
    CALENDAR_ROLE_EDITOR,
    CALENDAR_ROLE_OWNER,
    CALENDAR_ROLE_VIEWER,
)
from tldw_Server_API.app.core.Calendar.errors import CalendarPermissionDenied
from tldw_Server_API.app.core.DB_Management.Calendar_DB import (
    CalendarMembershipRow,
    CalendarRow,
)

CalendarRole = Literal["owner", "editor", "commenter", "viewer"]
CalendarAccessAction = Literal["read", "write", "comment", "manage"]
OrgRoleResolver = Callable[[int, int | None, str], bool]

_ROLE_RANK: dict[str, int] = {
    CALENDAR_ROLE_VIEWER: 10,
    CALENDAR_ROLE_COMMENTER: 20,
    CALENDAR_ROLE_EDITOR: 30,
    CALENDAR_ROLE_OWNER: 40,
}


@dataclass(frozen=True)
class CalendarAccessContext:
    """Inputs needed to evaluate an actor's calendar role."""

    actor_user_id: int
    calendar: CalendarRow
    memberships: Sequence[CalendarMembershipRow]
    org_role_resolver: OrgRoleResolver | None = None


def can_read_calendar(context: CalendarAccessContext) -> bool:
    """Return whether the actor can read calendar metadata and items."""

    return _best_role_rank(context) >= _ROLE_RANK[CALENDAR_ROLE_VIEWER]


def can_write_items(context: CalendarAccessContext) -> bool:
    """Return whether the actor can create/edit local tldw-owned items."""

    return _best_role_rank(context) >= _ROLE_RANK[CALENDAR_ROLE_EDITOR]


def can_comment(context: CalendarAccessContext) -> bool:
    """Return whether the actor can add annotation-style local overlays."""

    return _best_role_rank(context) >= _ROLE_RANK[CALENDAR_ROLE_COMMENTER]


def can_manage_calendar(context: CalendarAccessContext) -> bool:
    """Return whether the actor can manage calendar settings and membership."""

    return _best_role_rank(context) >= _ROLE_RANK[CALENDAR_ROLE_OWNER]


def assert_calendar_access(
    context: CalendarAccessContext,
    action: CalendarAccessAction,
) -> None:
    """Raise when the actor lacks the requested calendar capability."""

    allowed = {
        "read": can_read_calendar,
        "write": can_write_items,
        "comment": can_comment,
        "manage": can_manage_calendar,
    }[action](context)
    if not allowed:
        raise CalendarPermissionDenied(
            f"User {context.actor_user_id} lacks calendar {action} access"
        )


def _best_role_rank(context: CalendarAccessContext) -> int:
    best = 0
    if context.calendar.owner_user_id == context.actor_user_id:
        best = _ROLE_RANK[CALENDAR_ROLE_OWNER]

    for membership in context.memberships:
        if _membership_applies(context, membership):
            best = max(best, _ROLE_RANK.get(membership.role, 0))
    return best


def _membership_applies(
    context: CalendarAccessContext,
    membership: CalendarMembershipRow,
) -> bool:
    if membership.principal_type == "user":
        return membership.principal_id == str(context.actor_user_id)
    if membership.principal_type == "org_role" and context.org_role_resolver is not None:
        return context.org_role_resolver(
            context.actor_user_id,
            context.calendar.org_id,
            membership.principal_id,
        )
    return False


__all__ = [
    "CalendarAccessAction",
    "CalendarAccessContext",
    "CalendarRole",
    "OrgRoleResolver",
    "assert_calendar_access",
    "can_comment",
    "can_manage_calendar",
    "can_read_calendar",
    "can_write_items",
]
