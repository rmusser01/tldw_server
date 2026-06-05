"""Permission-enforcing local Calendar domain service."""

from __future__ import annotations

from typing import Any, get_args

from tldw_Server_API.app.core.Calendar.constants import CALENDAR_SOURCE_OWNER_PROVIDER
from tldw_Server_API.app.core.Calendar.errors import (
    CalendarPermissionDenied,
    CalendarReadOnlyError,
    CalendarValidationError,
)
from tldw_Server_API.app.core.Calendar.permissions import (
    CalendarAccessAction,
    CalendarAccessContext,
    CalendarRole,
    OrgRoleResolver,
    assert_calendar_access,
    can_manage_calendar,
)
from tldw_Server_API.app.core.DB_Management.Calendar_DB import (
    CalendarAnnotationRow,
    CalendarDatabase,
    CalendarItemRow,
    CalendarLinkRow,
    CalendarMembershipRow,
    CalendarRow,
)

_VALID_ROLES = set(get_args(CalendarRole))
_VALID_PRINCIPAL_TYPES = {"user", "org_role"}


class CalendarService:
    """Wrap Calendar DB operations with local calendar authorization rules."""

    def __init__(
        self,
        *,
        db: CalendarDatabase,
        tenant_id: str = "default",
        org_role_resolver: OrgRoleResolver | None = None,
    ) -> None:
        self.db = db
        self.tenant_id = tenant_id
        self.org_role_resolver = org_role_resolver

    def create_calendar(
        self,
        *,
        actor_user_id: int,
        name: str,
        timezone: str = "UTC",
        org_id: int | None = None,
        color: str | None = None,
        description: str | None = None,
        visibility: str = "private",
        default_reminder_policy_json: str | dict[str, Any] | None = None,
        rbac_policy_ref: str | None = None,
    ) -> CalendarRow:
        return self.db.create_calendar(
            tenant_id=self.tenant_id,
            owner_user_id=actor_user_id,
            org_id=org_id,
            name=name,
            timezone=timezone,
            color=color,
            description=description,
            visibility=visibility,
            default_reminder_policy_json=default_reminder_policy_json,
            rbac_policy_ref=rbac_policy_ref,
        )

    def list_calendars(
        self,
        *,
        actor_user_id: int,
        include_archived: bool = False,
    ) -> list[CalendarRow]:
        calendars = self.db.list_calendars(
            tenant_id=self.tenant_id,
            include_archived=include_archived,
        )
        visible: list[CalendarRow] = []
        for calendar in calendars:
            context = self._access_context(actor_user_id, calendar)
            try:
                assert_calendar_access(context, "read")
            except CalendarPermissionDenied:
                continue
            visible.append(calendar)
        return visible

    def update_calendar(
        self,
        *,
        actor_user_id: int,
        calendar_id: int,
        **updates: Any,
    ) -> CalendarRow:
        self._assert_calendar_access(actor_user_id, calendar_id, "manage")
        return self.db.update_calendar(calendar_id, updates)

    def archive_calendar(self, *, actor_user_id: int, calendar_id: int) -> CalendarRow:
        self._assert_calendar_access(actor_user_id, calendar_id, "manage")
        return self.db.archive_calendar(calendar_id)

    def add_membership(
        self,
        *,
        actor_user_id: int,
        calendar_id: int,
        principal_type: str,
        principal_id: str | int,
        role: CalendarRole,
    ) -> CalendarMembershipRow:
        self._assert_calendar_access(actor_user_id, calendar_id, "manage")
        self._validate_membership(principal_type=principal_type, role=role)
        return self.db.create_membership(
            calendar_id=calendar_id,
            principal_type=principal_type,
            principal_id=principal_id,
            role=role,
        )

    def list_memberships(
        self,
        *,
        actor_user_id: int,
        calendar_id: int,
    ) -> list[CalendarMembershipRow]:
        self._assert_calendar_access(actor_user_id, calendar_id, "manage")
        return self.db.list_memberships(calendar_id)

    def remove_membership(
        self,
        *,
        actor_user_id: int,
        calendar_id: int,
        principal_type: str,
        principal_id: str | int,
    ) -> int:
        self._assert_calendar_access(actor_user_id, calendar_id, "manage")
        if principal_type not in _VALID_PRINCIPAL_TYPES:
            raise CalendarValidationError(f"Unsupported calendar principal type: {principal_type}")
        return self.db.remove_membership(
            calendar_id=calendar_id,
            principal_type=principal_type,
            principal_id=principal_id,
        )

    def create_item(
        self,
        *,
        actor_user_id: int,
        calendar_id: int,
        kind: str,
        title: str,
        description: str | None = None,
        location: str | None = None,
        start_at: str | None = None,
        end_at: str | None = None,
        due_at: str | None = None,
        timezone: str | None = None,
        all_day: bool = False,
        status: str = "confirmed",
        local_tags_json: str | list[str] | None = None,
        metadata_json: str | dict[str, Any] | None = None,
    ) -> CalendarItemRow:
        self._validate_item_time(kind=kind, start_at=start_at, due_at=due_at)
        self._assert_calendar_access(actor_user_id, calendar_id, "write")
        return self.db.create_item(
            calendar_id=calendar_id,
            kind=kind,
            title=title,
            description=description,
            location=location,
            start_at=start_at,
            end_at=end_at,
            due_at=due_at,
            timezone=timezone,
            all_day=all_day,
            status=status,
            local_tags_json=local_tags_json,
            metadata_json=metadata_json,
        )

    def get_item(self, *, actor_user_id: int, item_id: int) -> CalendarItemRow:
        item, _ = self._item_and_context(actor_user_id, item_id)
        return item

    def update_item(
        self,
        *,
        actor_user_id: int,
        item_id: int,
        **updates: Any,
    ) -> CalendarItemRow:
        item, _ = self._item_and_context(actor_user_id, item_id)
        self._raise_if_provider_owned(item)
        self._assert_calendar_access(actor_user_id, item.calendar_id, "write")
        if "kind" in updates:
            self._validate_item_time(
                kind=str(updates["kind"]),
                start_at=updates.get("start_at", item.start_at),
                due_at=updates.get("due_at", item.due_at),
            )
        return self.db.update_item(item_id, updates)

    def delete_item(self, *, actor_user_id: int, item_id: int) -> CalendarItemRow:
        item, _ = self._item_and_context(actor_user_id, item_id)
        self._raise_if_provider_owned(item)
        self._assert_calendar_access(actor_user_id, item.calendar_id, "write")
        return self.db.soft_delete_item(item_id)

    def list_items_window(
        self,
        *,
        actor_user_id: int,
        calendar_ids: list[int],
        window_start: str,
        window_end: str,
    ) -> list[CalendarItemRow]:
        readable_calendar_ids = [
            calendar_id
            for calendar_id in calendar_ids
            if self._can_read_calendar(actor_user_id, calendar_id)
        ]
        items = self.db.list_items_window(
            calendar_ids=readable_calendar_ids,
            window_start=window_start,
            window_end=window_end,
        )
        return [
            item
            for item in items
            if self._can_read_item(actor_user_id=actor_user_id, item=item)
        ]

    def create_annotation(
        self,
        *,
        actor_user_id: int,
        item_id: int,
        body: str,
        tags: list[str] | None = None,
    ) -> CalendarAnnotationRow:
        item, _ = self._item_and_context(actor_user_id, item_id)
        self._assert_calendar_access(actor_user_id, item.calendar_id, "comment")
        return self.db.create_annotation(
            calendar_item_id=item_id,
            author_user_id=actor_user_id,
            body=body,
            tags_json=tags,
        )

    def update_annotation(
        self,
        *,
        actor_user_id: int,
        annotation_id: int,
        body: str | None = None,
        tags: list[str] | None = None,
    ) -> CalendarAnnotationRow:
        annotation = self.db.get_annotation(annotation_id)
        item, context = self._item_and_context(actor_user_id, annotation.calendar_item_id)
        self._assert_calendar_access(actor_user_id, item.calendar_id, "comment")
        if annotation.author_user_id != actor_user_id and not can_manage_calendar(context):
            raise CalendarPermissionDenied("Only annotation authors or calendar owners can edit annotations")
        patch: dict[str, Any] = {}
        if body is not None:
            patch["body"] = body
        if tags is not None:
            patch["tags_json"] = tags
        return self.db.update_annotation(annotation_id, patch)

    def delete_annotation(self, *, actor_user_id: int, annotation_id: int) -> int:
        annotation = self.db.get_annotation(annotation_id)
        item, context = self._item_and_context(actor_user_id, annotation.calendar_item_id)
        self._assert_calendar_access(actor_user_id, item.calendar_id, "comment")
        if annotation.author_user_id != actor_user_id and not can_manage_calendar(context):
            raise CalendarPermissionDenied("Only annotation authors or calendar owners can delete annotations")
        return self.db.delete_annotation(annotation_id)

    def list_annotations(
        self,
        *,
        actor_user_id: int,
        item_id: int,
    ) -> list[CalendarAnnotationRow]:
        self._item_and_context(actor_user_id, item_id)
        return self.db.list_annotations(item_id)

    def update_local_tags(
        self,
        *,
        actor_user_id: int,
        item_id: int,
        tags: list[str],
    ) -> CalendarAnnotationRow:
        item, _ = self._item_and_context(actor_user_id, item_id)
        self._assert_calendar_access(actor_user_id, item.calendar_id, "comment")
        for annotation in self.db.list_annotations(item_id):
            if annotation.author_user_id == actor_user_id and annotation.body == "":
                return self.db.update_annotation(annotation.id, tags_json=tags)
        return self.db.create_annotation(
            calendar_item_id=item_id,
            author_user_id=actor_user_id,
            body="",
            tags_json=tags,
        )

    def create_link(
        self,
        *,
        actor_user_id: int,
        item_id: int,
        target_type: str,
        target_id: str | int,
        label: str | None = None,
        url: str | None = None,
        metadata_json: str | dict[str, Any] | None = None,
    ) -> CalendarLinkRow:
        item, _ = self._item_and_context(actor_user_id, item_id)
        self._assert_calendar_access(actor_user_id, item.calendar_id, "comment")
        return self.db.create_link(
            calendar_item_id=item_id,
            target_type=target_type,
            target_id=target_id,
            label=label,
            url=url,
            metadata_json=metadata_json,
        )

    def list_links(self, *, actor_user_id: int, item_id: int) -> list[CalendarLinkRow]:
        self._item_and_context(actor_user_id, item_id)
        return self.db.list_links(item_id)

    def delete_link(self, *, actor_user_id: int, link_id: int) -> int:
        link = self.db.get_link(link_id)
        item, _ = self._item_and_context(actor_user_id, link.calendar_item_id)
        self._assert_calendar_access(actor_user_id, item.calendar_id, "write")
        return self.db.delete_link(link_id)

    def copy_provider_item(
        self,
        *,
        actor_user_id: int,
        item_id: int,
        target_calendar_id: int | None = None,
        title: str | None = None,
    ) -> CalendarItemRow:
        source, _ = self._item_and_context(actor_user_id, item_id)
        if not source.provider_owned or source.source_owner != CALENDAR_SOURCE_OWNER_PROVIDER:
            raise CalendarValidationError("Only provider-owned calendar items can be copied")
        destination_calendar_id = target_calendar_id or source.calendar_id
        self._assert_calendar_access(actor_user_id, destination_calendar_id, "write")
        return self.db.create_item(
            calendar_id=destination_calendar_id,
            kind=source.kind,
            title=title or source.title,
            description=source.description,
            location=source.location,
            start_at=source.start_at,
            end_at=source.end_at,
            due_at=source.due_at,
            timezone=source.timezone,
            all_day=source.all_day,
            status=source.status,
            local_tags_json=source.local_tags_json,
            metadata_json=source.metadata_json,
            copied_from_item_id=source.id,
        )

    def _can_read_calendar(self, actor_user_id: int, calendar_id: int) -> bool:
        try:
            self._assert_calendar_access(actor_user_id, calendar_id, "read")
        except CalendarPermissionDenied:
            return False
        return True

    def _can_read_item(self, *, actor_user_id: int, item: CalendarItemRow) -> bool:
        try:
            context = self._assert_calendar_access(actor_user_id, item.calendar_id, "read")
            self._assert_provider_item_visible(actor_user_id, item, context)
        except CalendarPermissionDenied:
            return False
        return True

    def _assert_calendar_access(
        self,
        actor_user_id: int,
        calendar_id: int,
        action: CalendarAccessAction,
    ) -> CalendarAccessContext:
        calendar = self.db.get_calendar(calendar_id, include_archived=action == "manage")
        context = self._access_context(actor_user_id, calendar)
        assert_calendar_access(context, action)
        return context

    def _item_and_context(
        self,
        actor_user_id: int,
        item_id: int,
    ) -> tuple[CalendarItemRow, CalendarAccessContext]:
        item = self.db.get_item(item_id)
        context = self._assert_calendar_access(actor_user_id, item.calendar_id, "read")
        self._assert_provider_item_visible(actor_user_id, item, context)
        return item, context

    def _access_context(
        self,
        actor_user_id: int,
        calendar: CalendarRow,
    ) -> CalendarAccessContext:
        return CalendarAccessContext(
            actor_user_id=actor_user_id,
            calendar=calendar,
            memberships=self.db.list_memberships(calendar.id),
            org_role_resolver=self.org_role_resolver,
        )

    @staticmethod
    def _raise_if_provider_owned(item: CalendarItemRow) -> None:
        if item.provider_owned or item.source_owner == CALENDAR_SOURCE_OWNER_PROVIDER:
            raise CalendarReadOnlyError("Provider-owned items are read-only")

    def _assert_provider_item_visible(
        self,
        actor_user_id: int,
        item: CalendarItemRow,
        context: CalendarAccessContext,
    ) -> None:
        if not item.provider_owned and item.source_owner != CALENDAR_SOURCE_OWNER_PROVIDER:
            return
        if actor_user_id == context.calendar.owner_user_id:
            return
        if item.external_binding_id is not None:
            binding = self.db.get_external_binding(item.external_binding_id, include_deleted=True)
            account = self.db.get_external_account(binding.account_id, include_deleted=True)
            if actor_user_id == account.user_id:
                return
        raise CalendarPermissionDenied("Provider-owned personal calendar imports are private")

    @staticmethod
    def _validate_membership(*, principal_type: str, role: str) -> None:
        if principal_type not in _VALID_PRINCIPAL_TYPES:
            raise CalendarValidationError(f"Unsupported calendar principal type: {principal_type}")
        if role not in _VALID_ROLES:
            raise CalendarValidationError(f"Unsupported calendar role: {role}")

    @staticmethod
    def _validate_item_time(*, kind: str, start_at: str | None, due_at: str | None) -> None:
        if kind == "event" and not start_at:
            raise CalendarValidationError("Calendar events require start_at")
        if kind == "todo" and not (start_at or due_at):
            raise CalendarValidationError("Calendar todos require due_at or start_at")
        if kind not in {"event", "todo"}:
            raise CalendarValidationError(f"Unsupported calendar item kind: {kind}")


__all__ = ["CalendarService"]
