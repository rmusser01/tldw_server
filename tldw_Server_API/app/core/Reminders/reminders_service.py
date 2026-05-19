from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from tldw_Server_API.app.core.DB_Management.Collections_DB import (
    CollectionsDatabase,
    ReminderTaskRow,
    UserNotificationRow,
)

MIN_SNOOZE_MINUTES = 1
MAX_SNOOZE_MINUTES = 10080
SNOOZE_TASK_TITLE_PREFIX = "Snoozed: "
SNOOZED_NOTIFICATIONS_SCAN_SIZE = 250


@dataclass(frozen=True)
class NotificationSnoozeMatch:
    task_ids: tuple[str, ...]
    run_at: str | None


def _build_snooze_task_title(title: str) -> str:
    return f"{SNOOZE_TASK_TITLE_PREFIX}{title}"


def _notification_snooze_signature(notification: UserNotificationRow) -> tuple[str, str, str, str, str]:
    return (
        str(notification.title or ""),
        str(notification.message or ""),
        str(notification.link_type or ""),
        str(notification.link_id or ""),
        str(notification.link_url or ""),
    )


def _task_snooze_signature(task: ReminderTaskRow) -> tuple[str, str, str, str, str] | None:
    title = str(task.title or "")
    if (
        task.schedule_kind != "one_time"
        or not task.enabled
        or not task.run_at
        or not title.startswith(SNOOZE_TASK_TITLE_PREFIX)
    ):
        return None
    return (
        title[len(SNOOZE_TASK_TITLE_PREFIX) :],
        str(task.body or ""),
        str(task.link_type or ""),
        str(task.link_id or ""),
        str(task.link_url or ""),
    )


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _select_notification_snooze_task(
    notification: UserNotificationRow,
    candidates: Sequence[ReminderTaskRow],
) -> ReminderTaskRow | None:
    if not candidates:
        return None

    dismissed_at = _parse_iso_datetime(notification.dismissed_at)
    if dismissed_at is None:
        return min(candidates, key=lambda task: (str(task.created_at or ""), str(task.run_at or ""), task.id))

    ranked_candidates: list[tuple[float, str, str, str, ReminderTaskRow]] = []
    for task in candidates:
        created_at = _parse_iso_datetime(task.created_at) or _parse_iso_datetime(task.run_at)
        delta_seconds = abs((created_at - dismissed_at).total_seconds()) if created_at else float("inf")
        ranked_candidates.append(
            (
                delta_seconds,
                str(task.created_at or ""),
                str(task.run_at or ""),
                task.id,
                task,
            )
        )
    ranked_candidates.sort(key=lambda item: item[:-1])
    return ranked_candidates[0][-1]


def _match_legacy_notification_snoozes(
    notifications: Sequence[UserNotificationRow],
    tasks: Sequence[ReminderTaskRow],
) -> dict[int, NotificationSnoozeMatch]:
    tasks_by_signature: dict[tuple[str, str, str, str, str], list[ReminderTaskRow]] = defaultdict(list)
    for task in tasks:
        signature = _task_snooze_signature(task)
        if signature is None:
            continue
        tasks_by_signature[signature].append(task)

    matches: dict[int, NotificationSnoozeMatch] = {}
    notifications_by_signature: dict[tuple[str, str, str, str, str], list[UserNotificationRow]] = defaultdict(list)
    for notification in notifications:
        notifications_by_signature[_notification_snooze_signature(notification)].append(notification)

    for signature, signature_notifications in notifications_by_signature.items():
        remaining_tasks = list(tasks_by_signature.get(signature, []))
        if not remaining_tasks:
            continue
        signature_notifications.sort(key=lambda notification: (str(notification.dismissed_at or ""), notification.id))
        for notification in signature_notifications:
            matched_task = _select_notification_snooze_task(notification, remaining_tasks)
            if matched_task is None:
                continue
            remaining_tasks = [task for task in remaining_tasks if task.id != matched_task.id]
            matches[notification.id] = NotificationSnoozeMatch(
                task_ids=(matched_task.id,),
                run_at=matched_task.run_at,
            )
    return matches


class RemindersService:
    """Domain service for user reminder task lifecycle operations."""

    def __init__(
        self,
        user_id: int | str,
        *,
        collections_db: CollectionsDatabase | None = None,
    ) -> None:
        self.user_id = int(user_id)
        self.collections = collections_db or CollectionsDatabase.for_user(self.user_id)

    def create_task(self, **kwargs: Any) -> ReminderTaskRow:
        task_id = self.collections.create_reminder_task(**kwargs)
        return self.collections.get_reminder_task(task_id)

    def list_tasks(self, *, include_disabled: bool = True) -> list[ReminderTaskRow]:
        return self.collections.list_reminder_tasks(include_disabled=include_disabled)

    def get_task(self, task_id: str) -> ReminderTaskRow:
        return self.collections.get_reminder_task(task_id)

    def update_task(self, task_id: str, patch: dict[str, Any]) -> ReminderTaskRow:
        return self.collections.update_reminder_task(task_id, patch)

    def delete_task(self, task_id: str) -> bool:
        return self.collections.delete_reminder_task(task_id)

    def _list_active_snooze_tasks(self) -> dict[str, ReminderTaskRow]:
        return {
            task.id: task
            for task in self.collections.list_reminder_tasks(include_disabled=False)
            if _task_snooze_signature(task) is not None
        }

    def _match_notification_snoozes(
        self,
        *,
        notifications: Sequence[UserNotificationRow],
        active_tasks: dict[str, ReminderTaskRow],
    ) -> dict[int, NotificationSnoozeMatch]:
        dismissed_notifications = [notification for notification in notifications if notification.dismissed_at]
        if not dismissed_notifications:
            return {}

        matches: dict[int, NotificationSnoozeMatch] = {}
        legacy_notifications: list[UserNotificationRow] = []
        for notification in dismissed_notifications:
            if notification.snooze_task_id is not None:
                if notification.snooze_task_id:
                    task = active_tasks.get(notification.snooze_task_id)
                    if task is not None and _task_snooze_signature(task) == _notification_snooze_signature(notification):
                        matches[notification.id] = NotificationSnoozeMatch(
                            task_ids=(task.id,),
                            run_at=task.run_at,
                        )
                continue
            legacy_notifications.append(notification)

        if legacy_notifications:
            legacy_matches = _match_legacy_notification_snoozes(
                notifications=legacy_notifications,
                tasks=list(active_tasks.values()),
            )
            for notification_id, match in legacy_matches.items():
                matches.setdefault(notification_id, match)

        return matches

    def list_notification_snoozes(
        self,
        *,
        notifications: Sequence[UserNotificationRow],
    ) -> dict[int, NotificationSnoozeMatch]:
        return self._match_notification_snoozes(
            notifications=notifications,
            active_tasks=self._list_active_snooze_tasks(),
        )

    def list_snoozed_notifications(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[UserNotificationRow], dict[int, NotificationSnoozeMatch], int]:
        bounded_limit = max(1, int(limit))
        bounded_offset = max(0, int(offset))
        batch_size = max(SNOOZED_NOTIFICATIONS_SCAN_SIZE, bounded_limit)
        active_tasks = self._list_active_snooze_tasks()
        if not active_tasks:
            return [], {}, 0

        rows: list[UserNotificationRow] = []
        matches_for_rows: dict[int, NotificationSnoozeMatch] = {}
        total = 0
        scan_offset = 0

        while True:
            dismissed_rows = self.collections.list_user_dismissed_notifications(
                limit=batch_size,
                offset=scan_offset,
            )
            if not dismissed_rows:
                break

            dismissed_matches = self._match_notification_snoozes(
                notifications=dismissed_rows,
                active_tasks=active_tasks,
            )
            for row in dismissed_rows:
                match = dismissed_matches.get(row.id)
                if match is None:
                    continue
                if total >= bounded_offset and len(rows) < bounded_limit:
                    rows.append(row)
                    matches_for_rows[row.id] = match
                total += 1

            scan_offset += len(dismissed_rows)
            if len(dismissed_rows) < batch_size:
                break

        return rows, matches_for_rows, total

    def cancel_notification_snooze(self, *, notification_id: int) -> list[str]:
        notification = self.collections.get_user_notification(notification_id)
        match = self.list_notification_snoozes(notifications=[notification]).get(notification.id)
        if match is None:
            self.collections.set_user_notification_snooze_task(notification_id, "")
            return []

        deleted_task_ids: list[str] = []
        for task_id in match.task_ids:
            if self.collections.delete_reminder_task(task_id):
                deleted_task_ids.append(task_id)
        self.collections.set_user_notification_snooze_task(notification_id, "")
        return deleted_task_ids

    def snooze_notification(self, *, notification_id: int, minutes: int = 30) -> ReminderTaskRow:
        if minutes < MIN_SNOOZE_MINUTES or minutes > MAX_SNOOZE_MINUTES:
            raise ValueError(
                f"minutes must be between {MIN_SNOOZE_MINUTES} and {MAX_SNOOZE_MINUTES}"
            )

        source = self.collections.get_user_notification(notification_id)
        run_at = (datetime.now(timezone.utc) + timedelta(minutes=minutes)).isoformat()
        task_id = self.collections.create_reminder_task(
            title=_build_snooze_task_title(source.title),
            body=source.message,
            schedule_kind="one_time",
            run_at=run_at,
            cron=None,
            timezone="UTC",
            enabled=True,
            link_type=source.link_type,
            link_id=source.link_id,
            link_url=source.link_url,
        )
        self.collections.set_user_notification_snooze_task(notification_id, task_id)
        self.collections.dismiss_user_notification(notification_id)
        return self.collections.get_reminder_task(task_id)
