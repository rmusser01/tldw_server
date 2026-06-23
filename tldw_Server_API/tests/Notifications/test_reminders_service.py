from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from typing import NoReturn

import pytest

from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase, UserNotificationRow
from tldw_Server_API.app.core.Reminders.reminders_service import RemindersService
from tldw_Server_API.app.core.config import settings


@pytest.fixture()
def reminders_service(monkeypatch, tmp_path):
    base_dir = tmp_path / "test_reminders_service"
    base_dir.mkdir(parents=True, exist_ok=True)
    prev_base_dir = settings.get("USER_DB_BASE_DIR")
    settings.USER_DB_BASE_DIR = str(base_dir)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))

    try:
        yield RemindersService(user_id=777)
    finally:
        if prev_base_dir is not None:
            settings.USER_DB_BASE_DIR = prev_base_dir
        else:
            try:
                del settings.USER_DB_BASE_DIR
            except AttributeError:
                pass


def _seed_notification(user_id: int = 777) -> int:
    cdb = CollectionsDatabase.for_user(user_id=user_id)
    row = cdb.create_user_notification(
        kind="reminder_due",
        title="Review docs",
        message="Re-check design assumptions",
        severity="info",
        link_type="item",
        link_id="item-12",
    )
    return row.id


def test_snooze_notification_creates_one_time_task(reminders_service):
    notification_id = _seed_notification()

    task = reminders_service.snooze_notification(notification_id=notification_id, minutes=30)

    assert task.schedule_kind == "one_time"
    assert task.timezone == "UTC"
    assert task.link_type == "item"
    assert task.link_id == "item-12"
    assert task.title.startswith("Snoozed:")
    assert task.run_at is not None
    run_at = datetime.fromisoformat(task.run_at)
    delta_seconds = (run_at - datetime.now(timezone.utc)).total_seconds()
    assert 20 * 60 <= delta_seconds <= 40 * 60


@pytest.mark.parametrize("minutes", [0, 10081])
def test_snooze_notification_rejects_invalid_minutes(reminders_service, minutes: int):
    notification_id = _seed_notification()
    with pytest.raises(ValueError):
        reminders_service.snooze_notification(notification_id=notification_id, minutes=minutes)


def test_snooze_notification_raises_for_missing_notification(reminders_service):
    with pytest.raises(KeyError):
        reminders_service.snooze_notification(notification_id=999999, minutes=10)


def test_snooze_notification_replaces_existing_active_snooze(reminders_service: RemindersService) -> None:
    """Re-snoozing replaces the previous active snooze task."""
    notification_id = _seed_notification()

    first = reminders_service.snooze_notification(notification_id=notification_id, minutes=15)
    second = reminders_service.snooze_notification(notification_id=notification_id, minutes=45)

    assert second.id != first.id
    with pytest.raises(KeyError):
        reminders_service.collections.get_reminder_task(first.id)
    assert reminders_service.collections.get_user_notification(notification_id).snooze_task_id == second.id
    active_tasks = reminders_service.collections.list_reminder_tasks(include_disabled=False)
    assert [task.id for task in active_tasks] == [second.id]


def test_explicit_snooze_link_survives_task_title_body_edits(reminders_service: RemindersService) -> None:
    """Explicit snooze links survive user edits to reminder title/body."""
    notification_id = _seed_notification()
    task = reminders_service.snooze_notification(notification_id=notification_id, minutes=30)
    reminders_service.collections.update_reminder_task(
        task.id,
        {
            "title": "User edited reminder title",
            "body": "User edited reminder body",
        },
    )

    notification = reminders_service.collections.get_user_notification(notification_id)
    matches = reminders_service.list_notification_snoozes(notifications=[notification])

    assert matches[notification_id].task_ids == (task.id,)
    assert matches[notification_id].run_at == task.run_at
    assert reminders_service.cancel_notification_snooze(notification_id=notification_id) == [task.id]
    with pytest.raises(KeyError):
        reminders_service.collections.get_reminder_task(task.id)


def test_list_snoozed_notifications_uses_direct_snooze_links(
    reminders_service: RemindersService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Current-format snoozes use direct links without scanning legacy rows."""
    notification_id = _seed_notification()
    task = reminders_service.snooze_notification(notification_id=notification_id, minutes=30)
    unrelated_task_id = reminders_service.collections.create_reminder_task(
        title="Unrelated one-time reminder",
        body="Do not query this as a snooze",
        schedule_kind="one_time",
        run_at=task.run_at,
        cron=None,
        timezone="UTC",
        enabled=True,
    )
    reminders_service.collections.update_reminder_task(unrelated_task_id, {"next_run_at": task.run_at})
    queried_task_ids: set[str] = set()
    original_direct_lookup = reminders_service.collections.list_user_notifications_by_snooze_task_ids

    def _capture_direct_lookup(task_ids: Iterable[str]) -> list[UserNotificationRow]:
        """Capture direct lookup IDs while preserving DB behavior."""
        task_id_list = list(task_ids)
        queried_task_ids.update(task_id_list)
        return original_direct_lookup(task_id_list)

    def _unexpected_legacy_scan(*_args: object, **_kwargs: object) -> NoReturn:
        """Fail if the current-format snooze path falls back to legacy scanning."""
        raise AssertionError("legacy dismissed notification scan should not be required")

    monkeypatch.setattr(
        reminders_service.collections,
        "list_user_notifications_by_snooze_task_ids",
        _capture_direct_lookup,
    )
    monkeypatch.setattr(
        reminders_service.collections,
        "list_user_legacy_snooze_candidate_notifications",
        _unexpected_legacy_scan,
    )

    rows, matches, total = reminders_service.list_snoozed_notifications(limit=10, offset=0)

    assert total == 1
    assert [row.id for row in rows] == [notification_id]
    assert matches[notification_id].task_ids == (task.id,)
    assert matches[notification_id].run_at == task.run_at
    assert queried_task_ids == {task.id}
