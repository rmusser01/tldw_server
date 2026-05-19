from __future__ import annotations

import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase


pytestmark = pytest.mark.unit


@pytest.fixture()
def collections_db(monkeypatch: pytest.MonkeyPatch) -> CollectionsDatabase:
    base_dir = Path.cwd() / "Databases" / "test_user_dbs_reminders_notifications"
    shutil.rmtree(base_dir, ignore_errors=True)
    base_dir.mkdir(parents=True, exist_ok=True)
    prev_base_dir = settings.get("USER_DB_BASE_DIR")
    settings.USER_DB_BASE_DIR = str(base_dir)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))

    try:
        yield CollectionsDatabase.for_user(user_id=778)
    finally:
        if prev_base_dir is not None:
            settings.USER_DB_BASE_DIR = prev_base_dir
        else:
            try:
                del settings.USER_DB_BASE_DIR
            except AttributeError:
                pass


def test_create_and_list_reminder_task(collections_db: CollectionsDatabase) -> None:
    task_id = collections_db.create_reminder_task(
        title="Ping",
        body="Re-check this thread",
        schedule_kind="one_time",
        run_at="2026-03-01T10:00:00+00:00",
        cron=None,
        timezone=None,
        enabled=True,
    )

    rows = collections_db.list_reminder_tasks()
    assert any(row.id == task_id for row in rows)


def test_notification_unread_mark_read_and_dismiss(collections_db: CollectionsDatabase) -> None:
    n1 = collections_db.create_user_notification(
        kind="reminder_due",
        title="Reminder 1",
        message="Do the thing",
        severity="info",
    )
    n2 = collections_db.create_user_notification(
        kind="job_completed",
        title="Job done",
        message="Background job completed",
        severity="info",
    )

    assert collections_db.count_unread_user_notifications() == 2

    updated = collections_db.mark_user_notifications_read([n1.id])
    assert updated == 1
    assert collections_db.count_unread_user_notifications() == 1

    dismissed = collections_db.dismiss_user_notification(n2.id)
    assert dismissed is True
    assert collections_db.count_unread_user_notifications() == 0

    listed = collections_db.list_user_notifications()
    assert all(row.id != n2.id for row in listed)

    listed_after = collections_db.list_user_notifications_after_id(after_id=0)
    assert all(row.id != n2.id for row in listed_after)


def test_create_user_notification_returns_existing_row_for_duplicate_dedupe_key(
    collections_db: CollectionsDatabase,
) -> None:
    first = collections_db.create_user_notification(
        kind="job_completed",
        title="Job done",
        message="Background job completed",
        severity="info",
        dedupe_key="jobs-event:123",
    )

    second = collections_db.create_user_notification(
        kind="job_completed",
        title="Job done",
        message="Background job completed",
        severity="info",
        dedupe_key="jobs-event:123",
    )

    assert second.id == first.id
    rows = collections_db.list_user_notifications(limit=10, offset=0)
    assert [row.id for row in rows] == [first.id]


def test_notification_preferences_defaults_and_update(collections_db: CollectionsDatabase) -> None:
    prefs = collections_db.get_notification_preferences()
    assert prefs.reminder_enabled is True
    assert prefs.job_completed_enabled is True
    assert prefs.job_failed_enabled is True

    updated = collections_db.update_notification_preferences(
        reminder_enabled=False,
        job_completed_enabled=True,
        job_failed_enabled=False,
    )
    assert updated.reminder_enabled is False
    assert updated.job_completed_enabled is True
    assert updated.job_failed_enabled is False


def test_update_notification_delivery_status_returns_boolean(collections_db: CollectionsDatabase) -> None:
    row = collections_db.create_user_notification(
        kind="job_completed",
        title="Job done",
        message="Background job completed",
        severity="info",
    )

    delivered_at = "2026-03-01T10:00:00+00:00"
    updated = collections_db.update_notification_delivery_status(
        row.id,
        "delivered",
        delivered_at=delivered_at,
    )

    assert updated is True

    refreshed = collections_db.get_user_notification(row.id)
    assert refreshed.delivery_status == "delivered"
    assert refreshed.delivered_at == delivered_at

    missing = collections_db.update_notification_delivery_status(999999, "failed")
    assert missing is False


def test_update_notification_delivery_status_can_clear_existing_timestamp(
    collections_db: CollectionsDatabase,
) -> None:
    row = collections_db.create_user_notification(
        kind="job_completed",
        title="Job done",
        message="Background job completed",
        severity="info",
    )

    collections_db.update_notification_delivery_status(
        row.id,
        "delivered",
        delivered_at="2026-03-01T10:00:00+00:00",
    )
    cleared = collections_db.update_notification_delivery_status(
        row.id,
        "pending",
        delivered_at=None,
    )

    assert cleared is True
    refreshed = collections_db.get_user_notification(row.id)
    assert refreshed.delivery_status == "pending"
    assert refreshed.delivered_at is None


def test_prune_user_notifications_archives_and_deletes(collections_db: CollectionsDatabase) -> None:
    row = collections_db.create_user_notification(
        kind="reminder_due",
        title="Reminder 1",
        message="Do the thing",
        severity="info",
    )
    now = datetime.now(timezone.utc)
    collections_db.backend.execute(
        "UPDATE user_notifications SET created_at = ?, read_at = ? WHERE id = ? AND user_id = ?",
        (
            (now - timedelta(days=10)).isoformat(),
            (now - timedelta(days=31)).isoformat(),
            row.id,
            collections_db.user_id,
        ),
    )
    archived, deleted = collections_db.prune_user_notifications(
        retention_days_by_kind={"reminder_due": 90},
        read_dismissed_grace_days=30,
        archive_grace_days=7,
    )
    assert archived == 1
    assert deleted == 0

    collections_db.backend.execute(
        "UPDATE user_notifications SET archived_at = ? WHERE id = ? AND user_id = ?",
        ((now - timedelta(days=8)).isoformat(), row.id, collections_db.user_id),
    )
    archived2, deleted2 = collections_db.prune_user_notifications(
        retention_days_by_kind={"reminder_due": 90},
        read_dismissed_grace_days=30,
        archive_grace_days=7,
    )
    assert archived2 == 0
    assert deleted2 == 1
