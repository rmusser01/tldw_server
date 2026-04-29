from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from loguru import logger

from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.services import notifications_prune_service
from tldw_Server_API.app.services.notifications_prune_service import NotificationsPruneService


pytestmark = pytest.mark.unit


@pytest.fixture()
def notifications_base(monkeypatch, tmp_path):
    base_dir = tmp_path / "test_notifications_prune"
    base_dir.mkdir(parents=True, exist_ok=True)
    prev_base_dir = settings.get("USER_DB_BASE_DIR")
    settings.USER_DB_BASE_DIR = str(base_dir)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))
    try:
        yield
    finally:
        if prev_base_dir is not None:
            settings.USER_DB_BASE_DIR = prev_base_dir
        else:
            try:
                del settings.USER_DB_BASE_DIR
            except AttributeError:
                pass


def _update_notification_fields(cdb: CollectionsDatabase, notification_id: int, **fields: str | None) -> None:
    if not fields:
        return
    sets = ", ".join([f"{k} = ?" for k in fields])  # nosec B608
    values = list(fields.values())
    cdb.backend.execute(
        f"UPDATE user_notifications SET {sets} WHERE id = ? AND user_id = ?",  # nosec B608
        tuple(values + [notification_id, cdb.user_id]),
    )


@pytest.mark.asyncio
async def test_prune_archives_due_notification_by_default_retention(notifications_base):
    cdb = CollectionsDatabase.for_user(user_id=990)
    row = cdb.create_user_notification(
        kind="reminder_due",
        title="Reminder",
        message="Follow up",
        severity="info",
    )
    old_created = (datetime.now(timezone.utc) - timedelta(days=91)).isoformat()
    _update_notification_fields(cdb, row.id, created_at=old_created)

    svc = NotificationsPruneService()
    summary = await svc.run_once_for_user(user_id=990)

    assert summary["archived"] == 1
    assert summary["deleted"] == 0
    refreshed = cdb.get_user_notification(row.id)
    assert refreshed.archived_at is not None


@pytest.mark.asyncio
async def test_prune_deletes_archived_notification_after_grace(notifications_base):
    cdb = CollectionsDatabase.for_user(user_id=991)
    row = cdb.create_user_notification(
        kind="job_completed",
        title="Done",
        message="Finished",
        severity="info",
    )
    archived_at = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
    _update_notification_fields(cdb, row.id, archived_at=archived_at)

    svc = NotificationsPruneService()
    summary = await svc.run_once_for_user(user_id=991)

    assert summary["archived"] == 0
    assert summary["deleted"] == 1
    with pytest.raises(KeyError):
        cdb.get_user_notification(row.id)


@pytest.mark.asyncio
async def test_prune_uses_read_acceleration_window(notifications_base):
    cdb = CollectionsDatabase.for_user(user_id=992)
    row = cdb.create_user_notification(
        kind="reminder_due",
        title="Review",
        message="Review item",
        severity="info",
    )
    now = datetime.now(timezone.utc)
    _update_notification_fields(
        cdb,
        row.id,
        created_at=(now - timedelta(days=10)).isoformat(),
        read_at=(now - timedelta(days=31)).isoformat(),
    )

    svc = NotificationsPruneService()
    summary = await svc.run_once_for_user(user_id=992)

    assert summary["archived"] == 1
    assert summary["deleted"] == 0


def test_enumerate_user_ids_sanitizes_base_dir_resolution_failure(monkeypatch):
    secret_path = "/tmp/private/user-db-token-abc123"
    records: list[dict[str, object]] = []
    sink_id = logger.add(
        lambda message: records.append(message.record),
        level="DEBUG",
        format="{message}",
    )

    def fail_base_dir():
        raise RuntimeError(f"cannot inspect {secret_path}")

    monkeypatch.setattr(
        notifications_prune_service.DatabasePaths,
        "get_user_db_base_dir",
        fail_base_dir,
    )
    try:
        assert notifications_prune_service._enumerate_user_ids() == []
    finally:
        logger.remove(sink_id)

    matching = [
        record
        for record in records
        if "notifications_prune: failed to resolve user db base dir" in record["message"]
    ]
    assert matching
    assert all(secret_path not in record["message"] for record in matching)
    assert matching[-1]["extra"]["error_type"] == "RuntimeError"
