from __future__ import annotations

import asyncio
import contextlib
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


def test_enumerate_user_ids_sanitizes_single_user_fallback_failure(monkeypatch, tmp_path):
    secret_path = "/tmp/private/single-user-token-xyz"
    records: list[dict[str, object]] = []
    sink_id = logger.add(
        lambda message: records.append(message.record),
        level="DEBUG",
        format="{message}",
    )

    monkeypatch.setattr(
        notifications_prune_service.DatabasePaths,
        "get_user_db_base_dir",
        lambda: tmp_path,
    )

    def fail_single_user_id():
        raise RuntimeError(f"cannot derive {secret_path}")

    monkeypatch.setattr(
        notifications_prune_service.DatabasePaths,
        "get_single_user_id",
        fail_single_user_id,
    )
    try:
        assert notifications_prune_service._enumerate_user_ids() == []
    finally:
        logger.remove(sink_id)

    matching = [
        record
        for record in records
        if "notifications_prune: failed to derive single user id" in record["message"]
    ]
    assert matching
    assert all(secret_path not in record["message"] for record in matching)
    assert matching[-1]["extra"]["error_type"] == "RuntimeError"


def test_int_env_sanitizes_invalid_raw_value(monkeypatch):
    secret_value = "not-an-int /tmp/private-notification-token sk-live-notify"
    records: list[dict[str, object]] = []
    sink_id = logger.add(
        lambda message: records.append(message.record),
        level="DEBUG",
        format="{message}",
    )
    monkeypatch.setenv("NOTIFICATIONS_PRUNE_INTERVAL_SEC", secret_value)

    try:
        assert notifications_prune_service._int_env("NOTIFICATIONS_PRUNE_INTERVAL_SEC", 3600) == 3600
    finally:
        logger.remove(sink_id)

    matching = [
        record
        for record in records
        if "notifications_prune: invalid NOTIFICATIONS_PRUNE_INTERVAL_SEC" in record["message"]
    ]
    assert matching
    rendered = "\n".join(record["message"] for record in matching)
    assert secret_value not in rendered
    assert "/tmp/private-notification-token" not in rendered
    assert "sk-live-notify" not in rendered
    assert "defaulting to 3600" in rendered
    assert matching[-1]["extra"]["error_type"] == "ValueError"


@pytest.mark.asyncio
async def test_notifications_prune_runner_failure_log_is_sanitized(monkeypatch):
    secret_path = "/tmp/private/notifications-run-secret"
    records: list[dict[str, object]] = []
    sink_id = logger.add(
        lambda message: records.append(message.record),
        level="DEBUG",
        format="{message}",
    )

    monkeypatch.setenv("NOTIFICATIONS_PRUNE_ENABLED", "true")
    monkeypatch.setenv("NOTIFICATIONS_PRUNE_INTERVAL_SEC", "1")

    async def _fail_run_once(self, *, user_ids=None):
        raise RuntimeError(f"cannot prune {secret_path}")

    sleep_calls = {"count": 0}

    async def _fake_sleep(_seconds: float) -> None:
        sleep_calls["count"] += 1
        if sleep_calls["count"] >= 2:
            raise asyncio.CancelledError

    monkeypatch.setattr(NotificationsPruneService, "run_once", _fail_run_once)
    monkeypatch.setattr(notifications_prune_service.asyncio, "sleep", _fake_sleep)

    try:
        task = await notifications_prune_service.start_notifications_prune_scheduler()
        assert task is not None
        with contextlib.suppress(asyncio.CancelledError):
            await task
    finally:
        logger.remove(sink_id)

    matching = [
        record
        for record in records
        if "Notifications prune run failed" in record["message"]
    ]
    assert matching
    rendered = "\n".join(record["message"] for record in matching)
    assert "cannot prune" not in rendered
    assert secret_path not in rendered
    assert matching[-1]["extra"]["error_type"] == "RuntimeError"
