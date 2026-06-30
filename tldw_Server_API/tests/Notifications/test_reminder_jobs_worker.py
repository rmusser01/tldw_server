from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.Reminders.reminder_jobs import handle_reminder_job


pytestmark = pytest.mark.unit


@pytest.fixture()
def reminder_job_env(monkeypatch, tmp_path):
    base_dir = tmp_path / "test_reminder_jobs_worker"
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


def _seed_one_time_task(*, user_id: int) -> tuple[CollectionsDatabase, str, str]:
    cdb = CollectionsDatabase.for_user(user_id=user_id)
    run_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    task_id = cdb.create_reminder_task(
        title="Check reminder",
        body="Follow up on open item",
        schedule_kind="one_time",
        run_at=run_at,
        cron=None,
        timezone="UTC",
        enabled=True,
    )
    cdb.update_reminder_task(task_id, {"next_run_at": run_at})
    return cdb, task_id, run_at


@pytest.mark.asyncio
async def test_reminder_job_creates_notification(reminder_job_env):
    user_id = 900
    cdb, task_id, run_slot = _seed_one_time_task(user_id=user_id)
    fake_job = {
        "id": 123,
        "owner_user_id": str(user_id),
        "payload": {
            "task_id": task_id,
            "user_id": user_id,
            "scheduled_for": run_slot,
        },
    }

    result = await handle_reminder_job(fake_job)

    assert result["status"] == "succeeded"
    rows = cdb.list_user_notifications(limit=10, offset=0)
    assert len(rows) == 1
    assert rows[0].kind == "reminder_due"
    assert rows[0].source_task_id == task_id

    refreshed_task = cdb.get_reminder_task(task_id)
    assert refreshed_task.enabled is False
    assert refreshed_task.last_status == "succeeded"

    run = cdb.get_reminder_task_run_by_slot(task_id=task_id, run_slot_key=run_slot)
    assert run.status == "succeeded"


@pytest.mark.asyncio
async def test_reminder_job_deduplicates_by_run_slot(reminder_job_env):
    user_id = 901
    cdb, task_id, run_slot = _seed_one_time_task(user_id=user_id)
    fake_job = {
        "id": 124,
        "owner_user_id": str(user_id),
        "payload": {
            "task_id": task_id,
            "user_id": user_id,
            "scheduled_for": run_slot,
        },
    }

    first = await handle_reminder_job(fake_job)
    second = await handle_reminder_job(fake_job)

    assert first["status"] == "succeeded"
    assert second["status"] == "succeeded"
    rows = cdb.list_user_notifications(limit=10, offset=0)
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_reminder_job_skips_disabled_task_without_notification(reminder_job_env: None) -> None:
    """Disabled queued reminder tasks are marked skipped without notifications."""
    user_id = 902
    cdb, task_id, run_slot = _seed_one_time_task(user_id=user_id)
    cdb.update_reminder_task(task_id, {"enabled": False})
    fake_job = {
        "id": 125,
        "owner_user_id": str(user_id),
        "payload": {
            "task_id": task_id,
            "user_id": user_id,
            "scheduled_for": run_slot,
        },
    }

    result = await handle_reminder_job(fake_job)

    assert result == {
        "status": "skipped",
        "reason": "task_disabled",
        "task_id": task_id,
        "run_id": result["run_id"],
    }
    assert cdb.list_user_notifications(limit=10, offset=0) == []
    run = cdb.get_reminder_task_run_by_slot(task_id=task_id, run_slot_key=run_slot)
    assert run.status == "skipped"
    assert run.error == "task_disabled"
    assert cdb.get_reminder_task(task_id).last_status == "skipped"


@pytest.mark.asyncio
async def test_reminder_job_honors_disabled_reminder_notifications(reminder_job_env: None) -> None:
    """Disabled reminder notification preferences suppress due notifications."""
    user_id = 903
    cdb, task_id, run_slot = _seed_one_time_task(user_id=user_id)
    cdb.update_notification_preferences(reminder_enabled=False)
    fake_job = {
        "id": 126,
        "owner_user_id": str(user_id),
        "payload": {
            "task_id": task_id,
            "user_id": user_id,
            "scheduled_for": run_slot,
        },
    }

    result = await handle_reminder_job(fake_job)

    assert result == {
        "status": "skipped",
        "reason": "reminder_notifications_disabled",
        "task_id": task_id,
        "run_id": result["run_id"],
    }
    assert cdb.list_user_notifications(limit=10, offset=0) == []
    run = cdb.get_reminder_task_run_by_slot(task_id=task_id, run_slot_key=run_slot)
    assert run.status == "skipped"
    assert run.error == "reminder_notifications_disabled"
    refreshed_task = cdb.get_reminder_task(task_id)
    assert refreshed_task.enabled is False
    assert refreshed_task.last_status == "skipped"


@pytest.mark.asyncio
async def test_reminder_job_skips_missing_task_without_worker_failure(reminder_job_env: None) -> None:
    """Deleted queued snooze tasks are recorded as skipped instead of failed."""
    user_id = 904
    cdb, task_id, run_slot = _seed_one_time_task(user_id=user_id)
    assert cdb.delete_reminder_task(task_id) is True
    fake_job = {
        "id": 127,
        "owner_user_id": str(user_id),
        "payload": {
            "task_id": task_id,
            "user_id": user_id,
            "scheduled_for": run_slot,
        },
    }

    result = await handle_reminder_job(fake_job)

    assert result == {
        "status": "skipped",
        "reason": "task_missing",
        "task_id": task_id,
        "run_id": result["run_id"],
    }
    assert cdb.list_user_notifications(limit=10, offset=0) == []
    run = cdb.get_reminder_task_run_by_slot(task_id=task_id, run_slot_key=run_slot)
    assert run.status == "skipped"
    assert run.error == "task_missing"
