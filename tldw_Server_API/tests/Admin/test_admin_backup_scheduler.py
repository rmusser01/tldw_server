from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit


@pytest.fixture()
async def backup_scheduler_repo(tmp_path, monkeypatch: pytest.MonkeyPatch, fresh_services):
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.repos.backup_schedules_repo import (
        AuthnzBackupSchedulesRepo,
    )
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    db_path = tmp_path / "users_backup_scheduler.db"
    jobs_db_path = tmp_path / "jobs_backup_scheduler.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("JOBS_DB_PATH", str(jobs_db_path))

    reset_settings()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(str(db_path)))

    repo = AuthnzBackupSchedulesRepo(pool)
    await repo.ensure_schema()
    return repo


@pytest.mark.asyncio
async def test_due_schedule_enqueues_job_once_and_updates_schedule(backup_scheduler_repo, monkeypatch):
    from tldw_Server_API.app.core.Storage.backup_schedule_jobs import (
        BACKUP_SCHEDULE_DOMAIN,
        BACKUP_SCHEDULE_JOB_TYPE,
        backup_schedule_queue,
    )
    from tldw_Server_API.app.services.admin_backup_scheduler import _AdminBackupScheduler

    due = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    created = await backup_scheduler_repo.create_schedule(
        dataset="authnz",
        target_user_id=None,
        frequency="daily",
        time_of_day="02:00",
        timezone="UTC",
        anchor_day_of_week=None,
        anchor_day_of_month=None,
        retention_count=30,
        created_by_user_id=1,
        updated_by_user_id=1,
        next_run_at=due.isoformat(),
    )

    scheduler = _AdminBackupScheduler(repo=backup_scheduler_repo)
    captured: list[dict[str, object]] = []

    def _capture_create_job(**kwargs):
        captured.append(kwargs)
        return {"id": "job-1"}

    monkeypatch.setattr(scheduler._jobs, "create_job", _capture_create_job)

    await scheduler._run_schedule(str(created["id"]))

    assert len(captured) == 1
    assert captured[0]["domain"] == BACKUP_SCHEDULE_DOMAIN
    assert captured[0]["queue"] == backup_schedule_queue()
    assert captured[0]["job_type"] == BACKUP_SCHEDULE_JOB_TYPE
    assert captured[0]["owner_user_id"] is None

    updated = await backup_scheduler_repo.get_schedule(str(created["id"]))
    assert updated is not None
    assert updated["last_status"] == "queued"
    assert updated["last_job_id"] == "job-1"
    assert updated["next_run_at"] is not None
    assert datetime.fromisoformat(updated["next_run_at"]) > due


@pytest.mark.asyncio
async def test_due_schedule_concurrent_schedulers_enqueue_single_job(backup_scheduler_repo):
    from tldw_Server_API.app.core.Jobs.manager import JobManager
    from tldw_Server_API.app.core.Storage.backup_schedule_jobs import (
        BACKUP_SCHEDULE_DOMAIN,
        BACKUP_SCHEDULE_JOB_TYPE,
    )
    from tldw_Server_API.app.services.admin_backup_scheduler import _AdminBackupScheduler

    due = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    created = await backup_scheduler_repo.create_schedule(
        dataset="authnz",
        target_user_id=None,
        frequency="daily",
        time_of_day="02:00",
        timezone="UTC",
        anchor_day_of_week=None,
        anchor_day_of_month=None,
        retention_count=30,
        created_by_user_id=1,
        updated_by_user_id=1,
        next_run_at=due.isoformat(),
    )

    sched_a = _AdminBackupScheduler(repo=backup_scheduler_repo)
    sched_b = _AdminBackupScheduler(repo=backup_scheduler_repo)

    await asyncio.gather(
        sched_a._run_schedule(str(created["id"])),
        sched_b._run_schedule(str(created["id"])),
    )

    jobs = JobManager().list_jobs(domain=BACKUP_SCHEDULE_DOMAIN, job_type=BACKUP_SCHEDULE_JOB_TYPE)
    assert len(jobs) == 1


@pytest.mark.asyncio
async def test_due_schedule_enqueue_failure_advances_next_run_and_reschedules(
    backup_scheduler_repo,
    monkeypatch,
):
    from tldw_Server_API.app.services.admin_backup_scheduler import _AdminBackupScheduler

    due = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    created = await backup_scheduler_repo.create_schedule(
        dataset="authnz",
        target_user_id=None,
        frequency="daily",
        time_of_day="02:00",
        timezone="UTC",
        anchor_day_of_week=None,
        anchor_day_of_month=None,
        retention_count=30,
        created_by_user_id=1,
        updated_by_user_id=1,
        next_run_at=due.isoformat(),
    )

    scheduler = _AdminBackupScheduler(repo=backup_scheduler_repo)

    def _failing_create_job(**kwargs):
        raise RuntimeError("queue write failed")

    scheduled_ids: list[str] = []

    def _capture_add(item):
        scheduled_ids.append(str(item["id"]))

    monkeypatch.setattr(scheduler._jobs, "create_job", _failing_create_job)
    monkeypatch.setattr(scheduler, "_add_job", _capture_add)

    await scheduler._run_schedule(str(created["id"]))

    updated = await backup_scheduler_repo.get_schedule(str(created["id"]))
    assert updated is not None
    assert updated["last_status"] == "error"
    assert updated["last_error"] == "queue write failed"
    assert updated["next_run_at"] is not None
    assert datetime.fromisoformat(updated["next_run_at"]) > due
    assert scheduled_ids == [str(created["id"])]


@pytest.mark.asyncio
async def test_rescan_skips_paused_schedule(backup_scheduler_repo, monkeypatch):
    from tldw_Server_API.app.services.admin_backup_scheduler import _AdminBackupScheduler

    created = await backup_scheduler_repo.create_schedule(
        dataset="authnz",
        target_user_id=None,
        frequency="daily",
        time_of_day="02:00",
        timezone="UTC",
        anchor_day_of_week=None,
        anchor_day_of_month=None,
        retention_count=30,
        created_by_user_id=1,
        updated_by_user_id=1,
        next_run_at=(datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
    )
    await backup_scheduler_repo.pause_schedule(str(created["id"]), updated_by_user_id=1)

    scheduler = _AdminBackupScheduler(repo=backup_scheduler_repo)

    class _FakeAPS:
        def __init__(self) -> None:
            self.removed: list[str] = []

        def get_jobs(self):
            return []

        def remove_job(self, job_id: str) -> None:
            self.removed.append(job_id)

    scheduler._aps = _FakeAPS()
    scheduler._started = True

    added: list[str] = []

    def _capture_add(item):
        added.append(str(item["id"]))

    monkeypatch.setattr(scheduler, "_add_job", _capture_add)

    await scheduler._rescan_once()

    assert added == []


@pytest.mark.asyncio
async def test_rescan_removes_deleted_schedule_from_registry(backup_scheduler_repo):
    from tldw_Server_API.app.services.admin_backup_scheduler import _AdminBackupScheduler

    created = await backup_scheduler_repo.create_schedule(
        dataset="authnz",
        target_user_id=None,
        frequency="daily",
        time_of_day="02:00",
        timezone="UTC",
        anchor_day_of_week=None,
        anchor_day_of_month=None,
        retention_count=30,
        created_by_user_id=1,
        updated_by_user_id=1,
        next_run_at=(datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
    )
    await backup_scheduler_repo.delete_schedule(
        str(created["id"]),
        deleted_at=datetime.now(timezone.utc).isoformat(),
    )

    scheduler = _AdminBackupScheduler(repo=backup_scheduler_repo)

    class _FakeJob:
        def __init__(self, job_id: str) -> None:
            self.id = job_id

    class _FakeAPS:
        def __init__(self, schedule_id: str) -> None:
            self._jobs = [_FakeJob(schedule_id)]
            self.removed: list[str] = []

        def get_jobs(self):
            return list(self._jobs)

        def remove_job(self, job_id: str) -> None:
            self.removed.append(job_id)

    fake_aps = _FakeAPS(str(created["id"]))
    scheduler._aps = fake_aps
    scheduler._started = True

    await scheduler._rescan_once()

    assert fake_aps.removed == [str(created["id"])]
